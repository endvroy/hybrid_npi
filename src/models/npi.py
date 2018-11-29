import torch
from torch import nn
import torch.nn.functional as F
from models.npi_core import NPICore


class PKeyMem(nn.Module):
    def __init__(self, n_progs, pkey_dim, n_act=1):
        """
        The memory is always organized as [ACTs, PROGs]
        """
        super(PKeyMem, self).__init__()
        if n_act >= n_progs:
            raise ValueError(
                'program memory of size {} is not enough for {} ACTs'.format(n_progs, n_act))
        self.n_progs = n_progs
        self.pkey_dim = pkey_dim
        self.pkey_mem = nn.Parameter(torch.randn(n_progs, pkey_dim))
        self.n_act = n_act

    def is_act(self, prog_id):
        return prog_id < self.n_act

    def calc_correlation_scores(self, pkey):
        return (pkey.unsqueeze(1) @ self.pkey_mem.t()).squeeze(1)


class NPI(nn.Module):
    def __init__(self,
                 core,
                 task,
                 pkey_mem,
                 ret_threshold,
                 n_progs,
                 prog_dim):
        super(NPI, self).__init__()
        self.core = core
        self.task = task
        self.ret_threshold = ret_threshold
        self.n_progs = n_progs
        self.prog_dim = prog_dim
        self.pkey_mem = pkey_mem
        self.prog_mem = nn.Parameter(torch.randn(n_progs, prog_dim))

    def forward(self, prog_id, args):
        state = self.task.f_enc(args)
        prog = self.prog_mem[prog_id]
        ret, pkey, new_args = self.core(state, prog)
        scores = self.pkey_mem.calc_correlation_scores(pkey)
        prog_id_log_probs = F.log_softmax(scores, dim=1)  # log softmax is more numerically stable
        return ret, prog_id_log_probs, new_args

    def run(self, prog_id, args):
        self.core.reset()
        ret = 0
        stack = [(ret, prog_id, args)]
        while stack:
            ret, prog_id, args = stack.pop()
            while ret < self.ret_threshold:
                ret, prog_id_log_probs, args = self.forward(prog_id, args)
                prog_id = torch.argmax(prog_id_log_probs, dim=1)

                # return probability, NEXT program id, NEXT program args
                yield ret, prog_id, args
                if self.pkey_mem.is_act(prog_id):
                    self.task.f_env(prog_id, args)
                else:
                    stack.append((ret, prog_id, args))
                    ret = 0


def npi_factory(task,
                state_dim,  # state tensor dimension
                n_progs,  # number of programs
                prog_dim,  # program embedding dimension
                hidden_dim,  # LSTM hidden dimension (in core)
                n_lstm_layers,  # number of LSTM layers (in core)
                ret_threshold,  # return probability threshold
                pkey_dim,  # program key dimension
                args_dim,  # argument vector dimension
                n_act=1):  # num of ACTs
    core = NPICore(state_dim=state_dim,
                   prog_dim=prog_dim,
                   hidden_dim=hidden_dim,
                   n_lstm_layers=n_lstm_layers,
                   pkey_dim=pkey_dim,
                   args_dim=args_dim)

    pkey_mem = PKeyMem(n_progs=n_progs,
                       pkey_dim=pkey_dim,
                       n_act=n_act)

    model = NPI(core=core,
                task=task,
                pkey_mem=pkey_mem,
                ret_threshold=ret_threshold,
                n_progs=n_progs,
                prog_dim=prog_dim)

    return model


if __name__ == '__main__':
    import random
    import sys
    from ..tasks.task_base import TaskBase  # todo: fix

    seed = random.randrange(sys.maxsize)
    print('seed= {}'.format(seed))
    torch.manual_seed(seed)
    # good seeds: 6478152654801362860

    state_dim = 2
    args_dim = 3


    class DummyTask(TaskBase):
        def __init__(self, env, state_dim):
            super(DummyTask, self).__init__(env, state_dim)

        def f_enc(self, args):
            return torch.randn(args.size(0), self.state_dim)

        def f_env(self, prog_id, args):
            self.env = torch.randn(prog_id.size(0), self.batch_size)


    dummy_task = DummyTask(42, state_dim)

    model = npi_factory(task=dummy_task,
                        state_dim=state_dim,
                        n_progs=3,
                        prog_dim=5,
                        hidden_dim=3,
                        n_lstm_layers=2,
                        ret_threshold=0.38,
                        pkey_dim=4,
                        args_dim=args_dim)

    ret, prog_id_probs, new_args = model(torch.tensor([1, 2]), torch.randn(2, args_dim))
    it = model.run(torch.tensor([1]), torch.randn(1, args_dim))
    for x in it:
        print(x, dummy_task.env)
    print(dummy_task.env)
    # or run with next(it)
