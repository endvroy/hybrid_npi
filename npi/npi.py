import torch
from torch import nn
import torch.nn.functional as F
from npi_core import NPICore


class NPI(nn.Module):
    def __init__(self,
                 core,
                 f_enc,
                 f_env,
                 n_progs,
                 ret_threshold,
                 pkey_dim,
                 prog_dim):
        super(NPI, self).__init__()
        self.core = core
        self.f_enc = f_enc
        self.f_env = f_env
        self.n_progs = n_progs
        self.ret_threshold = ret_threshold
        self.prog_dim = prog_dim
        self.pkey_mem = nn.Parameter(torch.randn(n_progs, pkey_dim))
        self.prog_mem = nn.Parameter(torch.randn(n_progs, prog_dim))
        self.act = 0  # ACT always has prog_id = 0

    def forward(self, env, prog_id, args):
        state = self.f_enc(env, args)
        prog = self.prog_mem[prog_id]
        ret, pkey, new_args = self.core(state, prog)
        scores = (self.pkey_mem @ pkey.t()).view(-1)
        prog_id_log_probs = F.softmax(scores, dim=0)
        return ret, prog_id_log_probs, new_args

    def run(self, env, prog_id, args):
        self.core.reset()
        ret = 0
        while ret < self.ret_threshold:
            ret, prog_id_log_probs, args = self.forward(env, prog_id, args)
            prog_id = torch.argmax(prog_id_log_probs)
            # todo: yield values
            # print('--------')
            # print(ret, env, prog_id, args)
            # print('---------')
            if prog_id == self.act:
                env = self.f_env(env, prog_id, args)
            else:
                ret = 0


def npi_factory(f_enc,  # encoder fn:: (Env, args: Tensor[args_dim]) -> Tensor[state_dim]
                f_env,  # env fn:: (Env, args: Tensor[args_dim], prog_id: int) -> Env
                state_dim,  # state tensor dimension
                n_progs,  # number of programs
                prog_dim,  # program embedding dimension
                hidden_dim,  # LSTM hidden dimension (in core)
                n_lstm_layers,  # number of LSTM layers (in core)
                ret_threshold,  # return probability threshold
                pkey_dim,  # program key dimension
                args_dim):  # argument vector dimension
    core = NPICore(state_dim=state_dim,
                   prog_dim=prog_dim,
                   hidden_dim=hidden_dim,
                   n_lstm_layers=n_lstm_layers,
                   pkey_dim=pkey_dim,
                   args_dim=args_dim)
    npi = NPI(core=core,
              f_enc=f_enc,
              f_env=f_env,
              n_progs=n_progs,
              ret_threshold=ret_threshold,
              pkey_dim=pkey_dim,
              prog_dim=prog_dim)
    return npi


if __name__ == '__main__':
    state_dim = 2
    args_dim = 3


    def f_enc(env, args):
        return torch.randn(state_dim)


    def f_env(env, prog_id, args):
        return torch.randn(1)


    npi = npi_factory(f_enc=f_enc,
                      f_env=f_env,
                      state_dim=state_dim,
                      n_progs=4,
                      prog_dim=5,
                      hidden_dim=3,
                      n_lstm_layers=2,
                      ret_threshold=0.2,
                      pkey_dim=3,
                      args_dim=args_dim)

    ret, prog_id_probs, new_args = npi(42, 1, torch.randn(args_dim))
    npi.run(42, 1, torch.randn(args_dim))
