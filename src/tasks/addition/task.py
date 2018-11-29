from tasks.task_base import TaskBase
import torch
from torch import nn
import torch.nn.functional as F
from tasks.addition.env.scratchpad import ScratchPad
import copy

MOVE_PID, WRITE_PID = 0, 1
WRITE_OUT, WRITE_CARRY = 0, 1
IN1_PTR, IN2_PTR, CARRY_PTR, OUT_PTR = range(4)
LEFT, RIGHT = 0, 1


def build(in1s,
          in2s,
          hidden_dim,
          state_dim,
          environment_row,
          environment_col,
          environment_depth,
          argument_num,
          argument_depth,
          default_argument_num,
          program_embedding_size,
          program_size,
          batch_size=1):
    scratch_pads = []
    for i in range(batch_size):
        scratch_pads.append(ScratchPad(in1s[i], in2s[i], environment_row, environment_col))
    envs = [scratch_pads[i].get_env(environment_row,
                                    environment_col,
                                    environment_depth)
            for i in range(batch_size)]
    task = AdditionTask(scratch_pads=scratch_pads,
                        envs=envs,
                        hidden_dim=hidden_dim,
                        state_dim=state_dim,
                        environment_row=environment_row,
                        environment_col=environment_col,
                        environment_depth=environment_depth,
                        argument_num=argument_num,
                        argument_depth=argument_depth,
                        default_argument_num=default_argument_num,
                        program_embedding_size=program_embedding_size,
                        program_size=program_size,
                        batch_size=batch_size)
    return task


class AdditionTask(TaskBase):
    def __init__(self,
                 scratch_pads,
                 envs,
                 hidden_dim,
                 state_dim,
                 environment_row,
                 environment_col,
                 environment_depth,
                 argument_num,
                 argument_depth,
                 default_argument_num,
                 program_embedding_size,
                 program_size,
                 batch_size=1):
        super(AdditionTask, self).__init__(envs, state_dim)
        # config params
        self.environment_row = environment_row
        self.environment_col = environment_col
        self.environment_depth = environment_depth
        self.argument_num = argument_num
        self.argument_depth = argument_depth
        self.default_argument_num = default_argument_num
        self.program_embedding_size = program_embedding_size
        self.program_size = program_size

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.env_dim = environment_row * environment_depth  # 4 * 10 = 40
        # self.arg_dim = argument_num * argument_depth  # 3 * 10 = 30
        self.arg_dim = argument_num 
        self.in_dim = self.env_dim + self.arg_dim
        self.prog_embedding_dim = program_embedding_size
        self.prog_dim = program_size
        self.scratch_pads = copy.copy(scratch_pads)
        # for f_enc
        self.fenc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.fenc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fenc3 = nn.Linear(self.hidden_dim, self.state_dim)
        # for f_env
        self.fenv1 = nn.Embedding(self.prog_dim, self.prog_embedding_dim)

    def init_scratchpad(self, in1s, in2s):
        for i in range(self.batch_size):
            self.scratch_pads[i].init_scratchpad(in1s[i], in2s[i])

    def get_args(self, args, arg_in=True):
        if arg_in:
            arg_vec = torch.zeros((self.argument_num, self.argument_depth))
        else:
            arg_vec = [torch.zeros((self.argument_depth,)) for _ in
                       range(self.argument_num)]
        if len(args) > 0:
            for i in range(self.argument_num):
                if i >= len(args):
                    arg_vec[i][self.default_argument_num] = 1
                else:
                    arg_vec[i][args[i]] = 1
        else:
            for i in range(self.argument_num):
                arg_vec[i][self.default_argument_num] = 1
        return [arg_vec.view(arg_vec.numel()) if arg_in else arg_vec]

    def f_enc(self, args):
        env = [self.scratch_pads[i].get_env(self.environment_row,
                                     self.environment_col,
                                     self.environment_depth) 
                for i in range(self.batch_size)]
        # print(env)
        # if len(env) > 1:
        #     env = torch.stack(env, -1)
        # else:
        #     env = env[0]
        env = torch.stack(env)
        # print([env, args])
        # print(env.size(), args.size())

        merge = torch.cat([env, args.float()],1)
        # print(merge)
        elu = F.elu(self.fenc1(merge))
        elu = F.elu(self.fenc2(elu))
        out = self.fenc3(elu)
        return out

    def get_program_embedding(self, prog_id):
        embedding = self.fenv1(prog_id)
        return embedding

    def f_env(self, prog_ids, args):
        for i in range(self.batch_size):
            prog_id = prog_ids[i]
            if prog_id == MOVE_PID or prog_id == WRITE_PID:
                self.scratch_pads[i].execute(prog_id, args)
        return [self.scratch_pads[i].get_env(self.environment_row,
                                             self.environment_col,
                                             self.environment_depth)
                for i in range(self.batch_size)]
