from tasks.addition.env.config import CONFIG
from tasks.task_base import TaskBase
import torch
from torch import nn
import torch.nn.functional as F
from tasks.addition.env.scratchpad import ScratchPad
import numpy as np

MOVE_PID, WRITE_PID = 0, 1
WRITE_OUT, WRITE_CARRY = 0, 1
IN1_PTR, IN2_PTR, CARRY_PTR, OUT_PTR = range(4)
LEFT, RIGHT = 0, 1

def build(in1, in2, hidden_dim, state_dim, environment_row, environment_col, environment_depth, argument_num, argument_depth, default_argument_num, program_embedding_size, program_size, batch_size=1):
    scratchPad = ScratchPad(in1, in2, environment_row, environment_col))
    env = [scratchPad.get_env()]
    task = AdditionCore(env, hidden_dim, state_dim, environment_row, environment_col, environment_depth, argument_num, argument_depth, default_argument_num, program_embedding_size, program_size, batch_size=1)
    return task

class AdditionCore(TaskBase):
    def __init__(self, env, hidden_dim, state_dim, environment_row, environment_col, environment_depth, argument_num, argument_depth, default_argument_num, program_embedding_size, program_size, batch_size=1):
        super(AdditionCore, self).__init__(env, state_dim, batch_size)
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
        self.arg_dim = argument_num * argument_depth        # 3 * 10 = 30
        self.in_dim = self.env_dim + self.arg_dim
        self.prog_embedding_dim = program_embedding_size
        self.prog_dim = program_size
        self.scratchPad = None
        # for f_enc
        self.fenc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.fenc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fenc3 = nn.Linear(self.hidden_dim, self.state_dim)
        # for f_env
        self.fenv1 = nn.Embedding(self.prog_dim, self.prog_embedding_dim)
      
    def init_scratchPad(self, in1, in2):
        self.scratchPad.init_scratchpad(in1, in2)
    
    def get_args(self, args, arg_in=True):
        if arg_in:
            arg_vec = np.zeros((self.argument_num, self.argument_depth), dtype=np.int32)
        else:
            arg_vec = [np.zeros((self.argument_depth), dtype=np.int32) for _ in
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
        return [arg_vec.flatten() if arg_in else arg_vec]

    def f_enc(self, args):
        env = [self.scratchPad.get_env()]
        merge = torch.cat([env, args], -1)
        elu = F.elu(self.fenc1(merge))
        elu = F.elu(self.fenc2(elu))
        out = self.fenc3(elu)
        return out 

    def get_program_embedding(self, prog_id, args):
        embedding = self.fenv1(prog_id)
        return embedding
    
    def f_env(self, prog_id, args):
        if prog_id == MOVE_PID or prog_id == WRITE_PID:
            self.scratchPad.execute(prog_id, args)
        return [self.scratchPad.get_env()]