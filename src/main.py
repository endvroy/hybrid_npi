from tasks.addition.env import config as addition_config
from train.train import train
from tasks.addition.task import build, AdditionTask
from tasks.task_base import TaskBase
from models.npi import npi_factory
import random
import sys
import torch
import json
seed = random.randrange(sys.maxsize)
print('seed= {}'.format(seed))
torch.manual_seed(seed)
# good seeds: 6478152654801362860

# TODO: reorganize with argsparse
state_dim = 2
args_dim = addition_config.CONFIG["ARGUMENT_NUM"]


class DummyTask(TaskBase):
    def __init__(self, env, state_dim):
        super(DummyTask, self).__init__(env, state_dim)

    def f_enc(self, args):
        return torch.randn(args.size(0), self.state_dim)

    def f_env(self, prog_id, args):
        self.env = torch.randn(prog_id.size(0), self.batch_size)

data = []
trace = []  # element of trace is dict={'ret':xx,'prog_id:xx','args:xx'}
data_num = 2
hidden_dim = 3
state_dim = 2
for i in range(data_num):
    addition_task = build(in1s=[random.randint(1, 1000), random.randint(1, 1000)],
          in2s=[random.randint(1, 1000), random.randint(1, 1000)],
          hidden_dim = hidden_dim,
          state_dim = state_dim,
          environment_row = addition_config.CONFIG["ENVIRONMENT_ROW"],
          environment_col = addition_config.CONFIG["ENVIRONMENT_COL"],
          environment_depth = addition_config.CONFIG["ENVIRONMENT_DEPTH"],
          argument_num = args_dim,
          argument_depth = addition_config.CONFIG["ARGUMENT_DEPTH"],
          default_argument_num = addition_config.CONFIG["DEFAULT_ARG_VALUE"],
          program_embedding_size = addition_config.CONFIG["PROGRAM_EMBEDDING_SIZE"],
          program_size = addition_config.CONFIG["PROGRAM_KEY_SIZE"], # FIXME: typo
          batch_size=2)
    data.append(addition_task)
    
with open("./src/tasks/addition/data/train_trace_input.json", 'r') as fin:
    trace = json.load(fin)
    trace.append(trace[0])
    
npi = npi_factory(task=AdditionTask,
                    state_dim=state_dim,
                    n_progs=5,
                    prog_dim=5,
                    hidden_dim=hidden_dim,
                    n_lstm_layers=2,
                    ret_threshold=0.38,
                    pkey_dim=addition_config.CONFIG["PROGRAM_KEY_SIZE"],
                    args_dim=args_dim,
                    n_act=2)
print('Initializing NPI Model!')

assert len(data) <= len(trace)
print("Data:", len(data))
print("Traces:", len(trace))
train(npi, data, trace, batchsize=2)