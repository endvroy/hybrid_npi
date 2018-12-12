from tasks.addition.env import config as addition_config
from train.train import test
from tasks.addition.task import build, build_param, AdditionTask, TaskParams
from tasks.task_base import TaskBase
from models.npi import npi_factory
import random
import sys
import torch
import json
import os
seed = random.randrange(sys.maxsize)
print('seed= {}'.format(seed))
torch.manual_seed(seed)
# good seeds: 6478152654801362860

# TODO: reorganize with argsparse
state_dim = 2
args_dim = addition_config.CONFIG["ARGUMENT_NUM"]
args_depth = addition_config.CONFIG["ARGUMENT_DEPTH"]
batchsize = 1

data = []
traces = []  # element of trace is dict={'ret':xx,'prog_id:xx','args:xx'}
new_batch = True
exp_id = "exp1_5"
# TODO: better way to deal with interger?
with open(os.path.join(addition_config.DATA_DIR, exp_id + '_int.json'), 'r') as fin:
        # element of traces is list=[[["prog_name",prog_id],args,ret],[...],[...]]
        nums = json.load(fin)
        for i in range(0, len(nums), batchsize):
            if i + batchsize > len(nums):
                break
            in1s = []
            in2s = []
            for j in range(i, i+batchsize):
                in1s.append(nums[j][0])
                in2s.append(nums[j][1])
            data.append([in1s, in2s])
print(len(data))
data_num = len(data)

hidden_dim = 3
state_dim = 2

# tasks
mytasks = []
task_parameters = build_param(
    hidden_dim = hidden_dim,
    state_dim = state_dim,
    environment_row = addition_config.CONFIG["ENVIRONMENT_ROW"],
    environment_depth = addition_config.CONFIG["ENVIRONMENT_DEPTH"],
    argument_num = args_dim,
)
for i in range(data_num):
    in1s = data[i][0]
    in2s = data[i][1]
    addition_task = build(
        in1s=in1s,
        in2s=in2s,
        state_dim = state_dim,
        environment_row = addition_config.CONFIG["ENVIRONMENT_ROW"],
        environment_col = addition_config.CONFIG["ENVIRONMENT_COL"],
        environment_depth = addition_config.CONFIG["ENVIRONMENT_DEPTH"],
        argument_num=args_dim,
        argument_depth=args_depth,
        default_argument_num=addition_config.CONFIG["DEFAULT_ARG_VALUE"],
        program_embedding_size=addition_config.CONFIG["PROGRAM_EMBEDDING_SIZE"],
        program_size=addition_config.CONFIG["PROGRAM_KEY_SIZE"],
    )
    mytasks.append(addition_task)

# npi
npi = npi_factory(
    task_params=task_parameters,
    state_dim=state_dim,
    n_progs=5,
    prog_dim=5,
    hidden_dim=hidden_dim,
    n_lstm_layers=2,
    ret_threshold=0.5,
    pkey_dim=addition_config.CONFIG["PROGRAM_KEY_SIZE"],
    args_dim=args_dim,
    args_depth=args_depth,
    n_act=2
)
print('Initializing NPI Model!')
print("Data:", len(mytasks))
test(npi, mytasks, load_model='./model_640/npi_model_latest.net')
