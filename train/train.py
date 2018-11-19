from npi.npi import *
import sys, time, math
import torch.nn as nn
import torch.optim as optim
from npi.task_base import TaskBase

PRETRAIN_WRIGHT_DECAY = 0.00001
PRETRAIN_LR = 0.0001


class Agent(object):
  def __init__(self, npi):
    super(Agent, self).__init__()
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.npi = npi.to(device=self._device)
    list_of_params = []
    list_of_params.append({'params': self.npi.parameters()})
    self.pretrain_optimizer = optim.Adam(list_of_params, lr=PRETRAIN_LR,
                                         weight_decay=PRETRAIN_WRIGHT_DECAY)
    self.criterion = nn.CrossEntropyLoss()


def train(npi, data, trace, epochs=10):
  agent = Agent(npi)
  agent.npi.train()
  start_time = time.time()
  
  best_loss = math.inf
  for ep in range(1, epochs + 1):
    
    # parameters
    total_loss = 0
    total_trace = 0
    arg_error = torch.zeros(agent.npi.args_dim)
    prog_error = torch.zeros(agent.npi.n_progs)
    ret_error = torch.zeros(1)
    
    for d in data:
      agent.npi.task = d
      
      # Setup Environment
      agent.npi.core.reset()
      
      # trace one by one
      for t in range(len(trace) - 1):
        prog_id = trace[t]["prog_id"]
        prog_id_log_probs = torch.zeros(agent.npi.n_progs)
        prog_id_log_probs[int(prog_id[0])] = 1.0
        args = trace[t]["args"]
        
        # forward
        new_ret, new_prog_id_log_probs, new_args = agent.npi.forward(prog_id, args)
        new_prog_id = torch.argmax(new_prog_id_log_probs, dim=1)
        
        # loss
        new_para = torch.cat([new_ret, new_prog_id_log_probs, new_args], -1)
        truth_ret = trace[t + 1]["ret"]
        truth_prog_id = trace[t + 1]["prog_id"]
        truth_prog_id_log_probs = torch.zeros(agent.npi.n_progs)
        truth_prog_id_log_probs[int(truth_prog_id[0])] = 1.0
        truth_args = trace[t + 1]["args"]
        truth_para = torch.cat([truth_ret, truth_prog_id_log_probs, truth_args], -1)
        
        loss_batch = agent.criterion(new_para, truth_para)
        
        arg_error += (truth_args - new_args) ** 2
        ret_error += (truth_ret - new_ret) ** 2
        prog_error += (truth_prog_id_log_probs - new_prog_id_log_probs) ** 2
        total_loss += ((truth_para - new_para) ** 2).sum().to(dtype=torch.float32)
        
        # backpropagation
        agent.pretrain_optimizer.zero_grad()
        loss_batch.backward()
        agent.pretrain_optimizer.step()
      
      total_trace += (len(trace) - 1)
    
    arg_error /= total_trace
    ret_error /= total_trace
    prog_error /= total_trace
    end_time = time.time()
    print("Epoch {}: Ret error {}, Arg_error {}, Prog_error {}, Time {}"
          .format(ep, ret_error, arg_error, prog_error, end_time - start_time))
    
    # save model
    if total_loss < best_loss:
      best_loss = total_loss
      save_state = {
        'epoch': ep,
        'pkey_mem_state_dict': agent.npi.pkey_mem.state_dict(),
        'prog_mem_state_dict': agent.npi.prog_mem.state_dict(),
        'npi_core_state_dict': agent.npi.core.state_dict(),
        'optimizer': agent.pretrain_optimizer.state_dict()
      }
      torch.save(save_state, 'model/npi_model_latest.net')


if __name__ == "__main__":
  seed = random.randrange(sys.maxsize)
  print('seed= {}'.format(seed))
  torch.manual_seed(seed)
  # good seeds: 6478152654801362860
  
  state_dim = 2
  args_dim = 3
  data_num = 100
  
  
  class DummyTask(TaskBase):
    def __init__(self, env, state_dim, batch_size=1):
      super(DummyTask, self).__init__(env, state_dim, batch_size=batch_size)
    
    def f_enc(self, args):
      return torch.randn(args.size(0), self.state_dim)
    
    def f_env(self, prog_id, args):
      self.env = torch.randn(prog_id.size(0), self.batch_size)
  
  
  data = []
  trace = []  # element of trace is dict={'ret':xx,'prog_id:xx','args:xx'}
  for i in range(data_num):
    dummy_task = DummyTask(random.randint(1, 1000), state_dim)
    data.append(dummy_task)
  
  npi = npi_factory(task=dummy_task,
                    state_dim=state_dim,
                    n_progs=3,
                    prog_dim=5,
                    hidden_dim=3,
                    n_lstm_layers=2,
                    ret_threshold=0.38,
                    pkey_dim=4,
                    args_dim=args_dim)
  print('Initializing NPI Model!')
  
  train(npi, data, trace)
