from models.npi import npi_factory
import sys, time, math, json, random, os
import torch
import torch.nn as nn
import torch.optim as optim
from tasks.task_base import TaskBase

PRETRAIN_WRIGHT_DECAY = 0.0001
PRETRAIN_LR = 0.01


class Agent(object):
    def __init__(self, npi):
        super(Agent, self).__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.npi = npi.to(device=self._device)
        list_of_params = []
        list_of_params.append({'params': self.npi.parameters()})
        self.pretrain_optimizer = optim.Adam(list_of_params, lr=PRETRAIN_LR,
                                             weight_decay=PRETRAIN_WRIGHT_DECAY)
        self.criterion_nll = nn.NLLLoss()
        self.criterion_binary = nn.BCELoss()
        self.criterion_mse = nn.MSELoss()


def eval(agent, data, traces, hidden_dim, n_lstm_layers):
    with torch.no_grad():
        agent.npi.eval()
        total_loss = 0
    
        for i in range(len(data)):
            agent.npi.task = data[i]
            agent.npi.task.task_params = agent.npi.task_params
        
            # Setup Environment
            agent.npi.core.reset()
            agent.npi.task.reset()
            
            # trace one by one
            trace = traces[i]
            hidden = torch.zeros(n_lstm_layers, 1, hidden_dim), \
                     torch.zeros(n_lstm_layers, 1, hidden_dim)
            for t in range(trace['len'] - 1):
                prog_id = torch.tensor(trace["prog_id"][t]).to(device=agent._device)
                args = torch.tensor(trace["args"][t]).to(device=agent._device)
          
                # forward
                new_ret, new_prog_id_log_probs, new_args, hidden = agent.npi(prog_id, args, hidden)
          
                # update env
                truth_ret = torch.tensor(trace["ret"][t + 1], dtype=torch.float32).to(device=agent._device)
                truth_prog_id = torch.tensor(trace["prog_id"][t + 1]).to(device=agent._device)
                truth_args = torch.tensor(trace["args"][t + 1], dtype=torch.float32).to(device=agent._device)
                agent.npi.task.f_env(truth_prog_id, truth_args)
          
                # loss
                total_loss += (agent.criterion_mse(new_args, truth_args)
                               + agent.criterion_mse(torch.squeeze(new_ret), truth_ret)
                               + agent.criterion_nll(new_prog_id_log_probs, truth_prog_id))
        agent.npi.train()
        return total_loss


def test(agent, data, hidden_dim, n_lstm_layers):
    with torch.no_grad():
        agent.npi.eval()
        total_loss = 0
    
        for i in range(len(data)):
            agent.npi.task = data[i]
            agent.npi.task.task_params = agent.npi.task_params
        
            # Setup Environment
            agent.npi.core.reset()
            agent.npi.task.reset()
            
            # no trace
            hidden = torch.zeros(n_lstm_layers, 1, hidden_dim), \
                     torch.zeros(n_lstm_layers, 1, hidden_dim)
            new_ret = torch.zeros(agent.npi.task.batch_size).to(device=agent._device)
            new_prog_id = torch.zeros(agent.npi.task.batch_size, dtype=torch.int64).to(device=agent._device)
            new_args = torch.zeros(agent.npi.task.batch_size, agent.npi.core.args_dim).to(device=agent._device)
            
            while max(new_ret) < agent.npi.ret_threshold:
                # forward
                new_ret, new_prog_id_log_probs, new_args, hidden = agent.npi(new_prog_id, new_args, hidden)
                new_prog_id = torch.argmax(new_prog_id_log_probs, dim=1)

                # update env
                print('Ret:', new_ret[0], ' prog id:', new_prog_id[0], ' Args:', new_args[0])
                agent.npi.task.f_env(new_prog_id, new_args)
            
            print(new_ret)
            agent.npi.task.scratch_pads[int(torch.argmax(new_ret))].pretty_print()
        agent.npi.train()
        return total_loss


def train(npi, data, traces, hidden_dim=3,
          n_lstm_layers=2, epochs=100, train_ratio=0.8,
          load_model=None, save_dir="./model_512"):
    agent = Agent(npi)
    train_data = data[0:int(train_ratio*len(data))]
    eval_data = data[int(train_ratio*len(data)):]
    train_traces = traces[0:int(train_ratio*len(data))]
    eval_traces = traces[int(train_ratio*len(data)):]
    agent.npi.train()
    start_time = time.time()

    best_loss = math.inf
    start_epoch = 1
    
    # load
    if load_model != None:
        checkpoint = torch.load(load_model)
        agent.npi.load_state_dict(checkpoint['state_dict'])
        agent.pretrain_optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print("Load model. Epoch={}, Loss={}".format(start_epoch, best_loss))
    
    for ep in range(start_epoch, epochs + 1):

        # parameters
        total_trace = 0
        arg_loss = 0
        prog_loss = 0
        ret_loss = 0
        prog_id_error = 0
        
        test(agent, eval_data, hidden_dim, n_lstm_layers)

        for i in range(len(train_data)):
            agent.npi.task = train_data[i]
            agent.npi.task.task_params = agent.npi.task_params

            # Setup Environment
            agent.npi.core.reset()
            agent.npi.task.reset()

            # trace one by one
            trace = train_traces[i]
            hidden = torch.zeros(n_lstm_layers, 1, hidden_dim), \
                     torch.zeros(n_lstm_layers, 1, hidden_dim)
            for t in range(trace['len'] - 1):
                prog_id = torch.tensor(trace["prog_id"][t]).to(device=agent._device)
                args = torch.tensor(trace["args"][t]).to(device=agent._device)
    
                # forward
                agent.pretrain_optimizer.zero_grad()
                new_ret, new_prog_id_log_probs, new_args, hidden = agent.npi(prog_id, args, hidden)
                new_prog_id = torch.argmax(new_prog_id_log_probs, dim=1)
  
                # update env
                truth_ret = torch.tensor(trace["ret"][t+1], dtype = torch.float32).to(device=agent._device)
                truth_prog_id = torch.tensor(trace["prog_id"][t+1]).to(device=agent._device)
                truth_args = torch.tensor(trace["args"][t+1], dtype=torch.float32).to(device=agent._device)
                agent.npi.task.f_env(truth_prog_id, truth_args)
  
                # loss
                arg_loss_now = agent.criterion_mse(new_args, truth_args)
                ret_loss_now = agent.criterion_mse(torch.squeeze(new_ret), truth_ret)
                prog_loss_now = agent.criterion_nll(new_prog_id_log_probs, truth_prog_id)
                prog_id_error += float(((new_prog_id - truth_prog_id) ** 2).sum()) / len(new_prog_id)
                loss_batch = arg_loss_now + ret_loss_now + prog_loss_now
                arg_loss += arg_loss_now
                ret_loss += ret_loss_now
                prog_loss += prog_loss_now
  
                # backpropagation
                loss_batch.backward(retain_graph=True)
                agent.pretrain_optimizer.step()
              
            total_trace += trace['len'] - 1

        # print result
        total_loss = eval(agent, eval_data, eval_traces, hidden_dim, n_lstm_layers)
        arg_loss /= total_trace
        ret_loss /= total_trace
        prog_loss /= total_trace
        prog_id_error /= total_trace
        end_time = time.time()
        print("Epoch {}: Ret err {}, Arg err {}, Prog err {}, Prog_id err {}, Tot loss {}, T {}s"
              .format(ep,
                      round(float(ret_loss), 4),
                      round(float(arg_loss), 4),
                      round(float(prog_loss), 4),
                      round(float(prog_id_error), 4),
                      round(float(total_loss), 4),
                      round((end_time - start_time), 4)))

        # save model
        if total_loss < best_loss:
            best_loss = total_loss
            save_state = {
                'epoch': ep,
                'loss': best_loss,
                'state_dict': agent.npi.state_dict(),
                'optimizer': agent.pretrain_optimizer.state_dict()
            }
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            torch.save(save_state, save_dir + '/npi_model_latest.net')


if __name__ == "__main__":
    seed = random.randrange(sys.maxsize)
    print('seed= {}'.format(seed))
    torch.manual_seed(seed)
    # good seeds: 6478152654801362860

    state_dim = 2
    args_dim = 3
    data_num = 2


    class DummyTask(TaskBase):
        def __init__(self, env, state_dim):
            super(DummyTask, self).__init__(env, state_dim)

        def f_enc(self, args):
            return torch.randn(args.size(0), self.state_dim)

        def f_env(self, prog_id, args):
            self.env = torch.randn(prog_id.size(0), self.batch_size)


    data = []
    trace = []  # element of trace is dict={'ret':xx,'prog_id:xx','args:xx'}
    for i in range(data_num):
        dummy_task = DummyTask([random.randint(1, 1000), random.randint(1, 1000)], state_dim)
        data.append(dummy_task)
    with open("./src/tasks/addition/data/train_trace_input.json", 'r') as fin:
        trace = json.load(fin)
        trace.append(trace[0])
      
    npi = npi_factory(task=dummy_task,
                      state_dim=state_dim,
                      n_progs=5,
                      prog_dim=5,
                      hidden_dim=3,
                      n_lstm_layers=2,
                      ret_threshold=0.38,
                      pkey_dim=4,
                      args_dim=args_dim,
                      n_act=2)
    print('Initializing NPI Model!')

    assert len(data) == len(trace)
    print("Data:", len(data))
    print("Traces:", len(trace))
    train(npi, data, trace)
