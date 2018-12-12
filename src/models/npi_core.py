import torch
from torch import nn
import torch.nn.functional as F


class NPICore(nn.Module):
    def __init__(self,
                 state_dim,
                 prog_dim,
                 hidden_dim,  # 2 by default
                 n_lstm_layers,
                 pkey_dim,
                 args_dim,
                 args_depth):
        super(NPICore, self).__init__()
        self.state_dim = state_dim
        self.prog_dim = prog_dim
        self.hidden_dim = hidden_dim
        self.n_lstm_layers = n_lstm_layers
        self.pkey_dim = pkey_dim
        self.args_dim = args_dim
        self.args_depth = args_depth
        self.in_dim = self.state_dim + self.prog_dim
        self.lstm = nn.LSTM(input_size=self.in_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.n_lstm_layers)
        self.ret_fc = nn.Linear(self.hidden_dim, 2)
        self.pkey_fc = nn.Linear(self.hidden_dim, self.pkey_dim)
        self.pkey_hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.args_fc = []
        for _ in range(self.args_dim):
            args = nn.Linear(self.hidden_dim, self.args_depth)
            self.args_fc.append(args)
        # self.last_lstm_state = torch.zeros(self.n_lstm_layers, 1, self.hidden_dim), \
        #                        torch.zeros(self.n_lstm_layers, 1, self.hidden_dim)

    def reset(self):
        pass
        # self.last_lstm_state = torch.zeros(self.n_lstm_layers, 1, self.hidden_dim), \
        #                        torch.zeros(self.n_lstm_layers, 1, self.hidden_dim)

    def forward(self, state, prog, hidden):
        inp = torch.cat([state, prog], -1)
        # for LSTM, out and h are the same
        lstm_h, last_h = self.lstm(inp.unsqueeze(1), hidden)
        emb = F.relu(lstm_h)
        ret = self.ret_fc(emb).squeeze(1)
        pkey_hidden = self.pkey_hidden(emb).squeeze(1)
        pkey = self.pkey_fc(pkey_hidden).squeeze(1)
        args = torch.zeros(self.args_dim, pkey.size()[0], self.args_depth)
        for i in range(self.args_dim):
            args[i] = self.args_fc[i](emb).squeeze(1)
        args = args.permute(1, 2, 0)
        return ret, pkey, args, last_h


if __name__ == '__main__':
    state = torch.randn(3, 2, 4)
    prog = torch.randn(3, 2, 6)
    core = NPICore(state_dim=4,
                   prog_dim=6,
                   hidden_dim=5,
                   n_lstm_layers=2,
                   pkey_dim=4,
                   args_dim=5,
                   args_depth=11)
    ret, pkey, args = core(state, prog)
