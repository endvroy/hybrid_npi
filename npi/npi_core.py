import torch
from torch import nn
import numpy as np


class NPICore(nn.Module):
    def __init__(self, state_dim, prog_dim, hidden_dim, n_lstm_layers, ret_dim, pkey_dim, args_dim):
        super(NPICore, self).__init__()
        self.state_dim = state_dim
        self.prog_dim = prog_dim
        self.hidden_dim = hidden_dim
        self.n_lstm_layers = n_lstm_layers
        self.ret_dim = ret_dim
        self.pkey_dim = pkey_dim
        self.args_dim = args_dim
        self.in_dim = self.state_dim + self.prog_dim
        self.lstm = nn.LSTM(input_size=self.in_dim,
                            hidden_size=self.hidden_dim)
        self.ret_fc = nn.Linear(self.hidden_dim, self.ret_dim)
        self.pkey_fc = nn.Linear(self.hidden_dim, self.pkey_dim)
        self.args_fc = nn.Linear(self.hidden_dim, self.args_dim)
        self.init_lstm_h = nn.Parameter(torch.randn(self.n_lstm_layers, 1, self.hidden_dim))
        self.init_lstm_c = nn.Parameter(torch.randn(self.n_lstm_layers, 1, self.hidden_dim))
        self.last_lstm_state = self.init_lstm_h, self.init_lstm_c

    def reset(self):
        self.last_lstm_state = self.init_lstm_h, self.init_lstm_c

    def forward(self, state, prog):
        inp = torch.cat([state, prog], -1)
        # for LSTM, out and h are the same
        lstm_h, self.last_lstm_state = self.lstm(inp.view(1, 1, -1), self.last_lstm_state)
        ret = self.ret_fc(lstm_h).squeeze(0)
        pkey = self.pkey_fc(lstm_h).squeeze(0)
        args = self.args_fc(lstm_h).squeeze(0)
        return ret, pkey, args


if __name__ == '__main__':
    state = torch.randn(1, 3)
    prog = torch.randn(1, 4)
    core = NPICore(3, 4, 5, 1, 4, 5, 6)
    ret, pkey, args = core(state, prog)
