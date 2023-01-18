import random
import torch
import torch.nn as nn

import esn


class RNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        
        hidden = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        out, _ = self.rnn(x, hidden)

        out = out[:, -1, :]
        out = self.linear(out)

        return out

    def clone(self):
        return RNN(self.input_size, self.output_size, self.hidden_dim, self.n_layers)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class LSTM(nn.Module):

    def __init__(self, input_size, output_size,hidden_dim, n_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        
        hidden = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x)

        out = out[:, -1, :]
        out = self.linear(out)

        return out

    def clone(self):
        return LSTM(self.input_size, self.output_size, self.hidden_dim, self.n_layers)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class ESN(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers=1):
        super(ESN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.esn = esn.DeepReservoirClassifier(input_size, output_size, 
                                           hidden_dim, n_layers, concat=False)

    def forward(self, x):
        out = self.esn(x)
        return out

    def clone(self):
        return ESN(self.input_size, self.output_size,
                   self.hidden_dim, self.n_layers)