import random
import torch
import torch.nn as nn
import torchesn.nn as esn


class DummyModel(torch.nn.Module):

    def __init__(self, start_pred=0):
        super(RNN, self).__init__()

        self.start_pred = start_pred

    def forward(self, x):
        r1 = random.uniform(-0.2, 0.2)
        r2 = random.uniform(-0.2, 0.2)
        return torch.tensor([self.start_pred + r1, self.start_pred + r2])


class RNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        # Defining some parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    

    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, _ = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.view(-1, self.hidden_dim)[-1, :]
        out = self.fc(out)

        return out


    def clone(self):
        return RNN(self.input_size, self.output_size, self.hidden_dim, self.n_layers)
        

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class ESN(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, washout_rate=0.2):
        super(ESN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.washout_rate = washout_rate
        self.esn = esn.ESN(input_size, hidden_dim,
                           input_size, batch_first=True, readout_training='gd')
    

    def forward(self, x):
        
        batch_size = x.size(0)
        washout = [int(self.washout_rate * x.size(1))] * x.size(0)

        out, boh = self.esn(x, washout)

        return out[0, -1]


    def clone(self):
        return ESN(self.input_size, self.output_size, self.hidden_dim)
