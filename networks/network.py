import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor Network"""

    def __init__(self, state_size, output_size, fc_layer_sizes, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            output_size (int): Dimension of each action
            fc_layer_sizes (list of int): Layer size of each FC layer
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        torch.manual_seed(seed)
        self.output_size = output_size

        # define fc and bn layers
        self.linears = nn.ModuleList([])
        self.batch_norms = nn.ModuleList([])
        if len(fc_layer_sizes) == 0:
            self.batch_norms.append(nn.BatchNorm1d(state_size))
            self.linears.append(nn.Linear(state_size, output_size))
        else:
            self.batch_norms.append(nn.BatchNorm1d(state_size))
            self.linears.append(nn.Linear(state_size, fc_layer_sizes[0]))
            for i in range(1, len(fc_layer_sizes)):
                self.batch_norms.append(nn.BatchNorm1d(fc_layer_sizes[i-1]))
                self.linears.append(
                    nn.Linear(fc_layer_sizes[i-1], fc_layer_sizes[i]))
            self.batch_norms.append(nn.BatchNorm1d(fc_layer_sizes[-1]))
            self.linears.append(nn.Linear(fc_layer_sizes[-1], output_size))

        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.linears) - 1):
            self.linears[i].weight.data.uniform_(*hidden_init(self.linears[i]))
        self.linears[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        #x = self.batch_norm(x)
        for i in range(len(self.linears) - 1):
            x = F.relu(self.linears[i](self.batch_norms[i](x)))
        x = self.linears[-1](self.batch_norms[-1](x))
        x = torch.tanh(x)  # tanh maps output to [-1, 1]
        return x


class Critic(nn.Module):
    """Critic Network"""

    def __init__(self, state_size, action_size, fc_layer_sizes, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc_layer_sizes (list of int): Layer size of each FC layer
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.action_size = action_size

        # define fc and bn layers
        self.linears = nn.ModuleList([])
        self.batch_norm = nn.BatchNorm1d(state_size)
        self.linears.append(nn.Linear(state_size, fc_layer_sizes[0]))
        self.linears.append(
            nn.Linear(fc_layer_sizes[0]+action_size, fc_layer_sizes[1]))
        for i in range(2, len(fc_layer_sizes)):
            self.linears.append(
                nn.Linear(fc_layer_sizes[i-1], fc_layer_sizes[i]))
        self.linears.append(nn.Linear(fc_layer_sizes[-1], 1))

        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.linears) - 1):
            self.linears[i].weight.data.uniform_(*hidden_init(self.linears[i]))
        self.linears[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x, a):
        x = self.batch_norm(x)
        x = F.relu(self.linears[0](x))
        x = torch.cat((x, a), dim=-1)
        for i in range(1, len(self.linears) - 1):
            x = F.relu(self.linears[i](x))
        x = self.linears[-1](x)
        return x
