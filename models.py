import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class Actor(nn.Module):
    "Actor model for Actor-Critic neural nets"
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(state_size, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, action_size)
    
    def forward(self, state):
        """Compute Forward pass"""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return torch.tanh(x)

class Critic(nn.Module):
    """ Critic network for Actor-Critic neural nets"""
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 64)
        self.linear2 = nn.Linear(64 + action_size, 32)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, state, action):
        """Compute forward pass"""
        x = F.relu(self.linear1(state))
        # concatenate action with x on the second layer
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x