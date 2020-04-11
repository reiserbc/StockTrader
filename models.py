import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class Actor(nn.Module):
    "Actor model for Actor-Critic neural nets"
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.state_size = input_size
        self.action_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, state):
        """Compute Forward pass"""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return torch.relu(x)

class Critic(nn.Module):
    """ Critic network for Actor-Critic neural nets"""
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.state_size = input_size
        self.action_size = output_size
        self.linear1 = nn.Linear(self.state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        """Compute forward pass"""
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
