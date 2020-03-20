import random
from collections import deque
import torch
import typing
from torch import nn, optim, FloatTensor
import numpy as np
from models import Actor, Critic
from helpers import copy_params
from noise import OrnsteinUhlenbeckProcess

class AgentDDPG:
    """Deep Deterministic Policy Gradient implementation for continuous action space reinforcement learning tasks"""
    def __init__(self, state_size: int, action_size: int, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, use_cuda=False):
        # Params
        self.state_size, self.action_size = state_size, action_size
        self.gamma, self.tau = gamma, tau
        self.use_cuda = use_cuda

        # Networks
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)

        # Create target networks for training actor-critic nets properly 
        self.actor_target = Actor(state_size, action_size)
        self.critic_target = Critic(state_size, action_size)
        copy_params(self.actor, self.actor_target)
        copy_params(self.critic, self.critic_target)

        if self.use_cuda:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()

        # Create replay buffer for storing experience
        self.replay_buffer = ReplayBuffer(cache_size=int(1e6))

        # Training
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def set_noise_process(self, noise_process):
        self.noise_process = noise_process

    def get_action(self, state: np.ndarray, noise=True):
        """Select action with respect to state according to current policy and exploration noise"""
        state = FloatTensor(state)
        if self.use_cuda:
            state = state.cuda()
        a = self.actor.forward(state).detach()
        if noise:
            try:
                n = FloatTensor(self.noise_process.sample())
                if self.use_cuda:
                    n = n.cuda()
                return a + n
            except NameError as e:
                print("Please set a noice process with self.set_noise_process(noise_process)")
        return a

    def save_experience(self, state_t: np.ndarray, action_t: any, reward_t: float, state_t1: np.ndarray):
        self.replay_buffer.add_sample(state_t, action_t, reward_t, state_t1)

    def update(self, batch_size: int):
        samples = self.replay_buffer.get_samples(batch_size)
        states, actions, rewards, next_states = unpack_replay_buffer(samples, self.use_cuda)

        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    
    def add_noise_to_weights(self, amount=0.1):
        self.actor.apply(lambda x: _add_noise_to_weights(x, amount, self.use_cuda))
        self.critic.apply(lambda x: _add_noise_to_weights(x, amount, self.use_cuda))
        self.actor_target.apply(lambda x: _add_noise_to_weights(x, amount, self.use_cuda))
        self.critic_target.apply(lambda x: _add_noise_to_weights(x, amount, self.use_cuda))

class ReplayBuffer:
    """
    Replay Buffer containing cache of interactions with the environment according to training Policy.
    Samples stored in tuples of <state_t, action_t, reward_t, state_t1>
    """
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.buffer = deque(maxlen=cache_size)
    
    def add_sample(self, state_t, action_t, reward_t, state_t1):
        """Store a sample tuple <state_t, action_t, reward_t, state_t1> into finite cache memory"""
        x = (state_t, action_t, reward_t, state_t1)
        self.buffer.append(x)
    
    def get_samples(self, batch_size):
        """Return list of tuples of <state_t, action_t, reward_t, state_t1>"""
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

def unpack_replay_buffer(experiences, using_cuda=False):
    # Unpack list of tuple experiences into Tensors with 
    # dimensions (len(experiences), x), where x varies in states, actions, etc.
    batch_size = len(experiences)

    states, actions, rewards, new_states = [], [], [], []
    for s, a, r, ns in experiences:
        if using_cuda:
            s = torch.Tensor(s).cuda()
            a = torch.Tensor(a).cuda() #TODO see if this line slows down the program
            r = torch.Tensor(r).cuda()
            ns = torch.Tensor(ns).cuda()
        else:
            s = torch.Tensor(s)
            a = torch.Tensor(a)
            r = torch.Tensor(r)
            ns = torch.Tensor(ns)

        states.append(s)
        actions.append(a)
        rewards.append(r)
        new_states.append(ns)

    # stack list of tensors into one big tensors with <batch_size, x> dimensions
    s_t, a_t, r_t, ns_t = torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(new_states)
    return s_t, a_t, r_t, ns_t

def _add_noise_to_weights(m, amount=0.1, use_cuda=False):
    """call model.apply(add_noise_to_weights) to apply noise to a models weights """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            if use_cuda:
                m.weight.add_(torch.randn(m.weight.size()).cuda() * amount)
            else:
                m.weight.add_(torch.randn(m.weight.size()) * amount)
            
