import random
from collections import deque
import torch
import typing
from torch import nn, optim, FloatTensor
from torch.autograd import Variable
import numpy as np
from models import Actor, Critic
from helpers import copy_params, soft_copy_params

class AgentDDPG:
    """Deep Deterministic Policy Gradient implementation for continuous action space reinforcement learning tasks"""
    def __init__(self, state_size, hidden_size, action_size, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, use_cuda=False):
        # Params
        self.state_size, self.hidden_size, self.action_size = state_size, hidden_size, action_size
        self.gamma, self.tau = gamma, tau
        self.use_cuda = use_cuda

        # Networks
        self.actor = Actor(state_size, hidden_size, action_size)
        self.actor_target = Actor(state_size, hidden_size, action_size)

        self.critic = Critic(state_size + action_size , hidden_size, action_size)
        self.critic_target = Critic(state_size + action_size, hidden_size, action_size)
        
        # Hard copy params from original networks to target networks
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

    def get_action(self, state):
        """Select action with respect to state according to current policy and exploration noise"""
        state = Variable(torch.from_numpy(state).float())

        if self.use_cuda:
            state = state.cuda()

        a = self.actor.forward(state)

        if self.use_cuda:
            return a.detach().cpu().numpy()

        return a.detach().numpy()


    def save_experience(self, state_t, action_t, reward_t, state_t1):
        self.replay_buffer.add_sample(state_t, action_t, reward_t, state_t1)

    def update(self, batch_size):
        states, actions, rewards, next_states = self.replay_buffer.get_samples(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        if self.use_cuda:
            states = states.cuda()
            next_states = next_states.cuda()

        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # update target networks 
        soft_copy_params(self.actor, self.actor_target, self.tau)
        soft_copy_params(self.critic, self.critic_target, self.tau)
    
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
        x = (state_t, action_t, np.array([reward_t]), state_t1)
        self.buffer.append(x)
    
    def get_samples(self, batch_size):
        """Return lists of state_batch, action_batch, reward_batch, next_state_batch"""
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
        
        return state_batch, action_batch, reward_batch, next_state_batch
        
    def __len__(self):
        return len(self.buffer)

def _add_noise_to_weights(m, amount=0.1, use_cuda=False):
    """call model.apply(add_noise_to_weights) to apply noise to a models weights """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            if use_cuda:
                m.weight.add_(torch.randn(m.weight.size()).cuda() * amount)
            else:
                m.weight.add_(torch.randn(m.weight.size()) * amount)
            
