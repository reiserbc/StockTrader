import torch
import gym
import numpy as np
from matplotlib import pyplot as plt
from ddpg import AgentDDPG
from noise import OUNoise
from train import ReinforcementTrainer

def main():
    env = gym.make('BipedalWalker-v3')
    agent = torch.load('bipedal_ddpg.pkl')
    noise = OUNoise(env.action_space)
    trainer = ReinforcementTrainer(env, agent, noise)
    trainer.simulate(5, 500, render_env=True)

if __name__ == '__main__':
    main()
