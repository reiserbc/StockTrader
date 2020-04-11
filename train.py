import torch
import gym
import numpy as np
from matplotlib import pyplot as plt
from ddpg import AgentDDPG
from noise import OUNoise
from helpers import print_gym_info, format_ochlv_df
from gyms import StockTraderEnv
from stock_data import AlphaVantage
    
class ReinforcementTrainer:
    def __init__(self, gym, agent, noise):
        self.gym = gym
        self.agent = agent
        self.noise = noise

    def train(self, episodes, timesteps, batch_size, save_path=None, mut_alg_episode=None, 
                                        mut_alg_step=None, render_env=False, plot=False, log=False):
        """
        Trains agent within gym environment, taking timestep steps for each episode.
        Learning happens with batch_size experiences from ReplayBuffer
        mut_alg_episode is applied to the agent model every episode
        mut_alg_step is applied to the agent model every step
        """
        rewards = []
    
        for e in range(episodes):
            state = self.gym.reset()
            self.noise.reset()
            episode_reward = 0

            for t in range(timesteps):
                if render_env:
                    self.gym.render()

                action = self.agent.get_action(state)
                action = self.noise.get_action(action, t)
                new_state, reward, done, info = self.gym.step(action)
                
                if t % 10 == 0:
                    print(action)
                if done:
                    break

                self.agent.save_experience(state, action, reward, new_state)

                if len(self.agent.replay_buffer) > batch_size:
                    self.agent.update(batch_size) 
        
                state = new_state
                episode_reward += reward
		
                if mut_alg_step:
                    mut_alg_step(self.agent)
                    
            if mut_alg_episode:
                mut_alg_episode(self.agent)

            rewards.append(episode_reward)
            
            if log:
                print("Episode {} | Reward {}".format(e, episode_reward))

            if save_path and (e % 20 == 0 or e == episodes):
                torch.save(self.agent, save_path)
        
            if plot:
                plt.plot(rewards)
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.draw()
                plt.pause(0.1)

    def simulate(self, episodes, timesteps, render_env=False, plot=False, log=False):
        rewards = []
    
        for e in range(episodes):
            state = self.gym.reset()
            self.noise.reset()
            episode_reward = 0

            for t in range(timesteps):
                if render_env:
                    self.gym.render()

                action = self.agent.get_action(state)
                action = self.noise.get_action(action, t)
                new_state, reward, done, info = self.gym.step(action)
                if done:
                    break

                state = new_state
                episode_reward += reward
	
            rewards.append(episode_reward)
            
            if log:
                print("Episode {} | Reward {}".format(e, episode_reward))

            if plot:
                plt.plot(rewards)
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.show()

# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """
    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
        

def main():
    stock_df = AlphaVantage('PYXRW3T85X6XF44V').get_intraday('goog', '1min')
    stock_df = format_ochlv_df(stock_df)
    # Initialize OpenAI Gym environment
    env = NormalizedEnv(StockTraderEnv(stock_df, max_steps=1000, lookback_period=5, init_balance=1000))

    print_gym_info(env)

    # Initialize networks
    agent = AgentDDPG(state_size=30, hidden_size=256, action_size=2, use_cuda=False)
    noise = OUNoise(env.action_space)

    trainer = ReinforcementTrainer(env, agent, noise)
    trainer.train(episodes=1000, timesteps=1000, batch_size=128, plot=True, log=True, render_env=False, save_path='stock_ddpg.pkl')

if __name__ == '__main__':
    main()     


