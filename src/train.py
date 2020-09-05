from tqdm import tqdm
from stock_data import YahooFinance
import gym
import torch
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ddpg import AgentDDPG
from noise import OUNoise
from helpers import print_gym_info, format_ochlv_df
from gyms import HistoricalStockTraderEnv
from agent import TradingAgent   
from constants import TRADING_SYMBOLS, ACTOR_PATH, CRITIC_PATH, DDPG_ACTION_SIZE, DDPG_HIDDEN_SIZE, DDPG_STATE_SIZE, LOOKBACK_PERIOD, USE_CUDA

class ReinforcementTrainer:
    def __init__(self, model, noise, env):
        self.model = model
        self.noise = noise
        self.env = env

    def train(self, episodes,  batch_size, save: bool, log=False):
        """
        Trains agent within gym environment, taking timestep steps for each episode.
        Learning happens with batch_size experiences from ReplayBuffer
        """
        rewards = []
    
        for e in range(episodes):
            state = self.env.reset()
            self.noise.reset()
            episode_reward = 0

            # train with the entire time length every episode 
            for t in tqdm(range(self.env._timesteps)):
                action = self.model.get_action(state)
                action = self.noise.get_action(action, t)
                new_state, reward, done, _ = self.env.step(action)
                
                if done:
                    break

                self.model.save_experience(state, action, reward, new_state)

                if len(self.model.replay_buffer) > batch_size:
                    self.model.update(batch_size) 
        
                state = new_state
                episode_reward += reward
		
            rewards.append(episode_reward)
            
            if log:
                print("Episode {} | Reward {}".format(e, episode_reward))

            # Save model as file only if it performed better than before
            if save and episode_reward >= max(rewards) :
                self.model.save_to_file(ACTOR_PATH, CRITIC_PATH)
  

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
        
if __name__ == '__main__':
    # # Command Line Args: 
    # # argv[1] = path/to/train_data.csv
    
    # # Initialize stock_data dataframe from argv[1]
    # stock_df = pd.read_csv(sys.argv[1])
    # print('Loaded stock_df with shape {}\n'.format(stock_df.shape))

    # # Initialize OpenAI Gym environment
    # env = HistoricalStockTraderEnv(stock_df, lookback_period=LOOKBACK_PERIOD)

    # # Initialize DDPG model with exploration noise
    # model = AgentDDPG(state_size=DDPG_STATE_SIZE, hidden_size=DDPG_HIDDEN_SIZE, action_size=DDPG_ACTION_SIZE, use_cuda=USE_CUDA,
    #         actor_path=ACTOR_PATH, critic_path=CRITIC_PATH)
    # noise = OUNoise(env.action_space)

    # # Initialize Trainer and train the agent
    # trainer = ReinforcementTrainer(model, noise, env)
    # trainer.train(episodes=10, batch_size=256, log=True, save=True)
    # print("Done Training!\n")

    # Train on LIVE DATA

    # Initialize stock_data dataframe from argv[1]
    yf = YahooFinance()
    for symbol in tqdm(TRADING_SYMBOLS):
        print("Trading symbols: {}\n".format(TRADING_SYMBOLS))

        try:
            yf.get_current_price(symbol)
        except:
            continue

        stock_df = yf.get_intraday(symbol, '1m', 1000)

        if stock_df.shape[0] < 1000:
            print("{} had stock_df with shape {} so its skipped\n".format(symbol, stock_df.shape))
            continue

        print('Loaded {} stock_df with shape {}\n'.format(symbol, stock_df.shape))

        # Initialize OpenAI Gym environment
        env = HistoricalStockTraderEnv(stock_df, lookback_period=LOOKBACK_PERIOD)

        # Initialize DDPG model with exploration noise
        model = AgentDDPG(state_size=DDPG_STATE_SIZE, hidden_size=DDPG_HIDDEN_SIZE, action_size=DDPG_ACTION_SIZE, use_cuda=USE_CUDA,
                actor_path=ACTOR_PATH, critic_path=CRITIC_PATH)
        noise = OUNoise(env.action_space)

        # Initialize Trainer and train the agent
        trainer = ReinforcementTrainer(model, noise, env)
        trainer.train(episodes=2, batch_size=128, log=True, save=True)
        print("Done Training on {}!\n".format(symbol))