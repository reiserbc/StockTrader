import torch
import gym
import numpy as np
from matplotlib import pyplot as plt
from ddpg import AgentDDPG
from noise import OUNoise
from train import ReinforcementTrainer
import pandas as pd
import sys
from gyms import HistoricalStockTraderEnv
from agent import TradingAgent   
from helpers import print_gym_info, format_ochlv_df
from constants import USE_CUDA, ACTOR_PATH, CRITIC_PATH, DDPG_ACTION_SIZE, DDPG_HIDDEN_SIZE, DDPG_STATE_SIZE, LOOKBACK_PERIOD

class TradingSimulator:
    def __init__(self, agent):
        self.agent = agent
        self.model = agent.model
        self.env = agent.env
        
        self.summary = None

    def simulate(self):
        state = self.env.reset()

        states = []
        actions = []
        rewards = []
        while True:
            action = self.model.get_action(state)
            new_state, reward, done, _ = self.env.step(action)
            
            if done:
                break
    
            state = new_state
            
            # update summary
            states += state
            actions += action
            rewards += reward
    
        self.summary = {"states": states, "actions": actions, "rewards": rewards}

    def summary(self):
        return self.summary

if __name__ == '__main__':
    # Command Line Args: 
    # argv[1] = path/to/train_data.csv
    
    # Initialize stock_data dataframe from argv[1]
    try:
        df = pd.read_csv(sys.argv[1])
    except:
        print("argv[1] = path/to/train_data.csv")

    stock_df = format_ochlv_df(df)

    # Initialize OpenAI Gym environment
    env = HistoricalStockTraderEnv(stock_df, lookback_period=LOOKBACK_PERIOD)

    # Initialize Trading agent

    # Initialize saved DDPG model from file
    model = AgentDDPG(state_size=DDPG_STATE_SIZE, hidden_size=DDPG_HIDDEN_SIZE, action_size=DDPG_ACTION_SIZE, 
            use_cuda=USE_CUDA, actor_path=ACTOR_PATH, critic_path=CRITIC_PATH)
    
    agent = TradingAgent("", model, env)

    # Simulate on env
    simulator = TradingSimulator(agent)
    simulator.simulate()
    print(simulator.summary())
