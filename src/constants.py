import torch
from helpers import load_symbols_from_file

ACTOR_PATH = "models/ddpg_actor.pkl"
CRITIC_PATH = "models/ddpg_critic.pkl"

DDPG_STATE_SIZE = 5005
DDPG_HIDDEN_SIZE = 512
DDPG_ACTION_SIZE = 2

TRADING_INTERVAL = '1m'
LOOKBACK_PERIOD = 1000


TRADING_SYMBOLS = load_symbols_from_file('symbols/TSX.txt') + load_symbols_from_file('symbols/NYSE.txt') + load_symbols_from_file('symbols/NASDAQ.txt')

USE_CUDA =  torch.cuda.is_available()
