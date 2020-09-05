import numpy as np
import gym
from collections import deque
import random
from tqdm import tqdm
import pickle

def copy_params(from_net, to_net):
    """Copy parameter weights in-place from from_net to to_net"""
    for target_param, param in zip(to_net.parameters(), from_net.parameters()):
        target_param.data.copy_(param.data)

def soft_copy_params(from_net, to_net, tau):
    for target_param, param in zip(to_net.parameters(), from_net.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

def normalize_ochlv(df, max_price, max_vol):
    # Return a 0-1 normalized df
    df = df.copy()
    df["open"] = df["open"] / max_price
    df["high"] = df["high"] / max_price
    df["low"] = df["low"] / max_price
    df["close"] = df["close"] / max_price
    df["volume"] = df["volume"] / max_vol
    return df

def format_ochlv_df(df):
    # only works for colums
    assert '1. open' in df.columns
    df = df.rename(columns={"1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close", "5. volume": "volume"})
    return df
    
def match_shape(of, to):
    """Adds columns of 0 into of DataFrame s.t. len(of.columns) == len(to.columns)
    len(of.columns) must be <= len(to.columns)
    """
    fill_val = 0
    while len(of.columns) < len(to.columns):
        column_name = to.columns[len(of.columns)-1]
        of[column_name] = fill_val
    return of

def print_gym_info(env):
    print("Env action space: {}. Env observation space: {}".format(env.action_space, env.observation_space))


from stock_data import YahooFinance

def load_symbols_from_file(path):
   with open(path, 'rb') as fp:
       return pickle.load(fp)

def fix_symbol_file(path):
    f = open(path, "r")
    lines = f.readlines()
    yf = YahooFinance()
    symbols = []
    failed = []
    for line in tqdm(lines):
        symbol = line.split()[0]
        
        # check if we can get data for this symbol
        try:
            yf.get_current_price(symbol)
            symbols.append(symbol)

        except:
            failed.append(symbol)

    print("The following symbols have failed to work with YahooFinance: {}".format(failed))
    symbols = [line.split()[0] for line in lines]
    
    with open(path, 'wb') as fp:
        pickle.dump(symbols, fp)

if __name__ == '__main__':
    import pickle
    fix_symbol_file("symbols/TSX.txt")
    fix_symbol_file("symbols/NYSE.txt")
    fix_symbol_file("symbols/NASDAQ.txt")