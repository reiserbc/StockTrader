import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
from helpers import normalize_ochlv, match_shape
from data_structures import Portfolio

""" 
LOOKBACK_PERIOD: # of history periods seen in observation
INITIAL_BALANCE: initial trading balance
MAX_STEPS: maximum number of steps an agent can take wihin this environment
"""
LOOKBACK_PERIOD = 5
INITIAL_BALANCE = 1000
MAX_STEPS = 20000

class StockTraderEnv(gym.Env):
    """ Stock Trader Environment following OpenAI Gym Interface """
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_df):
        """ Initialize environment.
        stock_df:
            - Pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        super(StockTraderEnv, self).__init__()
        self.stock_df = stock_df

        self.portfolio = Portfolio(init_balance=INITIAL_BALANCE)
        self._max_price = stock_df[['open', 'high', 'low', 'close']].max().max()
        self._max_vol = stock_df['volume'].max()
        
        # Actions of format Buy x%, Sell x%, Hold x% (NOTE: Hold % is irrelevant)
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3,1], dtype=np.float32))
        # Observations of normalized OCHLV values for the last LOOKBACK_PERIOD periods
        self.observation_space = spaces.Box(low=0, high=1, shape=(LOOKBACK_PERIOD, 5 + 1), dtype=np.float32)
        self.state = self._new_state()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached the environment's state is reset.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self._take_action(action)
        
        self._next_step()
    
        observation = self._get_observation()
        reward = self._calculate_reward(action)
        done = self._is_done()
        info = {}

        return observation, reward, done, info

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        self.state = self._new_state()
        return self._get_observation()

    def render(self, mode='human', close=False):
        """ Render environment to the screen """
        print(f'Step: {self.state["curr_step"]}')
        print(f'Df Loc ind: {self.state["df_loc"]}')
        print(f'Balance: {self.portfolio.get_balance()}')
        print(f'Shares held: {self.portfolio.get_shares_owned()}')
        print(f'Avg cost for held shares: {self.portfolio.get_cost_basis()}')
        print(f'Net worth: {self.portfolio.get_net_worth(self.current_price())}')
        print(f'Profit: {self.portfolio.get_profit(self.current_price())}')
    
    def current_price(self):
        """ Returns the stock price at df_loc in self.stock_df """
        i = self.state["df_loc"]
        price = random.uniform(self.stock_df.iloc[i]["open"], self.stock_df.iloc[i]["close"])
        return price

    def _take_action(self, action): 
        """Take an action in form ACTION: int, PERCENT: float, where 0=buy, 1=sell, and 2=hold. """
        # Set price to a random price within [Open, Close]
        price = self.current_price()

        action_type, amount = action

        if action_type == 0:
            # Buy Action
            # buy % of balance worth of shares
            max_possible_shares = self.portfolio.get_balance() // price
            shares_to_buy = int(max_possible_shares * amount)
            self.portfolio.buy(shares_to_buy, price)
            
        elif action_type == 1:
            # Sell Action
            # sell % of owned shares
            shares_to_sell = int(self.portfolio.get_shares_owned() * amount)
            self.portfolio.sell(shares_to_sell, price)
    
    def _next_step(self): 
        self.state["df_loc"] += 1
        if self.state["df_loc"] >= len(self.stock_df - LOOKBACK_PERIOD):
            self.state["df_loc"] = 0
        
        self.state['curr_step'] += 1

    def _new_state(self) -> dict:
        # Returns an initial state for the environment
        return {
            "df_loc": random.randint(0, len(self.stock_df)-1), 
            "curr_step": 0, 
        }
    
    def _get_observation(self) -> object:
        # Returns the observation for curr_period 
        i = self.state["df_loc"]
        obs_df = self.stock_df[i : i + LOOKBACK_PERIOD]

        # normalize to between 0-1
        obs_df = normalize_ochlv(obs_df, self._max_price, self._max_vol)

        # add empty entries into portfolio_arr to match obs_arr shape
        portfolio_df = match_shape(of=self.portfolio.as_df(), to=obs_df)
        return np.concatenate((obs_df.to_numpy(), portfolio_df.to_numpy()))

    def _calculate_reward(self, action) -> float:
        """ Return the Reward for action """

        # Reward maintaining a large balance for a long time
        delay_modifier = self.state["curr_step"] / MAX_STEPS
        return self.portfolio.get_balance() * delay_modifier

    def _is_done(self) -> bool:
        """ Returns True when net_worth <= 0 """
        return self.portfolio.get_net_worth(self.current_price()) <= 0 or self.state['curr_step'] >= MAX_STEPS