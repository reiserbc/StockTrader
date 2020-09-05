import numpy as np

class TradingAgent:
    def __init__(self, symbol, model, env):
        self.symbol = symbol
        self.model = model
        self.env = env
        self.history = {}
    
    def perform(self, command):
        action = command_to_action(command)
        # run user action through live stock trading environment
        new_state, reward, done, info = self.env.step( action )

    def predict(self):
        curr_state = self.env._get_observation()
        action = self.model.get_action(curr_state)
        command = action_to_command(action, self.symbol)

        print("curr_state.shape: {}".format(curr_state.shape))
        print("action: {}".format(action))

        return command

class TradingAgentCommand:
    def __init__(self, action_type, symbol, amount):
        self.action_type = action_type
        self.symbol = symbol
        self.amount = amount
    
    def __str__(self):
        return "{} {} {}".format(self.action_type, self.symbol, self.amount)

def buy_command(symbol, amount):
    return TradingAgentCommand("BUY", symbol, amount)

def sell_command(symbol, amount):
   return TradingAgentCommand("SELL", symbol, amount)

def hold_command(symbol):
   return TradingAgentCommand("HOLD", symbol, 0)

# returns a TradingAgentCommand from its string encoded representation
def parse_command(line: str):
    action_type, symbol, amount = line.split()
    return TradingAgentCommand(action_type, symbol, amount)

# returns a StockTraderEnv action from TradingAgentCommand
def command_to_action(command: TradingAgentCommand):
    # TODO
    print("TODO ensure that command.amount is less than 1")
    print(command.amount)
    
    if command.action_type == "BUY":
        return np.array([0, command.amount])
    elif command.action_type == "SELL":
        return np.array([1, command.amount])
    elif command.action_type == "HOLD":
        return np.array([2, 0])

def action_to_command(action: np.array, symbol: str):
    action_type, amount = action[0], action[1]

    if action_type == 0:
        return TradingAgentCommand("BUY", symbol, amount)
    elif action_type == 1:
        return TradingAgentCommand("SELL", symbol, amount)
    elif action_type == 1:
        return TradingAgentCommand("HOLD", symbol, 0)

from collections import deque

class TradingAgentCommandFeeder:
    def __init__(self):
        self.queue = deque()
        
    def get_next(self):
        self.queue.popleft()
    
    def add_command(self, command: TradingAgentCommand):
        self.queue.append(command)

    def __str__(self):
        s = ""
        for x in self.queue:
            s += str(x) +"\n" 
        return s