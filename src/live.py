# Live-running Trading Agent on a web server
import asyncio
import time
import fileinput
from sanic import Sanic
from sanic.response import json
from agent import TradingAgent
from constants import USE_CUDA, ACTOR_PATH, CRITIC_PATH, DDPG_ACTION_SIZE, DDPG_HIDDEN_SIZE, DDPG_STATE_SIZE, TRADING_INTERVAL, TRADING_SYMBOLS, LOOKBACK_PERIOD
from gyms import LiveStockTraderEnv
from stock_data import YahooFinance
from agent import TradingAgent, TradingAgentCommand, parse_command, TradingAgentCommandFeeder
from ddpg import AgentDDPG

# in an infinite loop, read commands from stdin, have trading_agent perform commands, and proceed to next timestep
def run_live_trading_agent(trading_agent, command_feeder):
    while True:
        command = command_feeder.get_next()
        if command:
            trading_agent.perform(command)
        time.sleep(60)
        

# spawn the async live trading agent process
async def spawn_live_trading_agent(trading_agent, command_feeder):
    process = await asyncio.create_subprocess_exec(
        run_live_trading_agent(trading_agent, command_feeder),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return process


# Initialize what we need to run live
try:
    # Initialize TradingAgent
    model = AgentDDPG(state_size=DDPG_STATE_SIZE, hidden_size=DDPG_HIDDEN_SIZE, action_size=DDPG_ACTION_SIZE, use_cuda=USE_CUDA, 
                        actor_path=ACTOR_PATH, critic_path=CRITIC_PATH)

    # initialize trading environment and trading agent for each symbol
    data_puller = YahooFinance()
    envs = []
    agents = []
    
    # initialize async process for trading agent and command feeder to communicate with them
    command_feeders = []
    processes = []

    for ts in TRADING_SYMBOLS:
        env = LiveStockTraderEnv(ts, TRADING_INTERVAL, data_puller, LOOKBACK_PERIOD, data_puller.get_current_price(ts) * 100)
        agent = TradingAgent(ts, model, env)
        
        envs.append(env)
        agents.append(agent)

        # spawn the async trading agent processes
        command_feeder = TradingAgentCommandFeeder()
        process = spawn_live_trading_agent(agent, command_feeder)

        command_feeders.append(command_feeder)
        processes.append(process)


except Exception as e:
    print(e)

# initialize web-server routes
app = Sanic("StockTrader")

@app.route('/ping')
async def ping(request):
    return json({"ping": "pong"})

@app.route("/symbols")
async def symbols(request):
    return json({"trading_symbols": TRADING_SYMBOLS})

@app.route("/predict")
async def predict(request):
    symbol = request.args['symbol'][0]
    agent = agents[TRADING_SYMBOLS.index(symbol)]
    command = agent.predict()
    return json({"symbol": symbol, "action_type": command.action_type, "amount": int(command.amount)})

@app.route("/save-trade")
async def save_trade(request):
    action, symbol, quantity = request.args['action'][0], request.args['symbol'][0], request.args['quantity'][0]

    command_feeder = command_feeders[TRADING_SYMBOLS.index(symbol)]
    command = TradingAgentCommand(action.upper(), symbol, quantity)
    command_feeder.add_command(command)

    return json(request.args)

@app.route("/view-commands")
async def view_commands(request):
    commands = [str(cf) for cf in command_feeders]
    
    return json({"commands": commands})

if __name__ == "__main__":
    # start web-server
    app.run(host='localhost', port=8080, debug=True)
