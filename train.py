import torch
import gym
import numpy as np
from matplotlib import pyplot as plt
from ddpg import AgentDDPG, add_noise_to_weights
from noise import OrnsteinUhlenbeckProcess

# Initialize OpenAI Gym environment
env = gym.make('BipedalWalker-v3')
env.reset()

print("Env action space: {}. Env observation space: {}".format(env.action_space, env.observation_space))
print("act_space high: {}, low: {}".format(env.action_space.high, env.action_space.low))
print("obs_space high: {}, low: {}".format(env.observation_space.high, env.observation_space.low))
# Initialize networks
agent = AgentDDPG(24, 4, use_cuda=True)
noise_process = OrnsteinUhlenbeckProcess(theta=0.15, sigma=0.5)
agent.set_noise_process(noise_process)
# Training params
batch_size = 64
rewards = []
avg_rewards = []

for episode in range(1000):
    state = env.reset()
    # reset noice_process episodically
    agent.noise_process.reset_states()
    
    episode_reward = 0
    for t in range(400):
        #env.render()
        action = agent.get_action(state, noise=True)
        new_state, reward, done, info = env.step(action.cpu())
        agent.save_experience(state, action, reward, new_state)
        
        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size) 
        
        state = new_state
        episode_reward += reward

        if done:
            print("episode: {}, reward: {}, average_reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break
        
        if t % 15 == 0:
            with torch.no_grad():
                agent.actor.apply(lambda x: add_noise_to_weights(x, amount=0.3))
                agent.critic.apply(lambda x: add_noise_to_weights(x, amount=0.3))
    
    with torch.no_grad():
        agent.actor.apply(lambda x: add_noise_to_weights(x, amount=0.7))
        agent.critic.apply(lambda x: add_noise_to_weights(x, amount=0.7))

    
    rewards.append(episode_reward)

    # plt.plot(rewards)
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.draw()
    # plt.pause(0.1)


