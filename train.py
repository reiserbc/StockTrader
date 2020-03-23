import torch
import gym
import numpy as np
from matplotlib import pyplot as plt
from ddpg import AgentDDPG
from noise import OUNoise
from helpers import print_gym_info

def main():
    # Initialize OpenAI Gym environment
    env = NormalizedEnv(gym.make('Pendulum-v0'))

    print_gym_info(env)

    # Initialize networks
    agent = AgentDDPG(state_size=3, hidden_size=32, action_size=1, use_cuda=False)
    noise = OUNoise(env.action_space)

    trainer = ReinforcementTrainer(env, agent, noise)

    trainer.train(episodes=200, timesteps=500, batch_size=128, plot=True, render_env=True, save_path='pendulum_ddpg.pkl')
        
def simulate(model_file):
    # env = gym.make('BipedalWalker-v3')
    # agent = AgentDDPG(24, 4, use_cuda=False)
    # noise_process = OrnsteinUhlenbeckProcess(theta=0.15, sigma=0.5)
    # agent.set_noise_process(noise_process)
    # trainer = ReinforcementTrainer(env, agent)
    # trainer.simulate(5, 250, model_file, render_env=True)
    pass

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
        avg_rewards = []
    
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

            if plot:
                plt.plot(rewards)
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.draw()
                plt.pause(0.1)

        if save_path:
            torch.save(self.agent, save_path)

    def simulate(self, episodes, timesteps, load_path=None, render_env=False, plot=False, log=False):
        agent = torch.load(load_path) if load_path else self.agent

        rewards = []
        avg_rewards = []
    
        for e in range(episodes):
            state = self.gym.reset()
            agent.noise_process.reset()
            episode_reward = 0

            for t in range(timesteps):
                if render_env:
                    self.gym.render()

                action = agent.get_action(state, noise=True)
                new_state, reward, done, info = self.gym.step(action.cpu())

                state = new_state
                episode_reward += reward
	
            rewards.append(episode_reward)
            
            if log:
                print("Episode {} | Reward {}".format(e, episode_reward))

        if plot:
            plt.plot(rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.draw()
            plt.pause(0.1)

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
    main() 
#    simulate('ddpg.pkl')       


