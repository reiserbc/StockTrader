import torch
import gym
import numpy as np
from matplotlib import pyplot as plt
from ddpg import AgentDDPG
from noise import OrnsteinUhlenbeckProcess

def main():
    # Initialize OpenAI Gym environment
    env = gym.make('BipedalWalker-v3')

    print("Env action space: {}. Env observation space: {}".format(env.action_space, env.observation_space))
    print("act_space high: {}, low: {}".format(env.action_space.high, env.action_space.low))
    print("obs_space high: {}, low: {}".format(env.observation_space.high, env.observation_space.low))
    # Initialize networks
    agent = AgentDDPG(24, 4, use_cuda=False)
    noise_process = OrnsteinUhlenbeckProcess(theta=0.15, sigma=0.5)
    agent.set_noise_process(noise_process)

    trainer = ReinforcementTrainer(env, agent)
    f1 = lambda a: a.add_noise_to_weights(0.3)
    f2 = lambda a: a.add_noise_to_weights(0.1)
    trainer.train(episodes=200, timesteps=100, batch_size=64, mut_alg_episode=f1, mut_alg_step=f2)

def simulate(model_file):
    env = gym.make('BipedalWalker-v3')
    agent = AgentDDPG(24, 4, use_cuda=False)
    noise_process = OrnsteinUhlenbeckProcess(theta=0.15, sigma=0.5)
    agent.set_noise_process(noise_process)
    trainer = ReinforcementTrainer(env, agent)
    trainer.simulate(5, 250, model_file, render_env=True)

class ReinforcementTrainer:
    def __init__(self, gym, agent):
        self.gym = gym
        self.agent = agent

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
            self.agent.noise_process.reset()
            episode_reward = 0

            for t in range(timesteps):
                if render_env:
                    self.gym.render()

                action = self.agent.get_action(state, noise=True)
                new_state, reward, done, info = self.gym.step(action.cpu())
                print(type(state), type(action), type(reward), type(new_state))
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
            torch.save(self.agent, 'ddpg.pkl')

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

if __name__ == '__main__':
	#main() 
    simulate('ddpg.pkl')       
