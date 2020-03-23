import torch
import gym
import numpy as np
from matplotlib import pyplot as plt
from ddpg import AgentDDPG
from noise import OrnsteinUhlenbeckProcess

def main():
    # Initialize OpenAI Gym environment
    env = gym.make('Pendulum-v0')

    print("Env action space: {}. Env observation space: {}".format(env.action_space, env.observation_space))
    print("act_space high: {}, low: {}".format(env.action_space.high, env.action_space.low))
    print("obs_space high: {}, low: {}".format(env.observation_space.high, env.observation_space.low))
    # Initialize networks
    agent = AgentDDPG(3, 1, use_cuda=True)
    noise_process = OrnsteinUhlenbeckProcess(theta=0.15, sigma=0.5)
    agent.set_noise_process(noise_process)

    trainer = ReinforcementTrainer(env, agent)
    weight_noise = lambda a: a.add_noise_to_weights(0.1)
    trainer.train(episodes=200, timesteps=500, batch_size=int(1e5), mut_alg_episode=weight_noise,
        log=True, render_env=False, save_path='pendulum_ddpg.pkl')
        
def simulate(model_file):
    env = gym.make('BipedalWalker-v3')
    agent = AgentDDPG(24, 4, use_cuda=False)
    noise_process = OrnsteinUhlenbeckProcess(theta=0.15, sigma=0.5)
    agent.set_noise_process(noise_process)
    trainer = ReinforcementTrainer(env, agent)
    trainer.simulate(5, 250, model_file, render_env=True)

def float_reward(reward) -> float:
    # try to put reward into a float
    if type(reward) == torch.Tensor:
        return float(reward.item())
    elif type(reward) != float:
        return float(reward)

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
                self.agent.save_experience(state, action, float_reward(reward), new_state)

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
            
            # Perform update
            if len(self.agent.replay_buffer) > batch_size:
                self.agent.update(batch_size) 

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

if __name__ == '__main__':
    main() 
#    simulate('ddpg.pkl')       
