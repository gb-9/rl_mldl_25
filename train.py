"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')         # backend non-interattivo che non cerca matplotlib_inline
import matplotlib.pyplot as plt

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)


	episode_rewards = [] #graf1
	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
		
		agent.update_policy() #update policy qua?
		episode_rewards.append(train_reward) #graf2
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)


	torch.save(agent.policy.state_dict(), "model.mdl")

	# graf3
	window = 20
	rewards = np.array(episode_rewards)

	# media mobile “same” mantiene la dimensione:
	smoothed = np.convolve(rewards, np.ones(window)/window, mode='same')

	# asse x unico
	x = np.arange(1, len(rewards)+1)

	plt.figure(figsize=(10,5))
	plt.plot(x, rewards,  color='lightgray', label='raw')
	plt.plot(x, smoothed, color='C0',        label=f'smoothed (w={window})')
	plt.xlabel('Episode')
	plt.ylabel('Return')
	plt.title('REINFORCE on CustomHopper')
	plt.legend()
	plt.tight_layout()
	plt.savefig("training_curve.png")
	#plt.show()
	

if __name__ == '__main__':
	main()