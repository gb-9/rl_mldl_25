"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import importlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")           # in order not to plot inline
import matplotlib.pyplot as plt

import torch
import gym

from env.custom_hopper import *  

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes",  default=100_000, type=int,
                        help="Number of training episodes")
    parser.add_argument("--print-every", default=20_000, type=int,
                        help="Print info every N episodes")
    parser.add_argument("--device",      default="cpu", choices=["cpu", "cuda"],
                        help="Network device")
    parser.add_argument("--agent",       default="REINFORCE",
                        help="Name of the Python file inside 'agents/' "
                             "without the .py extension "
                             "(must expose classes Policy and Agent)")
    return parser.parse_args()


def import_agent_module(agent_name: str):
    """
    Dynamically imports agents
    """
    try:
        module = importlib.import_module(f"agents.{agent_name}")
    except ModuleNotFoundError as exc:
        print(f"[ERROR] agent {agent_name} not found!", file=sys.stderr)
        raise exc

    # check if you have policy and agent classes
    if not all(hasattr(module, cls) for cls in ("Policy", "Agent")):
        raise AttributeError(
            f"[ERROR] {agent_name} should contain 'Policy' and 'Agent' classes!"
        )
    return module.Policy, module.Agent

def main() -> None:
    args = parse_args()

    # --- import environment ------------------------------------------------
    env = gym.make("CustomHopper-source-v0")
    # env = gym.make('CustomHopper-target-v0')
    
    print("Action space: ", env.action_space)
    print("State space:  ", env.observation_space)
    print("Dynamics parameters:", env.get_parameters())

    # --- import agent ------------------------------------------------------
    PolicyClass, AgentClass = import_agent_module(args.agent)

    obs_dim  = env.observation_space.shape[-1]
    act_dim  = env.action_space.shape[-1]

    policy = PolicyClass(obs_dim, act_dim)
    agent  = AgentClass(policy, device=args.device)

    # --- training loop -----------------------------------------------------
    episode_rewards = []
    for episode in range(args.n_episodes):
        state = env.reset()        
        done, reward_tot = False, 0.0

        while not done:
            action, action_prob = agent.get_action(state)
            prev_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(prev_state, state, action_prob, reward, done)
            reward_tot += reward

        agent.update_policy()
        episode_rewards.append(reward_tot)

        if (episode + 1) % args.print_every == 0:
            print(f"Episode {episode + 1}: return = {reward_tot:.2f}")

    # --- save -------------------------------------------------------------
    torch.save(agent.policy.state_dict(), f"models/{args.agent}_model.mdl")

    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards  = np.array(episode_rewards)

    window = 20
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="same")

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards,  label="raw", alpha=0.3)
    plt.plot(episodes, smoothed, label=f"smoothed (w={window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"Learning Curve ({args.agent})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"training_curves/{args.agent}_training_curve.png")


if __name__ == "__main__":
    main()