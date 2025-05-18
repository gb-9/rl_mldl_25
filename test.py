"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse
import importlib
import sys
from pathlib import Path

import torch
import gym

from env.custom_hopper import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",   default="REINFORCE",
                        help="Python file in agents/ (no .py) to load")
    parser.add_argument("--model",   default=None,
                        help="Path of .mdl weights file "
                             "(defaults to models_weights/<agent>_model.mdl)")
    parser.add_argument("--device",  default="cpu", choices=["cpu", "cuda"],
                        help="Device for networks")
    parser.add_argument("--episodes", default=10, type=int,
                        help="Number of evaluation episodes")
    parser.add_argument("--render",  action="store_true",
                        help="Render simulation")
    return parser.parse_args()

def import_agent_module(agent_name: str):
    try:
        module = importlib.import_module(f"agents.{agent_name}")
    except ModuleNotFoundError as exc:
        print(f"[ERROR] agent '{agent_name}' not found!", file=sys.stderr)
        raise exc

    if not all(hasattr(module, cls) for cls in ("Policy", "Agent")):
        raise AttributeError(
            f"[ERROR] {agent_name} should expose 'Policy' and 'Agent' classes"
        )
    return module.Policy, module.Agent


def main() -> None:
    args = parse_args()

    # --- environment -------------------------------------------------------
    env = gym.make("CustomHopper-source-v0")
    # env = gym.make("CustomHopper-target-v0")

    print("Action space:", env.action_space)
    print("State space :", env.observation_space)
    print("Dynamics parameters:", env.get_parameters())

    # --- import agent ------------------------------------------------------
    PolicyClass, AgentClass = import_agent_module(args.agent)

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    policy = PolicyClass(obs_dim, act_dim).to(args.device)

    model_path = (Path(args.model)
                  if args.model is not None
                  else Path(f"models_weights/{args.agent}_model.mdl"))

    if not model_path.exists():
        raise FileNotFoundError(
            f"[ERROR] weights file '{model_path}' does not exist "
            "(pass --model <path> if different)"
        )

    policy.load_state_dict(torch.load(model_path, map_location=args.device))

    agent = AgentClass(policy, device=args.device)

    # --- evaluation loop ---------------------------------------------------
    for episode in range(args.episodes):
        state = env.reset()
        done, test_reward = False, 0.0

        while not done:
            with torch.no_grad():
                action, _ = agent.get_action(state, evaluation=True)

            state, reward, done, info = env.step(action.cpu().numpy())
            if args.render:
                env.render()
            test_reward += reward

        print(f"Episode: {episode} | Return: {test_reward:.2f}")

if __name__ == "__main__":
    main()