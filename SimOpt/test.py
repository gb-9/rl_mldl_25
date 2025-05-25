"""Evaluate a trained SimOpt PPO policy on the target domain."""

import argparse
import gym
from stable_baselines3 import PPO
from env.custom_hopper import CustomHopper  # needed for gym registry side‑effect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models_weights/SimOpt/simopt_ppo_final.zip", help="Path to trained policy")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = gym.make("CustomHopper-target-v0")
    mean_rwd, std_rwd = 0.0, 0.0
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_rwd, std_rwd = evaluate_policy(model, env, n_eval_episodes=args.episodes, render=True)
    print(f"Evaluation over {args.episodes} episodes: {mean_rwd:.1f} ± {std_rwd:.1f}")


if __name__ == "__main__":
    main()
