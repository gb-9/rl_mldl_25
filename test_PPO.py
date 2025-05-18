import argparse
import numpy as np
import gym
import os
import random
import torch

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Fix seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate trained PPO models on CustomHopper envs')
    parser.add_argument(
        '--device', default='cpu', type=str,
        help='Device for policy inference (cpu or cuda)')
    parser.add_argument(
        '--render', dest='render', action='store_true',
        help='Render the environment during evaluation')
    parser.add_argument(
        '--no-render', dest='render', action='store_false',
        help='Do not render during evaluation')
    parser.set_defaults(render=True)
    parser.add_argument(
        '--episodes', default=50, type=int,
        help='Number of episodes per test case')
    return parser.parse_args()


def test_sb3_model(model_path, env_id, episodes, render):
    # Create and wrap environment
    env_raw = gym.make(env_id)
    env_raw.seed(SEED)
    env = Monitor(env_raw)
    vec_env = DummyVecEnv([lambda: env])

    # Load observation normalization if available
    log_dir = './ppo_hopper_logs_source' if 'source' in env_id else './ppo_hopper_logs_target'
    norm_path = os.path.join(log_dir, 'vecnormalize.pkl')
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print(f"⚠️ No VecNormalize found at {norm_path}, continuing without normalization.")

    # Load the trained model
    model = PPO.load(model_path, env=vec_env)
    model.set_env(vec_env)

    returns = []
    for ep in range(1, episodes+1):
        obs = vec_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = vec_env.step(action)
            total_reward += float(reward)
            if render:
                vec_env.render()
        returns.append(total_reward)
        print(f"Episode {ep}: Return = {total_reward:.2f}")

    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    vec_env.close()
    return mean_ret, std_ret


def main():
    args = parse_args()
    test_cases = [
        ('source→source', './ppo_hopper_final_model_source.zip', 'CustomHopper-source-v0'),
        ('source→target', './ppo_hopper_final_model_source.zip', 'CustomHopper-target-v0'),
        ('target→target', './ppo_hopper_final_model_target.zip', 'CustomHopper-target-v0'),
    ]

    print(f"Running evaluation for {args.episodes} episodes per case (render={args.render})\n")
    for label, model_path, env_id in test_cases:
        mean_ret, std_ret = test_sb3_model(
            model_path, env_id,
            episodes=args.episodes,
            render=args.render
        )
        print(f"{label} | Env: {env_id} | Mean Return: {mean_ret:.2f} ± {std_ret:.2f}")


if __name__ == '__main__':
    main()
