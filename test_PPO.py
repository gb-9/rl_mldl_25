"""Test two agent"""
import argparse

import numpy as np

import gym
from env.custom_hopper import *

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=50, type=int, help='Number of test episodes')
    
    return parser.parse_args()


def test_sb3_model(model_path, env_id, episodes, render):

    env_raw = gym.make(env_id)
    env_raw = Monitor(env_raw)
    # DummyVecEnv è richiesto per VecNormalize
    vec_env = DummyVecEnv([lambda: env_raw])
    # Percorso del file vecnormalize salvato durante il training
    log_dir = './ppo_hopper_logs_source' if 'source' in env_id else './ppo_hopper_logs_target'
    normalize_path = os.path.join(log_dir, "vecnormalize.pkl")

    # Carica VecNormalize, se esiste
    if os.path.exists(normalize_path):
        vec_env = VecNormalize.load(normalize_path, vec_env)
        vec_env.training = False  # Disattiva aggiornamento statistiche
        vec_env.norm_reward = False  # Non normalizzare reward a test time
    else:
        print(f"⚠️ Warning: VecNormalize non trovato a {normalize_path}. Test senza normalizzazione.")

    model = PPO.load(model_path, env=vec_env)
    env = vec_env

    
    returns = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            
            if render:
                env.render()
        
        print(f"Episode {ep+1}: Return = {float(total_reward):.2f}")
        returns.append(total_reward)
        
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    env.close()
    return mean_return, std_return

def main():

	args = parse_args()
	
	test_cases = [
        ('source→source', './ppo_hopper_final_model_source.zip', 'CustomHopper-source-v0'),
        ('source→target', './ppo_hopper_final_model_source.zip', 'CustomHopper-target-v0'),
        ('target→target', './ppo_hopper_final_model_target.zip', 'CustomHopper-target-v0'),
    ]

	print(f"Running tests on fixed configurations with {args.episodes} episodes each\n")
     
	for label, model_path, env_id in test_cases:
         mean_ret, std_ret = test_sb3_model(model_path, env_id, episodes=args.episodes, render=args.render)
         print(f"{label} | Env: {env_id} | Mean Return: {mean_ret:.2f} ± {std_ret:.2f}")


if __name__ == '__main__':
	main()
