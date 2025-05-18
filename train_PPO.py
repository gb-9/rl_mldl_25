import gym
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class HopperMassRandomWrapper(gym.Wrapper):
    """
    Simple domain randomization for Hopper link masses:
    multiply thigh(2), leg(3), foot(4) masses by a uniform factor each reset.
    Torso(1) mass remains fixed.
    """
    def __init__(self, env, ranges):
        super().__init__(env)
        self.base_mass = env.sim.model.body_mass.copy()
        self.ranges = ranges

    def reset(self):
        # Restore original masses
        self.env.sim.model.body_mass[:] = self.base_mass
        # Randomize specified link masses
        for idx, (low, high) in self.ranges.items():
            factor = np.random.uniform(low, high)
            self.env.sim.model.body_mass[idx] *= factor
        return self.env.reset()

    def get_parameters(self):
        return self.env.sim.model.body_mass.copy()



def plot_learning_curve(monitor_env, file_path):
    rewards = np.array(monitor_env.get_episode_rewards())
    if rewards.size == 0:
        print("No rewards to plot.")
        return
    smoothed = np.convolve(rewards, np.ones(20)/20, mode='same')
    plt.figure(figsize=(8,4))
    plt.plot(rewards, alpha=0.3, label='raw')
    plt.plot(smoothed, label='smoothed')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend()
    os.makedirs('training_curves', exist_ok=True)
    base = os.path.basename(file_path).replace('.monitor.csv', '')
    folder = os.path.basename(os.path.dirname(file_path))
    plt.savefig(f"training_curves/{folder}_{base}.png")
    plt.close()


def train_and_save(env_id, log_dir, model_path, use_udr=False):
    print(f"\n1) Prepare environment {env_id} (UDR={use_udr})...")
    # 1) Environment creation and optional UDR
    if env_id == 'CustomHopper-source-v0' and use_udr:
        base_env = gym.make(env_id)
        env = HopperMassRandomWrapper(
            base_env,
            ranges={2:(0.7,1.3), 3:(0.7,1.3), 4:(0.7,1.3)}
        )
    else:
        env = gym.make(env_id)
    
    # 2) Monitor and vectorize
    env = Monitor(env, f"{log_dir}/train_monitor", allow_early_resets=True)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    print('State space:', vec_env.observation_space)
    print('Action space:', vec_env.action_space)
    print('Link masses:', vec_env.envs[0].get_parameters())

    # 3) Setup evaluation environment
    eval_base = gym.make(env_id)
    eval_env = Monitor(eval_base, f"{log_dir}/eval_monitor", allow_early_resets=True)
    eval_vec = DummyVecEnv([lambda: eval_env])
    eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False)

    # 4) Check environment
    check_env(vec_env.envs[0])

    # 5) Create PPO model with linear lr schedule
    lr_schedule = get_linear_fn(start=1e-4, end=0.0, end_fraction=1.0)
    model = PPO(
        'MlpPolicy',
        vec_env,
        verbose=0,
        n_steps=8192,
        batch_size=64,
        gae_lambda=0.9,
        gamma=0.99,
        n_epochs=15,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=lr_schedule
    )

    # 6) Evaluation callback
    eval_callback = EvalCallback(
        eval_vec,
        best_model_save_path='./ppo_hopper_logs/',
        log_path='./ppo_hopper_logs/',
        eval_freq=6000,
        deterministic=True
    )

    # 7) Train
    model.learn(total_timesteps=2_000_000, callback=eval_callback)

    # 8) Save model and normalization stats
    model.save(model_path)
    vec_env.save(f"{log_dir}/vecnormalize.pkl")

    # 9) Final evaluation
    mean_ret, std_ret = evaluate_policy(
        model, eval_vec, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean return on {env_id}: {mean_ret:.2f} ± {std_ret:.2f}")

    # 10) Plot learning curves
    plot_learning_curve(env, f"{log_dir}/train_monitor.monitor.csv")
    plot_learning_curve(eval_env, f"{log_dir}/eval_monitor.monitor.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-udr', action='store_true')
    args = parser.parse_args()

    train_and_save(
        'CustomHopper-source-v0',
        './ppo_hopper_logs_source',
        './ppo_hopper_final_model_source',
        use_udr=args.use_udr
    )
    train_and_save(
        'CustomHopper-target-v0',
        './ppo_hopper_logs_target',
        './ppo_hopper_final_model_target',
        use_udr=False
    )

if __name__ == '__main__':
    main()
