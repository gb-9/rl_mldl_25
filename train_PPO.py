"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class HopperUDREnv(gym.Env):
    """
    A Gym Env that wraps 'CustomHopper-source-v0' and, at every reset(),
    randomises thigh (2), leg (3) and foot (4) masses uniformly within
    the given ranges, *around* the original source values.
    The torso (1) remains fixed (–30% wrt target).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    DEFAULT_RANGES = {
        2: (0.7, 1.3),   # thigh
        3: (0.7, 1.3),   # leg
        4: (0.7, 1.3)    # foot
    }

    def __init__(self, ranges=None, seed=None):
        super().__init__()
        # 1) create the underlying Hopper env
        self.inner = gym.make("CustomHopper-source-v0")
        # 2) set up RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        # 3) store your randomisation ranges
        self.ranges = ranges if ranges is not None else self.DEFAULT_RANGES
        # 4) grab the *fixed* source masses once and for all
        self.inner.reset()
        self.base_mass = self.inner.sim.model.body_mass.copy()
        # 5) mirror action/obs spaces so SB3 sees them correctly
        self.action_space      = self.inner.action_space
        self.observation_space = self.inner.observation_space

    def reset(self, seed=None, options=None):
        # (re-seed if user asked)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        # restore the original source masses
        self.inner.sim.model.body_mass[:] = self.base_mass
        # apply new uniform random factors
        for idx, (low, high) in self.ranges.items():
            factor = self.np_random.uniform(low, high)
            self.inner.sim.model.body_mass[idx] *= factor
        # now reset the inner env and return its obs+info
        # Gym ≥0.26 returns (obs, info)
        result = self.inner.reset(seed=seed, options=options)
        return result

    def step(self, action):
        # simply delegate
        return self.inner.step(action)

    def render(self):
        return self.inner.render()

    def close(self):
        return self.inner.close()
    
    def get_parameters(self):
        # Return the current body masses so your print() call works
        return self.inner.sim.model.body_mass.copy()


def plot_learningcurves(monitor_env, file_path):
    
    returns = monitor_env.get_episode_rewards()
    if len(returns) == 0:
        print("⚠️ Nessuna reward episodica trovata.")
        return

    returns = np.array(returns)

    window = 20
    smoothed = np.convolve(returns, np.ones(window) / window, mode="same")

    plt.figure(figsize=(8, 4))
    plt.plot(returns, label="Returns")
    plt.plot(smoothed, label=f"smoothed (w={window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()

    # Save
    if not os.path.exists('training_curves'):
        os.makedirs('training_curves', exist_ok=True)

    base_name = os.path.basename(file_path).replace('.monitor.csv', '')
    folder_name = os.path.basename(os.path.dirname(file_path))
    
    plt.savefig(f"training_curves/returns_plot_{folder_name}_{base_name}.png")

def train_and_save(env_id, log_dir, model_path, seed=42, use_udr=False):
    print(f"\n Training on {env_id}...")

    seed = 42
    if "source" in env_id and use_udr:
        env = HopperUDREnv(ranges={2:(0.7,1.3),3:(0.7,1.3),4:(0.7,1.3)}, seed=seed)
    else:
        env = gym.make(env_id)
        env.seed(seed)

    # Aggiungi il wrapper Monitor all'ambiente di train
    # A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
    # per allenare e valutare un agente, è consigliato avvolgere l'ambiente con il Monitor wrapper, 
    # per evitare che venga modificata la durata degli episodi o le ricompense in modo non 
    # intenzionale da parte di altri wrapper
    env = Monitor(env, f"{log_dir}/train_monitor", allow_early_resets=True)
    monitor_train_env = env
    train_env = DummyVecEnv([lambda: env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.envs[0].get_parameters())  # masses of each link of the Hopper

    # Learning rate che va da 2.5e-4 a 0 durante il training
    lr_schedule = get_linear_fn(start=1e-4, end=0.0, end_fraction=1.0)

    eval_env_raw = gym.make(env_id)
    eval_env_raw.seed(seed + 1)
    # Aggiungi il wrapper Monitor all'ambiente di valutazione
    eval_env_raw = Monitor(eval_env_raw, f"{log_dir}/eval_monitor", allow_early_resets=True)
    monitor_eval_env = eval_env_raw
    eval_env = DummyVecEnv([lambda: eval_env_raw])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    # check if environment is compatible with stable baseline
    check_env(train_env.envs[0])

    
    # Ogni eval_freq timesteps, il modello viene valutato.
    # Se la reward media è la migliore ottenuta finora, il modello viene salvato in ./ppo_hopper_logs/best_model.zip.
    # I risultati (media, deviazione standard, numero di episodi) vengono loggati in ./ppo_hopper_logs/
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path='./ppo_hopper_logs/',
                                 log_path='./ppo_hopper_logs/',
                                 eval_freq=6000,
                                 deterministic=True, 
                                 render=False)


    model = PPO(
        policy="MlpPolicy", #Which neural-network architecture to use for the actor–critic. 
                            #MlpPolicy = a feed-forward multilayer perceptron (the SB3 default: two hidden layers of 64 units)
        env=train_env,      #environment
        verbose=0,          #silent if =0
        n_steps=8192,       # Number of environment steps per parallel environment to gather before each policy update
        batch_size=64,      # Size of the mini-batches sampled from the collected rollout when running gradient descent
        gae_lambda=0.9,    # parameter in Generalized Advantage Estimation. the closest to 1 the lest bias, the more variance we have
        gamma=0.99,         # discount factor for future rewards
        n_epochs=15,        # How many passes of SGD to perform over each rollout batch
        clip_range=0.2,     # Range of clipping between new and previous policy. limits how far the new policy is allowed to deviate 
                            # from the old one in a single update
        ent_coef=0.005,     # coefficient of entropy of loss: the bigger, the more exploration we have
        vf_coef=0.5,        # Weight of the value-function loss term relative to the policy loss
        max_grad_norm=0.5,  # Gradient-clipping threshold to avoid gradient exploding
        learning_rate = lr_schedule   # dynamical Learning rate previously computed
    )

    model.learn(total_timesteps=2_000_000, callback=eval_callback)
    model.save(model_path)
    train_env.save(f"{log_dir}/vecnormalize.pkl")

    mean_reward, std_reward = evaluate_policy(
        model,
        env=eval_env,
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Mean reward on {env_id}: {mean_reward:.2f} ± {std_reward:.2f}")

    plot_learningcurves(monitor_train_env, f'{log_dir}/train_monitor.monitor.csv')
    plot_learningcurves(monitor_eval_env, f'{log_dir}/eval_monitor.monitor.csv')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-udr", action="store_true",
                        help="Enable Uniform Domain Randomization on the source env")
    args = parser.parse_args()

    train_and_save(
        env_id='CustomHopper-source-v0',
        log_dir='./ppo_hopper_logs_source',
        model_path='./ppo_hopper_final_model_source',
        use_udr=args.use_udr
    )

    train_and_save(
        env_id='CustomHopper-target-v0',
        log_dir='./ppo_hopper_logs_target',
        model_path='./ppo_hopper_final_model_target',
        use_udr=False        # never UDR on target
    )

if __name__ == '__main__':
    main()
