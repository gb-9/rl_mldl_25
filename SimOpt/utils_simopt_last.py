from typing import List
import numpy as np
import random
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from scipy.ndimage import gaussian_filter1d



SEED = 42  # Global seed for reproducibility

def gap(real_obs, sim_obs, w1: float = 1.0, w2: float = 0.1, sigma: float = 1.0) -> float:
    """Compute the discrepancy between two batches of trajectories.

    The metric is a weighted sum of smoothed L1 and L2 norms, similar to
    the formulation in the original SimOpt paper.

    Args:
        real_obs: list/array with shape (episodes, timesteps, obs_dim)
        sim_obs:  list/array with identical shape
        w1, w2:   weighting coefficients for L1 and squared L2 terms
        sigma:    Gaussian smoothing parameter

    Returns:
        Scalar discrepancy value.
    """
    real_arr = np.asarray(real_obs)
    sim_arr = np.asarray(sim_obs)
    if real_arr.shape != sim_arr.shape:
        raise ValueError("Observation tensors must have the same shape")

    diff = sim_arr - real_arr
    l1 = np.sum(np.abs(diff), axis=-1)
    l2 = np.sum(diff ** 2, axis=-1)

    l1_smoothed = gaussian_filter1d(l1, sigma=sigma)
    l2_smoothed = gaussian_filter1d(l2, sigma=sigma)
    gap = float(w1 * np.sum(l1_smoothed) + w2 * np.sum(l2_smoothed))
    # POTREBBE AVER SENSO NORMALIZZARE PER IL NUMERO DI OSSERVAZIONI COSì DA NON AVERE NUMERI ALTI?
    #n = np.prod(l1.shape)
    #gap = float((w1 * np.sum(l1_smoothed) + w2 * np.sum(l2_smoothed)) / n)

    return gap


#DA METTERE A POSTO
class HopperMassRandomGaussianWrapper(gym.Wrapper):
    """
    Domain randomization for Hopper link masses:
    sample thigh(2), leg(3), foot(4) masses from a Gaussian distribution each reset.
    Torso(1) mass remains fixed.
    ARGS:
        env: the Hopper environment to wrap
        phi: dictionary with keys 'thigh', 'leg', 'foot' and values as lists [mean, variance]
    
    """
    def __init__(self, env, phi):
        super().__init__(env)
        self.base_mass = env.sim.model.body_mass.copy()
        self.phi = phi

    def reset(self):
        # Restore original masses
        self.env.sim.model.body_mass[:] = self.base_mass

        # Randomize specified link masses, avoiding negative values
        mass_thigh = max(0.1,np.random.normal(self.phi['thigh'][0], self.phi['thigh'][1], 1))
        mass_leg = max(0.1,np.random.normal(self.phi['leg'][0], self.phi['leg'][1], 1))
        mass_foot = max(0.1,np.random.normal(self.phi['foot'][0], self.phi['foot'][1], 1)) 
        # Set new masses
        self.env.sim.model.body_mass[2] = mass_thigh
        self.env.sim.model.body_mass[3] = mass_leg
        self.env.sim.model.body_mass[4] = mass_foot

        return self.env.reset()

    def get_parameters(self):
        return self.env.sim.model.body_mass.copy()


def train_policy(env, log_dir: str, model_path: str, total_timesteps: int):
    """Train a PPO policy on the specified environment and save the model, without evaluation.
         partial implementation of the training function, without evaluation callback."""
   
    # 1) Monitor and vectorize
    env = Monitor(env, f"{log_dir}/train_monitor", allow_early_resets=True)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

     # 2) Check environment
    check_env(vec_env.envs[0])

    # 3) Create PPO model with linear lr schedule 
    lr_schedule = get_linear_fn(start=1e-4, end=0.0, end_fraction=1.0)     #PARAMETRI DA GRID SEARCH
    model = PPO(
        'MlpPolicy',
        vec_env,
        seed=SEED,
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

    # 4) Train
    model.learn(total_timesteps=total_timesteps)

    # 5) Save model 
    model.save(model_path)
    vec_env.save(f"{log_dir}/vecnormalize.pkl") ###########QUESTO MI SERVE IN QUESTO CASO?

    return model

###############CONTROLLARE CHE ABBIA SENSO CON UN PRINT DEI ROLLOUT
####vogliamo vedere anche le reward per fare un check?
def get_obs(model, env, n_episodes: int = 10):
    """Collect n_episodes rollouts from the environment using the trained model."""
    trajectories = []
    
    for _ in range(n_episodes):
        obs = env.reset()  # Reset the environment for each episode
        done = False
        ep_traj = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            ep_traj.append(obs)
        
        trajectories.append(np.array(ep_traj))
         
    return trajectories


#BO_objective mette tutto insieme
def BO_obj(phi: dict) -> float: 
    """Objective function for Bayesian Optimization.

    This function is called by the optimizer to evaluate the discrepancy
    between real and simulated environments with given parameters.
    """
    # 1) Build a temporary environment to train the policy
    tmp_env = HopperMassRandomGaussianWrapper(gym.make("CustomHopper-source-v0"), phi)
    tmp_env.seed(SEED)
    tmp_env.action_space.seed(SEED)
    tmp_env.observation_space.seed(SEED)
    
    total_timesteps = 50_000 

    # 2) Train a policy in the simulated temporary environment
    model = train_policy(tmp_env, log_dir="logs", model_path="model.zip", total_timesteps=total_timesteps)

    # 3) real rollouts
    real_env = gym.make("CustomHopper-target-v0")
    real_env.seed(SEED) #DEVO SETTARE SEED VARI?
    real_obs = get_obs(model, real_env, n_episodes=10)  

    # 4) sample masses from initial distributions
    sampled_masses = [np.random.normal(mu, var, 1)[0] for mu, var in phi.values()]
    print(f"Sampled masses: {sampled_masses}")   #per check

    # 5) simulated rollouts
    sim_env = gym.make("CustomHopper-source-v0")
    masses = sim_env.get_parameters()            # masses = [torso, thigh, leg, foot]
    masses[1] = sampled_masses[0]    # thigh
    masses[2] = sampled_masses[1]      # leg
    masses[3] = sampled_masses[2]     # foot
    sim_env.set_parameters(masses)  

    sim_obs = get_obs(model, sim_env, n_episodes=10)  

    # Align lengths of trajectories
    min_len = min(min(len(t) for t in real_obs), min(len(t) for t in sim_obs))
    real_obs = [t[:min_len] for t in real_obs]
    sim_obs  = [t[:min_len] for t in sim_obs]

    # 5) Compute discrepancy (gap)
    gap_value = gap(real_obs, sim_obs, w1=1.0, w2=0.1, sigma=1.0)
    print(f"Discrepancy (gap): {gap_value:.3f}")

    return gap_value  # Return the discrepancy as the objective value for BO

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
    base = os.path.basename(file_path).replace('.csv', '')
    folder = os.path.basename(os.path.dirname(file_path))
    plt.savefig(f"training_curves/{folder}_{base}.png")
    plt.close()

def train_and_save(env_id, log_dir, model_path, total_timesteps, phi):
    print(f"\n Prepare environment {env_id} for training...")

    # 1) Environment creation with domain randomization
    base_env = gym.make(env_id)
    base_env.seed(SEED)
    env = HopperMassRandomGaussianWrapper(
        base_env,
        phi 
    )
    
    # 2) Monitor and vectorize
    env = Monitor(env, f"{log_dir}/train_monitor", allow_early_resets=True)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    print('State space:', vec_env.observation_space)
    print('Action space:', vec_env.action_space)
    print('Link masses:', vec_env.envs[0].get_parameters())

    # 3) Setup evaluation environment
    eval_base = gym.make(env_id)
    eval_base.seed(SEED+1)
    eval_env = Monitor(eval_base, f"{log_dir}/eval_monitor", allow_early_resets=True)
    eval_vec = DummyVecEnv([lambda: eval_env])
    eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False)

    # 4) Check environment
    check_env(vec_env.envs[0])

    # 5) Create PPO model with linear lr schedule 
    lr_schedule = get_linear_fn(start=1e-4, end=0.0, end_fraction=1.0)     #PARAMETRI DA GRID SEARCH
    model = PPO(
        'MlpPolicy',
        vec_env,
        seed=SEED,
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
        best_model_save_path=f'{log_dir}/best_model', #'./ppo_hopper_logs/' oppure '{log_dir}/best_model'
        log_path=log_dir, #'./ppo_hopper_logs/' oppure log_dir
        eval_freq=6000,
        deterministic=True
    )

    # 7) Train
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # 8) Save model and normalization stats
    model.save(model_path)
    vec_env.save(f"{log_dir}/vecnormalize.pkl")

    # 9) Final evaluation
    mean_ret, std_ret = evaluate_policy(
        model, eval_vec, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean return on {env_id}: {mean_ret:.2f} ± {std_ret:.2f}")

    # 10) Plot learning curves
    plot_learning_curve(env, f"{log_dir}/train_monitor.csv")
    plot_learning_curve(eval_env, f"{log_dir}/eval_monitor.csv")
