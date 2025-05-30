"""Utility functions shared by SimOpt training / evaluation scripts."""

from typing import List
import numpy as np
from scipy.ndimage import gaussian_filter1d
import gym
from env.custom_hopper import CustomHopper  # Local import – adjust if path differs


def build_env(domain: str, mass_dist_params=None):
    env = gym.make(f"CustomHopper-{domain}-v0")
    if mass_dist_params is not None:
        if len(mass_dist_params) != 4:
            raise ValueError(f"Expected 4 masses, got {len(mass_dist_params)}")
        env.set_parameters(mass_dist_params)
    return env

def simulate_and_gap(model, real_env, sim_env, EPISODES_EVAL=50):
    """Roll-out della policy in real_env e sim_env e calcolo del gap."""
    real_obs, sim_obs = [], []

    for _ in range(EPISODES_EVAL):
        # --- rollout reale ---------------------------------------------------
        obs_r, done_r, ep_r = real_env.reset(), False, []
        while not done_r:
            act, _ = model.predict(obs_r, deterministic=True)
            obs_r, _, done_r, _ = real_env.step(act)
            ep_r.append(obs_r)
        real_obs.append(np.concatenate(ep_r))

        # --- rollout simulato -----------------------------------------------
        obs_s, done_s, ep_s = sim_env.reset(), False, []
        while not done_s:
            act, _ = model.predict(obs_s, deterministic=True)
            obs_s, _, done_s, _ = sim_env.step(act)
            ep_s.append(obs_s)
        sim_obs.append(np.concatenate(ep_s))

    # ---------- uniforma la lunghezza delle sequenze ------------------------
    min_len = min(min(len(t) for t in real_obs), min(len(t) for t in sim_obs))
    real_obs = [t[:min_len] for t in real_obs]
    sim_obs  = [t[:min_len] for t in sim_obs]

    return trajectory_gap(real_obs, sim_obs)



def trajectory_gap(real_obs, sim_obs, w1: float = 1.0, w2: float = 0.1, sigma: float = 1.0) -> float:
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

    return float(w1 * np.sum(l1_smoothed) + w2 * np.sum(l2_smoothed))
