"""Utility functions shared by SimOpt training / evaluation scripts."""

from typing import List
import numpy as np
from scipy.ndimage import gaussian_filter1d
import gym
from env.custom_hopper import CustomHopper  # Local import – adjust if path differs


def build_env(domain: str, mass_dist_params: List[np.ndarray]):
    """Create a CustomHopper environment with a given domain and parameters.

    Args:
        domain: Either "source" or "target" (controls physical engine setup).
        mass_dist_params: Sequence of mass values to set in the environment.

    Returns:
        An initialised `gym.Env` instance.
    """
    if domain not in {"source", "target"}:
        raise ValueError("domain must be 'source' or 'target'")
    env_id = f"CustomHopper-{domain}-v0"
    env = gym.make(env_id)
    # the first element in masses returned by `get_parameters` is often a dummy
    if mass_dist_params is not None:
        env.set_parameters(mass_dist_params)
    return env


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
