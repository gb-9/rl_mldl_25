from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import nevergrad as ng
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import build_env, trajectory_gap

# Hyper‑parameters

TOTAL_TIMESTEPS = 100_000
EVAL_INTERVAL = 1_000
N_EVAL_EPISODES = 50
CMA_BUDGET = 1_300
TOL_VAR = 1e-3
LR = 1e-3
GAMMA = 0.99
OUTPUT_DIR = Path("models_weights/SimOpt")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Helper functions
# -------------------------

def train_policy(env: gym.Env, total_timesteps: int = 10_000):
    """Train a PPO policy for `total_timesteps` and return the model."""
    model = PPO("MlpPolicy", env, learning_rate=LR, gamma=GAMMA, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return model


# -------------------------
# Main SimOpt loop
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Mean and variance for three body‑part masses (hip, thigh, foot)
    mass_stats = {
        "hip":   [3.92699082, 0.5],
        "thigh": [2.71433605, 0.5],
        "foot":  [5.08938010, 0.5],
    }

    step = 0
    while all(var > TOL_VAR for _, var in mass_stats.values()):
        # ------- Sample masses & build simulation env -------
        masses = [np.random.normal(mu, var, 1)[0] for mu, var in mass_stats.values()]
        sim_env = build_env("source", masses)

        # ------- Train a candidate policy in simulation -------
        model = train_policy(sim_env, total_timesteps=10_000)
        model_path = OUTPUT_DIR / "simopt_candidate.zip"
        model.save(model_path.as_posix())
        print(f"Saved temporary policy → {model_path.relative_to(Path.cwd())}")

        # ------- Roll‑outs in real and sim domains -------
        real_env = build_env("target", masses)
        sim_env_eval = build_env("source", masses)  # fresh eval env

        real_obs, sim_obs = [], []
        for domain, env, storage in [("REAL", real_env, real_obs), ("SIM", sim_env_eval, sim_obs)]:
            for _ in range(N_EVAL_EPISODES):
                obs = env.reset()
                done = False
                ep_traj = []
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, _ = env.step(action)
                    ep_traj.append(obs)
                storage.append(np.array(ep_traj))
            mean_rwd, _ = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES, render=False)
            print(f"Average return {domain}: {mean_rwd:.1f}")

        # ------- Discrepancy & CMA‑ES update -------
        # Pad / trim to common length
        min_len = min(min(len(t) for t in real_obs), min(len(t) for t in sim_obs))
        real_obs = [t[:min_len] for t in real_obs]
        sim_obs = [t[:min_len] for t in sim_obs]
        disc = trajectory_gap(real_obs, sim_obs)
        print(f"Discrepancy: {disc:.3f}")

        # Optimiser variables
        param = ng.p.Dict(
            hip=ng.p.Scalar(init=mass_stats["hip"][0]).set_mutation(sigma=mass_stats["hip"][1]),
            thigh=ng.p.Scalar(init=mass_stats["thigh"][0]).set_mutation(sigma=mass_stats["thigh"][1]),
            foot=ng.p.Scalar(init=mass_stats["foot"][0]).set_mutation(sigma=mass_stats["foot"][1]),
        )
        optim = ng.optimizers.CMA(parametrization=param, budget=CMA_BUDGET)
        for _ in range(optim.budget):
            x = optim.ask()
            optim.tell(x, disc)
        rec = optim.recommend().value
        print("Recommended masses from CMA‑ES:", rec)

        # Update distributions
        for key in mass_stats:
            samples = np.random.normal(mass_stats[key][0], mass_stats[key][1], 300)
            samples = np.append(samples, rec[key])
            mass_stats[key][0] = float(np.mean(samples))
            mass_stats[key][1] = float(np.var(samples))
        print("Updated mass distributions:")
        for k, (mu, var) in mass_stats.items():
            print(f"  {k:<6}: μ={mu:.4f}, σ²={var:.4f}")

        step += 1

    # ---------- Final training with converged distributions ----------
    print("\nStarting final training phase …")
    test_env = build_env("target", masses)

    source_rewards = {}
    model = PPO("MlpPolicy", build_env("source", masses), learning_rate=LR, gamma=GAMMA, verbose=0)
    for t in range(EVAL_INTERVAL, TOTAL_TIMESTEPS + 1, EVAL_INTERVAL):
        model.learn(total_timesteps=EVAL_INTERVAL, reset_num_timesteps=False)
        mean_r, _ = evaluate_policy(model, test_env, n_eval_episodes=N_EVAL_EPISODES, render=False)
        source_rewards[t] = mean_r
        print(f"Steps {t:6d}: mean reward on target = {mean_r:.1f}")

    final_path = OUTPUT_DIR / "simopt_ppo_final.zip"
    model.save(final_path.as_posix())
    print(f"Saved final policy → {final_path.relative_to(Path.cwd())}")

    # Optional: plot
    try:
        steps, rewards = zip(*sorted(source_rewards.items()))
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=steps, y=rewards)
        plt.title("SimOpt performance on target domain")
        plt.xlabel("Training steps")
        plt.ylabel("Mean reward")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plotting failed:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
