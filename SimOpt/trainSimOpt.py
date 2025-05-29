#from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from skopt import gp_minimize
from skopt.space import Real 
from skopt.plots import plot_convergence
#import nevergrad as ng
import matplotlib.pyplot as plt
import seaborn as sns
import random

from utils import build_env, trajectory_gap, simulate_and_gap, objective

#seed per mantenere risultati
SEED = 42                       # stesso numero per tutto il gruppo

os.environ["PYTHONHASHSEED"] = str(SEED)   # hash deterministico
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)

# Hyper‑parameters ----> non dovrebbero essere uguali come per ppo o ppo con udr?
total_timesteps = 100_000
eval_interval = 1_000 # How often to evaluate 
n_eval_episodes = 15
tol_var = 1e-3
lr = 1e-3
gamma = 0.99
BO_calls = 30 # Numero di iterazioni per la BO   
output_dir = Path.cwd() / "models_weights" / "SimOpt"
output_dir.mkdir(parents=True, exist_ok=True)


def train_policy(env: gym.Env, total_timesteps: int = 10_000):
    """Train a PPO policy for `total_timesteps` and return the model."""
    #PARAMETRI DA GRID SEARCH
    model = PPO("MlpPolicy", env, learning_rate=lr, gamma=gamma, verbose=0) 
    model.learn(total_timesteps=total_timesteps)
    return model

def objective(x):
    """
    Funzione obiettivo per la Bayesian Optimization.
    Minimizza la discrepanza tra traiettorie simulate e reali.

    Args:
        x (list): Lista delle masse [mass_thigh, mass_leg, mass_foot].

    Returns:
        float: Valore del trajectory gap tra env simulato e reale.
    """
    mass_thigh, mass_leg, mass_foot = x

    # Costruisci le masse con torso costante
    tmp_env = gym.make("CustomHopper-source-v0")
    tmp_env.seed(SEED)
    tmp_env.action_space.seed(SEED)
    tmp_env.observation_space.seed(SEED)
    masses = tmp_env.get_parameters()
    masses[1] = mass_thigh
    masses[2] = mass_leg
    masses[3] = mass_foot

    # Crea ambienti
    sim_env = build_env("source", masses)
    real_env = build_env("target", masses)

    # Allena una policy breve nel simulato
    model = train_policy(sim_env, total_timesteps=10_000)

    # Rollout nei due ambienti
    real_obs, sim_obs = [], []
    for domain, env, storage in [("REAL", real_env, real_obs), ("SIM", sim_env, sim_obs)]:
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            ep_traj = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = env.step(action)
                ep_traj.append(obs)
            storage.append(np.array(ep_traj))

    # Allinea lunghezze e calcola il gap
    min_len = min(min(len(t) for t in real_obs), min(len(t) for t in sim_obs))
    real_obs = [t[:min_len] for t in real_obs]
    sim_obs  = [t[:min_len] for t in sim_obs]

    disc = trajectory_gap(real_obs, sim_obs)
    print(f"[BO] Masses: {x} → Gap: {disc:.4f}")
    return disc

# Main SimOpt loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()


    # Mean and variance for thigh, leg and foot masses distributions (torso mass is fixed)
    # DOBBIAMO PRENDERE QUESTE MEDIE ?
    mass_dist = {
        "thigh":   [3.92699082, 0.5],
        "leg": [2.71433605, 0.5],
        "foot":  [5.08938010, 0.5],
    }

    step = 0
    while all(var > tol_var for _, var in mass_dist.values()):
        # Sample masses
        mass_thigh, mass_leg, mass_foot = [np.random.normal(mu, var, 1)[0] for mu, var in mass_dist.values() ]  
        tmp_env = gym.make("CustomHopper-source-v0")      # env temporaneo
        tmp_env.seed(SEED)
        tmp_env.action_space.seed(SEED)
        tmp_env.observation_space.seed(SEED)
        masses = tmp_env.get_parameters()            # masses = [torso, thigh, leg, foot]
        masses[1] = mass_thigh    # thigh
        masses[2] = mass_leg      # leg
        masses[3] = mass_foot     # foot
        # masses[0] (torso) does not change

        #Build env
        sim_env = build_env("source", masses)    # 4 valori → OK
    
        #Train a candidate policy in simulation
        model = train_policy(sim_env, total_timesteps=50_000)     #almeno 50k
        model_path = output_dir / "simopt_candidate.zip"
        model.save(model_path.as_posix())
        print(f"Saved temporary policy → {model_path.relative_to(Path.cwd())}")
        
        #Roll-outs: reale vs. sim 
        real_env      = build_env("target", masses)  # ambiente reale valutato su quali masse?
        sim_env_eval  = build_env("source", masses)  # fresh eval env

    

        # Evaluate the candidate policy in both environments
        real_obs, sim_obs = [], []
        for domain, env, storage in [("REAL", real_env, real_obs), ("SIM", sim_env_eval, sim_obs)]:
            for _ in range(n_eval_episodes):
                obs = env.reset()
                done = False
                ep_traj = []
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, _ = env.step(action)
                    ep_traj.append(obs)
                storage.append(np.array(ep_traj))
            mean_rwd, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=False)
            print(f"Average return {domain}: {mean_rwd:.1f}")

        # ------- Discrepancy (gap) ------------------------------------------
        ## Pad / trim per avere vettori uguali in lunghezza
        min_len = min(min(len(t) for t in real_obs), min(len(t) for t in sim_obs))
        real_obs = [t[:min_len] for t in real_obs]
        sim_obs  = [t[:min_len] for t in sim_obs]
        disc = trajectory_gap(real_obs, sim_obs)
        print(f"Discrepancy: {disc:.3f}")


        #Bayesian optimization step
        
        #spazio di ricerca per le masse: mu ± 2σ 
        search_space = [
            Real(mass_dist["thigh"][0] - 2 * mass_dist["thigh"][1], mass_dist["thigh"][0] + 2 * mass_dist["thigh"][1], name="thigh"),
            Real(mass_dist["leg"][0]   - 2 * mass_dist["leg"][1],   mass_dist["leg"][0]   + 2 * mass_dist["leg"][1],   name="leg"),
            Real(mass_dist["foot"][0]  - 2 * mass_dist["foot"][1],  mass_dist["foot"][0]  + 2 * mass_dist["foot"][1],  name="foot"),
        ]

        res = gp_minimize(
            func=lambda x: objective(masses[1:]),    #NON HA SENSO PERCHè è COSTANTE
            dimensions=search_space,
            n_calls=BO_calls,
            random_state=args.seed,
            verbose=True
        )

        # BO results
        print("Best found parameters:", res.x)
        print("Best found value (discrepancy):", res.fun)
        rec = {"thigh": res.x[0], "leg": res.x[1], "foot": res.x[2]}
        print("Recommended masses from Bayesian Optimization:", rec)

        #check convergence
        plot_convergence(res)
        plt.title("Convergence of Bayesian Optimization")
        plt.show()

        # Update masses distributions
        for key in mass_dist.keys():
            samples = np.random.normal(mass_dist[key][0], mass_dist[key][1], 300)
            samples = np.append(samples, rec[key])
            mass_dist[key][0] = float(np.mean(samples))   # μ
            mass_dist[key][1] = float(np.var(samples))    # σ²

            print(f"Updated {key} mass distribution: μ={mass_dist[key][0]:.3f}, σ²={mass_dist[key][1]:.3f}")
        
        ###### Save the model
        #model_path = output_dir / f"simopt_step_{step}.zip"
        #model.save(model_path.as_posix())
        #print(f"Saved model for step {step} → {model_path.relative_to(Path.cwd())}")

        step += 1

    #una volta che il SimOpt loop ha terminato, alleniamo la policy finale
    #print("Training completed. Finalizing the policy...")
    #final_model = train_policy(sim_env, total_timesteps=total_timesteps)
    #final_model_path = output_dir / "simopt_final_policy.zip"
    #final_model.save(final_model_path.as_posix())
    #print(f"Final policy saved → {final_model_path.relative_to(Path.cwd())}")

# ---------- Final training with converged distributions ----------
    print("\nStarting final training phase …")
    opt_masses = env.get_parameters()   # Use the last converged masses
    print(f"Final optimized masses: {opt_masses}")
    #bisognerebbe capire se cambia il torso su TARGET/SOURCE o no

    test_env = build_env("target", opt_masses)

    source_rewards = {}
    model = PPO("MlpPolicy", build_env("source", opt_masses), learning_rate=lr, gamma=gamma, verbose=0)
    for t in range(eval_interval, total_timesteps + 1, eval_interval):    ###PERCHè +1???
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        mean_r, _ = evaluate_policy(model, test_env, n_eval_episodes=n_eval_episodes, render=False)
        source_rewards[t] = mean_r
        print(f"Steps {t:6d}: mean reward on target = {mean_r:.1f}")


    final_path = output_dir / "simopt_ppo_final.zip"
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
