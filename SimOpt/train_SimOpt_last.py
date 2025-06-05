#CONTROLLARE TUTTI GLI IMPORTI
import gym
import argparse
import numpy as np
import random
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sys

from pathlib import Path

from skopt import gp_minimize
from skopt.space import Real 
from skopt.plots import plot_convergence

import seaborn as sns
import random

from utils_SimOpt import train_and_save, BO_obj

# Fix seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

tol_var = 1e-4  # Variance threshold for convergence    
BO_calls = 30  # Number of Bayesian Optimization calls

# Main SimOpt loop
def main(): 
    
    # 1) distribution p_phi_0
    # Mean (μ) and variance (σ²) for initial distributions for thigh, leg and foot masses (torso mass is fixed)
    phi_masses = {
        "thigh":   [3.92699082, 0.5],
        "leg": [2.71433605, 0.5],
        "foot":  [5.08938010, 0.5],
    }

    step = 0
    while all(var > tol_var for _, var in phi_masses.values()):

        # 2) Bayesian Optimization
        # search space for masses: mu ± 2σ 
        search_space = [
            Real(phi_masses["thigh"][0] - 2 * phi_masses["thigh"][1], phi_masses["thigh"][0] + 2 * phi_masses["thigh"][1], name="thigh"),
            Real(phi_masses["leg"][0]   - 2 * phi_masses["leg"][1],   phi_masses["leg"][0]   + 2 * phi_masses["leg"][1],   name="leg"),
            Real(phi_masses["foot"][0]  - 2 * phi_masses["foot"][1],  phi_masses["foot"][0]  + 2 * phi_masses["foot"][1],  name="foot"),
        ]

        #scrivere qualcosa che dica che si sta iniziando la BO
        #print(f"\nStarting Bayesian Optimization step {step}")

        res = gp_minimize(
            func=lambda x: BO_obj({
                "thigh": [x[0], phi_masses["thigh"][1]],
                "leg":   [x[1], phi_masses["leg"][1]],
                "foot":  [x[2], phi_masses["foot"][1]],
            }),     
            dimensions=search_space,
            n_calls=BO_calls,   
            random_state=SEED,
            verbose=True
        )

        # Plot convergence
        plt.figure(figsize=(10, 5))
        plot_convergence(res)
        plt.title(f"Bayesian Optimization Convergence (Step {step})")
        plt.savefig(f"bo_convergence_step_{step}.png")
        plt.close()

        # BO results
        print("Best found parameters:", res.x)
        print("Best found value (discrepancy):", res.fun)
        rec = {"thigh": res.x[0], "leg": res.x[1], "foot": res.x[2]}
        print("Recommended masses from Bayesian Optimization:", rec)

        #HA SENSO FARLO COSì O MAGARI RIMPIAZZIAMO LE MEDIE DELL'ITERAZIONE PRECEDENTE?
        # Update masses distributions
        for key in phi_masses.keys():
            samples = np.random.normal(phi_masses[key][0], phi_masses[key][1], 300)
            samples = np.append(samples, rec[key])
            phi_masses[key][0] = float(np.mean(samples))   # μ
            phi_masses[key][1] = float(np.var(samples))    # σ² SIGMA O SIGMA QUADRO?

            print(f"Updated {key} mass distribution: μ={phi_masses[key][0]:.3f}, σ²={phi_masses[key][1]:.3f}")

        step += 1
    
    # After convergence, phi_masses contains the final distributions
    phi_optimal = phi_masses.copy()
    print("\nConverged distributions:")
    for key, (mu, var) in phi_optimal.items():
        print(f"{key} mass: μ={mu:.3f}, σ²={var:.3f}")

    # Final training with converged distributions
    print("\nStarting final training phase …")
    train_and_save(
        'CustomHopper-source-v0', 
        './simopt_hopper_logs_source',  # Log directory for training
        './simopt_hopper_final_source', # Model path for saving the final model
        total_timesteps=2_000_000,
        phi=phi_optimal
    )

    #
    

if __name__ == "__main__":
    main()
