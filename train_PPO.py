"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *

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


def plot_learningcurves(monitor_env, file_path):
    
    returns = monitor_env.get_episode_rewards()
    if len(returns) == 0:
        print("⚠️ Nessuna reward episodica trovata.")
        return

    returns = np.array(returns)
    # df = pd.read_csv(file_path, skiprows=2, header=None, names=["r", "l", "t"])
    # returns = df["r"].values

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

    # Salva la figura nella directory desiderata
    if not os.path.exists('learning_curves'):
        os.makedirs('learning_curves', exist_ok=True)

    # Costruisci un nome file coerente in base al path del CSV
    base_name = os.path.basename(file_path).replace('.monitor.csv', '')
    # Prendi la cartella genitore per differenziare i nomi
    folder_name = os.path.basename(os.path.dirname(file_path))
    plt.savefig(f"learning_curves/returns_plot_{folder_name}_{base_name}.png")
    # plt.show()

def train_and_save(env_id, log_dir, model_path, seed=42):
    print(f"\n🚀 Training on {env_id}...")

    seed = 42
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

    # Verifica che l’ambiente sia compatibile con Stable-Baselines3
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
        policy="MlpPolicy",
        env=train_env,
        verbose=0,
        n_steps=8192,       # Numero di passi di simulazione raccolti prima di ogni aggiornamento della policy
        batch_size=64,      # Numero di esempi usati in ogni minibatch durante il training
        gae_lambda=0.9,    # Parametro per Generalized Advantage Estimation (GAE). Più vicino a 1 → meno bias, più varianza
        gamma=0.99,         # Fattore di sconto per le ricompense future
        n_epochs=15,        # Quante volte ogni batch di dati viene riutilizzato per aggiornare la policy 
        clip_range=0.2,     # Range di clipping per il rapporto tra policy attuale e precedente
        ent_coef=0.005,     # Coefficiente dell’entropia nella loss: Maggiore → più esplorazione
        vf_coef=0.5,        # Coefficiente per della value function nella loss
        max_grad_norm=0.5,  # Serve a prevenire problemi di esplosione del gradiente.
        learning_rate = lr_schedule   # Learning rate dinamico
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
    print(f"✅ Mean reward on {env_id}: {mean_reward:.2f} ± {std_reward:.2f}")

    plot_learningcurves(monitor_train_env, f'{log_dir}/train_monitor.monitor.csv')
    plot_learningcurves(monitor_eval_env, f'{log_dir}/eval_monitor.monitor.csv')

def main():
    train_and_save(
        env_id='CustomHopper-source-v0',
        log_dir='./ppo_hopper_logs_source',
        model_path='./ppo_hopper_final_model_source'
    )

    train_and_save(
        env_id='CustomHopper-target-v0',
        log_dir='./ppo_hopper_logs_target',
        model_path='./ppo_hopper_final_model_target'
    )

if __name__ == '__main__':
    main()
