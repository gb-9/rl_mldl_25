"""Evaluate a trained SimOpt PPO policy on the target domain."""

"""Evaluate a trained SimOpt PPO policy on the target domain."""

from pathlib import Path
import argparse
import gym
from stable_baselines3 import PPO
from env.custom_hopper import CustomHopper        # registra l'env dentro gym
from stable_baselines3.common.evaluation import evaluate_policy

# Directory dove train.py salva i pesi
OUTPUT_DIR = Path.cwd() / "models_weights" / "SimOpt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=(OUTPUT_DIR / "simopt_ppo_final.zip").as_posix(),
        help="Path to trained policy",
    )
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    # Carica il modello
    model = PPO.load(args.model)

    # Crea l'ambiente target
    env = gym.make("CustomHopper-target-v0")

    # Valutazione
    mean_rwd, std_rwd = evaluate_policy(model, env, n_eval_episodes=args.episodes, render=True)
    print(f"Evaluation over {args.episodes} episodes: {mean_rwd:.1f} ± {std_rwd:.1f}")


if __name__ == "__main__":
    main()

