from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from ur5_grasp_env import UR5RobotiqGraspEnv


def make_env(seed: int, rank: int, render_mode: str | None = None):
    def _init():
        env = UR5RobotiqGraspEnv(render_mode=render_mode, seed=seed + rank)
        return Monitor(env)

    return _init


def build_vec_env(n_envs: int, seed: int):
    if n_envs <= 1:
        return VecMonitor(DummyVecEnv([make_env(seed, 0)]))
    return VecMonitor(SubprocVecEnv([make_env(seed, rank) for rank in range(n_envs)]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for UR5 + Robotiq85 grasping in MuJoCo.")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/ppo_ur5_grasp"))
    parser.add_argument("--check-env", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (args.log_dir / "best_model").mkdir(parents=True, exist_ok=True)

    if args.check_env:
        check_env(UR5RobotiqGraspEnv(seed=args.seed), warn=True, skip_render_check=True)

    env = build_vec_env(args.n_envs, args.seed)
    eval_env = VecMonitor(DummyVecEnv([make_env(args.seed + 10_000, 0)]))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.log_dir / "tensorboard"),
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.002,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(10_000 // max(args.n_envs, 1), 1),
            save_path=str(args.log_dir / "checkpoints"),
            name_prefix="ppo_ur5_grasp",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(args.log_dir / "best_model"),
            log_path=str(args.log_dir / "eval"),
            eval_freq=max(10_000 // max(args.n_envs, 1), 1),
            n_eval_episodes=10,
            deterministic=True,
        ),
    ]

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)
    final_path = args.log_dir / "ppo_ur5_grasp_final"
    model.save(final_path)
    env.close()
    eval_env.close()
    print(f"Saved final model to {final_path}.zip")


if __name__ == "__main__":
    main()
