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
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from ur5_grasp_env import UR5RobotiqGraspEnv
from ur5_grasp_env.callbacks import TrainingMetricsCallback
from ur5_grasp_env.plotting import plot_training_curves


def make_env(
    seed: int,
    rank: int,
    render_mode: str | None = None,
    curriculum_level: int = 0,
    max_episode_steps: int = 250,
):
    def _init():
        env = UR5RobotiqGraspEnv(
            render_mode=render_mode,
            seed=seed + rank,
            curriculum_level=curriculum_level,
            max_episode_steps=max_episode_steps,
        )
        return Monitor(env)

    return _init


def build_vec_env(n_envs: int, seed: int, curriculum_level: int, max_episode_steps: int):
    if n_envs <= 1:
        return VecMonitor(DummyVecEnv([make_env(seed, 0, curriculum_level=curriculum_level, max_episode_steps=max_episode_steps)]))
    return VecMonitor(
        SubprocVecEnv(
            [
                make_env(
                    seed,
                    rank,
                    curriculum_level=curriculum_level,
                    max_episode_steps=max_episode_steps,
                )
                for rank in range(n_envs)
            ]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for UR5 + Robotiq85 grasping in MuJoCo.")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/ppo_ur5_grasp"))
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--plot-after-train", action="store_true")
    parser.add_argument("--csv-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--append-metrics", action="store_true")
    parser.add_argument("--save-metrics-freq", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--curriculum-level", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (args.log_dir / "best_model").mkdir(parents=True, exist_ok=True)
    (args.log_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (args.log_dir / "eval").mkdir(parents=True, exist_ok=True)
    (args.log_dir / "tensorboard").mkdir(parents=True, exist_ok=True)

    rollout_batch = args.n_steps * max(args.n_envs, 1)
    if rollout_batch < args.batch_size:
        raise ValueError(
            f"Invalid PPO batch config: n_steps * n_envs = {rollout_batch} "
            f"must be >= batch_size = {args.batch_size}."
        )

    if args.check_env:
        check_env(
            UR5RobotiqGraspEnv(
                seed=args.seed,
                curriculum_level=args.curriculum_level,
                max_episode_steps=args.max_episode_steps,
            ),
            warn=True,
            skip_render_check=True,
        )

    env = build_vec_env(args.n_envs, args.seed, args.curriculum_level, args.max_episode_steps)
    eval_env = VecMonitor(
        DummyVecEnv(
            [
                make_env(
                    args.seed + 10_000,
                    0,
                    curriculum_level=args.curriculum_level,
                    max_episode_steps=args.max_episode_steps,
                )
            ]
        )
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.log_dir / "tensorboard"),
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            log_std_init=-1.0,
        ),
    )
    logger_formats = ["stdout"]
    if args.csv_log:
        logger_formats.append("csv")
    if args.tensorboard:
        logger_formats.append("tensorboard")
    model.set_logger(configure(str(args.log_dir / "tensorboard"), logger_formats))

    callbacks = [
        TrainingMetricsCallback(
            log_dir=args.log_dir,
            save_freq=args.save_metrics_freq,
            append=args.append_metrics,
        ),
        CheckpointCallback(
            save_freq=max(10_000 // max(args.n_envs, 1), 1),
            save_path=str(args.log_dir / "checkpoints"),
            name_prefix="ppo_ur5_grasp",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(args.log_dir / "best_model"),
            log_path=str(args.log_dir / "eval"),
            eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        ),
    ]

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)
    final_path = args.log_dir / "ppo_ur5_grasp_final"
    model.save(final_path)
    env.close()
    eval_env.close()
    print(f"Saved final model to {final_path}.zip")
    if args.plot_after_train:
        for path in plot_training_curves(args.log_dir):
            print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
