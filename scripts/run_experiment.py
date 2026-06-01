from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train, evaluate, and plotting for one experiment.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--ent-coef", type=float, default=0.002)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path("runs") / args.name
    train_cmd = [
        sys.executable,
        "train.py",
        "--log-dir",
        str(log_dir),
        "--total-timesteps",
        str(args.total_timesteps),
        "--n-envs",
        str(args.n_envs),
        "--eval-episodes",
        str(args.eval_episodes),
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--gamma",
        str(args.gamma),
        "--ent-coef",
        str(args.ent_coef),
        "--clip-range",
        str(args.clip_range),
        "--batch-size",
        str(args.batch_size),
        "--n-steps",
        str(args.n_steps),
        "--plot-after-train",
    ]
    _run(train_cmd)

    best_model = log_dir / "best_model" / "best_model.zip"
    final_model = log_dir / "ppo_ur5_grasp_final.zip"
    model_path = best_model if best_model.exists() else final_model
    eval_cmd = [
        sys.executable,
        "evaluate.py",
        "--model",
        str(model_path),
        "--episodes",
        str(args.eval_episodes),
        "--no-render",
        "--deterministic",
        "--csv-path",
        str(log_dir / "metrics" / "eval_episode_metrics.csv"),
        "--plot-after-eval",
    ]
    _run(eval_cmd)
    _run([sys.executable, "scripts/plot_training.py", "--log-dir", str(log_dir)])
    _run([sys.executable, "scripts/plot_evaluation.py", "--log-dir", str(log_dir)])


def _run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
