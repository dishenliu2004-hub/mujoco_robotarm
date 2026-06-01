from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(
    log_dir: Path,
    output_dir: Path | None = None,
    smoothing_window: int = 20,
) -> list[Path]:
    log_dir = Path(log_dir)
    output_dir = Path(output_dir) if output_dir else log_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    episode_csv = log_dir / "metrics" / "train_episode_metrics.csv"
    step_csv = log_dir / "metrics" / "train_step_metrics.csv"
    eval_npz = log_dir / "eval" / "evaluations.npz"
    progress_csv = _first_existing([log_dir / "progress.csv", log_dir / "tensorboard" / "progress.csv"])

    episode_df = _read_csv(episode_csv)
    if episode_df is not None and not episode_df.empty:
        x = _x_values(episode_df, "episode")
        saved.append(
            _line_plot(
                x,
                episode_df["episode_reward"],
                "Training Episode Reward",
                "Episode",
                "Reward",
                output_dir / "train_episode_reward.png",
                smoothing_window,
            )
        )
        saved.append(
            _line_plot(
                x,
                episode_df["episode_success"].astype(float),
                "Training Success Rate",
                "Episode",
                "Success rate",
                output_dir / "train_success_rate.png",
                smoothing_window,
                ylim=(-0.05, 1.05),
            )
        )
        saved.append(
            _line_plot(
                x,
                episode_df["episode_max_cube_lift_height"],
                "Training Max Cube Lift Height",
                "Episode",
                "Lift height (m)",
                output_dir / "train_cube_lift_height.png",
                smoothing_window,
            )
        )
        distance_col = "episode_mean_ee_cube_distance"
        if distance_col in episode_df:
            saved.append(
                _line_plot(
                    x,
                    episode_df[distance_col],
                    "Training EE-Cube Distance",
                    "Episode",
                    "Distance (m)",
                    output_dir / "train_ee_cube_distance.png",
                    smoothing_window,
                )
            )
    else:
        print(f"Skipping training episode plots; missing or empty {episode_csv}")

    step_df = _read_csv(step_csv)
    reward_cols = [
        "reward_dist",
        "reward_lift",
        "reward_target",
        "reward_action_penalty",
    ]
    if step_df is not None and all(col in step_df for col in reward_cols):
        saved.append(
            _multi_line_plot(
                step_df,
                "global_step",
                reward_cols,
                "Training Reward Terms",
                "Global step",
                "Reward term",
                output_dir / "train_reward_terms.png",
                smoothing_window,
            )
        )
    else:
        print(f"Skipping reward term plot; missing or incomplete {step_csv}")

    if eval_npz.exists():
        with np.load(eval_npz) as data:
            timesteps = data.get("timesteps")
            results = data.get("results")
            if timesteps is not None and results is not None:
                mean_rewards = np.asarray(results, dtype=float).mean(axis=1)
                saved.append(
                    _line_plot(
                        timesteps,
                        mean_rewards,
                        "Evaluation Mean Reward During Training",
                        "Global step",
                        "Mean reward",
                        output_dir / "train_eval_mean_reward.png",
                        smoothing_window=1,
                    )
                )
    else:
        print(f"Skipping eval mean reward plot; missing {eval_npz}")

    progress_df = _read_csv(progress_csv) if progress_csv else None
    loss_cols = [
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
    ]
    if progress_df is not None:
        available = [col for col in loss_cols if col in progress_df]
        x_col = "time/total_timesteps" if "time/total_timesteps" in progress_df else None
        if available and x_col:
            saved.append(
                _multi_line_plot(
                    progress_df,
                    x_col,
                    available,
                    "PPO Training Diagnostics",
                    "Global step",
                    "Value",
                    output_dir / "train_ppo_losses.png",
                    smoothing_window,
                )
            )
        else:
            print(f"Skipping PPO loss plot; expected SB3 loss columns not found in {progress_csv}")
    else:
        print("Skipping PPO loss plot; missing progress.csv")

    return saved


def plot_evaluation_curves(
    log_dir: Path,
    output_dir: Path | None = None,
) -> list[Path]:
    log_dir = Path(log_dir)
    output_dir = Path(output_dir) if output_dir else log_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    episode_csv = log_dir / "metrics" / "eval_episode_metrics.csv"
    step_csv = log_dir / "metrics" / "eval_step_metrics.csv"
    episode_df = _read_csv(episode_csv)
    if episode_df is not None and not episode_df.empty:
        x = _x_values(episode_df, "episode")
        saved.append(
            _line_plot(
                x,
                episode_df["reward"],
                "Evaluation Episode Reward",
                "Episode",
                "Reward",
                output_dir / "eval_episode_reward.png",
                smoothing_window=1,
            )
        )
        success_counts = episode_df["success"].astype(bool).value_counts()
        saved.append(
            _bar_plot(
                ["Failure", "Success"],
                [int(success_counts.get(False, 0)), int(success_counts.get(True, 0))],
                "Evaluation Success Count",
                "Outcome",
                "Episodes",
                output_dir / "eval_success_bar.png",
            )
        )
        saved.append(
            _line_plot(
                x,
                episode_df["max_cube_lift_height"],
                "Evaluation Max Lift Height Per Episode",
                "Episode",
                "Lift height (m)",
                output_dir / "eval_lift_height_per_episode.png",
                smoothing_window=1,
            )
        )
        saved.append(
            _line_plot(
                x,
                episode_df["final_ee_cube_distance"],
                "Evaluation Final EE-Cube Distance",
                "Episode",
                "Distance (m)",
                output_dir / "eval_final_distance_per_episode.png",
                smoothing_window=1,
            )
        )
    else:
        print(f"Skipping evaluation episode plots; missing or empty {episode_csv}")

    step_df = _read_csv(step_csv)
    if step_df is not None and {"episode_step", "ee_cube_distance", "cube_lift_height"}.issubset(step_df.columns):
        fig, ax1 = plt.subplots(figsize=(9, 5))
        x = step_df["episode_step"]
        ax1.plot(x, step_df["ee_cube_distance"], label="EE-cube distance", color="#2563eb", alpha=0.8)
        ax1.set_xlabel("Episode step")
        ax1.set_ylabel("Distance (m)")
        ax2 = ax1.twinx()
        ax2.plot(x, step_df["cube_lift_height"], label="Cube lift height", color="#dc2626", alpha=0.8)
        ax2.set_ylabel("Lift height (m)")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], loc="best")
        ax1.set_title("Evaluation Step Distance and Height")
        ax1.grid(True, alpha=0.25)
        path = output_dir / "eval_step_distance_height.png"
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        saved.append(path)
    else:
        print(f"Skipping evaluation step plot; missing or incomplete {step_csv}")

    return saved


def plot_comparison(
    experiment_dirs: list[Path],
    labels: list[str],
    output_dir: Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    rows = []
    for exp_dir, label in zip(experiment_dirs, labels):
        csv_path = Path(exp_dir) / "metrics" / "eval_episode_metrics.csv"
        df = _read_csv(csv_path)
        if df is None or df.empty:
            print(f"Skipping comparison entry; missing or empty {csv_path}")
            continue
        rows.append(
            {
                "label": label,
                "success_rate": df["success"].astype(float).mean(),
                "mean_reward": df["reward"].mean(),
                "mean_lift_height": df["max_cube_lift_height"].mean(),
                "mean_final_distance": df["final_ee_cube_distance"].mean(),
            }
        )
    if not rows:
        print("No comparison data available.")
        return saved

    summary = pd.DataFrame(rows)
    saved.append(_summary_bar(summary, "success_rate", "Comparison Success Rate", "Success rate", output_dir / "comparison_success_rate.png"))
    saved.append(_summary_bar(summary, "mean_reward", "Comparison Mean Reward", "Mean reward", output_dir / "comparison_mean_reward.png"))
    saved.append(_summary_bar(summary, "mean_lift_height", "Comparison Mean Lift Height", "Lift height (m)", output_dir / "comparison_mean_lift_height.png"))
    saved.append(_summary_bar(summary, "mean_final_distance", "Comparison Final Distance", "Distance (m)", output_dir / "comparison_final_distance.png"))
    return saved


def write_comparison_summary(experiment_dirs: list[Path], labels: list[str], output_csv: Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for exp_dir, label in zip(experiment_dirs, labels):
        csv_path = Path(exp_dir) / "metrics" / "eval_episode_metrics.csv"
        df = _read_csv(csv_path)
        if df is None or df.empty:
            continue
        rows.append(
            {
                "label": label,
                "experiment_dir": str(exp_dir),
                "episodes": len(df),
                "success_rate": df["success"].astype(float).mean(),
                "mean_reward": df["reward"].mean(),
                "mean_lift_height": df["max_cube_lift_height"].mean(),
                "mean_final_distance": df["final_ee_cube_distance"].mean(),
            }
        )
    fieldnames = [
        "label",
        "experiment_dir",
        "episodes",
        "success_rate",
        "mean_reward",
        "mean_lift_height",
        "mean_final_distance",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _x_values(df: pd.DataFrame, preferred_col: str) -> pd.Series:
    if preferred_col in df:
        return df[preferred_col]
    return pd.Series(np.arange(len(df)))


def _smooth(values: Iterable[float], window: int) -> pd.Series:
    series = pd.Series(values, dtype=float)
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def _line_plot(
    x: Iterable[float],
    y: Iterable[float],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    smoothing_window: int,
    ylim: tuple[float, float] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    y_series = pd.Series(y, dtype=float)
    ax.plot(x, y_series, color="#94a3b8", alpha=0.45, linewidth=1.0, label="raw")
    ax.plot(x, _smooth(y_series, smoothing_window), color="#1d4ed8", linewidth=2.0, label="rolling mean")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _multi_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    smoothing_window: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = df[x_col] if x_col in df else np.arange(len(df))
    for col in y_cols:
        ax.plot(x, _smooth(df[col], smoothing_window), linewidth=1.8, label=col)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _bar_plot(
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=["#64748b", "#16a34a"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _summary_bar(summary: pd.DataFrame, value_col: str, title: str, ylabel: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["label"], summary[value_col], color="#0f766e")
    ax.set_title(title)
    ax.set_xlabel("Experiment")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path
