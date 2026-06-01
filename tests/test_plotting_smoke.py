from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ur5_grasp_env.plotting import plot_evaluation_curves, plot_training_curves


def test_plotting_smoke_generates_pngs(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    metrics_dir = log_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    _write_csv(
        metrics_dir / "train_episode_metrics.csv",
        [
            "episode",
            "episode_reward",
            "episode_success",
            "episode_max_cube_lift_height",
            "episode_mean_ee_cube_distance",
        ],
        [
            [0, 1.0, False, 0.01, 0.2],
            [1, 2.0, True, 0.12, 0.08],
        ],
    )
    _write_csv(
        metrics_dir / "train_step_metrics.csv",
        [
            "global_step",
            "reward_pregrasp",
            "reward_xy_align",
            "reward_height_align",
            "reward_close_gripper",
            "reward_lift",
            "reward_target",
            "reward_action_penalty",
            "xy_dist",
            "height_error",
            "ee_cube_distance",
            "cube_lift_height",
        ],
        [
            [1, 0.2, 0.3, 0.1, 0.0, 0.0, 0.0, -0.01, 0.10, 0.04, 0.20, 0.01],
            [2, 0.4, 0.5, 0.2, 1.0, 0.3, 0.2, -0.02, 0.04, 0.02, 0.08, 0.13],
        ],
    )
    _write_csv(
        metrics_dir / "eval_episode_metrics.csv",
        [
            "episode",
            "reward",
            "success",
            "max_cube_lift_height",
            "final_ee_cube_distance",
        ],
        [
            [0, 1.0, False, 0.01, 0.2],
            [1, 3.0, True, 0.13, 0.04],
        ],
    )
    _write_csv(
        metrics_dir / "eval_step_metrics.csv",
        ["episode", "episode_step", "ee_cube_distance", "cube_lift_height"],
        [[0, 1, 0.2, 0.01], [1, 1, 0.04, 0.13]],
    )

    train_paths = plot_training_curves(log_dir, smoothing_window=1)
    eval_paths = plot_evaluation_curves(log_dir)

    assert log_dir / "figures" / "train_episode_reward.png" in train_paths
    assert log_dir / "figures" / "train_grasp_phase_terms.png" in train_paths
    assert log_dir / "figures" / "eval_episode_reward.png" in eval_paths
    assert log_dir / "figures" / "eval_step_distance_height.png" in eval_paths
    assert all(path.exists() for path in train_paths + eval_paths)


def _write_csv(path: Path, fieldnames: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        writer.writerows(rows)
