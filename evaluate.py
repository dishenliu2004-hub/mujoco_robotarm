from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stable_baselines3 import PPO

from ur5_grasp_env import UR5RobotiqGraspEnv
from ur5_grasp_env.metrics import DictCsvLogger, step_metric_from_info
from ur5_grasp_env.plotting import plot_evaluation_curves


EPISODE_FIELDS = [
    "episode",
    "reward",
    "steps",
    "success",
    "max_cube_lift_height",
    "final_cube_lift_height",
    "mean_cube_lift_height",
    "mean_ee_cube_distance",
    "final_ee_cube_distance",
    "min_ee_cube_distance",
    "cube_out_of_workspace_count",
    "final_dist_cube_target",
]

STEP_FIELDS = [
    "episode",
    "episode_step",
    "reward",
    "cube_height",
    "cube_lift_height",
    "ee_cube_distance",
    "dist_cube_target",
    "gripper_open",
    "reward_dist",
    "reward_pregrasp",
    "reward_xy_align",
    "reward_height_align",
    "reward_close_gripper",
    "reward_lift",
    "reward_target",
    "reward_action_penalty",
    "xy_dist",
    "height_error",
    "cube_out_of_workspace",
    "is_success",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO grasping policy.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/ppo_ur5_grasp/best_model/best_model.zip"),
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("runs/ppo_ur5_grasp/metrics/eval_episode_metrics.csv"),
    )
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--save-step-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-after-eval", action="store_true")
    parser.add_argument("--curriculum-level", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = None if args.no_render else "human"
    env = UR5RobotiqGraspEnv(
        render_mode=render_mode,
        curriculum_level=args.curriculum_level,
        max_episode_steps=args.max_episode_steps,
    )
    model = PPO.load(args.model)
    episode_logger = DictCsvLogger(args.csv_path, EPISODE_FIELDS)
    step_csv_path = args.csv_path.parent / "eval_step_metrics.csv"
    step_logger = DictCsvLogger(step_csv_path, STEP_FIELDS) if args.save_step_metrics else None

    successes = 0
    try:
        for episode in range(args.episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            lift_heights: list[float] = []
            distances: list[float] = []
            cube_out_of_workspace_count = 0
            final_metric = None
            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                done = terminated or truncated
                metric = step_metric_from_info(
                    info,
                    global_step=steps,
                    episode=episode,
                    episode_step=steps,
                    reward=float(reward),
                )
                final_metric = metric
                if math.isfinite(metric.cube_lift_height):
                    lift_heights.append(metric.cube_lift_height)
                if math.isfinite(metric.ee_cube_distance):
                    distances.append(metric.ee_cube_distance)
                cube_out_of_workspace_count += int(metric.cube_out_of_workspace)
                if step_logger is not None:
                    step_logger.write_row(
                        {
                            "episode": episode,
                            "episode_step": steps,
                            "reward": float(reward),
                            "cube_height": metric.cube_height,
                            "cube_lift_height": metric.cube_lift_height,
                            "ee_cube_distance": metric.ee_cube_distance,
                            "dist_cube_target": metric.dist_cube_target,
                            "gripper_open": metric.gripper_open,
                            "reward_dist": metric.reward_dist,
                            "reward_pregrasp": metric.reward_pregrasp,
                            "reward_xy_align": metric.reward_xy_align,
                            "reward_height_align": metric.reward_height_align,
                            "reward_close_gripper": metric.reward_close_gripper,
                            "reward_lift": metric.reward_lift,
                            "reward_target": metric.reward_target,
                            "reward_action_penalty": metric.reward_action_penalty,
                            "xy_dist": metric.xy_dist,
                            "height_error": metric.height_error,
                            "cube_out_of_workspace": metric.cube_out_of_workspace,
                            "is_success": metric.is_success,
                        }
                    )
            success = bool(info.get("is_success", False))
            successes += int(success)
            final_lift = final_metric.cube_lift_height if final_metric else math.nan
            final_distance = final_metric.ee_cube_distance if final_metric else math.nan
            final_target_distance = final_metric.dist_cube_target if final_metric else math.nan
            episode_logger.write_row(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "steps": steps,
                    "success": success,
                    "max_cube_lift_height": max(lift_heights) if lift_heights else math.nan,
                    "final_cube_lift_height": final_lift,
                    "mean_cube_lift_height": float(np.mean(lift_heights)) if lift_heights else math.nan,
                    "mean_ee_cube_distance": float(np.mean(distances)) if distances else math.nan,
                    "final_ee_cube_distance": final_distance,
                    "min_ee_cube_distance": min(distances) if distances else math.nan,
                    "cube_out_of_workspace_count": cube_out_of_workspace_count,
                    "final_dist_cube_target": final_target_distance,
                }
            )
            print(
                f"episode={episode + 1} reward={total_reward:.2f} "
                f"steps={steps} success={success}"
            )
    finally:
        episode_logger.close()
        if step_logger is not None:
            step_logger.close()

    print(f"success_rate={successes / max(args.episodes, 1):.2%}")
    print(f"Saved episode metrics to {args.csv_path}")
    if args.save_step_metrics:
        print(f"Saved step metrics to {step_csv_path}")
    if args.plot_after_eval:
        log_dir = args.csv_path.parent.parent if args.csv_path.parent.name == "metrics" else args.csv_path.parent
        for path in plot_evaluation_curves(log_dir, output_dir=args.plot_path):
            print(f"Saved figure: {path}")
    env.close()


if __name__ == "__main__":
    main()
