from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ur5_grasp_env.metrics import CsvMetricLogger, DictCsvLogger, step_metric_from_info


EPISODE_FIELDS = [
    "episode",
    "global_step",
    "episode_reward",
    "episode_length",
    "episode_success",
    "episode_cube_out_of_workspace",
    "episode_max_cube_lift_height",
    "episode_mean_ee_cube_distance",
    "episode_final_ee_cube_distance",
]


class TrainingMetricsCallback(BaseCallback):
    def __init__(
        self,
        log_dir: Path,
        save_freq: int = 1,
        append: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.log_dir = Path(log_dir)
        self.metrics_dir = self.log_dir / "metrics"
        self.save_freq = max(int(save_freq), 1)
        self.append = append
        self.step_logger: CsvMetricLogger | None = None
        self.episode_logger: DictCsvLogger | None = None
        self.active_episode_ids: list[int] = []
        self.next_episode_id = 0
        self.episode_steps: list[int] = []
        self.episode_rewards: list[float] = []
        self.episode_values: list[dict[str, list[float]]] = []

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.training_env, "num_envs", 1))
        self.active_episode_ids = list(range(n_envs))
        self.next_episode_id = n_envs
        self.episode_steps = [0 for _ in range(n_envs)]
        self.episode_rewards = [0.0 for _ in range(n_envs)]
        self.episode_values = [defaultdict(list) for _ in range(n_envs)]
        self.step_logger = CsvMetricLogger(
            self.metrics_dir / "train_step_metrics.csv",
            append=self.append,
        )
        self.episode_logger = DictCsvLogger(
            self.metrics_dir / "train_episode_metrics.csv",
            EPISODE_FIELDS,
            append=self.append,
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float).reshape(-1)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        if len(infos) == 0:
            return True

        for env_idx, info in enumerate(infos):
            reward = float(rewards[env_idx]) if env_idx < len(rewards) else math.nan
            done = bool(dones[env_idx]) if env_idx < len(dones) else False
            self._record_env_step(env_idx, info or {}, reward, done)
        return True

    def _on_training_end(self) -> None:
        if self.step_logger is not None:
            self.step_logger.close()
        if self.episode_logger is not None:
            self.episode_logger.close()

    def _record_env_step(self, env_idx: int, info: dict[str, Any], reward: float, done: bool) -> None:
        episode_id = self.active_episode_ids[env_idx]
        self.episode_steps[env_idx] += 1
        self.episode_rewards[env_idx] += reward
        episode_step = self.episode_steps[env_idx]

        metric = step_metric_from_info(
            info,
            global_step=int(self.num_timesteps),
            episode=episode_id,
            episode_step=episode_step,
            reward=reward,
        )
        self._record_tensorboard(metric)

        values = self.episode_values[env_idx]
        values["cube_lift_height"].append(metric.cube_lift_height)
        values["ee_cube_distance"].append(metric.ee_cube_distance)
        values["cube_out_of_workspace"].append(float(metric.cube_out_of_workspace))

        if self.n_calls % self.save_freq == 0 and self.step_logger is not None:
            self.step_logger.write_step(metric)

        if done:
            self._write_episode(env_idx, metric)
            self.active_episode_ids[env_idx] = self.next_episode_id
            self.next_episode_id += 1
            self.episode_steps[env_idx] = 0
            self.episode_rewards[env_idx] = 0.0
            self.episode_values[env_idx] = defaultdict(list)

    def _record_tensorboard(self, metric: Any) -> None:
        self.logger.record_mean("env/cube_lift_height", metric.cube_lift_height)
        self.logger.record_mean("env/cube_height", metric.cube_height)
        self.logger.record_mean("env/ee_cube_distance", metric.ee_cube_distance)
        self.logger.record_mean("env/dist_cube_target", metric.dist_cube_target)
        self.logger.record_mean("env/reward_dist", metric.reward_dist)
        self.logger.record_mean("env/reward_pregrasp", metric.reward_pregrasp)
        self.logger.record_mean("env/reward_xy_align", metric.reward_xy_align)
        self.logger.record_mean("env/reward_height_align", metric.reward_height_align)
        self.logger.record_mean("env/reward_close_gripper", metric.reward_close_gripper)
        self.logger.record_mean("env/reward_lift", metric.reward_lift)
        self.logger.record_mean("env/reward_target", metric.reward_target)
        self.logger.record_mean("env/reward_action_penalty", metric.reward_action_penalty)
        self.logger.record_mean("env/xy_dist", metric.xy_dist)
        self.logger.record_mean("env/height_error", metric.height_error)
        self.logger.record_mean("env/cube_out_of_workspace", float(metric.cube_out_of_workspace))
        self.logger.record_mean("env/is_success", float(metric.is_success))

    def _write_episode(self, env_idx: int, final_metric: Any) -> None:
        if self.episode_logger is None:
            return
        lift_values = _finite_values(self.episode_values[env_idx]["cube_lift_height"])
        distance_values = _finite_values(self.episode_values[env_idx]["ee_cube_distance"])
        out_values = _finite_values(self.episode_values[env_idx]["cube_out_of_workspace"])
        max_lift = max(lift_values) if lift_values else math.nan
        mean_distance = float(np.mean(distance_values)) if distance_values else math.nan
        row = {
            "episode": self.active_episode_ids[env_idx],
            "global_step": int(self.num_timesteps),
            "episode_reward": self.episode_rewards[env_idx],
            "episode_length": self.episode_steps[env_idx],
            "episode_success": bool(final_metric.is_success),
            "episode_cube_out_of_workspace": bool(final_metric.cube_out_of_workspace or any(out_values)),
            "episode_max_cube_lift_height": max_lift,
            "episode_mean_ee_cube_distance": mean_distance,
            "episode_final_ee_cube_distance": final_metric.ee_cube_distance,
        }
        self.episode_logger.write_row(row)


def _finite_values(values: list[float]) -> list[float]:
    return [float(value) for value in values if np.isfinite(value)]
