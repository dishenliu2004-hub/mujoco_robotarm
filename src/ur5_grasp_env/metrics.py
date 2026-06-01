from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class StepMetric:
    global_step: int
    episode: int
    episode_step: int
    reward: float
    is_success: bool
    cube_height: float
    cube_lift_height: float
    ee_cube_distance: float
    dist_cube_target: float
    gripper_open: float
    reward_dist: float
    reward_lift: float
    reward_target: float
    reward_action_penalty: float


class CsvMetricLogger:
    def __init__(self, csv_path: Path, append: bool = False) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = [field.name for field in fields(StepMetric)]
        mode = "a" if append else "w"
        self._file = self.csv_path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        should_write_header = not append or self.csv_path.stat().st_size == 0
        if should_write_header:
            self._writer.writeheader()

    def write_step(self, metric: StepMetric) -> None:
        row = asdict(metric)
        row["is_success"] = bool(row["is_success"])
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()


class DictCsvLogger:
    def __init__(self, csv_path: Path, fieldnames: Iterable[str], append: bool = False) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        mode = "a" if append else "w"
        self._file = self.csv_path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
        should_write_header = not append or self.csv_path.stat().st_size == 0
        if should_write_header:
            self._writer.writeheader()

    def write_row(self, row: dict[str, Any]) -> None:
        self._writer.writerow(_normalize_csv_row(row, self.fieldnames))

    def close(self) -> None:
        self._file.close()


def flatten_info(info: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in info.items():
        if key in {"cube_pos", "ee_pos"}:
            arr = np.asarray(value, dtype=float).reshape(-1)
            prefix = "cube" if key == "cube_pos" else "ee"
            for axis, axis_value in zip(("x", "y", "z"), arr[:3]):
                flat[f"{prefix}_{axis}"] = float(axis_value)
            continue
        if isinstance(value, np.ndarray):
            arr = np.asarray(value, dtype=float).reshape(-1)
            for idx, item in enumerate(arr):
                flat[f"{key}_{idx}"] = float(item)
            continue
        if isinstance(value, (np.floating, np.integer)):
            flat[key] = value.item()
        else:
            flat[key] = value
    return flat


def step_metric_from_info(
    info: dict[str, Any],
    *,
    global_step: int,
    episode: int,
    episode_step: int,
    reward: float,
) -> StepMetric:
    flat = flatten_info(info)
    return StepMetric(
        global_step=global_step,
        episode=episode,
        episode_step=episode_step,
        reward=float(reward),
        is_success=bool(flat.get("is_success", False)),
        cube_height=_float_or_nan(flat.get("cube_height")),
        cube_lift_height=_float_or_nan(flat.get("cube_lift_height")),
        ee_cube_distance=_float_or_nan(flat.get("ee_cube_distance")),
        dist_cube_target=_float_or_nan(flat.get("dist_cube_target")),
        gripper_open=_float_or_nan(flat.get("gripper_open")),
        reward_dist=_float_or_nan(flat.get("reward_dist")),
        reward_lift=_float_or_nan(flat.get("reward_lift")),
        reward_target=_float_or_nan(flat.get("reward_target")),
        reward_action_penalty=_float_or_nan(flat.get("reward_action_penalty")),
    )


def _normalize_csv_row(row: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name in fieldnames:
        value = row.get(name, math.nan)
        if isinstance(value, np.ndarray):
            normalized[name] = np.asarray(value).reshape(-1).tolist()
        elif isinstance(value, (np.floating, np.integer)):
            normalized[name] = value.item()
        elif isinstance(value, np.bool_):
            normalized[name] = bool(value)
        else:
            normalized[name] = value
    return normalized


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
