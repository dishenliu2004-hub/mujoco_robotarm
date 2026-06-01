from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ur5_grasp_env.metrics import CsvMetricLogger, StepMetric, flatten_info


def test_csv_metric_logger_writes_header_and_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics" / "steps.csv"
    logger = CsvMetricLogger(csv_path)
    logger.write_step(
        StepMetric(
            global_step=1,
            episode=0,
            episode_step=1,
            reward=0.5,
            is_success=False,
            cube_out_of_workspace=False,
            cube_height=0.33,
            cube_lift_height=0.02,
            ee_cube_distance=0.1,
            dist_cube_target=0.2,
            gripper_open=0.03,
            reward_dist=-0.2,
            reward_pregrasp=0.4,
            reward_xy_align=0.3,
            reward_height_align=0.2,
            reward_close_gripper=0.0,
            reward_lift=0.1,
            reward_target=0.0,
            reward_action_penalty=-0.01,
            xy_dist=0.05,
            height_error=0.02,
        )
    )
    logger.close()

    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["global_step"] == "1"
    assert rows[0]["cube_lift_height"] == "0.02"
    assert "reward_action_penalty" in rows[0]
    assert "reward_pregrasp" in rows[0]
    assert rows[0]["cube_out_of_workspace"] == "False"


def test_flatten_info_splits_positions() -> None:
    flat = flatten_info({"cube_pos": [1, 2, 3], "ee_pos": [4, 5, 6]})
    assert flat["cube_x"] == 1.0
    assert flat["cube_y"] == 2.0
    assert flat["cube_z"] == 3.0
    assert flat["ee_x"] == 4.0
    assert flat["ee_y"] == 5.0
    assert flat["ee_z"] == 6.0
