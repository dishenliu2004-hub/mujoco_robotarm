from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ur5_grasp_env import UR5RobotiqGraspEnv


def test_env_reset_and_step_expose_stability_metrics() -> None:
    env = UR5RobotiqGraspEnv(seed=7, curriculum_level=0, max_episode_steps=5)
    try:
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert "is_success" in info
        assert "cube_out_of_workspace" in info

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        for key in [
            "is_success",
            "cube_out_of_workspace",
            "reward_pregrasp",
            "reward_xy_align",
            "reward_height_align",
            "reward_close_gripper",
            "reward_lift",
            "reward_target",
            "reward_action_penalty",
        ]:
            assert key in info
    finally:
        env.close()


def test_out_of_workspace_reward_overrides_positive_terms() -> None:
    env = UR5RobotiqGraspEnv(seed=7, curriculum_level=0)
    try:
        env.reset()
        env.data.qpos[env.cube_qpos_addr : env.cube_qpos_addr + 7] = np.array(
            [0.52, 0.0, env.table_top_z + 0.80, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        mujoco.mj_forward(env.model, env.data)

        reward, terms = env._reward()

        assert env._cube_out_of_workspace()
        assert reward == -50.0
        assert terms["reward_lift"] == 0.0
        assert terms["reward_target"] == 0.0
        assert terms["cube_lift_height"] == 0.25
    finally:
        env.close()
