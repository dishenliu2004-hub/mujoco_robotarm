from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class UR5RobotiqGraspEnv(gym.Env):
    """MuJoCo UR5 + Robotiq85-style block lifting task.

    The bundled MJCF uses simple primitive geometry so the project runs without
    external meshes. Keep the joint, actuator, site, and body names if you swap
    in a higher fidelity UR5/Robotiq85 asset.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    ARM_JOINTS = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    ARM_ACTUATORS = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    )
    FINGER_JOINTS = ("left_finger_joint", "right_finger_joint")
    FINGER_ACTUATORS = ("left_finger", "right_finger")

    def __init__(
        self,
        xml_path: str | Path | None = None,
        render_mode: str | None = None,
        max_episode_steps: int = 250,
        frame_skip: int = 10,
        seed: int | None = None,
        curriculum_level: int = 0,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        root = Path(__file__).resolve().parents[2]
        self.xml_path = Path(xml_path) if xml_path else root / "assets" / "ur5_robotiq85_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.curriculum_level = int(curriculum_level)
        self._step_count = 0
        self._viewer = None
        self._renderer = None
        self.rng = np.random.default_rng(seed)
        self.last_action = np.zeros(7, dtype=np.float64)

        self.arm_joint_ids = [self._joint_id(name) for name in self.ARM_JOINTS]
        self.finger_joint_ids = [self._joint_id(name) for name in self.FINGER_JOINTS]
        self.arm_act_ids = [self._actuator_id(name) for name in self.ARM_ACTUATORS]
        self.finger_act_ids = [self._actuator_id(name) for name in self.FINGER_ACTUATORS]
        self.ee_site_id = self._site_id("pinch_site")
        self.cube_body_id = self._body_id("cube")
        self.target_site_id = self._site_id("lift_target")

        self.arm_qpos_addr = [self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids]
        self.arm_qvel_addr = [self.model.jnt_dofadr[jid] for jid in self.arm_joint_ids]
        self.finger_qpos_addr = [self.model.jnt_qposadr[jid] for jid in self.finger_joint_ids]
        self.finger_qvel_addr = [self.model.jnt_dofadr[jid] for jid in self.finger_joint_ids]
        self.cube_qpos_addr = self.model.jnt_qposadr[self._joint_id("cube_freejoint")]

        self.home_qpos = np.array([-0.35, -0.95, 1.45, -1.15, -1.57, 0.0], dtype=np.float64)
        self.arm_delta = np.array([0.035, 0.035, 0.035, 0.035, 0.045, 0.045], dtype=np.float64)
        self.table_top_z = 0.305
        self.success_lift_height = 0.12

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        sample_obs, _ = self.reset(seed=seed)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count = 0
        self.last_action = np.zeros(7, dtype=np.float64)
        mujoco.mj_resetData(self.model, self.data)

        noise = self.rng.uniform(-0.04, 0.04, size=6)
        for addr, value in zip(self.arm_qpos_addr, self.home_qpos + noise):
            self.data.qpos[addr] = value

        for addr in self.finger_qpos_addr:
            self.data.qpos[addr] = 0.038

        cube_xy = self._sample_cube_xy(options)
        self.data.qpos[self.cube_qpos_addr : self.cube_qpos_addr + 7] = np.array(
            [cube_xy[0], cube_xy[1], self.table_top_z + 0.026, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self._set_initial_ctrl()
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        return obs, self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action.copy()
        self._step_count += 1

        current_qpos = self.data.qpos[self.arm_qpos_addr].copy()
        target_qpos = current_qpos + action[:6] * self.arm_delta
        target_qpos = self._clip_arm_to_joint_ranges(target_qpos)
        for act_id, value in zip(self.arm_act_ids, target_qpos):
            self.data.ctrl[act_id] = value

        gripper_open = float(np.interp(action[6], [-1.0, 1.0], [0.04, 0.0]))
        for act_id in self.finger_act_ids:
            self.data.ctrl[act_id] = gripper_open

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        obs = self._get_obs()
        reward, reward_terms = self._reward()
        success = self._is_success()
        cube_out_of_workspace = self._cube_out_of_workspace()
        terminated = success or cube_out_of_workspace
        truncated = self._step_count >= self.max_episode_steps
        info = self._info(reward_terms)
        info["cube_out_of_workspace"] = bool(cube_out_of_workspace)
        info["is_success"] = bool(success)

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="track")
            return self._renderer.render()

        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_obs(self) -> np.ndarray:
        arm_qpos = self.data.qpos[self.arm_qpos_addr]
        arm_qvel = self.data.qvel[self.arm_qvel_addr]
        finger_qpos = self.data.qpos[self.finger_qpos_addr]
        finger_qvel = self.data.qvel[self.finger_qvel_addr]
        ee_pos = self.data.site_xpos[self.ee_site_id]
        cube_pos = self.data.xpos[self.cube_body_id]
        cube_quat = self.data.xquat[self.cube_body_id]
        target_pos = self.data.site_xpos[self.target_site_id]
        rel_cube_to_ee = cube_pos - ee_pos

        obs = np.concatenate(
            [
                arm_qpos,
                arm_qvel,
                finger_qpos,
                finger_qvel,
                ee_pos,
                cube_pos,
                cube_quat,
                target_pos,
                rel_cube_to_ee,
            ]
        )
        return obs.astype(np.float32)

    def _reward(self) -> tuple[float, dict[str, float]]:
        ee_pos = self.data.site_xpos[self.ee_site_id]
        cube_pos = self.data.xpos[self.cube_body_id]
        target_pos = self.data.site_xpos[self.target_site_id]

        gripper_open = float(np.mean(self.data.qpos[self.finger_qpos_addr]))
        raw_cube_lift_height = float(cube_pos[2] - self.table_top_z)
        cube_lift_height = float(np.clip(raw_cube_lift_height, 0.0, 0.25))
        ee_cube_distance = float(np.linalg.norm(ee_pos - cube_pos))
        dist_cube_target = float(np.linalg.norm(cube_pos - target_pos))
        cube_out = self._cube_out_of_workspace()

        pregrasp_pos = cube_pos.copy()
        pregrasp_pos[2] = self.table_top_z + 0.16
        pregrasp_dist = float(np.linalg.norm(ee_pos - pregrasp_pos))
        reward_pregrasp = 2.0 * np.exp(-8.0 * pregrasp_dist)

        xy_dist = float(np.linalg.norm(ee_pos[:2] - cube_pos[:2]))
        reward_xy_align = 2.0 * np.exp(-12.0 * xy_dist)

        grasp_height = self.table_top_z + 0.065
        height_error = abs(float(ee_pos[2] - grasp_height))
        reward_height_align = 1.5 * np.exp(-18.0 * height_error)

        is_near_grasp = xy_dist < 0.055 and height_error < 0.055
        reward_close_gripper = 1.0 if is_near_grasp and gripper_open < 0.025 else 0.0

        is_near_cube = ee_cube_distance < 0.12
        is_gripper_closed = gripper_open < 0.03
        is_likely_grasping = is_near_cube and is_gripper_closed

        reward_lift = 25.0 * cube_lift_height if is_likely_grasping else 0.0
        reward_target = 3.0 * np.exp(-6.0 * dist_cube_target) if cube_lift_height > 0.04 else 0.0
        if cube_out:
            reward_lift = 0.0
            reward_target = 0.0

        last_action = getattr(self, "last_action", np.zeros(7, dtype=np.float64))
        action_penalty = 0.01 * float(np.sum(np.square(last_action[:6])))

        reward = (
            reward_pregrasp
            + reward_xy_align
            + reward_height_align
            + reward_close_gripper
            + reward_lift
            + reward_target
            - action_penalty
        )
        if self._is_success():
            reward += 25.0

        if cube_out:
            reward = -50.0

        return reward, {
            "reward_dist": float(-ee_cube_distance),
            "reward_pregrasp": float(reward_pregrasp),
            "reward_xy_align": float(reward_xy_align),
            "reward_height_align": float(reward_height_align),
            "reward_close_gripper": float(reward_close_gripper),
            "reward_lift": float(reward_lift),
            "reward_target": float(reward_target),
            "reward_action_penalty": float(-action_penalty),
            "cube_height": float(cube_pos[2]),
            "cube_lift_height": float(cube_lift_height),
            "raw_cube_lift_height": float(raw_cube_lift_height),
            "ee_cube_distance": float(ee_cube_distance),
            "dist_cube_target": float(dist_cube_target),
            "gripper_open": float(gripper_open),
            "xy_dist": float(xy_dist),
            "height_error": float(height_error),
            "is_likely_grasping": float(is_likely_grasping),
        }

    def _is_success(self) -> bool:
        ee_pos = self.data.site_xpos[self.ee_site_id]
        cube_pos = self.data.xpos[self.cube_body_id]

        cube_lift_height = float(cube_pos[2] - self.table_top_z)
        ee_cube_distance = float(np.linalg.norm(ee_pos - cube_pos))

        return (
            cube_lift_height > self.success_lift_height
            and ee_cube_distance < 0.10
            and cube_pos[2] < self.table_top_z + 0.45
        )

    def _cube_out_of_workspace(self) -> bool:
        ee_pos = self.data.site_xpos[self.ee_site_id]
        cube_pos = self.data.xpos[self.cube_body_id]
        ee_cube_distance = float(np.linalg.norm(ee_pos - cube_pos))

        return (
            cube_pos[2] > self.table_top_z + 0.60
            or cube_pos[2] < self.table_top_z - 0.05
            or abs(cube_pos[0]) > 1.2
            or abs(cube_pos[1]) > 1.2
            or ee_cube_distance > 1.5
        )

    def _info(self, reward_terms: dict[str, float] | None = None) -> dict[str, Any]:
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        target_pos = self.data.site_xpos[self.target_site_id].copy()
        if reward_terms is None:
            _, reward_terms = self._reward()
        return {
            "step": self._step_count,
            "is_success": bool(self._is_success()),
            "cube_out_of_workspace": bool(self._cube_out_of_workspace()),
            "cube_pos": cube_pos,
            "ee_pos": ee_pos,
            "cube_height": float(cube_pos[2]),
            "cube_lift_height": float(np.clip(cube_pos[2] - self.table_top_z, 0.0, 0.25)),
            "ee_cube_distance": float(np.linalg.norm(ee_pos - cube_pos)),
            "dist_cube_target": float(np.linalg.norm(cube_pos - target_pos)),
            "gripper_open": float(np.mean(self.data.qpos[self.finger_qpos_addr])),
            **reward_terms,
        }

    def _set_initial_ctrl(self) -> None:
        for act_id, value in zip(self.arm_act_ids, self.data.qpos[self.arm_qpos_addr]):
            self.data.ctrl[act_id] = value
        for act_id in self.finger_act_ids:
            self.data.ctrl[act_id] = 0.038

    def _clip_arm_to_joint_ranges(self, qpos: np.ndarray) -> np.ndarray:
        clipped = qpos.copy()
        for i, jid in enumerate(self.arm_joint_ids):
            if self.model.jnt_limited[jid]:
                low, high = self.model.jnt_range[jid]
                clipped[i] = np.clip(clipped[i], low, high)
        return clipped

    def _sample_cube_xy(self, options: dict[str, Any] | None) -> np.ndarray:
        if options and "cube_xy" in options:
            cube_xy = np.asarray(options["cube_xy"], dtype=np.float64)
            if cube_xy.shape != (2,):
                raise ValueError("options['cube_xy'] must be a 2D position.")
            return cube_xy

        level = self.curriculum_level
        if options and "curriculum_level" in options:
            level = int(options["curriculum_level"])

        if level <= 0:
            x = self.rng.uniform(0.49, 0.53)
            y = self.rng.uniform(-0.03, 0.03)
        elif level == 1:
            x = self.rng.uniform(0.46, 0.56)
            y = self.rng.uniform(-0.08, 0.08)
        elif level == 2:
            x = self.rng.uniform(0.43, 0.59)
            y = self.rng.uniform(-0.12, 0.12)
        else:
            x = self.rng.uniform(0.42, 0.60)
            y = self.rng.uniform(-0.16, 0.16)
        return np.array([x, y], dtype=np.float64)

    def _joint_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _actuator_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _site_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
