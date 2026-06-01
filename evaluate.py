from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stable_baselines3 import PPO

from ur5_grasp_env import UR5RobotiqGraspEnv


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = None if args.no_render else "human"
    env = UR5RobotiqGraspEnv(render_mode=render_mode)
    model = PPO.load(args.model)

    successes = 0
    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        successes += int(info.get("is_success", False))
        print(
            f"episode={episode + 1} reward={total_reward:.2f} "
            f"steps={steps} success={info.get('is_success', False)}"
        )

    print(f"success_rate={successes / max(args.episodes, 1):.2%}")
    env.close()


if __name__ == "__main__":
    main()
