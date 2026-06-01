from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ur5_grasp_env.plotting import plot_evaluation_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot evaluation metrics for a UR5 grasping run.")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/ppo_ur5_grasp"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = plot_evaluation_curves(args.log_dir, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
