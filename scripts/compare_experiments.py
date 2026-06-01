from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ur5_grasp_env.plotting import plot_comparison, write_comparison_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evaluation metrics across experiments.")
    parser.add_argument("--experiments", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.experiments) != len(args.labels):
        raise SystemExit("--experiments and --labels must have the same length.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = write_comparison_summary(
        args.experiments,
        args.labels,
        args.output_dir / "comparison_summary.csv",
    )
    print(summary_path)
    for path in plot_comparison(args.experiments, args.labels, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
