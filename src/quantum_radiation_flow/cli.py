"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conditional-flow black-hole information toy model"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/latest"))
    parser.add_argument("--steps", type=int, default=None, help="override optimization steps")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    if args.steps is not None:
        payload.setdefault("training", {})["steps"] = args.steps
    config = ExperimentConfig.from_mapping(payload)
    result = run_experiment(config, output_directory=args.output)
    final = result["curve"][-1]
    print(f"status={result['status']}")
    print(f"final_decoder_fidelity={final['decoder_fidelity']:.6f}")
    print(f"artifacts={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
