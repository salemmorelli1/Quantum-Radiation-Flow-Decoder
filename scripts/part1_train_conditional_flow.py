#!/usr/bin/env python3
"""Train the conditional RealNVP baseline."""

import json
from pathlib import Path

from quantum_radiation_flow.config import ExperimentConfig
from quantum_radiation_flow.pipeline import run_experiment

config = ExperimentConfig.from_mapping(
    json.loads(Path("configs/baseline.json").read_text(encoding="utf-8"))
)
result = run_experiment(config, output_directory="artifacts/run-training")
history = result["training_history"]
print(f"records={len(history)} final_loss={history[-1]['loss']:.6f}")
