#!/usr/bin/env python3
"""Run training and write the entropy/decryption curve."""

import json
from pathlib import Path

from quantum_radiation_flow.config import ExperimentConfig
from quantum_radiation_flow.pipeline import run_experiment

config = ExperimentConfig.from_mapping(
    json.loads(Path("configs/baseline.json").read_text(encoding="utf-8"))
)
result = run_experiment(config, output_directory="artifacts/latest")
print(result["page_curve_diagnostic"])
