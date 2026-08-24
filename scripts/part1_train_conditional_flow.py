#!/usr/bin/env python3
"""Train the conditional RealNVP baseline."""

import json
from pathlib import Path

from quantum_radiation_flow.config import ExperimentConfig
from quantum_radiation_flow.flow import ConditionalRealNVP
from quantum_radiation_flow.target import ReferenceRadiationModel
from quantum_radiation_flow.training import train_conditional_flow

config = ExperimentConfig.from_mapping(
    json.loads(Path("configs/baseline.json").read_text(encoding="utf-8"))
)
target = ReferenceRadiationModel(config.physics, seed=config.training.seed)
model = ConditionalRealNVP(config.flow)
history = train_conditional_flow(model, target, config.training)
print(f"records={len(history)} final_loss={history[-1].loss:.6f}")
