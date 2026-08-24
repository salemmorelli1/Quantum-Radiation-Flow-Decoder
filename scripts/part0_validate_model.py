#!/usr/bin/env python3
"""Validate configuration and the exact flow round trip."""

import json
from pathlib import Path

import torch

from quantum_radiation_flow.config import ExperimentConfig
from quantum_radiation_flow.flow import ConditionalRealNVP

payload = json.loads(Path("configs/baseline.json").read_text(encoding="utf-8"))
config = ExperimentConfig.from_mapping(payload)
torch.set_default_dtype(torch.float64)
model = ConditionalRealNVP(config.flow)
latent = torch.randn(16, config.flow.data_dim)
condition = torch.rand(16, config.flow.condition_dim)
forward = model.decode(latent, condition)
inverse = model.encode(forward.value, condition)
torch.testing.assert_close(inverse.value, latent, rtol=1.0e-9, atol=1.0e-9)
torch.testing.assert_close(
    forward.log_abs_det_jacobian + inverse.log_abs_det_jacobian,
    torch.zeros(16),
    rtol=1.0e-9,
    atol=1.0e-9,
)
print("PASS: exact conditional-flow round trip")
