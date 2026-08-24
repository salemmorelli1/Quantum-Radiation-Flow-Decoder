#!/usr/bin/env python3
"""Demonstrate post-Page inversion on a paired toy pure state."""

import torch

from quantum_radiation_flow.config import FlowConfig, PhysicsConfig
from quantum_radiation_flow.flow import ConditionalRealNVP
from quantum_radiation_flow.quantum import complex_state_from_real_coordinates, pure_state_fidelity
from quantum_radiation_flow.target import ReferenceRadiationModel


torch.set_default_dtype(torch.float64)
physics = PhysicsConfig()
target = ReferenceRadiationModel(physics)
model = ConditionalRealNVP(FlowConfig(data_dim=physics.data_dim))
batch = target.sample(32, time=torch.tensor(0.95), dtype=torch.float64)
latent = model.encode(batch.observation, batch.condition).value
decoded = complex_state_from_real_coordinates(latent)
secret = complex_state_from_real_coordinates(batch.secret_coordinates)
print(f"untrained_fidelity={pure_state_fidelity(decoded, secret).mean():.6f}")
print("Train the flow before interpreting this diagnostic.")
