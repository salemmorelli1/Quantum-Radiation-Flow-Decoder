from dataclasses import replace

import torch

from quantum_radiation_flow.config import PhysicsConfig
from quantum_radiation_flow.target import ReferenceRadiationModel


def test_reference_density_matches_generated_log_prob() -> None:
    torch.set_default_dtype(torch.float64)
    physics = replace(
        PhysicsConfig(),
        data_dim=4,
        target_hidden_dim=24,
        target_layers=2,
    )
    target = ReferenceRadiationModel(physics, seed=19)
    batch = target.sample(64, time=torch.tensor(0.8), dtype=torch.float64)
    recomputed = target.log_prob(batch.observation, batch.condition)
    torch.testing.assert_close(recomputed, batch.target_log_prob, rtol=1.0e-9, atol=1.0e-9)


def test_visibility_transitions_through_page_time() -> None:
    target = ReferenceRadiationModel(PhysicsConfig())
    time = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float64)
    visibility = target.visibility(time)
    assert visibility[0] < visibility[1] < visibility[2]
    torch.testing.assert_close(visibility[1], torch.tensor([0.5], dtype=torch.float64))
