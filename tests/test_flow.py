from dataclasses import replace

import torch

from quantum_radiation_flow.config import FlowConfig, PhysicsConfig, TrainingConfig
from quantum_radiation_flow.flow import ConditionalRealNVP
from quantum_radiation_flow.target import ReferenceRadiationModel
from quantum_radiation_flow.training import train_conditional_flow


def test_flow_round_trip_and_jacobian_cancellation() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(7)
    model = ConditionalRealNVP(
        FlowConfig(data_dim=6, hidden_dim=32, context_dim=12, num_layers=4)
    )
    latent = torch.randn(24, 6)
    condition = torch.rand(24, 2)
    forward = model.decode(latent, condition)
    inverse = model.encode(forward.value, condition)
    torch.testing.assert_close(inverse.value, latent, rtol=1.0e-9, atol=1.0e-9)
    torch.testing.assert_close(
        forward.log_abs_det_jacobian + inverse.log_abs_det_jacobian,
        torch.zeros(24),
        rtol=1.0e-9,
        atol=1.0e-9,
    )


def test_log_prob_and_entropy_are_finite() -> None:
    torch.set_default_dtype(torch.float64)
    model = ConditionalRealNVP(FlowConfig(data_dim=4, hidden_dim=24, num_layers=4))
    condition = torch.tensor([[0.7, 0.6]])
    observation, log_prob = model.sample(128, condition)
    assert observation.shape == (128, 4)
    assert torch.isfinite(log_prob).all()
    entropy, standard_error = model.entropy(128, condition)
    assert torch.isfinite(entropy)
    assert standard_error > 0.0


def test_short_training_pass_updates_parameters() -> None:
    torch.set_default_dtype(torch.float64)
    target = ReferenceRadiationModel(
        physics=replace(
            PhysicsConfig(),
            data_dim=4,
            target_hidden_dim=24,
            target_layers=2,
        ),
        seed=13,
    )
    model = ConditionalRealNVP(
        FlowConfig(data_dim=4, hidden_dim=24, context_dim=12, num_layers=2)
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    history = train_conditional_flow(
        model,
        target,
        TrainingConfig(
            steps=3,
            batch_size=16,
            evaluation_times=3,
            evaluation_samples=16,
            seed=13,
        ),
    )
    assert history[-1].step == 3
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )
