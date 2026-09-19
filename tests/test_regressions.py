import math
from dataclasses import replace

import pytest
import torch

from quantum_radiation_flow.artifacts import write_json
from quantum_radiation_flow.config import ExperimentConfig, PhysicsConfig
from quantum_radiation_flow.flow import ConditionalRealNVP
from quantum_radiation_flow.quantum import radiation_density_matrix
from quantum_radiation_flow.target import ReferenceRadiationModel
from quantum_radiation_flow.training import reverse_kl_objective


def test_entropy_rejects_single_draw_before_nan_mcse():
    torch.set_default_dtype(torch.float64)
    model = ConditionalRealNVP(
        replace(ExperimentConfig().flow, data_dim=4, hidden_dim=16, num_layers=2)
    )
    with pytest.raises(ValueError, match="at least two"):
        model.entropy(1, torch.tensor([[0.5, 0.5]], dtype=torch.float64))


def test_partial_trace_normalizes_an_unnormalized_state():
    state = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex128)
    rho_r = radiation_density_matrix(state, black_hole_dimension=2, radiation_dimension=2)
    torch.testing.assert_close(torch.trace(rho_r), torch.tensor(1.0, dtype=torch.complex128))


def test_reverse_kl_preserves_per_sample_conditions():
    torch.set_default_dtype(torch.float64)
    physics = replace(PhysicsConfig(), data_dim=4, target_hidden_dim=16, target_layers=2)
    target = ReferenceRadiationModel(physics, seed=4)
    model = ConditionalRealNVP(
        replace(ExperimentConfig().flow, data_dim=4, hidden_dim=16, num_layers=2)
    )
    conditions = torch.tensor([[0.2, 0.9], [0.8, 0.6]], dtype=torch.float64)
    value = reverse_kl_objective(
        model,
        target,
        conditions,
        sample_count=2,
        generator=torch.Generator().manual_seed(5),
    )
    assert torch.isfinite(value)
    bad_conditions = torch.cat((conditions, conditions[:1]))
    with pytest.raises(ValueError, match="one or sample_count"):
        reverse_kl_objective(model, target, bad_conditions, sample_count=2)


def test_configuration_rejects_nonfinite_optimizer_values():
    with pytest.raises(ValueError, match="learning_rate"):
        ExperimentConfig.from_mapping({"training": {"learning_rate": math.nan}})


def test_target_rejects_non_normalized_time():
    target = ReferenceRadiationModel(PhysicsConfig())
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        target.sample(2, time=torch.tensor(1.1))


def test_json_artifacts_fail_closed_on_nonfinite_values(tmp_path):
    with pytest.raises(ValueError, match="Out of range float values"):
        write_json(tmp_path / "invalid.json", {"value": math.nan})
