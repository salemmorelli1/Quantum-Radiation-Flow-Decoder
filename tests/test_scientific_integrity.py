import copy
import math
from dataclasses import replace

import pytest
import torch

from quantum_radiation_flow.config import ExperimentConfig, FlowConfig, PhysicsConfig
from quantum_radiation_flow.diagnostics import curve_diagnostic, monte_carlo_standard_error
from quantum_radiation_flow.flow import AffineCoupling, ConditionalRealNVP, Permutation
from quantum_radiation_flow.quantum import (
    complex_state_from_real_coordinates,
    density_matrix,
    haar_state,
    harmonic_number,
    mass_schedule,
    page_expected_entropy,
    pure_state_fidelity,
    radiation_density_matrix,
    von_neumann_entropy,
)
from quantum_radiation_flow.target import ReferenceRadiationModel
from quantum_radiation_flow.training import reverse_kl_objective


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("physics", "total_qubits", True),
        ("physics", "data_dim", 4.0),
        ("physics", "target_layers", 0),
        ("physics", "target_hidden_dim", 3),
        ("physics", "target_scale_clip", 0),
        ("physics", "initial_mass", math.inf),
        ("flow", "num_layers", 2.5),
        ("flow", "condition_dim", 3),
        ("flow", "hidden_dim", False),
        ("training", "steps", 3.5),
        ("training", "seed", -1),
        ("training", "seed", 2**63),
        ("training", "alignment_weight", math.nan),
        ("training", "learning_rate", True),
        ("training", "device", 0),
        ("training", "dtype", []),
    ],
)
def test_invalid_configuration_is_rejected(section, field, value):
    with pytest.raises(ValueError):
        ExperimentConfig.from_mapping({section: {field: value}})


@pytest.mark.parametrize("mapping", [{"typo": {}}, {"flow": []}, []])
def test_invalid_config_sections_are_rejected(mapping):
    with pytest.raises(ValueError):
        ExperimentConfig.from_mapping(mapping)


@pytest.mark.parametrize(
    "rho",
    [
        torch.zeros(2, 2),
        torch.diag(torch.tensor([1.2, -0.2])),
        torch.eye(2),
        torch.tensor([[0.5, 0.4], [0.0, 0.5]]),
        torch.full((2, 2), math.nan),
        torch.empty(0, 0),
        torch.ones(2, 3),
        torch.ones(2),
    ],
)
def test_entropy_rejects_invalid_density_operators(rho):
    with pytest.raises(ValueError):
        von_neumann_entropy(rho)


def test_entropy_handles_zero_eigenvalues_and_roundoff_only():
    pure = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert von_neumann_entropy(pure).item() == 0.0
    perturbed = torch.diag(torch.tensor([1.0 + 1e-16, -1e-16], dtype=torch.float64))
    assert abs(von_neumann_entropy(perturbed).item()) < 1e-14
    states = torch.stack((pure, torch.eye(2, dtype=torch.float64) / 2))
    torch.testing.assert_close(
        von_neumann_entropy(states), torch.tensor([0.0, math.log(2)], dtype=torch.float64)
    )


@pytest.mark.parametrize("amplitude", [1e308, 1e-308, 2.0])
@pytest.mark.parametrize("complex_value", [True, False])
def test_pure_state_normalization_is_invariant_to_extreme_scaling(amplitude, complex_value):
    state = torch.tensor([amplitude, -amplitude], dtype=torch.float64)
    if complex_value:
        state = torch.complex(state, state)
    rho = density_matrix(state)
    torch.testing.assert_close(rho.diagonal().sum().real, torch.tensor(1.0, dtype=torch.float64))
    assert abs(von_neumann_entropy(rho).item()) < 1e-12
    torch.testing.assert_close(
        pure_state_fidelity(state, state), torch.tensor(1.0, dtype=torch.float64)
    )


@pytest.mark.parametrize("time", [-0.1, 1.1, math.inf, math.nan])
def test_mass_schedule_and_visibility_reject_invalid_time(time):
    t = torch.tensor(time, dtype=torch.float64)
    with pytest.raises(ValueError):
        mass_schedule(t)
    with pytest.raises(ValueError):
        ReferenceRadiationModel(PhysicsConfig()).visibility(t)


def test_haar_mean_agrees_with_finite_page_benchmark():
    generator = torch.Generator().manual_seed(37)
    states = torch.stack([haar_state(16, generator=generator) for _ in range(512)])
    rho = radiation_density_matrix(states, 4, 4)
    entropy = von_neumann_entropy(rho)
    se = entropy.std() / math.sqrt(entropy.numel())
    assert abs(entropy.mean().item() - page_expected_entropy(2, 4)) < 5 * se


def test_large_harmonic_evaluation_preserves_float64_accuracy_and_page_symmetry():
    for n in (10001, 20000):
        exact = math.fsum(1.0 / k for k in range(1, n + 1))
        assert abs(harmonic_number(n) - exact) < 5e-15
    assert abs(page_expected_entropy(30, 60) - (30 * math.log(2) - 0.5)) < 1e-12
    assert page_expected_entropy(20, 60) == page_expected_entropy(40, 60)


def test_squared_visibility_proxy_is_not_expected_normalized_gaussian_fidelity():
    generator = torch.Generator().manual_seed(71)
    secret = torch.randn(65536, 8, generator=generator, dtype=torch.float64)
    noise = torch.randn(65536, 8, generator=generator, dtype=torch.float64)
    accessible = 0.5 * secret + math.sqrt(0.75) * noise
    scores = pure_state_fidelity(
        complex_state_from_real_coordinates(secret), complex_state_from_real_coordinates(accessible)
    )
    mcse = scores.std() / math.sqrt(scores.numel())
    # The report's 0.25+0.75*v^2 formula is deliberately labeled heuristic.
    assert abs(scores.mean().item() - 0.4375) > 5 * mcse


def test_nontrivial_flow_matches_full_autograd_jacobian_and_inverse():
    torch.set_default_dtype(torch.float64)
    target = ReferenceRadiationModel(
        replace(PhysicsConfig(), data_dim=4, target_layers=2, target_hidden_dim=16), seed=41
    )
    model = target.scrambler
    latent = torch.tensor([0.2, -0.3, 0.6, 0.9])
    condition = target.condition(torch.tensor(0.7))
    result = model.decode(latent[None], condition)
    jacobian = torch.autograd.functional.jacobian(
        lambda z: model.decode(z[None], condition).value[0], latent
    )
    logdet = torch.linalg.slogdet(jacobian).logabsdet
    assert abs(logdet.item()) > 1e-4
    torch.testing.assert_close(logdet, result.log_abs_det_jacobian[0], atol=1e-12, rtol=1e-10)
    inverse = model.encode(result.value, condition)
    torch.testing.assert_close(inverse.value[0], latent, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(inverse.log_abs_det_jacobian[0], -logdet, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("which", ["value", "condition"])
@pytest.mark.parametrize("invalid", [math.inf, math.nan])
def test_flow_rejects_nonfinite_inputs(which, invalid):
    model = ConditionalRealNVP(FlowConfig(data_dim=4, hidden_dim=16, num_layers=2))
    values = torch.ones(2, 4)
    condition = torch.ones(2, 2)
    (values if which == "value" else condition)[0, 0] = invalid
    with pytest.raises(ValueError, match="finite"):
        model.log_prob(values, condition)


def test_flow_rejects_arithmetic_overflow_in_finite_input():
    model = ConditionalRealNVP(FlowConfig(data_dim=4, hidden_dim=16, num_layers=2)).double()
    with pytest.raises(FloatingPointError):
        model.log_prob(
            torch.full((1, 4), 1e308, dtype=torch.float64), torch.ones(1, 2, dtype=torch.float64)
        )


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_flow_rejects_invalid_sample_counts(count):
    model = ConditionalRealNVP(FlowConfig(data_dim=4, hidden_dim=16, num_layers=2))
    with pytest.raises(ValueError):
        model.sample(count, torch.ones(1, 2))


def test_entropy_does_not_mislabel_multiple_conditions():
    model = ConditionalRealNVP(FlowConfig(data_dim=4, hidden_dim=16, num_layers=2))
    with pytest.raises(ValueError, match="shared condition"):
        model.entropy(2, torch.tensor([[0.2, 0.9], [0.7, 0.6]]))


def test_invalid_mask_and_duplicate_permutation_are_rejected():
    with pytest.raises(ValueError, match="binary"):
        AffineCoupling(4, 4, 16, torch.tensor([0.0, 0.5, 1.0, 1.0]), 1.0)
    with pytest.raises(ValueError, match="binary"):
        AffineCoupling(4, 4, 16, torch.zeros(4), 1.0)
    with pytest.raises(ValueError, match="exactly once"):
        Permutation(torch.tensor([0, 1, 1, 3]))


def test_reverse_kl_is_zero_for_same_target_and_is_differentiable():
    torch.set_default_dtype(torch.float64)
    target = ReferenceRadiationModel(
        replace(PhysicsConfig(), data_dim=4, target_layers=2, target_hidden_dim=16), seed=43
    )
    model = copy.deepcopy(target.scrambler).requires_grad_(True)
    condition = target.condition(torch.tensor(0.7))
    loss = reverse_kl_objective(
        model, target, condition, 16, generator=torch.Generator().manual_seed(23)
    )
    assert abs(loss.item()) < 1e-12
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_rotated_latent_preserves_density_but_changes_paired_fidelity():
    generator = torch.Generator().manual_seed(31)
    secret = torch.randn(4096, 8, generator=generator, dtype=torch.float64)
    rotated = secret.roll(2, dims=-1)
    # Orthogonal permutations leave the Gaussian density exactly unchanged.
    torch.testing.assert_close(secret.square().sum(-1), rotated.square().sum(-1))
    state = complex_state_from_real_coordinates(secret)
    rotated_state = complex_state_from_real_coordinates(rotated)
    assert pure_state_fidelity(state, state).mean() > 0.999
    assert pure_state_fidelity(state, rotated_state).mean() < 0.3


@pytest.mark.parametrize(
    "times,values",
    [
        ([0.0, 0.5, 0.5], [0.0, 1.0, 0.0]),
        ([0.0, 0.7, 0.4], [0.0, 1.0, 0.0]),
        ([0.0, 0.5, 1.0], [0.0, math.nan, 0.0]),
    ],
)
def test_curve_diagnostic_rejects_invalid_data(times, values):
    with pytest.raises(ValueError):
        curve_diagnostic(times, values)


def test_mcse_rejects_nonfinite_observations():
    with pytest.raises(ValueError):
        monte_carlo_standard_error([0.0, math.inf])
