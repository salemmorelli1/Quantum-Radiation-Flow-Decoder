"""Finite-dimensional quantum-information benchmarks for the toy model."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _safe_entropy(probabilities: Tensor, *, epsilon: float = 1.0e-15) -> Tensor:
    probabilities = probabilities.real.clamp_min(epsilon)
    return -(probabilities * probabilities.log()).sum(dim=-1)


def density_matrix(state: Tensor) -> Tensor:
    """Return |psi><psi| for normalized batched complex state vectors."""

    state = state / state.norm(dim=-1, keepdim=True).clamp_min(1.0e-15)
    return state.unsqueeze(-1) * state.conj().unsqueeze(-2)


def radiation_density_matrix(
    state: Tensor,
    black_hole_dimension: int,
    radiation_dimension: int,
) -> Tensor:
    """Trace a pure bipartite state over the black-hole factor."""

    expected = black_hole_dimension * radiation_dimension
    if state.shape[-1] != expected:
        raise ValueError(f"state dimension must be {expected}")
    psi = state.reshape(*state.shape[:-1], black_hole_dimension, radiation_dimension)
    return torch.einsum("...br,...bs->...rs", psi, psi.conj())


def von_neumann_entropy(rho: Tensor, *, epsilon: float = 1.0e-15) -> Tensor:
    """Compute -Tr(rho log rho) in nats for Hermitian density matrices."""

    rho = 0.5 * (rho + rho.conj().transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(rho).real.clamp_min(0.0)
    eigenvalues = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    return _safe_entropy(eigenvalues, epsilon=epsilon)


def haar_state(
    dimension: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Draw a normalized complex vector from the Haar-induced Gaussian law."""

    real = torch.randn(dimension, generator=generator, device=device, dtype=dtype)
    imag = torch.randn(dimension, generator=generator, device=device, dtype=dtype)
    state = torch.complex(real, imag)
    return state / state.norm().clamp_min(1.0e-15)


def harmonic_number(n: int) -> float:
    if n < 1:
        raise ValueError("n must be positive")
    return math.fsum(1.0 / k for k in range(1, n + 1))


def page_expected_entropy(radiation_qubits: int, total_qubits: int) -> float:
    """Page's exact average entropy for a Haar-random bipartite pure state.

    The returned entropy is in nats.  The smaller Hilbert-space dimension is
    called m and the larger n, so the formula applies on both sides of Page time.
    """

    if not 0 <= radiation_qubits <= total_qubits:
        raise ValueError("radiation_qubits must lie in [0, total_qubits]")
    d_r = 2**radiation_qubits
    d_b = 2 ** (total_qubits - radiation_qubits)
    m, n = min(d_r, d_b), max(d_r, d_b)
    if m == 1:
        return 0.0
    return harmonic_number(m * n) - harmonic_number(n) - (m - 1.0) / (2.0 * n)


def page_curve(total_qubits: int) -> tuple[Tensor, Tensor]:
    times = torch.linspace(0.0, 1.0, total_qubits + 1, dtype=torch.float64)
    entropy = torch.tensor(
        [page_expected_entropy(k, total_qubits) for k in range(total_qubits + 1)],
        dtype=torch.float64,
    )
    return times, entropy


def mass_schedule(
    time: Tensor,
    *,
    initial_mass: float = 1.0,
    mass_floor: float = 0.08,
) -> Tensor:
    """Stopped semiclassical M(t) proportional to (1-t)^(1/3)."""

    clipped = time.clamp(0.0, 1.0)
    mass = initial_mass * (1.0 - clipped).clamp_min(0.0).pow(1.0 / 3.0)
    return mass.clamp_min(mass_floor)


def complex_state_from_real_coordinates(coordinates: Tensor) -> Tensor:
    """Interpret an even real vector as normalized complex amplitudes."""

    if coordinates.shape[-1] % 2:
        raise ValueError("the final coordinate dimension must be even")
    half = coordinates.shape[-1] // 2
    state = torch.complex(coordinates[..., :half], coordinates[..., half:])
    return state / state.norm(dim=-1, keepdim=True).clamp_min(1.0e-15)


def pure_state_fidelity(state_a: Tensor, state_b: Tensor) -> Tensor:
    """Return |<a|b>|^2 for batches of normalized pure states."""

    state_a = state_a / state_a.norm(dim=-1, keepdim=True).clamp_min(1.0e-15)
    state_b = state_b / state_b.norm(dim=-1, keepdim=True).clamp_min(1.0e-15)
    overlap = (state_a.conj() * state_b).sum(dim=-1)
    return overlap.abs().square().real.clamp(0.0, 1.0)
