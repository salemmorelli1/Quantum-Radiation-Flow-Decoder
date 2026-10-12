"""Finite-dimensional quantum-information benchmarks for the toy model."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor

from .validation import finite_tensor, sample_count


def _normalize_state(state: Tensor) -> Tensor:
    if state.ndim < 1:
        raise ValueError("state must have at least one dimension")
    finite_tensor(state, "state", complex_allowed=True)
    # Scaling components first avoids overflow in complex magnitudes and norms,
    # and also permits valid states with arbitrarily small finite amplitudes.
    scale = state.real.abs().amax(dim=-1, keepdim=True)
    if state.is_complex():
        scale = torch.maximum(scale, state.imag.abs().amax(dim=-1, keepdim=True))
    if torch.any(scale == 0):
        raise ValueError("state norm must be positive")
    scaled = state / scale
    return cast(Tensor, scaled / scaled.norm(dim=-1, keepdim=True))


def density_matrix(state: Tensor) -> Tensor:
    """Return |psi><psi| for normalized batched complex state vectors."""

    state = _normalize_state(state)
    return state.unsqueeze(-1) * state.conj().unsqueeze(-2)


def radiation_density_matrix(
    state: Tensor,
    black_hole_dimension: int,
    radiation_dimension: int,
) -> Tensor:
    """Trace a pure bipartite state over the black-hole factor."""

    sample_count(black_hole_dimension)
    sample_count(radiation_dimension)
    expected = black_hole_dimension * radiation_dimension
    if state.ndim < 1:
        raise ValueError("state must have at least one dimension")
    if state.shape[-1] != expected:
        raise ValueError(f"state dimension must be {expected}")
    state = _normalize_state(state)
    psi = state.reshape(*state.shape[:-1], black_hole_dimension, radiation_dimension)
    return torch.einsum("...br,...bs->...rs", psi, psi.conj())


def von_neumann_entropy(rho: Tensor, *, epsilon: float = 1.0e-15) -> Tensor:
    """Entropy in nats of a finite Hermitian, PSD, trace-one density matrix.

    Only roundoff within the dtype tolerance is repaired. Invalid operators are
    rejected instead of being silently projected into a different state.
    """

    if rho.ndim < 2 or rho.shape[-1] != rho.shape[-2]:
        raise ValueError("rho must contain square density matrices")
    finite_tensor(rho, "rho", complex_allowed=True)
    if rho.dtype not in {torch.float32, torch.float64, torch.complex64, torch.complex128}:
        raise ValueError("rho requires float32/float64 or their complex counterparts")
    if not math.isfinite(epsilon) or not 0 < epsilon < 1.0e-3:
        raise ValueError("epsilon must be a positive roundoff tolerance below 1e-3")
    tolerance = max(epsilon, 32 * torch.finfo(rho.real.dtype).eps)
    adjoint = rho.conj().transpose(-1, -2)
    if torch.any((rho - adjoint).abs() > tolerance):
        raise ValueError("rho must be Hermitian")
    trace = rho.diagonal(dim1=-2, dim2=-1).sum(-1)
    if torch.any((trace - 1).abs() > tolerance):
        raise ValueError("rho must have trace one")
    eigenvalues = torch.linalg.eigvalsh(0.5 * (rho + adjoint)).real
    if torch.any(eigenvalues < -tolerance):
        raise ValueError("rho must be positive semidefinite")
    eigenvalues = eigenvalues.clamp_min(0.0)
    eigenvalues = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True)
    # xlogy implements the exact limit 0 log(0)=0, without artificial entropy.
    return cast(Tensor, -torch.special.xlogy(eigenvalues, eigenvalues).sum(dim=-1))


def haar_state(
    dimension: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Draw a normalized complex vector from the Haar-induced Gaussian law."""

    sample_count(dimension)
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("Haar coordinates require float32 or float64")
    real = torch.randn(dimension, generator=generator, device=device, dtype=dtype)
    imag = torch.randn(dimension, generator=generator, device=device, dtype=dtype)
    state = torch.complex(real, imag)
    return _normalize_state(state)


def harmonic_number(n: int) -> float:
    sample_count(n)
    if n <= 10000:
        return math.fsum(1.0 / k for k in range(1, n + 1))
    reciprocal = 1 / n
    return (
        math.log(n)
        + 0.5772156649015329
        + reciprocal / 2
        - reciprocal**2 / 12
        + reciprocal**4 / 120
        - reciprocal**6 / 252
    )


def page_expected_entropy(radiation_qubits: int, total_qubits: int) -> float:
    """Page's exact average entropy for a Haar-random bipartite pure state.

    The returned entropy is in nats.  The smaller Hilbert-space dimension is
    called m and the larger n, so the formula applies on both sides of Page time.
    """

    sample_count(total_qubits, minimum=0)
    sample_count(radiation_qubits, minimum=0)
    if not 0 <= radiation_qubits <= total_qubits:
        raise ValueError("radiation_qubits must lie in [0, total_qubits]")
    d_r = 2**radiation_qubits
    d_b = 2 ** (total_qubits - radiation_qubits)
    m, n = min(d_r, d_b), max(d_r, d_b)
    if m == 1:
        return 0.0
    return float(harmonic_number(m * n) - harmonic_number(n) - (m - 1.0) / (2.0 * n))


def page_curve(total_qubits: int) -> tuple[Tensor, Tensor]:
    sample_count(total_qubits)
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

    finite_tensor(time, "time")
    if torch.any((time < 0) | (time > 1)):
        raise ValueError("time must lie in [0, 1]")
    if not all(math.isfinite(v) for v in (initial_mass, mass_floor)):
        raise ValueError("mass parameters must be finite")
    if not 0 < mass_floor < initial_mass:
        raise ValueError("mass_floor must lie between zero and initial_mass")
    mass = initial_mass * (1.0 - time).pow(1.0 / 3.0)
    return mass.clamp_min(mass_floor)


def complex_state_from_real_coordinates(coordinates: Tensor) -> Tensor:
    """Interpret an even real vector as normalized complex amplitudes."""

    finite_tensor(coordinates, "coordinates")
    if coordinates.ndim < 1 or coordinates.shape[-1] < 2 or coordinates.shape[-1] % 2:
        raise ValueError("the final coordinate dimension must be even")
    half = coordinates.shape[-1] // 2
    state = torch.complex(coordinates[..., :half], coordinates[..., half:])
    return _normalize_state(state)


def pure_state_fidelity(state_a: Tensor, state_b: Tensor) -> Tensor:
    """Return |<a|b>|^2 for batches of normalized pure states."""

    if state_a.ndim < 1 or state_b.ndim < 1 or state_a.shape[-1] != state_b.shape[-1]:
        raise ValueError("states must have the same final dimension")
    state_a = _normalize_state(state_a)
    state_b = _normalize_state(state_b)
    overlap = (state_a.conj() * state_b).sum(dim=-1)
    return overlap.abs().square().real.clamp(0.0, 1.0)
