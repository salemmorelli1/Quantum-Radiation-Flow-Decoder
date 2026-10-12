"""End-to-end training, decryption, and entropy evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .artifacts import write_curve_csv, write_json
from .config import ExperimentConfig
from .diagnostics import curve_diagnostic
from .flow import ConditionalRealNVP
from .provenance import runtime_provenance, stream_seed, write_model_bundle
from .quantum import (
    complex_state_from_real_coordinates,
    page_expected_entropy,
    pure_state_fidelity,
)
from .target import ReferenceRadiationModel
from .training import resolve_dtype, train_conditional_flow
from .validation import finite_result


def _interpolated_page_entropy(time: float, total_qubits: int) -> float:
    position = min(max(time, 0.0), 1.0) * total_qubits
    lower = int(math.floor(position))
    upper = min(total_qubits, lower + 1)
    fraction = position - lower
    left = page_expected_entropy(lower, total_qubits)
    right = page_expected_entropy(upper, total_qubits)
    return (1.0 - fraction) * left + fraction * right


@torch.no_grad()
def evaluate_model(
    model: ConditionalRealNVP,
    target: ReferenceRadiationModel,
    config: ExperimentConfig,
) -> list[dict[str, float]]:
    """Held-out snapshot estimates, conditional on one trained seed.

    Fidelity intervals are pointwise normal Monte Carlo intervals over fresh
    independent secrets. They do not quantify retraining/seed uncertainty and
    are not simultaneous intervals over the time grid.
    """

    config.validate()
    dtype = resolve_dtype(config.training.dtype)
    device = torch.device(config.training.device)
    model.eval().to(device=device, dtype=dtype)
    target.eval().to(device=device, dtype=dtype)
    generator = torch.Generator(device=device).manual_seed(
        stream_seed(config.training.seed, "evaluation-target")
    )
    flow_generator = torch.Generator(device=device).manual_seed(
        stream_seed(config.training.seed, "evaluation-flow")
    )
    null_generator = torch.Generator(device=device).manual_seed(
        stream_seed(config.training.seed, "evaluation-null")
    )
    rows: list[dict[str, float]] = []
    time_grid = torch.linspace(
        0.0,
        1.0,
        config.training.evaluation_times,
        device=device,
        dtype=dtype,
    )

    for time_scalar in time_grid:
        batch = target.sample(
            config.training.evaluation_samples,
            time=time_scalar,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        log_q_on_p = model.log_prob(batch.observation, batch.condition)
        encoded = model.encode(batch.observation, batch.condition)
        decoded_state = complex_state_from_real_coordinates(encoded.value)
        secret_state = complex_state_from_real_coordinates(batch.secret_coordinates)
        fidelity = pure_state_fidelity(decoded_state, secret_state)
        oracle_state = complex_state_from_real_coordinates(batch.accessible_latent)
        oracle_fidelity = pure_state_fidelity(oracle_state, secret_state)
        null_state = complex_state_from_real_coordinates(
            torch.randn(
                batch.secret_coordinates.shape, generator=null_generator, device=device, dtype=dtype
            )
        )
        null_fidelity = pure_state_fidelity(null_state, secret_state)
        gain = fidelity - null_fidelity
        reconstructed = model.decode(encoded.value, batch.condition)

        single_condition = batch.condition[:1]
        q_samples, log_q_on_q = model.sample(
            config.training.evaluation_samples,
            single_condition,
            generator=flow_generator,
        )
        repeated_condition = single_condition.expand(config.training.evaluation_samples, -1)
        log_p_on_q = target.log_prob(q_samples, repeated_condition)
        forward_contributions = batch.target_log_prob - log_q_on_p
        reverse_contributions = log_q_on_q - log_p_on_q
        forward_kl = forward_contributions.mean()
        reverse_kl = reverse_contributions.mean()
        forward_kl_se = forward_contributions.std(unbiased=True) / math.sqrt(
            forward_contributions.numel()
        )
        reverse_kl_se = reverse_contributions.std(unbiased=True) / math.sqrt(
            reverse_contributions.numel()
        )
        entropy_samples = -log_q_on_q
        entropy_se = entropy_samples.std(unbiased=True) / math.sqrt(entropy_samples.numel())
        time_value = float(time_scalar)
        fidelity_mcse = fidelity.std(unbiased=True) / math.sqrt(fidelity.numel())
        gain_mcse = gain.std(unbiased=True) / math.sqrt(gain.numel())

        rows.append(
            {
                "time": time_value,
                "mass": float(single_condition[0, 1]),
                "page_entropy_nats": _interpolated_page_entropy(
                    time_value, config.physics.total_qubits
                ),
                "flow_shannon_entropy_nats": float(entropy_samples.mean()),
                "flow_entropy_mcse": float(entropy_se),
                "target_cross_entropy_nats": float(-log_q_on_p.mean()),
                "forward_kl_nats": float(forward_kl),
                "forward_kl_mcse": float(forward_kl_se),
                "reverse_kl_nats": float(reverse_kl),
                "reverse_kl_mcse": float(reverse_kl_se),
                "decoder_fidelity": float(fidelity.mean()),
                "decoder_fidelity_mcse": float(fidelity_mcse),
                "decoder_fidelity_ci95_low": float(
                    fidelity.mean() - 1.959963984540054 * fidelity_mcse
                ),
                "decoder_fidelity_ci95_high": float(
                    fidelity.mean() + 1.959963984540054 * fidelity_mcse
                ),
                "null_fidelity": float(null_fidelity.mean()),
                "null_fidelity_mcse": float(
                    null_fidelity.std(unbiased=True) / math.sqrt(fidelity.numel())
                ),
                "null_fidelity_expectation": 2.0 / config.physics.data_dim,
                "oracle_inverse_fidelity": float(oracle_fidelity.mean()),
                "oracle_inverse_fidelity_mcse": float(
                    oracle_fidelity.std(unbiased=True) / math.sqrt(fidelity.numel())
                ),
                "paired_fidelity_gain": float(gain.mean()),
                "paired_fidelity_gain_mcse": float(gain_mcse),
                "paired_fidelity_gain_ci95_low": float(gain.mean() - 1.959963984540054 * gain_mcse),
                "paired_fidelity_gain_ci95_high": float(
                    gain.mean() + 1.959963984540054 * gain_mcse
                ),
                "round_trip_max_error": float(
                    (reconstructed.value - batch.observation).abs().max()
                ),
                "jacobian_cancellation_max_error": float(
                    (encoded.log_abs_det_jacobian + reconstructed.log_abs_det_jacobian).abs().max()
                ),
                "visibility": float(batch.visibility.mean()),
            }
        )
    return rows


def run_experiment(
    config: ExperimentConfig,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Run a simulated experiment while preserving caller RNG and dtype state."""

    config.validate()
    previous_dtype = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]):
        try:
            torch.set_default_dtype(resolve_dtype(config.training.dtype))
            torch.default_generator.manual_seed(
                stream_seed(config.training.seed, "model-initialization")
            )
            return _run_experiment(config, output_directory=output_directory)
        finally:
            torch.set_default_dtype(previous_dtype)


def _run_experiment(
    config: ExperimentConfig, *, output_directory: str | Path | None
) -> dict[str, Any]:
    dtype = resolve_dtype(config.training.dtype)
    target = ReferenceRadiationModel(
        config.physics, seed=stream_seed(config.training.seed, "target-initialization")
    )
    model = ConditionalRealNVP(config.flow)
    history = train_conditional_flow(model, target, config.training)
    curve = evaluate_model(model, target, config)
    finite_result(
        torch.tensor([value for row in curve for value in row.values()], dtype=dtype),
        "evaluation curve",
    )
    page_summary = curve_diagnostic(
        [row["time"] for row in curve],
        [row["page_entropy_nats"] for row in curve],
    )
    flow_summary = curve_diagnostic(
        [row["time"] for row in curve],
        [row["flow_shannon_entropy_nats"] for row in curve],
    )
    result: dict[str, Any] = {
        "schema": "quantum-radiation-flow-experiment-v2",
        "status": "COMPLETED",
        "scientific_validation": {
            "status": "NOT_ASSESSED",
            "reason": "No preregistered tolerances, seed-family inference, or trained ablation comparison was executed.",
            "fidelity_uncertainty": "Pointwise approximate normal Monte Carlo intervals, conditional on one trained model; not simultaneous or seed-level intervals.",
            "controls": "Independent random-state null and reference inverse diagnostic; neither is an unconditioned or no-alignment trained ablation.",
            "oracle_scope": "Inverting the known scrambler yields accessible y, not the unknown secret; this diagnostic is not an optimal-decoder fidelity bound.",
        },
        "interpretation": {
            "quantum_entropy": "von Neumann/Page benchmark for the toy bipartition",
            "flow_entropy": "differential Shannon entropy of classical telemetry",
            "decoder_fidelity": "paired toy-state recovery, not physical black-hole decoding",
        },
        "config": config.to_dict(),
        "provenance": runtime_provenance(model, target),
        "random_streams": {
            name: stream_seed(config.training.seed, name)
            for name in (
                "model-initialization",
                "target-initialization",
                "training",
                "evaluation-target",
                "evaluation-flow",
                "evaluation-null",
            )
        },
        "training_history": [asdict(record) for record in history],
        "curve": curve,
        "page_curve_diagnostic": asdict(page_summary),
        "flow_entropy_diagnostic": asdict(flow_summary),
    }
    if output_directory is not None:
        directory = Path(output_directory)
        write_json(directory / "experiment_summary.json", result)
        write_curve_csv(directory / "page_curve.csv", curve)
        write_model_bundle(directory, model, target)
    return result
