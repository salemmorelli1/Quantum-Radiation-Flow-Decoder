import copy
import json
import math
from dataclasses import replace

import pytest
import torch

from quantum_radiation_flow.artifacts import write_curve_csv
from quantum_radiation_flow.config import (
    ExperimentConfig,
    FlowConfig,
    PhysicsConfig,
    TrainingConfig,
)
from quantum_radiation_flow.flow import ConditionalRealNVP
from quantum_radiation_flow.pipeline import evaluate_model, run_experiment
from quantum_radiation_flow.provenance import state_fingerprint, stream_seed, verify_model_bundle
from quantum_radiation_flow.target import ReferenceRadiationModel


def small_config():
    return ExperimentConfig(
        physics=replace(PhysicsConfig(), data_dim=4, target_layers=2, target_hidden_dim=16),
        flow=FlowConfig(data_dim=4, hidden_dim=16, context_dim=8, num_layers=2),
        training=TrainingConfig(
            steps=2, batch_size=16, evaluation_times=3, evaluation_samples=32, seed=47
        ),
    )


def test_run_is_reproducible_preserves_caller_state_and_writes_reloadable_evidence(tmp_path):
    config = small_config()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(53)
    rng = torch.random.get_rng_state().clone()
    result = run_experiment(config, output_directory=tmp_path)
    assert torch.get_default_dtype() == torch.float32
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert result == run_experiment(config)
    assert result["status"] == "COMPLETED"
    assert result["scientific_validation"]["status"] == "NOT_ASSESSED"
    assert len(set(result["random_streams"].values())) == 6
    verify_model_bundle(tmp_path)
    saved = json.loads((tmp_path / "experiment_summary.json").read_text())
    assert saved == result
    checkpoint = torch.load(tmp_path / "model_checkpoint.pt", weights_only=True)
    model = ConditionalRealNVP(config.flow).double()
    target = ReferenceRadiationModel(config.physics).double()
    model.load_state_dict(checkpoint["model"])
    target.load_state_dict(checkpoint["target"])
    assert state_fingerprint(model) == result["provenance"]["model_state_sha256"]
    assert state_fingerprint(target) == result["provenance"]["target_state_sha256"]
    assert evaluate_model(model, target, config) == result["curve"]


def test_run_restores_dtype_and_rng_after_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise FloatingPointError("synthetic optimizer failure")

    monkeypatch.setattr("quantum_radiation_flow.pipeline.train_conditional_flow", fail)
    torch.set_default_dtype(torch.float32)
    state = torch.random.get_rng_state().clone()
    with pytest.raises(FloatingPointError):
        run_experiment(small_config())
    assert torch.get_default_dtype() == torch.float32
    assert torch.equal(torch.random.get_rng_state(), state)


def test_oracle_flow_produces_zero_kl_and_equal_decoder_oracle_scores():
    torch.set_default_dtype(torch.float64)
    config = replace(
        small_config(), training=replace(small_config().training, evaluation_samples=512)
    )
    target = ReferenceRadiationModel(config.physics, seed=61)
    model = copy.deepcopy(target.scrambler)
    rows = evaluate_model(model, target, config)
    for row in rows:
        assert abs(row["forward_kl_nats"]) < 1e-12
        assert abs(row["reverse_kl_nats"]) < 1e-12
        assert abs(row["decoder_fidelity"] - row["oracle_inverse_fidelity"]) < 1e-12
        assert row["round_trip_max_error"] < 1e-10
        assert row["jacobian_cancellation_max_error"] < 1e-10
        assert abs(row["null_fidelity"] - 0.5) < 5 * row["null_fidelity_mcse"]
        assert row["paired_fidelity_gain"] == pytest.approx(
            row["decoder_fidelity"] - row["null_fidelity"]
        )
        assert row["paired_fidelity_gain_ci95_high"] - row["paired_fidelity_gain"] == pytest.approx(
            1.959963984540054 * row["paired_fidelity_gain_mcse"]
        )
        assert all(math.isfinite(value) for value in row.values())
    assert rows[-1]["decoder_fidelity"] > 0.99
    assert rows[0]["decoder_fidelity"] < 0.6


def test_stream_seeds_are_separate_and_repeatable():
    assert stream_seed(47, "training") == stream_seed(47, "training")
    assert stream_seed(47, "training") != stream_seed(47, "evaluation-target")
    assert stream_seed(47, "training") != stream_seed(48, "training")


@pytest.mark.parametrize(
    "rows",
    [
        [{"a": math.nan}],
        [{"a": math.inf}],
        [{"a": True}],
        [{"a": 1.0}, {"b": 2.0}],
        [{"a": 1.0}, {"a": "bad"}],
        [],
    ],
)
def test_csv_rejects_invalid_data_without_overwriting_previous_artifact(tmp_path, rows):
    path = tmp_path / "curve.csv"
    path.write_text("previous artifact\n")
    with pytest.raises(ValueError):
        write_curve_csv(path, rows)
    assert path.read_text() == "previous artifact\n"


@pytest.mark.parametrize(
    "name", ["experiment_summary.json", "page_curve.csv", "model_checkpoint.pt"]
)
def test_bundle_detects_tampering(tmp_path, name):
    run_experiment(small_config(), output_directory=tmp_path)
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        verify_model_bundle(tmp_path)


def test_float32_experiment_is_finite_and_labeled():
    config = small_config()
    config = replace(config, training=replace(config.training, dtype="float32"))
    result = run_experiment(config)
    assert result["status"] == "COMPLETED"
    assert all(math.isfinite(value) for row in result["curve"] for value in row.values())
