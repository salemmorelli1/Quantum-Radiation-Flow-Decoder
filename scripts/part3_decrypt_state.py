#!/usr/bin/env python3
"""Reload a verified trained bundle and repeat its held-out toy evaluation."""

import argparse
import json
from pathlib import Path

import torch

from quantum_radiation_flow.config import ExperimentConfig
from quantum_radiation_flow.flow import ConditionalRealNVP
from quantum_radiation_flow.pipeline import evaluate_model
from quantum_radiation_flow.provenance import state_fingerprint, verify_model_bundle
from quantum_radiation_flow.target import ReferenceRadiationModel
from quantum_radiation_flow.training import resolve_dtype


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    verify_model_bundle(args.artifacts)
    result = json.loads((args.artifacts / "experiment_summary.json").read_text("utf-8"))
    config = ExperimentConfig.from_mapping(result["config"])
    dtype = resolve_dtype(config.training.dtype)
    device = torch.device(config.training.device)
    model = ConditionalRealNVP(config.flow).to(device=device, dtype=dtype)
    target = ReferenceRadiationModel(config.physics).to(device=device, dtype=dtype)
    checkpoint = torch.load(args.artifacts / "model_checkpoint.pt", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    target.load_state_dict(checkpoint["target"])
    for name, module in (("model", model), ("target", target)):
        if state_fingerprint(module) != result["provenance"][f"{name}_state_sha256"]:
            raise ValueError(f"{name} state does not match the recorded experiment")
    curve = evaluate_model(model, target, config)
    print(
        json.dumps(
            {
                "time": curve[-1]["time"],
                "decoder_fidelity": curve[-1]["decoder_fidelity"],
                "oracle_inverse_fidelity": curve[-1]["oracle_inverse_fidelity"],
                "scientific_validation": result["scientific_validation"]["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
