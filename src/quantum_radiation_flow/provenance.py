"""Fingerprints and reloadable model artifacts for simulated experiments."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import __version__
from .artifacts import write_json


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream_seed(root: int, label: str) -> int:
    digest = hashlib.sha256(f"qrf-v2:{root}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def state_fingerprint(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        # Hash the native CPU storage without introducing a NumPy dependency
        # into the otherwise pure-PyTorch simulation package.
        digest.update(bytes(value.reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()


def runtime_provenance(model: nn.Module, target: nn.Module) -> dict[str, Any]:
    source = Path(__file__).parent
    sources = {path.name: sha256_file(path) for path in sorted(source.glob("*.py"))}
    return {
        "package_version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "source_sha256": sources,
        "model_state_sha256": state_fingerprint(model),
        "target_state_sha256": state_fingerprint(target),
        "scope": "integrity fingerprints; not physical-source authentication or execution attestation",
    }


def write_model_bundle(directory: Path, model: nn.Module, target: nn.Module) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model_checkpoint.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "target": {k: v.detach().cpu() for k, v in target.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(checkpoint)
    paths = ("experiment_summary.json", "page_curve.csv", "model_checkpoint.pt")
    write_json(
        directory / "artifact_manifest.json",
        {
            "schema": "quantum-radiation-flow-artifacts-v1",
            "files": {name: sha256_file(directory / name) for name in paths},
        },
    )


def verify_model_bundle(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory)
    manifest = json.loads((directory / "artifact_manifest.json").read_text("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("artifact manifest must be an object")
    expected = {"experiment_summary.json", "page_curve.csv", "model_checkpoint.pt"}
    if manifest.get("schema") != "quantum-radiation-flow-artifacts-v1":
        raise ValueError("unsupported artifact manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError("incomplete artifact manifest")
    for name, expected_digest in files.items():
        if sha256_file(directory / name) != expected_digest:
            raise ValueError(f"artifact fingerprint mismatch: {name}")
    return manifest
