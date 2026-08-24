"""Typed configuration objects for the conditional-flow experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PhysicsConfig:
    """Parameters for the toy evaporation and telemetry model."""

    total_qubits: int = 10
    data_dim: int = 8
    page_time: float = 0.5
    visibility_width: float = 0.075
    initial_mass: float = 1.0
    mass_floor: float = 0.08
    target_layers: int = 4
    target_hidden_dim: int = 64
    target_scale_clip: float = 0.8

    def validate(self) -> None:
        if self.total_qubits < 2:
            raise ValueError("total_qubits must be at least 2")
        if self.data_dim < 2 or self.data_dim % 2:
            raise ValueError("data_dim must be an even integer of at least 2")
        if not 0.0 < self.page_time < 1.0:
            raise ValueError("page_time must lie strictly between 0 and 1")
        if self.visibility_width <= 0.0:
            raise ValueError("visibility_width must be positive")
        if not 0.0 < self.mass_floor < self.initial_mass:
            raise ValueError("mass_floor must lie between zero and initial_mass")


@dataclass(frozen=True)
class FlowConfig:
    """Architecture parameters for conditional RealNVP."""

    data_dim: int = 8
    condition_dim: int = 2
    context_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 8
    scale_clip: float = 1.5

    def validate(self) -> None:
        if self.data_dim < 2:
            raise ValueError("data_dim must be at least 2")
        if self.condition_dim < 1 or self.context_dim < 2:
            raise ValueError("condition dimensions must be positive")
        if self.hidden_dim < 8 or self.num_layers < 2:
            raise ValueError("flow must contain at least two nontrivial layers")
        if self.scale_clip <= 0.0:
            raise ValueError("scale_clip must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and evaluation parameters."""

    steps: int = 1200
    batch_size: int = 256
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-6
    alignment_weight: float = 0.20
    gradient_clip: float = 5.0
    evaluation_times: int = 21
    evaluation_samples: int = 1024
    seed: int = 20260824
    dtype: str = "float64"
    device: str = "cpu"

    def validate(self) -> None:
        if self.steps < 1 or self.batch_size < 2:
            raise ValueError("steps and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer configuration")
        if self.alignment_weight < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("invalid regularization configuration")
        if self.evaluation_times < 3 or self.evaluation_samples < 16:
            raise ValueError("evaluation grid is too small")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete serializable experiment configuration."""

    physics: PhysicsConfig = PhysicsConfig()
    flow: FlowConfig = FlowConfig()
    training: TrainingConfig = TrainingConfig()

    def validate(self) -> None:
        self.physics.validate()
        self.flow.validate()
        self.training.validate()
        if self.physics.data_dim != self.flow.data_dim:
            raise ValueError("physics.data_dim must equal flow.data_dim")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ExperimentConfig:
        config = cls(
            physics=PhysicsConfig(**values.get("physics", {})),
            flow=FlowConfig(**values.get("flow", {})),
            training=TrainingConfig(**values.get("training", {})),
        )
        config.validate()
        return config
