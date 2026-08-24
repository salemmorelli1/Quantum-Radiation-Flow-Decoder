"""Differentiable quantum-radiation flow decoder."""

from .config import ExperimentConfig, FlowConfig, PhysicsConfig, TrainingConfig
from .flow import ConditionalRealNVP, TransformResult, standard_normal_log_prob
from .pipeline import evaluate_model, run_experiment
from .quantum import (
    complex_state_from_real_coordinates,
    page_curve,
    page_expected_entropy,
    pure_state_fidelity,
    radiation_density_matrix,
    von_neumann_entropy,
)
from .target import RadiationBatch, ReferenceRadiationModel
from .training import reverse_kl_objective, train_conditional_flow

__version__ = "0.1.0"

__all__ = [
    "ConditionalRealNVP",
    "ExperimentConfig",
    "FlowConfig",
    "PhysicsConfig",
    "RadiationBatch",
    "ReferenceRadiationModel",
    "TrainingConfig",
    "TransformResult",
    "complex_state_from_real_coordinates",
    "evaluate_model",
    "page_curve",
    "page_expected_entropy",
    "pure_state_fidelity",
    "radiation_density_matrix",
    "reverse_kl_objective",
    "run_experiment",
    "standard_normal_log_prob",
    "train_conditional_flow",
    "von_neumann_entropy",
]
