"""Optimization routines for conditional maximum likelihood and alignment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import TrainingConfig
from .flow import ConditionalRealNVP, standard_normal_log_prob
from .target import ReferenceRadiationModel


@dataclass(frozen=True)
class TrainingRecord:
    step: int
    loss: float
    negative_log_likelihood: float
    latent_alignment: float
    gradient_norm: float


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {name}")


def train_conditional_flow(
    model: ConditionalRealNVP,
    target: ReferenceRadiationModel,
    config: TrainingConfig,
) -> list[TrainingRecord]:
    """Fit q_theta(x|t,M) with paired latent-coordinate regularization.

    Maximum likelihood minimizes KL(p||q).  The alignment term is deliberately
    separated: density fitting alone cannot identify a physically privileged
    Gaussian latent basis because rotations preserve the base distribution.
    """

    config.validate()
    dtype = resolve_dtype(config.dtype)
    device = torch.device(config.device)
    model.to(device=device, dtype=dtype)
    target.to(device=device, dtype=dtype)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device=device).manual_seed(config.seed)
    history: list[TrainingRecord] = []
    report_every = max(1, config.steps // 100)

    for step in range(1, config.steps + 1):
        batch = target.sample(
            config.batch_size,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        encoded = model.encode(batch.observation, batch.condition)
        log_q = standard_normal_log_prob(encoded.value) + encoded.log_abs_det_jacobian
        nll = -log_q.mean()
        alignment = torch.nn.functional.mse_loss(encoded.value, batch.accessible_latent)
        loss = nll + config.alignment_weight * alignment

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite optimization state")
        optimizer.step()

        if step == 1 or step == config.steps or step % report_every == 0:
            history.append(
                TrainingRecord(
                    step=step,
                    loss=float(loss.detach()),
                    negative_log_likelihood=float(nll.detach()),
                    latent_alignment=float(alignment.detach()),
                    gradient_norm=float(gradient_norm.detach()),
                )
            )
    return history


def reverse_kl_objective(
    model: ConditionalRealNVP,
    target: ReferenceRadiationModel,
    condition: Tensor,
    sample_count: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Monte Carlo variational free energy E_q[log q - log p]."""

    if condition.ndim == 1:
        condition = condition.unsqueeze(0)
    if condition.ndim != 2 or condition.shape[-1] != model.config.condition_dim:
        raise ValueError("condition must have shape [1 or sample_count, condition_dim]")
    if condition.shape[0] not in {1, sample_count}:
        raise ValueError("condition batch must be one or sample_count")
    observation, log_q = model.sample(sample_count, condition, generator=generator)
    expanded = condition.expand(sample_count, -1) if condition.shape[0] == 1 else condition
    log_p = target.log_prob(observation, expanded)
    return (log_q - log_p).mean()
