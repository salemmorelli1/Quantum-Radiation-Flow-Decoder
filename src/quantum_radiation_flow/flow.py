"""Conditional RealNVP with exact change-of-variables accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import FlowConfig

LOG_2PI = math.log(2.0 * math.pi)


def standard_normal_log_prob(value: Tensor) -> Tensor:
    """Log density of an independent standard Gaussian, reduced by row."""

    return -0.5 * (value.square() + LOG_2PI).sum(dim=-1)


@dataclass(frozen=True)
class TransformResult:
    value: Tensor
    log_abs_det_jacobian: Tensor


class ConditionEncoder(nn.Module):
    """Embed normalized evaporation time and remaining mass."""

    def __init__(self, condition_dim: int, context_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(condition_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
        )

    def forward(self, condition: Tensor) -> Tensor:
        if condition.ndim != 2:
            raise ValueError("condition must have shape [batch, condition_dim]")
        return self.network(condition)


class AffineCoupling(nn.Module):
    """One conditional affine coupling bijection.

    Coordinates selected by ``mask`` remain unchanged and condition the scale
    and translation of the complementary coordinates.  Bounding log-scale by
    tanh prevents singular Jacobians while retaining exact invertibility.
    """

    def __init__(
        self,
        data_dim: int,
        context_dim: int,
        hidden_dim: int,
        mask: Tensor,
        scale_clip: float,
    ) -> None:
        super().__init__()
        if mask.shape != (data_dim,):
            raise ValueError("mask shape must equal [data_dim]")
        self.register_buffer("mask", mask)
        self.scale_clip = float(scale_clip)
        self.conditioner = nn.Sequential(
            nn.Linear(data_dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * data_dim),
        )
        final = self.conditioner[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _coupling_parameters(self, fixed: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        raw_scale, shift = self.conditioner(torch.cat((fixed, context), dim=-1)).chunk(2, dim=-1)
        active = 1.0 - self.mask
        log_scale = self.scale_clip * torch.tanh(raw_scale / self.scale_clip) * active
        shift = shift * active
        return log_scale, shift

    def forward(self, value: Tensor, context: Tensor) -> TransformResult:
        fixed = value * self.mask
        log_scale, shift = self._coupling_parameters(fixed, context)
        transformed = fixed + (1.0 - self.mask) * (value * log_scale.exp() + shift)
        return TransformResult(transformed, log_scale.sum(dim=-1))

    def inverse(self, value: Tensor, context: Tensor) -> TransformResult:
        fixed = value * self.mask
        log_scale, shift = self._coupling_parameters(fixed, context)
        transformed = fixed + (1.0 - self.mask) * ((value - shift) * (-log_scale).exp())
        return TransformResult(transformed, -log_scale.sum(dim=-1))


class Permutation(nn.Module):
    """Fixed coordinate permutation with zero log-Jacobian."""

    def __init__(self, permutation: Tensor) -> None:
        super().__init__()
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", inverse)

    def forward(self, value: Tensor, context: Tensor) -> TransformResult:
        del context
        zeros = value.new_zeros(value.shape[0])
        return TransformResult(value[:, self.permutation], zeros)

    def inverse(self, value: Tensor, context: Tensor) -> TransformResult:
        del context
        zeros = value.new_zeros(value.shape[0])
        return TransformResult(value[:, self.inverse_permutation], zeros)


class ConditionalRealNVP(nn.Module):
    """A time- and mass-conditioned normalizing flow."""

    def __init__(self, config: FlowConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.context_encoder = ConditionEncoder(config.condition_dim, config.context_dim)
        transforms: list[nn.Module] = []
        base_mask = torch.tensor(
            [(index % 2) for index in range(config.data_dim)],
            dtype=torch.get_default_dtype(),
        )
        for layer in range(config.num_layers):
            mask = base_mask if layer % 2 == 0 else 1.0 - base_mask
            transforms.append(
                AffineCoupling(
                    data_dim=config.data_dim,
                    context_dim=config.context_dim,
                    hidden_dim=config.hidden_dim,
                    mask=mask,
                    scale_clip=config.scale_clip,
                )
            )
            permutation = torch.roll(torch.arange(config.data_dim), shifts=layer + 1)
            transforms.append(Permutation(permutation))
        self.transforms = nn.ModuleList(transforms)

    def decode(self, latent: Tensor, condition: Tensor) -> TransformResult:
        """Map base coordinates to radiation telemetry."""

        self._validate_inputs(latent, condition)
        context = self.context_encoder(condition)
        value = latent
        logdet = latent.new_zeros(latent.shape[0])
        for transform in self.transforms:
            result = transform(value, context)
            value = result.value
            logdet = logdet + result.log_abs_det_jacobian
        return TransformResult(value, logdet)

    def encode(self, observation: Tensor, condition: Tensor) -> TransformResult:
        """Invert radiation telemetry to statistical latent coordinates."""

        self._validate_inputs(observation, condition)
        context = self.context_encoder(condition)
        value = observation
        logdet = observation.new_zeros(observation.shape[0])
        for transform in reversed(self.transforms):
            result = transform.inverse(value, context)
            value = result.value
            logdet = logdet + result.log_abs_det_jacobian
        return TransformResult(value, logdet)

    def log_prob(self, observation: Tensor, condition: Tensor) -> Tensor:
        encoded = self.encode(observation, condition)
        return standard_normal_log_prob(encoded.value) + encoded.log_abs_det_jacobian

    def sample(
        self,
        sample_count: int,
        condition: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Generate samples and their exact sample-wise log densities."""

        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        condition = _expand_condition(condition, sample_count)
        reference = next(self.parameters())
        latent = torch.randn(
            sample_count,
            self.config.data_dim,
            generator=generator,
            device=reference.device,
            dtype=reference.dtype,
        )
        decoded = self.decode(latent, condition)
        log_prob = standard_normal_log_prob(latent) - decoded.log_abs_det_jacobian
        return decoded.value, log_prob

    def entropy(
        self,
        sample_count: int,
        condition: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Monte Carlo entropy estimate and its standard error.

        Log density is exact for every draw; the expectation defining entropy is
        estimated unless the transformation admits a closed-form expectation.
        """

        if sample_count < 2:
            raise ValueError("at least two samples are required for an entropy standard error")
        _, log_prob = self.sample(sample_count, condition, generator=generator)
        values = -log_prob
        estimate = values.mean()
        standard_error = values.std(unbiased=True) / math.sqrt(sample_count)
        return estimate, standard_error

    def _validate_inputs(self, value: Tensor, condition: Tensor) -> None:
        if value.ndim != 2 or value.shape[-1] != self.config.data_dim:
            raise ValueError("value must have shape [batch, data_dim]")
        if condition.ndim != 2 or condition.shape[0] != value.shape[0]:
            raise ValueError("condition batch dimension must match value")
        if condition.shape[-1] != self.config.condition_dim:
            raise ValueError("condition has the wrong final dimension")


def _expand_condition(condition: Tensor, sample_count: int) -> Tensor:
    if condition.ndim == 1:
        condition = condition.unsqueeze(0)
    if condition.ndim != 2:
        raise ValueError("condition must be one- or two-dimensional")
    if condition.shape[0] == 1:
        return condition.expand(sample_count, -1)
    if condition.shape[0] != sample_count:
        raise ValueError("condition batch must be one or sample_count")
    return condition
