"""A transparent toy scrambling channel with an exact reference density."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import FlowConfig, PhysicsConfig
from .flow import ConditionalRealNVP, standard_normal_log_prob
from .quantum import mass_schedule


@dataclass(frozen=True)
class RadiationBatch:
    observation: Tensor
    condition: Tensor
    secret_coordinates: Tensor
    accessible_latent: Tensor
    target_log_prob: Tensor
    visibility: Tensor


class ReferenceRadiationModel(nn.Module):
    """Generate paired secret/radiation data through a frozen invertible map.

    The accessible latent is y_t = v_t s + sqrt(1-v_t^2) epsilon.  Because both
    s and epsilon are standard Gaussian, y_t remains standard Gaussian, giving
    the frozen flow an exact normalized target density.  Correlation with the
    secret grows through Page time, so inverse-coordinate fidelity is testable.
    """

    def __init__(self, physics: PhysicsConfig, *, seed: int = 20260824) -> None:
        super().__init__()
        physics.validate()
        self.physics = physics
        target_config = FlowConfig(
            data_dim=physics.data_dim,
            condition_dim=2,
            context_dim=24,
            hidden_dim=physics.target_hidden_dim,
            num_layers=physics.target_layers,
            scale_clip=physics.target_scale_clip,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.scrambler = ConditionalRealNVP(target_config)
            self._randomize_target()
        self.scrambler.requires_grad_(False)

    def _randomize_target(self) -> None:
        """Replace identity initialization with a stable nontrivial bijection."""

        with torch.no_grad():
            for name, parameter in self.scrambler.named_parameters():
                if "conditioner.4.weight" in name:
                    parameter.normal_(mean=0.0, std=0.035)
                elif "conditioner.4.bias" in name:
                    parameter.normal_(mean=0.0, std=0.025)

    def visibility(self, time: Tensor) -> Tensor:
        return torch.sigmoid((time - self.physics.page_time) / self.physics.visibility_width)

    def condition(self, time: Tensor) -> Tensor:
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        mass = mass_schedule(
            time,
            initial_mass=self.physics.initial_mass,
            mass_floor=self.physics.mass_floor,
        )
        return torch.cat((time, mass), dim=-1)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        *,
        time: Tensor | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> RadiationBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if time is None:
            time = torch.rand(batch_size, 1, generator=generator, device=device, dtype=dtype)
        else:
            time = time.to(device=device, dtype=dtype)
            if time.ndim == 0:
                time = time.reshape(1, 1).expand(batch_size, 1)
            elif time.ndim == 1:
                time = time.reshape(-1, 1)
            if time.shape[0] == 1:
                time = time.expand(batch_size, 1)
            if time.shape != (batch_size, 1):
                raise ValueError("time must broadcast to [batch_size, 1]")
        condition = self.condition(time)
        secret = torch.randn(
            batch_size,
            self.physics.data_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        nuisance = torch.randn(
            batch_size,
            self.physics.data_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        visibility = self.visibility(time)
        inaccessible_weight = (1.0 - visibility.square()).clamp_min(0.0).sqrt()
        accessible = visibility * secret + inaccessible_weight * nuisance
        self.scrambler.to(device=device, dtype=dtype)
        decoded = self.scrambler.decode(accessible, condition)
        target_log_prob = standard_normal_log_prob(accessible) - decoded.log_abs_det_jacobian
        return RadiationBatch(
            observation=decoded.value,
            condition=condition,
            secret_coordinates=secret,
            accessible_latent=accessible,
            target_log_prob=target_log_prob,
            visibility=visibility.squeeze(-1),
        )

    def log_prob(self, observation: Tensor, condition: Tensor) -> Tensor:
        encoded = self.scrambler.encode(observation, condition)
        return standard_normal_log_prob(encoded.value) + encoded.log_abs_det_jacobian
