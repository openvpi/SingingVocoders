from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class Batch:
    audio: Tensor
    mel: Tensor
    f0: Tensor
    uv: Tensor
    pe: Tensor
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def to(self, device: torch.device | str, non_blocking: bool = False) -> Batch:
        return Batch(
            audio=self.audio.to(device, non_blocking=non_blocking),
            mel=self.mel.to(device, non_blocking=non_blocking),
            f0=self.f0.to(device, non_blocking=non_blocking),
            uv=self.uv.to(device, non_blocking=non_blocking),
            pe=self.pe.to(device, non_blocking=non_blocking),
            metadata=self.metadata,
        )


@dataclass(slots=True)
class GeneratorOutput:
    audio: Tensor
    aux: dict[str, Tensor] = field(default_factory=dict)


@dataclass(slots=True)
class DiscriminatorOutput:
    scores: list[Tensor]
    feature_maps: list[list[Tensor]]


class GeneratorComponent(nn.Module):
    def forward(self, batch: Batch) -> GeneratorOutput:
        raise NotImplementedError


class DiscriminatorComponent(nn.Module):
    def forward(self, audio: Tensor) -> DiscriminatorOutput:
        raise NotImplementedError
