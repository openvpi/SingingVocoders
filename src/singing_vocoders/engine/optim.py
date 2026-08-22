from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from singing_vocoders.config.schema import (
    AdamWConfig,
    MuonAdamWConfig,
    OptimizerSlotConfig,
)
from singing_vocoders.engine.muon import MuonAdamW


@dataclass(slots=True)
class BuiltOptimizer:
    optimizer: Optimizer
    scheduler: LRScheduler | None


def build_optimizer(slot: OptimizerSlotConfig, module: nn.Module) -> BuiltOptimizer:
    config = slot.optimizer
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("optimizer target has no trainable parameters")
    if isinstance(config, AdamWConfig):
        optimizer: Optimizer = torch.optim.AdamW(
            parameters,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    elif isinstance(config, MuonAdamWConfig):
        optimizer = MuonAdamW(
            module,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=config.nesterov,
            ns_steps=config.ns_steps,
            adamw_lr=config.adamw_lr,
            adamw_betas=config.adamw_betas,
            adamw_eps=config.adamw_eps,
            adamw_weight_decay=config.adamw_weight_decay,
        )
    else:
        raise TypeError(f"unsupported optimizer config: {type(config).__name__}")
    scheduler: LRScheduler | None = None
    if slot.scheduler is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=slot.scheduler.gamma)
    return BuiltOptimizer(optimizer, scheduler)
