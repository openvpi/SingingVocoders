from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from singing_vocoders.config.schema import (
    AdamWConfig,
    MuonConfig,
    OptimizerRouteConfig,
    OptimizerSlotConfig,
)
from singing_vocoders.engine.muon import Muon


@dataclass(slots=True)
class BuiltOptimizer:
    route: OptimizerRouteConfig
    optimizer: Optimizer
    scheduler: LRScheduler | None


def _muon_parameters(module: nn.Module) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for child in module.modules():
        for name, parameter in child.named_parameters(recurse=False):
            if not parameter.requires_grad or name == "weight_g":
                continue
            if not isinstance(child, nn.Embedding) and parameter.ndim >= 2:
                parameters.append(parameter)
    return parameters


def _select_parameters(
    module: nn.Module,
    selector: str,
    muon_parameter_ids: set[int],
) -> list[nn.Parameter]:
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if selector == "all":
        return trainable
    if selector == "muon":
        return [parameter for parameter in trainable if id(parameter) in muon_parameter_ids]
    if selector == "remaining":
        return [parameter for parameter in trainable if id(parameter) not in muon_parameter_ids]
    raise ValueError(f"unsupported optimizer parameter route: {selector}")


def _build_optimizer(route: OptimizerRouteConfig, parameters: list[nn.Parameter]) -> BuiltOptimizer:
    config = route.optimizer
    if not parameters:
        raise ValueError(f"optimizer parameter route is empty: {route.parameters}")
    if isinstance(config, AdamWConfig):
        optimizer: Optimizer = torch.optim.AdamW(
            parameters,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    elif isinstance(config, MuonConfig):
        optimizer = Muon(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=config.nesterov,
            ns_steps=config.ns_steps,
        )
    else:
        raise TypeError(f"unsupported optimizer config: {type(config).__name__}")
    scheduler: LRScheduler | None = None
    if route.scheduler is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=route.scheduler.gamma)
    return BuiltOptimizer(route, optimizer, scheduler)


def build_optimizers(slot: OptimizerSlotConfig, module: nn.Module) -> list[BuiltOptimizer]:
    muon_parameter_ids = {id(parameter) for parameter in _muon_parameters(module)}
    built = [
        _build_optimizer(
            route,
            _select_parameters(module, route.parameters, muon_parameter_ids),
        )
        for route in slot.optimizers
    ]
    routed_ids = [
        id(parameter)
        for item in built
        for group in item.optimizer.param_groups
        for parameter in group["params"]
    ]
    expected_ids = [id(parameter) for parameter in module.parameters() if parameter.requires_grad]
    if len(routed_ids) != len(set(routed_ids)) or set(routed_ids) != set(expected_ids):
        raise ValueError(
            "optimizer parameter routes must be disjoint and cover all trainable parameters"
        )
    return built
