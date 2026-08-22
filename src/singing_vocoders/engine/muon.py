from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import repeat
from typing import Any, overload

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

_COEFFICIENTS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
_COEFFICIENTS = [(a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _COEFFICIENTS[:-1]] + [
    _COEFFICIENTS[-1]
]


def _orthogonalize(gradient: Tensor, steps: int, use_bfloat16: bool) -> Tensor:
    if gradient.ndim != 3:
        raise ValueError("Muon expects a batch of matrices")
    value = gradient.to(dtype=torch.bfloat16 if use_bfloat16 else torch.float32)
    value = value / torch.linalg.vector_norm(value, dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    coefficients = _COEFFICIENTS[:steps] + list(
        repeat(_COEFFICIENTS[-1], max(0, steps - len(_COEFFICIENTS)))
    )
    if value.size(-2) < value.size(-1):
        for a, b, c in coefficients:
            product = torch.bmm(value, value.mT)
            product = torch.baddbmm(product, product, product, beta=b, alpha=c)
            value = torch.baddbmm(value, product, value, beta=a, alpha=1)
    else:
        for a, b, c in coefficients:
            product = torch.bmm(value.mT, value)
            product = torch.baddbmm(product, product, product, beta=b, alpha=c)
            value = torch.baddbmm(value, value, product, beta=a, alpha=1)
    return value


class Muon(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        ns_steps: int,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
            },
        )

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            loss = None if closure is None else closure()
        for group in self.param_groups:
            shape_groups: dict[tuple[Any, ...], dict[str, list[Tensor]]] = {}
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                buffer = state.setdefault("momentum_buffer", torch.zeros_like(gradient))
                key = (tuple(parameter.shape), parameter.device, parameter.dtype)
                values = shape_groups.setdefault(key, {"params": [], "grads": [], "buffers": []})
                values["params"].append(parameter)
                values["grads"].append(gradient)
                values["buffers"].append(buffer)
            for values in shape_groups.values():
                parameters = values["params"]
                gradients = values["grads"]
                buffers = values["buffers"]
                momentum = group["momentum"]
                torch._foreach_lerp_(buffers, gradients, 1 - momentum)
                if group["nesterov"]:
                    torch._foreach_lerp_(gradients, buffers, momentum)
                    stacked = torch.stack(gradients)
                else:
                    stacked = torch.stack(buffers)
                original_shape = stacked.shape
                if stacked.ndim >= 4:
                    stacked = stacked.view(stacked.size(0), stacked.size(1), -1)
                use_bfloat16 = (
                    stacked.is_cuda and torch.cuda.get_device_capability(stacked.device)[0] >= 8
                )
                update = _orthogonalize(stacked, group["ns_steps"], use_bfloat16)
                if group["weight_decay"]:
                    torch._foreach_mul_(parameters, 1 - group["lr"] * group["weight_decay"])
                torch._foreach_add_(
                    parameters,
                    update.view(original_shape).unbind(0),
                    alpha=-group["lr"] * max(update[0].shape) ** 0.5,
                )
        return loss


def partition_muon_parameters(
    module: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    muon_ids: set[int] = set()
    muon_parameters: list[nn.Parameter] = []
    for child in module.modules():
        for name, parameter in child.named_parameters(recurse=False):
            if (
                parameter.requires_grad
                and parameter.ndim >= 2
                and name != "weight_g"
                and not isinstance(child, nn.Embedding)
                and id(parameter) not in muon_ids
            ):
                muon_ids.add(id(parameter))
                muon_parameters.append(parameter)
    adamw_parameters = [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad and id(parameter) not in muon_ids
    ]
    return muon_parameters, adamw_parameters


class MuonAdamW(Optimizer):
    def __init__(
        self,
        module: nn.Module,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        ns_steps: int,
        adamw_lr: float,
        adamw_betas: tuple[float, float],
        adamw_eps: float,
        adamw_weight_decay: float,
    ) -> None:
        muon_parameters, adamw_parameters = partition_muon_parameters(module)
        groups: list[dict[str, Any]] = []
        if muon_parameters:
            groups.append(
                {
                    "params": muon_parameters,
                    "role": "muon",
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )
        if adamw_parameters:
            groups.append(
                {
                    "params": adamw_parameters,
                    "role": "adamw",
                    "lr": adamw_lr,
                    "weight_decay": adamw_weight_decay,
                }
            )
        if not groups:
            raise ValueError("optimizer target has no trainable parameters")
        super().__init__(groups, {})
        self.muon = (
            Muon(
                muon_parameters,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
            )
            if muon_parameters
            else None
        )
        self.adamw = (
            torch.optim.AdamW(
                adamw_parameters,
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=adamw_weight_decay,
            )
            if adamw_parameters
            else None
        )

    @property
    def muon_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.muon is None:
            return ()
        return tuple(self.muon.param_groups[0]["params"])

    @property
    def adamw_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.adamw is None:
            return ()
        return tuple(self.adamw.param_groups[0]["params"])

    def _synchronize_groups(self) -> None:
        for group in self.param_groups:
            if group["role"] == "muon" and self.muon is not None:
                self.muon.param_groups[0]["lr"] = group["lr"]
                self.muon.param_groups[0]["weight_decay"] = group["weight_decay"]
            elif group["role"] == "adamw" and self.adamw is not None:
                self.adamw.param_groups[0]["lr"] = group["lr"]
                self.adamw.param_groups[0]["weight_decay"] = group["weight_decay"]

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "outer": super().state_dict(),
            "muon": None if self.muon is None else self.muon.state_dict(),
            "adamw": None if self.adamw is None else self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("format_version") != 1:
            raise ValueError("unsupported MuonAdamW state format")
        super().load_state_dict(state_dict["outer"])
        if (state_dict["muon"] is None) != (self.muon is None):
            raise ValueError("Muon parameter partition does not match")
        if (state_dict["adamw"] is None) != (self.adamw is None):
            raise ValueError("AdamW parameter partition does not match")
        if self.muon is not None:
            self.muon.load_state_dict(state_dict["muon"])
        if self.adamw is not None:
            self.adamw.load_state_dict(state_dict["adamw"])
        self._synchronize_groups()

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            loss = None if closure is None else closure()
        self._synchronize_groups()
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()
        return loss
