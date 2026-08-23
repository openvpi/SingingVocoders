from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import repeat
from typing import Any, overload

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer

COEFFICIENTS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
COEFFICIENTS = [(a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in COEFFICIENTS[:-1]] + [
    COEFFICIENTS[-1]
]


def get_bf16_support_map() -> dict[torch.device, bool]:
    if not torch.cuda.is_available():
        return {}
    return {
        device: torch.cuda.get_device_capability(device)[0] >= 8
        for device in (torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count()))
    }


def zeropower_via_newtonschulz5(gradient: Tensor, steps: int, use_bf16: bool) -> Tensor:
    if gradient.ndim != 3:
        raise ValueError("Muon expects a batch of matrices")
    value = gradient.to(dtype=torch.bfloat16 if use_bf16 else torch.float32)
    value = F.normalize(value, p=2.0, dim=(-2, -1), eps=1e-7)  # type: ignore[arg-type]
    coefficients = COEFFICIENTS[:steps] + list(repeat(COEFFICIENTS[-1], steps - len(COEFFICIENTS)))
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
        self.bf16_support_map = get_bf16_support_map()

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        with torch.enable_grad():
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
                use_bfloat16 = self.bf16_support_map.get(stacked.device, False)
                update = zeropower_via_newtonschulz5(
                    stacked,
                    steps=group["ns_steps"],
                    use_bf16=use_bfloat16,
                )
                if group["weight_decay"]:
                    torch._foreach_mul_(parameters, 1 - group["lr"] * group["weight_decay"])
                torch._foreach_add_(
                    parameters,
                    update.view(original_shape).unbind(0),
                    alpha=-group["lr"] * max(update[0].shape) ** 0.5,
                )
        return loss
