from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from math import prod
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm as remove_legacy_weight_norm
from torch.nn.utils import weight_norm

from singing_vocoders.modules.lvc.univnet import LVCBlock

LRELU_SLOPE = 0.1


def _required_int(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _required_float(values: Mapping[str, Any], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    return float(value)


def _required_int_sequence(values: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = values.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{key} must be a sequence of integers")
    result = tuple(value)
    if not result or any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise TypeError(f"{key} must contain integers")
    if any(item <= 0 for item in result):
        raise ValueError(f"{key} must contain positive integers")
    return result


def _upmel_stages(upmel: int | None) -> int:
    if upmel is None:
        return 0
    if isinstance(upmel, bool) or not isinstance(upmel, int) or upmel <= 0:
        raise ValueError("upmel must be a positive integer or null")
    stages = upmel // 2
    if 2**stages != upmel:
        raise ValueError("upmel must be one of 1, 2, or 4")
    return stages


@dataclass(frozen=True)
class UnivNetArchitecture:
    cond_in_channels: int
    out_channels: int
    cg_channels: int
    cond_channels: int
    hop_size: int
    upsample_rates: tuple[int, ...]
    num_lvc_blocks: int
    lvc_kernels: int
    lvc_hidden_channels: int
    lvc_conv_size: int
    dropout: float
    upmel: int | None

    @property
    def upmel_factor(self) -> int:
        return int(2 ** _upmel_stages(self.upmel))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> UnivNetArchitecture:
        model_args_value = config.get("model_args")
        if not isinstance(model_args_value, Mapping):
            raise TypeError("model_args must be a mapping")
        model_args: Mapping[str, Any] = model_args_value
        upsample_rates = _required_int_sequence(model_args, "upsample_rates")
        hop_size = _required_int(config, "hop_size")
        upmel_value = model_args.get("upmel")
        if upmel_value is not None and (
            isinstance(upmel_value, bool) or not isinstance(upmel_value, int)
        ):
            raise TypeError("upmel must be an integer or null")
        upmel = cast(int | None, upmel_value)
        upmel_factor = 2 ** _upmel_stages(upmel)
        if upmel_factor * prod(upsample_rates) != hop_size:
            raise ValueError(
                "upmel * product(upsample_rates) must equal hop_size "
                f"({upmel_factor} * {prod(upsample_rates)} != {hop_size})"
            )
        dropout = _required_float(model_args, "dropout")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")
        lvc_kernels = _required_int(model_args, "lvc_kernels")
        lvc_conv_size = _required_int(model_args, "lvc_conv_size")
        if lvc_kernels % 2 == 0 or lvc_conv_size % 2 == 0:
            raise ValueError("lvc_kernels and lvc_conv_size must be odd")
        return cls(
            cond_in_channels=_required_int(model_args, "cond_in_channels"),
            out_channels=_required_int(model_args, "out_channels"),
            cg_channels=_required_int(model_args, "cg_channels"),
            cond_channels=_required_int(config, "audio_num_mel_bins"),
            hop_size=hop_size,
            upsample_rates=upsample_rates,
            num_lvc_blocks=_required_int(model_args, "num_lvc_blocks"),
            lvc_kernels=lvc_kernels,
            lvc_hidden_channels=_required_int(model_args, "lvc_hidden_channels"),
            lvc_conv_size=lvc_conv_size,
            dropout=dropout,
            upmel=upmel,
        )


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        output, gate = x.chunk(2, dim=self.dim)
        return output * gate.sigmoid()


class Upspamper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, kernel_size=1)
        self.UP = nn.ConvTranspose2d(4, 8, (3, 32), stride=(1, 2), padding=(1, 15))
        self.Glu = GLU(1)
        self.c2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.Glu(self.c1(x.unsqueeze(1)))
        x = self.Glu(self.UP(x))
        x = self.Glu(self.c2(x)) + x
        x = self.Glu(self.c3(x))
        return x.squeeze(1)


class UnivNetBackbone(nn.Module):
    architecture: UnivNetArchitecture

    def __init__(
        self,
        architecture: UnivNetArchitecture,
        *,
        source_channels: int | None = None,
    ) -> None:
        super().__init__()
        self._initialize_backbone(architecture, source_channels=source_channels)

    def _initialize_backbone(
        self,
        architecture: UnivNetArchitecture,
        *,
        source_channels: int | None,
    ) -> None:
        self.architecture = architecture
        self.upblocke = (
            nn.Identity()
            if architecture.upmel is None
            else nn.Sequential(*(Upspamper() for _ in range(_upmel_stages(architecture.upmel))))
        )
        self.in_channels = architecture.cond_in_channels
        self.out_channels = architecture.out_channels
        self.cond_channels = architecture.cond_channels
        self.lvc_block_nums = len(architecture.upsample_rates)
        self.first_conv = nn.Conv1d(
            architecture.cond_in_channels,
            architecture.cg_channels,
            kernel_size=7,
            padding=3,
            dilation=1,
            bias=True,
        )
        self.lvc_blocks = nn.ModuleList()
        cond_hop_length = 1
        for upsample_ratio in architecture.upsample_rates:
            cond_hop_length *= upsample_ratio
            self.lvc_blocks.append(
                LVCBlock(
                    in_channels=architecture.cg_channels,
                    cond_channels=architecture.cond_channels,
                    upsample_ratio=upsample_ratio,
                    conv_layers=architecture.num_lvc_blocks,
                    conv_kernel_size=architecture.lvc_kernels,
                    cond_hop_length=cond_hop_length,
                    kpnet_hidden_channels=architecture.lvc_hidden_channels,
                    kpnet_conv_size=architecture.lvc_conv_size,
                    kpnet_dropout=architecture.dropout,
                    source_channels=source_channels,
                )
            )
        self.last_conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    architecture.cg_channels,
                    architecture.out_channels,
                    kernel_size=7,
                    padding=3,
                    dilation=1,
                    bias=True,
                )
            ]
        )

    def forward_backbone(
        self,
        x: Tensor,
        c: Tensor,
        spec: Sequence[Tensor] | None = None,
    ) -> Tensor:
        x = self.first_conv(x)
        c = self.upblocke(c)
        if spec is not None and len(spec) != self.lvc_block_nums:
            raise ValueError("one source tensor is required for each LVC block")
        for index in range(self.lvc_block_nums):
            source = None if spec is None else spec[index]
            x = self.lvc_blocks[index](x, c, source)
        for layer in self.last_conv_layers:
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = layer(x)
        return torch.tanh(x)

    def remove_weight_norm(self) -> None:
        def remove(module: nn.Module) -> None:
            with suppress(ValueError):
                remove_legacy_weight_norm(module)

        self.apply(remove)

    def apply_weight_norm(self) -> None:
        def apply(module: nn.Module) -> None:
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                weight_norm(module)

        self.apply(apply)


class UnivNet(UnivNetBackbone):
    def __init__(self, config: Mapping[str, Any], use_weight_norm: bool = True) -> None:
        architecture = UnivNetArchitecture.from_config(config)
        super().__init__(architecture)
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        return self.forward_backbone(x, c)
