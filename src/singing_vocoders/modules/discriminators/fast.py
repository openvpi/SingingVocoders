from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from singing_vocoders.components.contracts import (
    DiscriminatorComponent,
    DiscriminatorOutput,
)
from singing_vocoders.modules.discriminators.common import (
    WindowName,
    build_window,
    validate_resolutions,
)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class Transpose(nn.Module):
    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()
        if len(dims) != 2:
            raise ValueError("dims must contain exactly two dimensions")
        self.dims = tuple(dims)

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.dims)


class Permute(nn.Module):
    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()
        self.dims = tuple(dims)

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(*self.dims)


class SoftSignGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, out: Tensor, gate: Tensor) -> Tensor:
        denom_out = out.abs().add(1.0)
        denom_gate = gate.abs().add(1.0)
        out = out / denom_out
        gate = gate / denom_gate
        ctx.save_for_backward(
            out / denom_gate / denom_gate,
            gate / denom_out / denom_out,
        )
        return out * gate

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        out_d_gate, gate_d_out = ctx.saved_tensors
        return grad_output * gate_d_out, grad_output * out_d_gate


class SoftSignGLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return cast(Tensor, SoftSignGLUFunction.apply(out, gate))  # type: ignore[no-untyped-call]


class LYNXNet2Block(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 11, use_dwconv: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            Transpose((1, 2)),
            nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=dim if use_dwconv else 1,
            ),
            Transpose((1, 2)),
            nn.Linear(dim, dim * 2),
            SoftSignGLU(),
            nn.Linear(dim, dim * 2),
            SoftSignGLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: Tensor, norm_x: Tensor | None = None) -> Tensor:
        if norm_x is None:
            norm_x = F.rms_norm(x, (x.size(-1),))
        return x + cast(Tensor, self.net(norm_x))


class FastPD(nn.Module):
    def __init__(
        self,
        period: int,
        init_channel: int = 8,
        strides: Sequence[int] = (4, 4, 4),
        kernel_size: int = 11,
    ) -> None:
        super().__init__()
        self.period = _positive_integer(period, "period")
        init_channel = _positive_integer(init_channel, "init_channel")
        kernel_size = _positive_integer(kernel_size, "kernel_size")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.strides = tuple(_positive_integer(stride, "stride") for stride in strides)
        if not self.strides:
            raise ValueError("strides must not be empty")
        self.hop_length = self.period * prod(self.strides)
        self.pre = nn.Linear(self.strides[0], init_channel * self.strides[0])
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=init_channel * prod(self.strides[: index + 1]),
                    kernel_size=kernel_size,
                    use_dwconv=index > 0,
                )
                for index in range(len(self.strides))
            ]
        )
        self.post = nn.Linear(init_channel * prod(self.strides), 1)

    def forward(
        self,
        x: Tensor,
        infer_last_layer: bool = True,
    ) -> tuple[Tensor, list[Tensor]]:
        feature_maps: list[Tensor] = []
        batch, _, samples = x.shape
        usable_samples = samples // self.hop_length * self.hop_length
        if usable_samples == 0:
            raise ValueError(
                f"audio requires at least {self.hop_length} samples, received {samples}"
            )
        x = x[:, :, :usable_samples].view(batch, -1, self.period)
        x = x.transpose(1, 2).reshape(batch * self.period, -1, self.strides[0])
        x = F.gelu(self.pre(x))
        for index, layer in enumerate(self.residual_layers):
            if index > 0 and self.strides[index] > 1:
                x = x.view(batch * self.period, -1, x.size(2) * self.strides[index])
            norm_x = F.rms_norm(x, (x.size(-1),))
            if index > 0:
                feature_maps.append(norm_x.view(batch, -1))
            if not infer_last_layer and index == len(self.residual_layers) - 1:
                return norm_x, feature_maps
            x = layer(x, norm_x)
        x = F.rms_norm(x, (x.size(-1),))
        return self.post(x).view(batch, -1), feature_maps


class FastMPD(nn.Module):
    def __init__(
        self,
        periods: Sequence[int] | None = None,
        init_channel: int = 8,
        strides: Sequence[int] = (4, 4, 4),
        kernel_size: int = 11,
    ) -> None:
        super().__init__()
        self.periods = tuple((2, 3, 5, 7, 11) if periods is None else periods)
        if not self.periods:
            raise ValueError("periods must not be empty")
        self.discriminators = nn.ModuleList(
            FastPD(period, init_channel, strides, kernel_size) for period in self.periods
        )

    def forward(
        self,
        y: Tensor,
        infer_last_layer: bool = True,
    ) -> tuple[list[Tensor], list[list[Tensor]]]:
        outputs = [
            discriminator(y, infer_last_layer=infer_last_layer)
            for discriminator in self.discriminators
        ]
        return [score for score, _ in outputs], [features for _, features in outputs]


class ResBlock(nn.Module):
    def __init__(self, dim: int, use_dwconv: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            Permute((0, 3, 1, 2)),
            nn.Conv2d(
                dim,
                dim * 2,
                kernel_size=3,
                padding=1,
                groups=dim if use_dwconv else 1,
            ),
            Permute((0, 2, 3, 1)),
            SoftSignGLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: Tensor, norm_x: Tensor | None = None) -> Tensor:
        if norm_x is None:
            norm_x = F.rms_norm(x, (x.size(-1),))
        return x + cast(Tensor, self.net(norm_x))


class FastSpecD(nn.Module):
    window: Tensor
    freq_coords: Tensor

    def __init__(
        self,
        init_channel: int = 8,
        strides: Sequence[int] = (4, 2, 2),
        fft_size: int = 1024,
        shift_size: int = 128,
        win_length: int = 1024,
        window: WindowName = "hann_window",
    ) -> None:
        super().__init__()
        init_channel = _positive_integer(init_channel, "init_channel")
        self.strides = tuple(_positive_integer(stride, "stride") for stride in strides)
        if not self.strides:
            raise ValueError("strides must not be empty")
        self.expansion = prod(self.strides)
        self.fft_size = _positive_integer(fft_size, "fft_size")
        self.shift_size = _positive_integer(shift_size, "shift_size")
        self.win_length = _positive_integer(win_length, "win_length")
        if self.win_length > self.fft_size:
            raise ValueError("win_length must not exceed fft_size")
        self.register_buffer("window", build_window(window, self.win_length))
        self.register_buffer(
            "freq_coords",
            torch.linspace(-1, 1, self.fft_size // 2 + 1).view(-1, 1, 1),
        )
        self.pre = nn.Linear(self.strides[0], init_channel * self.strides[0])
        self.freq = nn.Linear(1, init_channel * self.strides[0])
        self.layers = nn.ModuleList(
            ResBlock(init_channel * prod(self.strides[: index + 1]), use_dwconv=False)
            for index in range(len(self.strides))
        )
        self.post = nn.Linear(init_channel * self.expansion, 1)

    def forward(
        self,
        x: Tensor,
        infer_last_layer: bool = True,
    ) -> tuple[Tensor, list[Tensor]]:
        feature_maps: list[Tensor] = []
        x = torch.stft(
            x.squeeze(1),
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        x = x.abs()
        x.clamp_(min=1e-5).log10_()

        batch, frequencies, frames = x.shape
        usable_frames = frames // self.expansion * self.expansion
        if usable_frames == 0:
            raise ValueError(
                f"spectrogram requires at least {self.expansion} frames, received {frames}"
            )
        x = x[:, :, :usable_frames, None]
        x = x.view(batch, frequencies, -1, self.strides[0])
        x = self.pre(x) + self.freq(self.freq_coords)
        for index, layer in enumerate(self.layers):
            if index > 0:
                x = x.view(batch, frequencies, -1, x.size(3) * self.strides[index])
            norm_x = F.rms_norm(x, (x.size(-1),))
            if index > 0:
                feature_maps.append(norm_x)
            if not infer_last_layer and index == len(self.layers) - 1:
                return norm_x, feature_maps
            x = layer(x, norm_x)
        x = F.rms_norm(x, (x.size(-1),))
        return self.post(x), feature_maps


class FastMRD(nn.Module):
    def __init__(
        self,
        init_channel: int = 8,
        strides: Sequence[int] = (4, 2, 2),
        fft_sizes: Sequence[int] = (1024, 2048, 512),
        hop_sizes: Sequence[int] = (128, 256, 64),
        win_lengths: Sequence[int] = (1024, 2048, 512),
        window: WindowName = "hann_window",
    ) -> None:
        super().__init__()
        resolutions = validate_resolutions(fft_sizes, hop_sizes, win_lengths)
        self.discriminators = nn.ModuleList(
            FastSpecD(
                init_channel,
                strides,
                fft_size,
                hop_size,
                win_length,
                window,
            )
            for fft_size, hop_size, win_length in resolutions
        )

    def forward(
        self,
        y: Tensor,
        infer_last_layer: bool = True,
    ) -> tuple[list[Tensor], list[list[Tensor]]]:
        outputs = [
            discriminator(y, infer_last_layer=infer_last_layer)
            for discriminator in self.discriminators
        ]
        return [score for score, _ in outputs], [features for _, features in outputs]


class FastMultiPeriodDiscriminator(DiscriminatorComponent):
    def __init__(
        self,
        periods: Sequence[int] | None = None,
        init_channel: int = 8,
        strides: Sequence[int] = (4, 4, 4),
        kernel_size: int = 11,
    ) -> None:
        super().__init__()
        self.backbone = FastMPD(periods, init_channel, strides, kernel_size)

    def forward(self, audio: Tensor) -> DiscriminatorOutput:
        scores, feature_maps = self.backbone(audio)
        return DiscriminatorOutput(scores, feature_maps)


class FastMultiResolutionDiscriminator(DiscriminatorComponent):
    def __init__(
        self,
        init_channel: int = 8,
        strides: Sequence[int] = (4, 2, 2),
        fft_sizes: Sequence[int] = (1024, 2048, 512),
        hop_sizes: Sequence[int] = (128, 256, 64),
        win_lengths: Sequence[int] = (1024, 2048, 512),
        window: WindowName = "hann_window",
    ) -> None:
        super().__init__()
        self.backbone = FastMRD(
            init_channel,
            strides,
            fft_sizes,
            hop_sizes,
            win_lengths,
            window,
        )

    def forward(self, audio: Tensor) -> DiscriminatorOutput:
        scores, feature_maps = self.backbone(audio)
        return DiscriminatorOutput(scores, feature_maps)
