from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


class AttrDict(dict[str, Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if "Conv" in module.__class__.__name__:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, Tensor):
            raise TypeError(f"{type(module).__name__} does not expose a tensor weight")
        with torch.no_grad():
            weight.normal_(mean=mean, std=std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def _validate_architecture(h: AttrDict) -> None:
    num_mels = h["num_mels"]
    initial_channels = h["upsample_initial_channel"]
    rates = h["upsample_rates"]
    kernels = h["upsample_kernel_sizes"]
    residual_kernels = h["resblock_kernel_sizes"]
    residual_dilations = h["resblock_dilation_sizes"]
    resblock = h["resblock"]

    if isinstance(num_mels, bool) or not isinstance(num_mels, int) or num_mels <= 0:
        raise ValueError("num_mels must be a positive integer")
    if (
        isinstance(initial_channels, bool)
        or not isinstance(initial_channels, int)
        or initial_channels <= 0
    ):
        raise ValueError("upsample_initial_channel must be a positive integer")
    if not isinstance(rates, Sequence) or isinstance(rates, (str, bytes)) or not rates:
        raise ValueError("upsample_rates must be a non-empty sequence")
    if not isinstance(kernels, Sequence) or isinstance(kernels, (str, bytes)):
        raise ValueError("upsample_kernel_sizes must be a sequence")
    if len(rates) != len(kernels):
        raise ValueError("upsample rates and kernel sizes must have equal lengths")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in rates):
        raise ValueError("upsample rates must be positive integers")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in kernels
    ):
        raise ValueError("upsample kernel sizes must be positive integers")
    if any(
        kernel < rate or (kernel - rate) % 2 for rate, kernel in zip(rates, kernels, strict=True)
    ):
        raise ValueError(
            "each upsample kernel must be at least its rate and differ from it by an even value"
        )
    if initial_channels % (2 ** len(rates)):
        raise ValueError("upsample_initial_channel must be divisible by two at every stage")
    if (
        not isinstance(residual_kernels, Sequence)
        or isinstance(residual_kernels, (str, bytes))
        or not residual_kernels
    ):
        raise ValueError("resblock_kernel_sizes must be a non-empty sequence")
    if not isinstance(residual_dilations, Sequence) or isinstance(residual_dilations, (str, bytes)):
        raise ValueError("resblock_dilation_sizes must be a sequence")
    if len(residual_kernels) != len(residual_dilations):
        raise ValueError("residual kernel and dilation groups must have equal lengths")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 2 == 0
        for value in residual_kernels
    ):
        raise ValueError("residual kernel sizes must be positive odd integers")
    if resblock not in {"1", "2"}:
        raise ValueError("resblock must be '1' or '2'")
    required_dilations = 3 if resblock == "1" else 2
    for dilations in residual_dilations:
        if not isinstance(dilations, Sequence) or isinstance(dilations, (str, bytes)):
            raise ValueError("each residual dilation group must be a sequence")
        if len(dilations) != required_dilations:
            raise ValueError(
                f"resblock {resblock} requires {required_dilations} dilations per residual block"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dilations
        ):
            raise ValueError("residual dilations must be positive integers")


class ResBlock1(nn.Module):
    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: Sequence[int] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[index],
                        padding=get_padding(kernel_size, dilation[index]),
                    )
                )
                for index in range(3)
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(3)
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            residual = F.leaky_relu(x, LRELU_SLOPE)
            residual = conv1(residual)
            residual = F.leaky_relu(residual, LRELU_SLOPE)
            residual = conv2(residual)
            x = residual + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class ResBlock2(nn.Module):
    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: Sequence[int] = (1, 3),
    ) -> None:
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[index],
                        padding=get_padding(kernel_size, dilation[index]),
                    )
                )
                for index in range(2)
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            residual = F.leaky_relu(x, LRELU_SLOPE)
            residual = conv(residual)
            x = residual + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs:
            remove_weight_norm(layer)


class Encoder(nn.Module):
    def __init__(self, h: AttrDict) -> None:
        super().__init__()
        _validate_architecture(h)
        self.h = h
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.out_channels = h["num_mels"]
        self.num_downsamples = len(h["upsample_rates"])
        self.conv_pre = weight_norm(
            nn.Conv1d(
                1,
                h["upsample_initial_channel"] // (2 ** len(h["upsample_rates"])),
                7,
                1,
                padding=3,
            )
        )
        resblock = ResBlock1 if h["resblock"] == "1" else ResBlock2
        self.ups = nn.ModuleList()
        for index, (rate, kernel) in enumerate(
            zip(
                reversed(h["upsample_rates"]),
                reversed(h["upsample_kernel_sizes"]),
                strict=True,
            )
        ):
            self.ups.append(
                weight_norm(
                    nn.Conv1d(
                        h["upsample_initial_channel"] // (2 ** (len(h["upsample_rates"]) - index)),
                        h["upsample_initial_channel"]
                        // (2 ** (len(h["upsample_rates"]) - index - 1)),
                        kernel,
                        rate,
                        padding=(kernel - rate + 1) // 2,
                    )
                )
            )
        self.resblocks = nn.ModuleList()
        for stage in range(len(self.ups), 0, -1):
            channels = h["upsample_initial_channel"] // (2 ** (stage - 1))
            for kernel, dilation in zip(
                h["resblock_kernel_sizes"], h["resblock_dilation_sizes"], strict=True
            ):
                self.resblocks.append(resblock(h, channels, kernel, dilation))

        self.conv_post = weight_norm(nn.Conv1d(channels, 2 * h["num_mels"], 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = math.prod(h["upsample_rates"])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.conv_pre(x)
        for stage in range(self.num_downsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[stage](x)
            combined: Tensor | None = None
            for kernel in range(self.num_kernels):
                residual = self.resblocks[stage * self.num_kernels + kernel](x)
                combined = residual if combined is None else combined + residual
            if combined is None:
                raise RuntimeError("encoder residual stage contains no blocks")
            x = combined / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        mean, log_scale = torch.split(x, self.out_channels, dim=1)
        latent = mean + torch.randn_like(mean) * torch.exp(log_scale)
        return latent, mean, log_scale


class Generators(nn.Module):
    def __init__(self, h: AttrDict) -> None:
        super().__init__()
        _validate_architecture(h)
        self.h = h
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.conv_pre = weight_norm(
            nn.Conv1d(h["num_mels"], h["upsample_initial_channel"], 7, 1, padding=3)
        )
        resblock = ResBlock1 if h["resblock"] == "1" else ResBlock2
        self.ups = nn.ModuleList()
        for index, (rate, kernel) in enumerate(
            zip(h["upsample_rates"], h["upsample_kernel_sizes"], strict=True)
        ):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        h["upsample_initial_channel"] // (2**index),
                        h["upsample_initial_channel"] // (2 ** (index + 1)),
                        kernel,
                        rate,
                        padding=(kernel - rate + 1) // 2,
                    )
                )
            )
        self.resblocks = nn.ModuleList()
        for stage in range(len(self.ups)):
            channels = h["upsample_initial_channel"] // (2 ** (stage + 1))
            for kernel, dilation in zip(
                h["resblock_kernel_sizes"], h["resblock_dilation_sizes"], strict=True
            ):
                self.resblocks.append(resblock(h, channels, kernel, dilation))

        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = math.prod(h["upsample_rates"])

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_pre(x)
        for stage in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[stage](x)
            combined: Tensor | None = None
            for kernel in range(self.num_kernels):
                residual = self.resblocks[stage * self.num_kernels + kernel](x)
                combined = residual if combined is None else combined + residual
            if combined is None:
                raise RuntimeError("generator residual stage contains no blocks")
            x = combined / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()  # type: ignore[operator]
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class HiFivae(nn.Module):
    def __init__(self, h: AttrDict) -> None:
        super().__init__()
        _validate_architecture(h)
        self.Generator = Generators(h)
        self.Encoder = Encoder(h)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        latent, mean, log_scale = self.Encoder(x)
        audio = self.Generator(latent)
        return audio, latent, mean, log_scale
