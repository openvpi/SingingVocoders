from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, prod
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


class AttributeDict(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _required_int(values: Mapping[str, Any], name: str) -> int:
    value: object = values[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _required_bool(values: Mapping[str, Any], name: str) -> bool:
    value: object = values[name]
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _optional_number(values: Mapping[str, Any], name: str) -> int | float | None:
    value: object = values[name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number or null")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer_tuple(values: Mapping[str, Any], name: str) -> tuple[int, ...]:
    value: object = values[name]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} must contain only integers")
        result.append(item)
    return tuple(result)


def _nested_integer_tuple(
    values: Mapping[str, Any],
    name: str,
) -> tuple[tuple[int, ...], ...]:
    value: object = values[name]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of integer sequences")
    result: list[tuple[int, ...]] = []
    for group in value:
        if isinstance(group, (str, bytes)) or not isinstance(group, Sequence):
            raise TypeError(f"{name} must contain only integer sequences")
        parsed_group: list[int] = []
        for item in group:
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(f"{name} must contain only integers")
            parsed_group.append(item)
        result.append(tuple(parsed_group))
    return tuple(result)


def _residual_block_name(values: Mapping[str, Any]) -> Literal["1", "2"]:
    value: object = values["resblock"]
    if value == "1":
        return "1"
    if value == "2":
        return "2"
    raise ValueError(f"resblock must be '1' or '2', got {value!r}")


@dataclass(frozen=True)
class _ModelConfig:
    sampling_rate: int
    num_mels: int
    hop_size: int
    mini_nsf: bool
    noise_sigma: int | float | None
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    upsample_initial_channel: int
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    resblock: Literal["1", "2"]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> _ModelConfig:
        config = cls(
            sampling_rate=_required_int(values, "sampling_rate"),
            num_mels=_required_int(values, "num_mels"),
            hop_size=_required_int(values, "hop_size"),
            mini_nsf=_required_bool(values, "mini_nsf"),
            noise_sigma=_optional_number(values, "noise_sigma"),
            upsample_rates=_integer_tuple(values, "upsample_rates"),
            upsample_kernel_sizes=_integer_tuple(values, "upsample_kernel_sizes"),
            upsample_initial_channel=_required_int(values, "upsample_initial_channel"),
            resblock_kernel_sizes=_integer_tuple(values, "resblock_kernel_sizes"),
            resblock_dilation_sizes=_nested_integer_tuple(values, "resblock_dilation_sizes"),
            resblock=_residual_block_name(values),
        )
        config.validate()
        return config

    def validate(self) -> None:
        scalar_fields = {
            "sampling_rate": self.sampling_rate,
            "num_mels": self.num_mels,
            "hop_size": self.hop_size,
            "upsample_initial_channel": self.upsample_initial_channel,
        }
        for name, value in scalar_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if not self.upsample_rates:
            raise ValueError("upsample_rates must not be empty")
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("upsample_rates and upsample_kernel_sizes must have equal lengths")
        if any(value <= 0 for value in self.upsample_rates):
            raise ValueError("upsample_rates must contain only positive values")
        if any(value <= 0 for value in self.upsample_kernel_sizes):
            raise ValueError("upsample_kernel_sizes must contain only positive values")
        if prod(self.upsample_rates) != self.hop_size:
            raise ValueError("the product of upsample_rates must equal hop_size")
        for rate, kernel_size in zip(
            self.upsample_rates,
            self.upsample_kernel_sizes,
            strict=True,
        ):
            if kernel_size < rate or (kernel_size - rate) % 2 != 0:
                raise ValueError(
                    "each upsample kernel must be at least its rate and differ by an even value"
                )

        stage_divisor = 2 ** len(self.upsample_rates)
        if self.upsample_initial_channel % stage_divisor != 0:
            raise ValueError(
                "upsample_initial_channel must be divisible by two at every upsampling stage"
            )
        if self.mini_nsf and len(self.upsample_rates) < 2:
            raise ValueError("mini_nsf requires at least two upsampling stages")

        if not self.resblock_kernel_sizes:
            raise ValueError("resblock_kernel_sizes must not be empty")
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilation_sizes):
            raise ValueError(
                "resblock_kernel_sizes and resblock_dilation_sizes must have equal lengths"
            )
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.resblock_kernel_sizes):
            raise ValueError("resblock kernels must be positive odd integers")
        required_dilations = 3 if self.resblock == "1" else 2
        for dilations in self.resblock_dilation_sizes:
            if len(dilations) != required_dilations:
                raise ValueError(
                    f"resblock {self.resblock} requires {required_dilations} dilations per kernel"
                )
            if any(dilation <= 0 for dilation in dilations):
                raise ValueError("resblock dilations must be positive")


def init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        with torch.no_grad():
            module.weight.normal_(mean, std)
            if module.bias is not None:
                module.bias.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(nn.Module):
    def __init__(
        self,
        h: AttributeDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, ...] = (1, 3, 5),
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
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
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
                        padding=get_padding(kernel_size),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        output = x
        for first_module, second_module in zip(self.convs1, self.convs2, strict=True):
            first = cast(nn.Conv1d, first_module)
            second = cast(nn.Conv1d, second_module)
            residual = F.leaky_relu(output, LRELU_SLOPE)
            residual = first(residual)
            residual = F.leaky_relu(residual, LRELU_SLOPE, True)
            residual = second(residual)
            output = residual + output
        return output

    def remove_weight_norm(self) -> None:
        for module in self.convs1:
            remove_weight_norm(cast(nn.Conv1d, module))
        for module in self.convs2:
            remove_weight_norm(cast(nn.Conv1d, module))


class ResBlock2(nn.Module):
    def __init__(
        self,
        h: AttributeDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, ...] = (1, 3),
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
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        output = x
        for module in self.convs:
            convolution = cast(nn.Conv1d, module)
            residual = F.leaky_relu(output, LRELU_SLOPE)
            residual = convolution(residual)
            output = residual + output
        return output

    def remove_weight_norm(self) -> None:
        for module in self.convs:
            remove_weight_norm(cast(nn.Conv1d, module))


class SineGen(nn.Module):
    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ) -> None:
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0: Tensor) -> Tensor:
        uv = torch.ones_like(f0)
        return uv * (f0 > self.voiced_threshold)

    def _f02sine(self, f0: Tensor, upp: int) -> Tensor:
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, device=f0.device)
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], -1, 1)
        harmonic_indices = torch.arange(1, self.dim + 1, device=f0.device).reshape(1, 1, -1)
        rad = torch.multiply(rad, harmonic_indices)
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        return torch.sin(2 * np.pi * rad)

    @torch.no_grad()
    def forward(self, f0: Tensor, upp: int) -> Tensor:
        expanded_f0 = f0.unsqueeze(-1)
        sine_waves = self._f02sine(expanded_f0, upp) * self.sine_amp
        uv = (expanded_f0 > self.voiced_threshold).float()
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode="nearest").transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        return sine_waves * uv + noise


class SourceModuleHnNSF(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ) -> None:
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(
            sampling_rate,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshold,
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x: Tensor, upp: int) -> Tensor:
        sine_wavs = self.l_sin_gen.forward(x, upp)
        return cast(Tensor, self.l_tanh(self.l_linear(sine_wavs)))


class NSFHiFiGAN(nn.Module):
    m_source: SourceModuleHnNSF
    noise_convs: nn.ModuleList
    source_conv: nn.Conv1d

    def __init__(self, h: AttributeDict) -> None:
        super().__init__()
        config = _ModelConfig.from_mapping(h)
        self.h = h
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.mini_nsf = config.mini_nsf
        self.noise_sigma = config.noise_sigma

        if config.mini_nsf:
            self.source_sr = config.sampling_rate / prod(config.upsample_rates[2:])
            self.upp = prod(config.upsample_rates[:2])
        else:
            self.source_sr = config.sampling_rate
            self.upp = prod(config.upsample_rates)
            self.m_source = SourceModuleHnNSF(
                sampling_rate=config.sampling_rate,
                harmonic_num=8,
            )
            self.noise_convs = nn.ModuleList()

        self.conv_pre = weight_norm(
            nn.Conv1d(
                config.num_mels,
                config.upsample_initial_channel,
                7,
                1,
                padding=3,
            )
        )

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        resblock: type[ResBlock1] | type[ResBlock2]
        resblock = ResBlock1 if config.resblock == "1" else ResBlock2
        channels = config.upsample_initial_channel
        for stage_index, (rate, kernel_size) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes, strict=True)
        ):
            channels //= 2
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        2 * channels,
                        channels,
                        kernel_size,
                        rate,
                        padding=(kernel_size - rate) // 2,
                    )
                )
            )
            for residual_kernel, dilations in zip(
                config.resblock_kernel_sizes,
                config.resblock_dilation_sizes,
                strict=True,
            ):
                self.resblocks.append(resblock(h, channels, residual_kernel, dilations))
            if not config.mini_nsf:
                if stage_index + 1 < len(config.upsample_rates):
                    stride_f0 = prod(config.upsample_rates[stage_index + 1 :])
                    self.noise_convs.append(
                        nn.Conv1d(
                            1,
                            channels,
                            kernel_size=stride_f0 * 2,
                            stride=stride_f0,
                            padding=stride_f0 // 2,
                        )
                    )
                else:
                    self.noise_convs.append(nn.Conv1d(1, channels, kernel_size=1))
            elif stage_index == 1:
                self.source_conv = nn.Conv1d(1, channels, 1)
                self.source_conv.apply(init_weights)

        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def fastsinegen(self, f0: Tensor) -> Tensor:
        sample_indices = torch.arange(1, self.upp + 1, device=f0.device)
        normalized_frequency = f0.unsqueeze(-1) / self.source_sr
        frequency_delta = F.pad(
            normalized_frequency[:, 1:, :] - normalized_frequency[:, :-1, :],
            (0, 0, 0, 1),
        )
        rad = (
            normalized_frequency * sample_indices
            + 0.5 * frequency_delta * sample_indices * (sample_indices - 1) / self.upp
        )
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], 1, -1)
        return torch.sin(2 * np.pi * rad)

    def forward(self, x: Tensor, f0: Tensor) -> Tensor:
        if self.mini_nsf:
            harmonic_source = self.fastsinegen(f0)
        else:
            harmonic_source = self.m_source.forward(f0, self.upp).transpose(1, 2)

        output = self.conv_pre(x)
        if self.noise_sigma is not None and self.noise_sigma > 0:
            output += self.noise_sigma * torch.randn_like(output)

        for stage_index in range(self.num_upsamples):
            output = F.leaky_relu(output, LRELU_SLOPE, True)
            upsampling_layer = cast(nn.ConvTranspose1d, self.ups[stage_index])
            output = upsampling_layer(output)
            if not self.mini_nsf:
                source_convolution = cast(nn.Conv1d, self.noise_convs[stage_index])
                output = output + source_convolution(harmonic_source)
            elif stage_index == 1:
                output = output + self.source_conv(harmonic_source)

            block_offset = stage_index * self.num_kernels
            first_block = cast(ResBlock1 | ResBlock2, self.resblocks[block_offset])
            residual_sum = first_block.forward(output)
            for block_index in range(1, self.num_kernels):
                block = cast(ResBlock1 | ResBlock2, self.resblocks[block_offset + block_index])
                residual_sum += block.forward(output)
            output = residual_sum / self.num_kernels

        output = F.leaky_relu(output, inplace=True)
        output = self.conv_post(output)
        return torch.tanh(output)

    def remove_weight_norm(self) -> None:
        for module in self.ups:
            remove_weight_norm(cast(nn.ConvTranspose1d, module))
        for module in self.resblocks:
            block = cast(ResBlock1 | ResBlock2, module)
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
