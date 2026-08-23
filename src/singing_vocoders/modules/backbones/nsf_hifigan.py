from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

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


def init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        with torch.no_grad():
            module.weight.normal_(mean, std)
            if module.bias is not None:
                module.bias.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):
    def __init__(
        self,
        h: AttributeDict,
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
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            residual = F.leaky_relu(x, LRELU_SLOPE)
            residual = conv1(residual)
            residual = F.leaky_relu(residual, LRELU_SLOPE, True)
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
        h: AttributeDict,
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
        for convolution in self.convs:
            residual = F.leaky_relu(x, LRELU_SLOPE)
            residual = convolution(residual)
            x = residual + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs:
            remove_weight_norm(layer)


class SineGen(nn.Module):
    def __init__(
        self,
        sample_rate: int,
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
        self.sampling_rate = sample_rate
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
        rad = torch.multiply(
            rad,
            torch.arange(1, self.dim + 1, device=f0.device).reshape(1, 1, -1),
        )
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        return torch.sin(2 * np.pi * rad)

    @torch.no_grad()
    def forward(self, f0: Tensor, upp: int) -> Tensor:
        f0 = f0.unsqueeze(-1)
        sine_waves = self._f02sine(f0, upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).float()
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
        sine_wavs = self.l_sin_gen(x, upp)
        return cast(Tensor, self.l_tanh(self.l_linear(sine_wavs)))


class NSFHiFiGAN(nn.Module):
    m_source: SourceModuleHnNSF
    noise_convs: nn.ModuleList
    source_conv: nn.Conv1d

    def __init__(self, h: AttributeDict) -> None:
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.mini_nsf = h.mini_nsf
        self.noise_sigma = h.noise_sigma

        if h.mini_nsf:
            self.source_sr = h.sampling_rate / int(np.prod(h.upsample_rates[2:]))
            self.upp = int(np.prod(h.upsample_rates[:2]))
        else:
            self.source_sr = h.sampling_rate
            self.upp = int(np.prod(h.upsample_rates))
            self.m_source = SourceModuleHnNSF(
                sampling_rate=h.sampling_rate,
                harmonic_num=8,
            )
            self.noise_convs = nn.ModuleList()

        self.conv_pre = weight_norm(
            nn.Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2
        channels = h.upsample_initial_channel
        for index, (rate, kernel_size) in enumerate(
            zip(h.upsample_rates, h.upsample_kernel_sizes, strict=True)
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
            for residual_kernel, dilation in zip(
                h.resblock_kernel_sizes,
                h.resblock_dilation_sizes,
                strict=True,
            ):
                self.resblocks.append(resblock(h, channels, residual_kernel, dilation))
            if not h.mini_nsf:
                if index + 1 < len(h.upsample_rates):
                    stride_f0 = int(np.prod(h.upsample_rates[index + 1 :]))
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
            elif index == 1:
                self.source_conv = nn.Conv1d(1, channels, 1)
                self.source_conv.apply(init_weights)

        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def fastsinegen(self, f0: Tensor) -> Tensor:
        sample_index = torch.arange(1, self.upp + 1, device=f0.device)
        normalized_f0 = f0.unsqueeze(-1) / self.source_sr
        f0_delta = F.pad(
            normalized_f0[:, 1:, :] - normalized_f0[:, :-1, :],
            (0, 0, 0, 1),
        )
        rad = (
            normalized_f0 * sample_index
            + 0.5 * f0_delta * sample_index * (sample_index - 1) / self.upp
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
            harmonic_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        if self.noise_sigma is not None and self.noise_sigma > 0:
            x += self.noise_sigma * torch.randn_like(x)
        for index in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE, True)
            x = self.ups[index](x)
            if not self.mini_nsf:
                x = x + self.noise_convs[index](harmonic_source)
            elif index == 1:
                x = x + self.source_conv(harmonic_source)
            residual_sum: Tensor | None = None
            for kernel_index in range(self.num_kernels):
                residual = self.resblocks[index * self.num_kernels + kernel_index](x)
                residual_sum = residual if residual_sum is None else residual_sum + residual
            if residual_sum is None:
                raise RuntimeError("NSF HiFiGAN requires at least one residual kernel")
            x = residual_sum / self.num_kernels
        x = F.leaky_relu(x, inplace=True)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            cast(ResBlock1 | ResBlock2, layer).remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
