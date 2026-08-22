from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from singing_vocoders.components.contracts import (
    DiscriminatorComponent,
    DiscriminatorOutput,
)
from singing_vocoders.modules.discriminators.common import (
    WindowName,
    build_window,
    validate_resolutions,
)


class TransposeDimensions(nn.Module):
    def __init__(self, first: int, second: int) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.transpose(self.first, self.second)


class PermuteDimensions(nn.Module):
    def __init__(self, dimensions: Sequence[int]) -> None:
        super().__init__()
        self.dimensions = tuple(dimensions)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.permute(*self.dimensions)


class SoftSignGatedLinearUnit(nn.Module):
    def __init__(self, dimension: int = -1) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, inputs: Tensor) -> Tensor:
        values, gates = inputs.chunk(2, dim=self.dimension)
        return values / values.abs().add(1.0) * (gates / gates.abs().add(1.0))


class LynxResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 11, depthwise: bool = True) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            TransposeDimensions(1, 2),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels if depthwise else 1,
            ),
            TransposeDimensions(1, 2),
            nn.Linear(channels, channels * 2),
            SoftSignGatedLinearUnit(),
            nn.Linear(channels, channels * 2),
            SoftSignGatedLinearUnit(),
            nn.Linear(channels, channels),
        )

    def forward(self, inputs: Tensor, normalized_inputs: Tensor | None = None) -> Tensor:
        if normalized_inputs is None:
            normalized_inputs = F.rms_norm(inputs, (inputs.size(-1),))
        return inputs + cast(Tensor, self.layers(normalized_inputs))


class FastPeriodDiscriminator(nn.Module):
    def __init__(
        self,
        period: int,
        initial_channels: int = 8,
        strides: Sequence[int] = (4, 4, 4),
        kernel_size: int = 11,
    ) -> None:
        super().__init__()
        if period <= 0:
            raise ValueError("period must be positive")
        if initial_channels <= 0:
            raise ValueError("initial_channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.period = period
        self.strides = tuple(strides)
        if not self.strides or any(stride < 1 for stride in self.strides):
            raise ValueError("strides must contain positive integers")
        self.hop_length = period * prod(self.strides)
        self.input_projection = nn.Linear(self.strides[0], initial_channels * self.strides[0])
        self.residual_blocks = nn.ModuleList(
            [
                LynxResidualBlock(
                    channels=initial_channels * prod(self.strides[: index + 1]),
                    kernel_size=kernel_size,
                    depthwise=index > 0,
                )
                for index in range(len(self.strides))
            ]
        )
        self.output_projection = nn.Linear(initial_channels * prod(self.strides), 1)

    def forward(self, audio: Tensor) -> tuple[Tensor, list[Tensor]]:
        batch_size, _, samples = audio.shape
        usable_samples = samples // self.hop_length * self.hop_length
        if usable_samples == 0:
            raise ValueError(
                f"audio requires at least {self.hop_length} samples, received {samples}"
            )
        hidden = audio[..., :usable_samples].reshape(batch_size, -1, self.period)
        hidden = hidden.transpose(1, 2).reshape(batch_size * self.period, -1, self.strides[0])
        hidden = F.gelu(self.input_projection(hidden))

        feature_maps: list[Tensor] = []
        for index, block in enumerate(self.residual_blocks):
            if index > 0 and self.strides[index] > 1:
                hidden = hidden.reshape(
                    batch_size * self.period,
                    -1,
                    hidden.size(2) * self.strides[index],
                )
            normalized = F.rms_norm(hidden, (hidden.size(-1),))
            if index > 0:
                feature_maps.append(normalized.reshape(batch_size, -1))
            hidden = block(hidden, normalized)

        hidden = F.rms_norm(hidden, (hidden.size(-1),))
        scores = self.output_projection(hidden).reshape(batch_size, -1)
        return scores, feature_maps


class FastMultiPeriodDiscriminator(DiscriminatorComponent):
    def __init__(
        self,
        periods: Sequence[int] = (2, 3, 5, 7, 11),
        initial_channels: int = 8,
        strides: Sequence[int] = (4, 4, 4),
        kernel_size: int = 11,
    ) -> None:
        super().__init__()
        period_values = tuple(periods)
        if not period_values or any(period <= 0 for period in period_values):
            raise ValueError("periods must contain positive integers")
        self.discriminators = nn.ModuleList(
            [
                FastPeriodDiscriminator(period, initial_channels, strides, kernel_size)
                for period in period_values
            ]
        )

    def forward(self, audio: Tensor) -> DiscriminatorOutput:
        outputs = [discriminator(audio) for discriminator in self.discriminators]
        return DiscriminatorOutput(
            [scores for scores, _ in outputs],
            [features for _, features in outputs],
        )


class SpectrogramResidualBlock(nn.Module):
    def __init__(self, channels: int, depthwise: bool = False) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            PermuteDimensions((0, 3, 1, 2)),
            nn.Conv2d(
                channels,
                channels * 2,
                kernel_size=3,
                padding=1,
                groups=channels if depthwise else 1,
            ),
            PermuteDimensions((0, 2, 3, 1)),
            SoftSignGatedLinearUnit(),
            nn.Linear(channels, channels),
        )

    def forward(self, inputs: Tensor, normalized_inputs: Tensor | None = None) -> Tensor:
        if normalized_inputs is None:
            normalized_inputs = F.rms_norm(inputs, (inputs.size(-1),))
        return inputs + cast(Tensor, self.layers(normalized_inputs))


class FastSpectrogramDiscriminator(nn.Module):
    def __init__(
        self,
        initial_channels: int = 8,
        strides: Sequence[int] = (4, 2, 2),
        fft_size: int = 1024,
        hop_size: int = 128,
        window_length: int = 1024,
        window: WindowName = "hann_window",
    ) -> None:
        super().__init__()
        if initial_channels <= 0:
            raise ValueError("initial_channels must be positive")
        self.strides = tuple(strides)
        if not self.strides or any(stride < 1 for stride in self.strides):
            raise ValueError("strides must contain positive integers")
        self.expansion = prod(self.strides)
        if min(fft_size, hop_size, window_length) <= 0:
            raise ValueError("spectrogram sizes must be positive")
        if window_length > fft_size:
            raise ValueError("window_length must not exceed fft_size")
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_length = window_length
        self.register_buffer("window", build_window(window, window_length))
        self.register_buffer(
            "frequency_coordinates",
            torch.linspace(-1, 1, fft_size // 2 + 1).view(-1, 1, 1),
        )
        self.input_projection = nn.Linear(self.strides[0], initial_channels * self.strides[0])
        self.frequency_projection = nn.Linear(1, initial_channels * self.strides[0])
        self.residual_blocks = nn.ModuleList(
            [
                SpectrogramResidualBlock(initial_channels * prod(self.strides[: index + 1]))
                for index in range(len(self.strides))
            ]
        )
        self.output_projection = nn.Linear(initial_channels * self.expansion, 1)

    def forward(self, audio: Tensor) -> tuple[Tensor, list[Tensor]]:
        magnitude = torch.stft(
            audio.squeeze(1),
            self.fft_size,
            self.hop_size,
            self.window_length,
            cast(Tensor, self.window),
            return_complex=True,
        ).abs()
        magnitude = magnitude.clamp_min_(1e-5).log10_()

        batch_size, frequencies, frames = magnitude.shape
        usable_frames = frames // self.expansion * self.expansion
        if usable_frames == 0:
            raise ValueError(
                f"spectrogram requires at least {self.expansion} frames, received {frames}"
            )
        hidden = magnitude[:, :, :usable_frames, None].reshape(
            batch_size, frequencies, -1, self.strides[0]
        )
        hidden = self.input_projection(hidden) + self.frequency_projection(
            self.frequency_coordinates
        )

        feature_maps: list[Tensor] = []
        for index, block in enumerate(self.residual_blocks):
            if index > 0:
                hidden = hidden.reshape(
                    batch_size,
                    frequencies,
                    -1,
                    hidden.size(3) * self.strides[index],
                )
            normalized = F.rms_norm(hidden, (hidden.size(-1),))
            if index > 0:
                feature_maps.append(normalized)
            hidden = block(hidden, normalized)

        hidden = F.rms_norm(hidden, (hidden.size(-1),))
        return self.output_projection(hidden), feature_maps


class FastMultiResolutionDiscriminator(DiscriminatorComponent):
    def __init__(
        self,
        initial_channels: int = 8,
        strides: Sequence[int] = (4, 2, 2),
        fft_sizes: Sequence[int] = (1024, 2048, 512),
        hop_sizes: Sequence[int] = (128, 256, 64),
        window_lengths: Sequence[int] = (1024, 2048, 512),
        window: WindowName = "hann_window",
    ) -> None:
        super().__init__()
        resolutions = validate_resolutions(fft_sizes, hop_sizes, window_lengths)
        self.discriminators = nn.ModuleList(
            [
                FastSpectrogramDiscriminator(
                    initial_channels,
                    strides,
                    fft_size,
                    hop_size,
                    window_length,
                    window,
                )
                for fft_size, hop_size, window_length in resolutions
            ]
        )

    def forward(self, audio: Tensor) -> DiscriminatorOutput:
        outputs = [discriminator(audio) for discriminator in self.discriminators]
        return DiscriminatorOutput(
            [scores for scores, _ in outputs],
            [features for _, features in outputs],
        )
