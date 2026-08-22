from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from torch import Tensor, nn


def mel_upsampling_stages(factor: int | None) -> int:
    normalized_factor = 1 if factor is None else factor
    if (
        not isinstance(normalized_factor, int)
        or isinstance(normalized_factor, bool)
        or normalized_factor <= 0
        or normalized_factor & (normalized_factor - 1)
    ):
        raise ValueError("mel_upsample_factor must be a positive power of two")
    return normalized_factor.bit_length() - 1


def validate_upsampling_configuration(
    *,
    hop_size: int,
    mel_upsample_factor: int | None,
    waveform_upsample_rates: Sequence[int],
) -> int:
    stages = mel_upsampling_stages(mel_upsample_factor)
    factor = 1 << stages
    if hop_size <= 0:
        raise ValueError("hop_size must be positive")
    if not waveform_upsample_rates or any(
        not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0
        for rate in waveform_upsample_rates
    ):
        raise ValueError("waveform_upsample_rates must contain positive integers")

    waveform_factor = 1
    for rate in waveform_upsample_rates:
        waveform_factor *= rate
    if factor * waveform_factor != hop_size:
        raise ValueError(
            "mel_upsample_factor * product(waveform_upsample_rates) must equal hop_size "
            f"({factor} * {waveform_factor} != {hop_size})"
        )
    return factor


class GatedLinearUnit(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        values, gates = inputs.chunk(2, dim=self.dim)
        return values * gates.sigmoid()


class MelUpsampler(nn.Module):
    """Double the temporal resolution without mixing mel-frequency bins."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(1, 8, kernel_size=1)
        self.upsample = nn.ConvTranspose2d(
            4,
            8,
            kernel_size=(3, 32),
            stride=(1, 2),
            padding=(1, 15),
        )
        self.gate = GatedLinearUnit(1)
        self.residual = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.output_projection = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, mel: Tensor) -> Tensor:
        hidden = self.gate(self.input_projection(mel.unsqueeze(1)))
        hidden = self.gate(self.upsample(hidden))
        hidden = self.gate(self.residual(hidden)) + hidden
        return cast(Tensor, self.gate(self.output_projection(hidden)).squeeze(1))


def build_mel_upsampler(factor: int | None) -> nn.Module:
    return nn.Sequential(*(MelUpsampler() for _ in range(mel_upsampling_stages(factor))))


class DownsampleBlock(nn.Module):
    def __init__(self, rate: int, input_channels: int, output_channels: int) -> None:
        super().__init__()
        if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
            raise ValueError("downsampling rate must be positive")
        self.downsample = nn.Conv1d(
            input_channels,
            output_channels * 2,
            kernel_size=rate * 2,
            stride=rate,
            padding=(rate + 1) // 2,
        )
        self.gate = GatedLinearUnit(1)
        self.projection = nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(
            Tensor,
            self.activation(self.projection(self.gate(self.downsample(inputs)))),
        )


class SourceDownsampler(nn.Module):
    def __init__(self, channels: int, rates: Sequence[int]) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("source feature channels must be positive")
        if not rates or any(
            not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0 for rate in rates
        ):
            raise ValueError("source downsampling rates must contain positive integers")

        reversed_rates = list(reversed(rates))
        self.input_projection = nn.Conv1d(1, channels, kernel_size=1)
        blocks: list[nn.Module] = []
        projections: list[nn.Module] = []
        for index, rate in enumerate(reversed_rates[:-1]):
            input_channels = 1 if index == 0 else channels * index
            output_channels = channels * (index + 1)
            blocks.append(DownsampleBlock(rate, input_channels, output_channels))
            projections.append(nn.Conv1d(output_channels, channels, kernel_size=1))
        self.blocks = nn.ModuleList(blocks)
        self.projections = nn.ModuleList(projections)

    def forward(self, source: Tensor) -> list[Tensor]:
        features = [self.input_projection(source)]
        for block, projection in zip(self.blocks, self.projections, strict=True):
            source = block(source)
            features.append(projection(source))
        features.reverse()
        return features
