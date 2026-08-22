from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from singing_vocoders.modules.backbones.univnet import (
    UnivNetArchitecture,
    UnivNetBackbone,
)
from singing_vocoders.modules.lvc.conditioning import SourceDownsampler


def _required_positive_int(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


class SineGenerator(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        harmonic_count: int = 0,
        sine_amplitude: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if harmonic_count < 0:
            raise ValueError("harmonic_count must be non-negative")
        self.sampling_rate = float(sampling_rate)
        self.harmonic_count = harmonic_count
        self.source_channels = harmonic_count + 1
        self.sine_amplitude = sine_amplitude
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    def _frequency_to_sine(self, frequencies: Tensor, upsample_factor: int) -> Tensor:
        input_dtype = frequencies.dtype
        phase_increments = torch.remainder(frequencies / self.sampling_rate, 1.0)
        initial_phase = torch.rand(
            (1, self.source_channels),
            device=frequencies.device,
            dtype=input_dtype,
        )
        initial_phase[:, 0] = 0
        phase_increments[:, 0, :] += initial_phase

        accumulated_phase = torch.cumsum(phase_increments.double(), dim=1)
        accumulated_phase = functional.interpolate(
            accumulated_phase.transpose(1, 2),
            scale_factor=upsample_factor,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        accumulated_phase = torch.remainder(accumulated_phase * upsample_factor, 1.0)
        phase_wraps = (accumulated_phase[:, 1:] - accumulated_phase[:, :-1] < 0).double()
        phase_wraps = torch.cat(
            (
                torch.zeros(
                    frequencies.shape[0],
                    1,
                    self.source_channels,
                    device=frequencies.device,
                    dtype=torch.float64,
                ),
                phase_wraps,
            ),
            dim=1,
        )
        upsampled_increments = functional.interpolate(
            phase_increments.double().transpose(1, 2),
            scale_factor=upsample_factor,
            mode="nearest",
        ).transpose(1, 2)
        phase = torch.cumsum(upsampled_increments + phase_wraps, dim=1)
        return torch.sin(phase * math.tau).to(dtype=input_dtype)

    @torch.no_grad()
    def forward(self, f0: Tensor, upsample_factor: int) -> Tensor:
        if upsample_factor <= 0:
            raise ValueError("upsample_factor must be positive")
        f0 = f0.unsqueeze(-1)
        harmonic_numbers = torch.arange(
            1,
            self.source_channels + 1,
            device=f0.device,
            dtype=f0.dtype,
        ).reshape(1, 1, -1)
        sine_waves = self._frequency_to_sine(f0 * harmonic_numbers, upsample_factor)
        sine_waves = sine_waves * self.sine_amplitude

        voiced = (f0 > self.voiced_threshold).to(dtype=f0.dtype)
        voiced = functional.interpolate(
            voiced.transpose(1, 2),
            scale_factor=upsample_factor,
            mode="nearest",
        ).transpose(1, 2)
        noise_amplitude = voiced * self.noise_std + (1.0 - voiced) * self.sine_amplitude / 3.0
        return sine_waves * voiced + noise_amplitude * torch.randn_like(sine_waves)


class HarmonicSource(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        harmonic_count: int = 0,
        sine_amplitude: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.sine_generator = SineGenerator(
            sampling_rate,
            harmonic_count,
            sine_amplitude,
            noise_std,
            voiced_threshold,
        )
        self.projection = nn.Linear(harmonic_count + 1, 1)
        self.activation = nn.Tanh()

    def forward(self, f0: Tensor, upsample_factor: int) -> Tensor:
        return cast(
            Tensor,
            self.activation(self.projection(self.sine_generator(f0, upsample_factor))),
        )


class NSFUnivNet(UnivNetBackbone):
    def __init__(self, config: Mapping[str, Any], use_weight_norm: bool = True) -> None:
        architecture = UnivNetArchitecture.from_config(config)
        super().__init__(architecture, source_channels=architecture.hidden_channels)
        self.harmonic_source = HarmonicSource(
            sampling_rate=_required_positive_int(config, "audio_sample_rate"),
            harmonic_count=8,
        )
        self.source_downsampler = SourceDownsampler(
            architecture.hidden_channels,
            architecture.waveform_upsample_rates,
        )
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, noise: Tensor, mel: Tensor, f0: Tensor) -> tuple[Tensor, Tensor]:
        harmonic_source = self.harmonic_source(f0, self.architecture.hop_size).transpose(1, 2)
        source_features = self.source_downsampler(harmonic_source)
        audio = self.forward_backbone(noise, mel, source_features)
        return audio, harmonic_source
