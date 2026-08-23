# Adapted from yxlllc/pc-ddsp under the MIT License.

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from singing_vocoders.modules.ddsp.core import (
    frequency_filter,
    remove_above_fmax,
    upsample,
)
from singing_vocoders.modules.ddsp.mel2control import Mel2Control


class Sins(nn.Module):
    sampling_rate: Tensor
    block_size: Tensor
    win_length: Tensor
    window: Tensor

    def __init__(
        self,
        sampling_rate: int,
        block_size: int,
        win_length: int,
        n_harmonics: int,
        n_mag_noise: int,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        split_map = {
            "harmonic_phase": win_length // 2 + 1,
            "amplitudes": n_harmonics,
            "noise_magnitude": n_mag_noise,
        }
        self.mel2ctrl = Mel2Control(n_mels, split_map)

    def forward(
        self,
        mel_frames: Tensor,
        f0_frames: Tensor,
        initial_phase: Tensor | None = None,
        infer: bool = True,
        max_upsample_dim: int = 32,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        block_size = int(self.block_size.item())
        win_length = int(self.win_length.item())
        f0 = upsample(f0_frames, block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, dim=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, dim=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / (2 * np.pi)
        x = x - torch.round(x)
        x = x.to(f0)

        phase = 2 * np.pi * x
        phase_frames = phase[:, ::block_size, :]
        controls = self.mel2ctrl(mel_frames, phase_frames)

        source_allpass = torch.exp(1.0j * np.pi * controls["harmonic_phase"])
        source_allpass = torch.cat((source_allpass, source_allpass[:, -1:, :]), 1)
        amplitude_frames = torch.exp(controls["amplitudes"]) / 128
        noise_magnitude = torch.exp(controls["noise_magnitude"]) / 128

        amplitude_frames = remove_above_fmax(
            amplitude_frames,
            f0_frames,
            float(self.sampling_rate.item()) / 2,
            level_start=1,
        )
        n_harmonic = amplitude_frames.shape[-1]
        harmonic_levels = torch.arange(1, n_harmonic + 1).to(phase)
        sinusoids = torch.zeros_like(phase[..., 0])
        for index in range((n_harmonic - 1) // max_upsample_dim + 1):
            start = index * max_upsample_dim
            end = (index + 1) * max_upsample_dim
            phases = phase * harmonic_levels[start:end]
            amplitudes = upsample(amplitude_frames[:, :, start:end], block_size)
            sinusoids += (torch.sin(phases) * amplitudes).sum(-1)

        harmonic_spectrum = torch.stft(
            sinusoids,
            n_fft=win_length,
            win_length=win_length,
            hop_length=block_size,
            window=self.window,
            center=True,
            return_complex=True,
        )
        harmonic_spectrum = harmonic_spectrum * source_allpass.permute(0, 2, 1)
        harmonic = torch.istft(
            harmonic_spectrum,
            n_fft=win_length,
            win_length=win_length,
            hop_length=block_size,
            window=self.window,
            center=True,
        )

        noise = torch.rand_like(harmonic).to(noise_magnitude) * 2 - 1
        noise = frequency_filter(
            noise,
            torch.complex(noise_magnitude, torch.zeros_like(noise_magnitude)),
            hann_window=True,
        )
        return harmonic + noise, phase, (harmonic, noise)


class CombSub(nn.Module):
    sampling_rate: Tensor
    block_size: Tensor
    win_length: Tensor
    window: Tensor

    def __init__(
        self,
        sampling_rate: int,
        block_size: int,
        win_length: int,
        n_mag_harmonic: int,
        n_mag_noise: int,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        split_map = {
            "harmonic_phase": win_length // 2 + 1,
            "harmonic_magnitude": n_mag_harmonic,
            "noise_magnitude": n_mag_noise,
        }
        self.mel2ctrl = Mel2Control(n_mels, split_map)

    def forward(
        self,
        mel_frames: Tensor,
        f0_frames: Tensor,
        initial_phase: Tensor | None = None,
        infer: bool = True,
        **_kwargs: object,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        block_size = int(self.block_size.item())
        win_length = int(self.win_length.item())
        f0 = upsample(f0_frames, block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, dim=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, dim=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / (2 * np.pi)
        x = x - torch.round(x)
        x = x.to(f0)

        phase_frames = 2 * np.pi * x[:, ::block_size, :]
        controls = self.mel2ctrl(mel_frames, phase_frames)

        source_allpass = torch.exp(1.0j * np.pi * controls["harmonic_phase"])
        source_allpass = torch.cat((source_allpass, source_allpass[:, -1:, :]), 1)
        source_magnitude = torch.exp(controls["harmonic_magnitude"])
        noise_magnitude = torch.exp(controls["noise_magnitude"]) / 128

        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3)).squeeze(-1)
        harmonic = frequency_filter(
            combtooth,
            torch.complex(source_magnitude, torch.zeros_like(source_magnitude)),
            hann_window=True,
            half_width_frames=1.5 * self.sampling_rate / (f0_frames + 1e-3),
        )
        harmonic_spectrum = torch.stft(
            harmonic,
            n_fft=win_length,
            win_length=win_length,
            hop_length=block_size,
            window=self.window,
            center=True,
            return_complex=True,
        )
        harmonic_spectrum = harmonic_spectrum * source_allpass.permute(0, 2, 1)
        harmonic = torch.istft(
            harmonic_spectrum,
            n_fft=win_length,
            win_length=win_length,
            hop_length=block_size,
            window=self.window,
            center=True,
        )

        noise = torch.rand_like(harmonic).to(noise_magnitude) * 2 - 1
        noise = frequency_filter(
            noise,
            torch.complex(noise_magnitude, torch.zeros_like(noise_magnitude)),
            hann_window=True,
        )
        return harmonic + noise, phase_frames, (harmonic, noise)


def build_vocoder(
    vocoder_type: Literal["Sins", "CombSub"],
    *,
    sampling_rate: int,
    block_size: int,
    win_length: int,
    n_mag_noise: int,
    n_mels: int,
    n_harmonics: int | None = None,
    n_mag_harmonic: int | None = None,
) -> Sins | CombSub:
    if vocoder_type == "Sins":
        if n_harmonics is None:
            raise ValueError("Sins requires n_harmonics")
        return Sins(
            sampling_rate,
            block_size,
            win_length,
            n_harmonics,
            n_mag_noise,
            n_mels,
        )
    if vocoder_type == "CombSub":
        if n_mag_harmonic is None:
            raise ValueError("CombSub requires n_mag_harmonic")
        return CombSub(
            sampling_rate,
            block_size,
            win_length,
            n_mag_harmonic,
            n_mag_noise,
            n_mels,
        )
    raise ValueError(f"unsupported DDSP type: {vocoder_type}")
