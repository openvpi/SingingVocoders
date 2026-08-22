# Adapted from yxlllc/pc-ddsp under the MIT License.

from __future__ import annotations

from math import pi
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from singing_vocoders.modules.ddsp.core import (
    frequency_filter,
    mean_filter,
    remove_above_fmax,
    upsample,
)
from singing_vocoders.modules.ddsp.mel2control import Mel2Control


def _validate_frames(mel_frames: Tensor, f0_frames: Tensor) -> None:
    if mel_frames.ndim != 3:
        raise ValueError("mel_frames must have shape [batch, frames, mel_bins]")
    if f0_frames.ndim != 3 or f0_frames.size(-1) != 1:
        raise ValueError("f0_frames must have shape [batch, frames, 1]")
    if mel_frames.shape[:2] != f0_frames.shape[:2]:
        raise ValueError("mel and F0 frame dimensions must match")


def _validate_architecture(
    sampling_rate: int,
    block_size: int,
    win_length: int,
    n_mels: int,
) -> None:
    if min(sampling_rate, block_size, win_length, n_mels) <= 0:
        raise ValueError("DDSP architecture sizes must be positive")
    if win_length < block_size:
        raise ValueError("win_length must not be smaller than block_size")


def _apply_phase_filter(
    audio: Tensor,
    phase_filter: Tensor,
    *,
    window: Tensor,
    win_length: int,
    block_size: int,
) -> Tensor:
    spectrum = torch.stft(
        audio,
        n_fft=win_length,
        win_length=win_length,
        hop_length=block_size,
        window=window,
        center=True,
        return_complex=True,
    )
    if spectrum.shape[-1] != phase_filter.shape[1]:
        raise ValueError("phase-filter frames do not align with the generated audio")
    filtered = spectrum * phase_filter.transpose(1, 2)
    return torch.istft(
        filtered,
        n_fft=win_length,
        win_length=win_length,
        hop_length=block_size,
        window=window,
        center=True,
        length=audio.shape[-1],
    )


class Sins(nn.Module):
    window: Tensor

    def __init__(
        self,
        sampling_rate: int,
        block_size: int,
        win_length: int,
        use_mean_filter: bool,
        n_harmonics: int,
        n_mag_noise: int,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        _validate_architecture(sampling_rate, block_size, win_length, n_mels)
        if n_harmonics <= 0 or n_mag_noise <= 0:
            raise ValueError("DDSP control sizes must be positive")
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        self.mel2ctrl = Mel2Control(
            n_mels,
            block_size,
            {
                "amplitudes": n_harmonics,
                "harmonic_phase": win_length // 2 + 1,
                "noise_magnitude": n_mag_noise,
                "noise_phase": n_mag_noise,
            },
        )
        self.mean_kernel_size = win_length // block_size if use_mean_filter else 1

    def _synthesize_noise(self, noise_exciter: Tensor, controls: dict[str, Tensor]) -> Tensor:
        noise_filter = (
            torch.exp(controls["noise_magnitude"] + 1j * pi * controls["noise_phase"]) / 128
        )
        return frequency_filter(noise_exciter, noise_filter)

    def fast_phase_gen(self, f0_frames: Tensor) -> Tensor:
        offsets = torch.arange(
            self.block_size,
            device=f0_frames.device,
            dtype=f0_frames.dtype,
        )
        normalized_f0 = f0_frames / self.sampling_rate
        slope = F.pad(
            normalized_f0[:, 1:, :] - normalized_f0[:, :-1, :],
            (0, 0, 0, 1),
        )
        phase = (
            normalized_f0 * (offsets + 1) + 0.5 * slope * offsets * (offsets + 1) / self.block_size
        )
        boundary = torch.fmod(phase[..., -1:].float() + 0.5, 1.0) - 0.5
        accumulated = boundary.cumsum(dim=1).fmod(1.0).to(f0_frames)
        phase += F.pad(accumulated[:, :-1, :], (0, 0, 1, 0))
        return 2 * pi * phase.reshape(f0_frames.shape[0], -1, 1)

    def forward(
        self,
        mel_frames: Tensor,
        f0_frames: Tensor,
        output_f0_frames: Tensor | None = None,
        infer: bool = True,
        max_upsample_dim: int = 32,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        _validate_frames(mel_frames, f0_frames)
        if max_upsample_dim < 1:
            raise ValueError("max_upsample_dim must be positive")

        control_phase = self.fast_phase_gen(f0_frames)
        sinusoid = control_phase.sin().squeeze(-1)
        sinusoid_frames = sinusoid.unfold(1, self.block_size, self.block_size)
        noise_exciter = torch.randn_like(sinusoid)
        noise_frames = noise_exciter.unfold(1, self.block_size, self.block_size)
        controls = self.mel2ctrl(mel_frames, sinusoid_frames, noise_frames)
        if self.mean_kernel_size > 1:
            controls["amplitudes"] = mean_filter(controls["amplitudes"], self.mean_kernel_size)
            controls["harmonic_phase"] = mean_filter(
                controls["harmonic_phase"], self.mean_kernel_size
            )

        phase_filter = torch.exp(1j * pi * controls["harmonic_phase"])
        phase_filter = torch.cat((phase_filter, phase_filter[:, -1:, :]), dim=1)
        amplitude_frames = torch.exp(controls["amplitudes"]) / 128
        synthesis_f0 = output_f0_frames if infer and output_f0_frames is not None else f0_frames
        _validate_frames(mel_frames, synthesis_f0)
        phase = self.fast_phase_gen(synthesis_f0)
        amplitude_frames = remove_above_fmax(
            amplitude_frames,
            synthesis_f0,
            self.sampling_rate / 2,
        )
        harmonic_indices = torch.arange(
            1,
            amplitude_frames.shape[-1] + 1,
            device=phase.device,
            dtype=phase.dtype,
        )
        sinusoids = torch.zeros_like(phase[..., 0])
        for start in range(0, amplitude_frames.shape[-1], max_upsample_dim):
            stop = start + max_upsample_dim
            phases = phase * harmonic_indices[start:stop]
            amplitudes = upsample(amplitude_frames[:, :, start:stop], self.block_size)
            sinusoids += (phases.sin() * amplitudes).sum(dim=-1)

        harmonic = _apply_phase_filter(
            sinusoids,
            phase_filter,
            window=self.window,
            win_length=self.win_length,
            block_size=self.block_size,
        )
        noise = self._synthesize_noise(noise_exciter, controls)
        return harmonic + noise, sinusoids, (harmonic, noise)


class CombSub(nn.Module):
    window: Tensor

    def __init__(
        self,
        sampling_rate: int,
        block_size: int,
        win_length: int,
        use_mean_filter: bool,
        n_mag_harmonic: int,
        n_mag_noise: int,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        _validate_architecture(sampling_rate, block_size, win_length, n_mels)
        if n_mag_harmonic <= 0 or n_mag_noise <= 0:
            raise ValueError("DDSP control sizes must be positive")
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        self.mel2ctrl = Mel2Control(
            n_mels,
            block_size,
            {
                "harmonic_magnitude": n_mag_harmonic,
                "harmonic_phase": win_length // 2 + 1,
                "noise_magnitude": n_mag_noise,
                "noise_phase": n_mag_noise,
            },
        )
        self.mean_kernel_size = win_length // block_size if use_mean_filter else 1

    def _synthesize_noise(self, noise_exciter: Tensor, controls: dict[str, Tensor]) -> Tensor:
        noise_filter = (
            torch.exp(controls["noise_magnitude"] + 1j * pi * controls["noise_phase"]) / 128
        )
        return frequency_filter(noise_exciter, noise_filter)

    def fast_source_gen(self, f0_frames: Tensor) -> Tensor:
        offsets = torch.arange(
            self.block_size,
            device=f0_frames.device,
            dtype=f0_frames.dtype,
        )
        normalized_f0 = f0_frames / self.sampling_rate
        slope = F.pad(
            normalized_f0[:, 1:, :] - normalized_f0[:, :-1, :],
            (0, 0, 0, 1),
        )
        phase = (
            normalized_f0 * (offsets + 1) + 0.5 * slope * offsets * (offsets + 1) / self.block_size
        )
        instantaneous_f0 = normalized_f0 + slope * offsets / self.block_size
        boundary = torch.fmod(phase[..., -1:].float() + 0.5, 1.0) - 0.5
        accumulated = boundary.cumsum(dim=1).fmod(1.0).to(f0_frames)
        phase += F.pad(accumulated[:, :-1, :], (0, 0, 1, 0))
        phase -= phase.round()
        return torch.sinc(phase / (instantaneous_f0 + 1e-5)).reshape(f0_frames.shape[0], -1)

    def forward(
        self,
        mel_frames: Tensor,
        f0_frames: Tensor,
        output_f0_frames: Tensor | None = None,
        infer: bool = True,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        _validate_frames(mel_frames, f0_frames)
        combtooth = self.fast_source_gen(f0_frames)
        combtooth_frames = combtooth.unfold(1, self.block_size, self.block_size)
        noise_exciter = torch.randn_like(combtooth)
        noise_frames = noise_exciter.unfold(1, self.block_size, self.block_size)
        controls = self.mel2ctrl(mel_frames, combtooth_frames, noise_frames)
        if self.mean_kernel_size > 1:
            controls["harmonic_magnitude"] = mean_filter(
                controls["harmonic_magnitude"], self.mean_kernel_size
            )
            controls["harmonic_phase"] = mean_filter(
                controls["harmonic_phase"], self.mean_kernel_size
            )

        phase_filter = torch.exp(1j * pi * controls["harmonic_phase"])
        phase_filter = torch.cat((phase_filter, phase_filter[:, -1:, :]), dim=1)
        harmonic_filter = torch.exp(controls["harmonic_magnitude"])
        synthesis_f0 = output_f0_frames if infer and output_f0_frames is not None else f0_frames
        _validate_frames(mel_frames, synthesis_f0)
        if synthesis_f0 is not f0_frames:
            combtooth = self.fast_source_gen(synthesis_f0)
        harmonic = frequency_filter(
            combtooth,
            torch.complex(harmonic_filter, torch.zeros_like(harmonic_filter)),
            hann_window=True,
            half_width_frames=1.5 * self.sampling_rate / (synthesis_f0 + 1e-3),
        )
        harmonic = _apply_phase_filter(
            harmonic,
            phase_filter,
            window=self.window,
            win_length=self.win_length,
            block_size=self.block_size,
        )
        noise = self._synthesize_noise(noise_exciter, controls)
        return harmonic + noise, combtooth, (harmonic, noise)


def build_vocoder(
    vocoder_type: Literal["Sins", "CombSub"],
    *,
    sampling_rate: int,
    block_size: int,
    win_length: int,
    use_mean_filter: bool,
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
            use_mean_filter,
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
            use_mean_filter,
            n_mag_harmonic,
            n_mag_noise,
            n_mels,
        )
    raise ValueError(f"unsupported DDSP type: {vocoder_type}")
