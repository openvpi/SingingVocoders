# Adapted from yxlllc/pc-ddsp under the MIT License.

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import torch
from torch import Tensor, nn

from singing_vocoders.modules.ddsp.core import (
    frequency_filter,
    remove_above_fmax,
    upsample,
)
from singing_vocoders.modules.ddsp.mel2control import Mel2Control
from singing_vocoders.modules.lvc.lvcnet import LVCNetGenerator


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


def _build_upnet(up_condition: int | None) -> tuple[nn.Module, int]:
    if up_condition is None:
        return nn.Identity(), 1
    if isinstance(up_condition, bool) or not isinstance(up_condition, int) or up_condition <= 0:
        raise ValueError("up_condition must be a positive integer or null")
    stages = up_condition // 2
    if 2**stages != up_condition:
        raise ValueError("up_condition must be one of 1, 2, or 4")
    return nn.Sequential(*(Upspamper() for _ in range(stages))), up_condition


class _LVCVocoder(nn.Module):
    sampling_rate: Tensor
    block_size: Tensor
    win_length: Tensor
    window: Tensor

    def _initialize_lvc(
        self,
        *,
        block_size: int,
        n_mag_noise: int,
        lvc_noise_channels: int,
        lvc_latent_channels: int,
        lvc_block_nums: int,
        lvc_layers_each_block: int,
        lvc_kernel_size: int,
        kpnet_hidden_channels: int,
        kpnet_conv_size: int,
        dropout: float,
        up_condition: int | None,
    ) -> None:
        upnet, actual_up_condition = _build_upnet(up_condition)
        if block_size % actual_up_condition:
            raise ValueError("up_condition must divide block_size")
        self.lvc = LVCNetGenerator(
            in_channels=lvc_noise_channels,
            out_channels=1,
            inner_channels=lvc_latent_channels,
            cond_channels=n_mag_noise,
            cond_hop_length=block_size // actual_up_condition,
            lvc_block_nums=lvc_block_nums,
            lvc_layers_each_block=lvc_layers_each_block,
            lvc_kernel_size=lvc_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            dropout=dropout,
        )
        self.lvc_noise_channels = lvc_noise_channels
        self.upnet = upnet

    def _lvc_noise(self, harmonic: Tensor, noise_parameter: Tensor) -> Tensor:
        noise = torch.randn(
            harmonic.shape[0],
            self.lvc_noise_channels,
            harmonic.shape[1],
            device=noise_parameter.device,
            dtype=noise_parameter.dtype,
        )
        condition = self.upnet(noise_parameter.transpose(1, 2))
        return cast(Tensor, self.lvc(noise, condition).squeeze(1))


class Sins(_LVCVocoder):
    def __init__(
        self,
        sampling_rate: int,
        block_size: int,
        win_length: int,
        n_harmonics: int,
        n_mag_noise: int,
        n_mels: int = 80,
        lvc_noise_channels: int = 8,
        lvc_latent_channels: int = 8,
        lvc_block_nums: int = 2,
        lvc_layers_each_block: int = 5,
        lvc_kernel_size: int = 3,
        kpnet_hidden_channels: int = 128,
        kpnet_conv_size: int = 1,
        dropout: float = 0.0,
        up_condition: int | None = 2,
    ) -> None:
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        self.mel2ctrl = Mel2Control(
            n_mels,
            {
                "harmonic_phase": win_length // 2 + 1,
                "amplitudes": n_harmonics,
                "noise_magnitude": n_mag_noise,
            },
        )
        self._initialize_lvc(
            block_size=block_size,
            n_mag_noise=n_mag_noise,
            lvc_noise_channels=lvc_noise_channels,
            lvc_latent_channels=lvc_latent_channels,
            lvc_block_nums=lvc_block_nums,
            lvc_layers_each_block=lvc_layers_each_block,
            lvc_kernel_size=lvc_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            dropout=dropout,
            up_condition=up_condition,
        )

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
        x = (x - torch.round(x)).to(f0)

        phase = 2 * np.pi * x
        phase_frames = phase[:, ::block_size]
        controls = self.mel2ctrl(mel_frames, phase_frames)
        source_allpass = torch.exp(1.0j * np.pi * controls["harmonic_phase"])
        source_allpass = torch.cat((source_allpass, source_allpass[:, -1:]), 1)
        amplitude_frames = torch.exp(controls["amplitudes"]) / 128
        noise_parameter = controls["noise_magnitude"]

        amplitude_frames = remove_above_fmax(
            amplitude_frames,
            f0_frames,
            float(self.sampling_rate.item()) / 2,
            level_start=1,
        )
        harmonic_count = amplitude_frames.shape[-1]
        harmonic_levels = torch.arange(1, harmonic_count + 1).to(phase)
        sinusoids = torch.zeros_like(phase[..., 0])
        for index in range((harmonic_count - 1) // max_upsample_dim + 1):
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
        noise = self._lvc_noise(harmonic, noise_parameter)
        return harmonic + noise, phase, (harmonic, noise)


class CombSub(_LVCVocoder):
    def __init__(
        self,
        sampling_rate: int,
        block_size: int,
        win_length: int,
        n_mag_harmonic: int,
        n_mag_noise: int,
        n_mels: int = 80,
        lvc_noise_channels: int = 8,
        lvc_latent_channels: int = 8,
        lvc_block_nums: int = 2,
        lvc_layers_each_block: int = 5,
        lvc_kernel_size: int = 3,
        kpnet_hidden_channels: int = 128,
        kpnet_conv_size: int = 1,
        dropout: float = 0.0,
        up_condition: int | None = 2,
    ) -> None:
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        self.mel2ctrl = Mel2Control(
            n_mels,
            {
                "harmonic_phase": win_length // 2 + 1,
                "harmonic_magnitude": n_mag_harmonic,
                "noise_magnitude": n_mag_noise,
            },
        )
        self._initialize_lvc(
            block_size=block_size,
            n_mag_noise=n_mag_noise,
            lvc_noise_channels=lvc_noise_channels,
            lvc_latent_channels=lvc_latent_channels,
            lvc_block_nums=lvc_block_nums,
            lvc_layers_each_block=lvc_layers_each_block,
            lvc_kernel_size=lvc_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            dropout=dropout,
            up_condition=up_condition,
        )

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
        x = (x - torch.round(x)).to(f0)

        phase_frames = 2 * np.pi * x[:, ::block_size]
        controls = self.mel2ctrl(mel_frames, phase_frames)
        source_allpass = torch.exp(1.0j * np.pi * controls["harmonic_phase"])
        source_allpass = torch.cat((source_allpass, source_allpass[:, -1:]), 1)
        source_magnitude = torch.exp(controls["harmonic_magnitude"])
        noise_parameter = controls["noise_magnitude"]

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
        noise = self._lvc_noise(harmonic, noise_parameter)
        return harmonic + noise, phase_frames, (harmonic, noise)


def build_lvc_vocoder(
    vocoder_type: Literal["Sins", "CombSub"],
    *,
    sampling_rate: int,
    block_size: int,
    win_length: int,
    n_mag_noise: int,
    n_mels: int,
    lvc_noise_channels: int,
    lvc_latent_channels: int,
    lvc_block_nums: int,
    lvc_layers_each_block: int,
    lvc_kernel_size: int,
    kpnet_hidden_channels: int,
    kpnet_conv_size: int,
    dropout: float,
    up_condition: int | None,
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
            lvc_noise_channels,
            lvc_latent_channels,
            lvc_block_nums,
            lvc_layers_each_block,
            lvc_kernel_size,
            kpnet_hidden_channels,
            kpnet_conv_size,
            dropout,
            up_condition,
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
            lvc_noise_channels,
            lvc_latent_channels,
            lvc_block_nums,
            lvc_layers_each_block,
            lvc_kernel_size,
            kpnet_hidden_channels,
            kpnet_conv_size,
            dropout,
            up_condition,
        )
    raise ValueError(f"unsupported DDSP type: {vocoder_type}")
