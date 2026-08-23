from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from singing_vocoders.modules.backbones.univnet import (
    UnivNetArchitecture,
    UnivNetBackbone,
)
from singing_vocoders.modules.lvc.conditioning import ddsp_down


class SineGen(nn.Module):
    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if samp_rate <= 0:
            raise ValueError("samp_rate must be positive")
        if harmonic_num < 0:
            raise ValueError("harmonic_num must not be negative")
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0: Tensor) -> Tensor:
        return torch.ones_like(f0) * (f0 > self.voiced_threshold)

    def _f02sine(self, f0_values: Tensor, upp: int) -> Tensor:
        input_dtype = f0_values.dtype
        rad_values = (f0_values / self.sampling_rate).fmod(1.0)
        rand_ini = torch.rand(
            1,
            self.dim,
            device=f0_values.device,
            dtype=f0_values.dtype,
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] += rand_ini
        tmp_over_one = torch.cumsum(rad_values.double(), 1).to(input_dtype)
        tmp_over_one *= upp
        tmp_over_one = F.interpolate(
            tmp_over_one.transpose(2, 1),
            scale_factor=upp,
            mode="linear",
            align_corners=True,
        ).transpose(2, 1)
        rad_values = F.interpolate(
            rad_values.transpose(2, 1),
            scale_factor=upp,
            mode="nearest",
        ).transpose(2, 1)
        tmp_over_one = tmp_over_one.fmod(1.0)
        difference_kernel = tmp_over_one.new_tensor([[[[-1.0], [1.0]]]])
        difference = F.conv2d(
            tmp_over_one.unsqueeze(1),
            difference_kernel,
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
        ).squeeze(1)
        cumsum_shift = (difference < 0).double()
        cumsum_shift = torch.cat(
            (
                torch.zeros(
                    (f0_values.size(0), 1, self.dim),
                    dtype=torch.double,
                    device=f0_values.device,
                ),
                cumsum_shift,
            ),
            dim=1,
        )
        sines = torch.sin(torch.cumsum(rad_values.double() + cumsum_shift, dim=1) * 2 * np.pi)
        return sines.to(input_dtype)

    @torch.no_grad()
    def forward(self, f0: Tensor, upp: int) -> Tensor:
        f0 = f0.unsqueeze(-1)
        harmonic_numbers = torch.arange(1, self.dim + 1, device=f0.device).reshape(1, 1, -1)
        sine_waves = self._f02sine(f0 * harmonic_numbers, upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).to(f0.dtype)
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode="nearest").transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        return sine_waves * uv + noise_amp * torch.randn_like(sine_waves)


class SourceModuleHnNSF(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
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
        sine_waves = self.l_sin_gen(x, upp)
        return cast(Tensor, self.l_tanh(self.l_linear(sine_waves)))


class nsfUnivNet(UnivNetBackbone):
    def __init__(self, config: Mapping[str, Any], use_weight_norm: bool = True) -> None:
        nn.Module.__init__(self)
        architecture = UnivNetArchitecture.from_config(config)
        sampling_rate = config.get("audio_sample_rate")
        if isinstance(sampling_rate, bool) or not isinstance(sampling_rate, int):
            raise TypeError("audio_sample_rate must be an integer")
        self.m_source = SourceModuleHnNSF(sampling_rate=sampling_rate, harmonic_num=8)
        self.upp = architecture.hop_size
        self.ddspd = ddsp_down(dims=architecture.cg_channels, downs=architecture.upsample_rates)
        self._initialize_backbone(
            architecture,
            source_channels=architecture.cg_channels,
        )
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        f0: Tensor,
        infer: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del infer
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        spec = self.ddspd(har_source)
        return self.forward_backbone(x, c, spec), har_source


NSFUnivNet = nsfUnivNet
