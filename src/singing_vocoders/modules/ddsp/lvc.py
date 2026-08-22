from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from singing_vocoders.modules.ddsp.mel2control import Mel2Control
from singing_vocoders.modules.ddsp.vocoder import CombSub as DirectCombSub
from singing_vocoders.modules.ddsp.vocoder import Sins as DirectSins
from singing_vocoders.modules.lvc.base import LocationVariableNoiseGenerator
from singing_vocoders.modules.lvc.conditioning import (
    build_mel_upsampler,
    mel_upsampling_stages,
)


class _LVCNoiseMixin:
    noise_channels: int
    noise_generator: LocationVariableNoiseGenerator
    noise_condition_upsampler: nn.Module

    def _initialize_lvc_noise(
        self,
        *,
        block_size: int,
        n_mag_noise: int,
        noise_channels: int,
        hidden_channels: int,
        block_count: int,
        layers_per_block: int,
        convolution_kernel_size: int,
        predictor_hidden_channels: int,
        predictor_kernel_size: int,
        predictor_dropout: float,
        conditioning_upsample_factor: int,
    ) -> None:
        stages = mel_upsampling_stages(conditioning_upsample_factor)
        factor = 1 << stages
        if block_size % factor:
            raise ValueError("conditioning_upsample_factor must divide block_size")
        self.noise_channels = noise_channels
        self.noise_generator = LocationVariableNoiseGenerator(
            noise_channels=noise_channels,
            output_channels=1,
            hidden_channels=hidden_channels,
            conditioning_channels=n_mag_noise,
            conditioning_hop_length=block_size // factor,
            block_count=block_count,
            layers_per_block=layers_per_block,
            convolution_kernel_size=convolution_kernel_size,
            predictor_hidden_channels=predictor_hidden_channels,
            predictor_kernel_size=predictor_kernel_size,
            predictor_dropout=predictor_dropout,
        )
        self.noise_condition_upsampler = build_mel_upsampler(factor)

    def _synthesize_noise(self, noise_exciter: Tensor, controls: dict[str, Tensor]) -> Tensor:
        del noise_exciter
        condition = self.noise_condition_upsampler(controls["noise_magnitude"].transpose(1, 2))
        hop_length = self.noise_generator.blocks[0].conditioning_hop_length
        noise = torch.randn(
            condition.shape[0],
            self.noise_channels,
            condition.shape[-1] * hop_length,
            device=condition.device,
            dtype=condition.dtype,
        )
        return cast(Tensor, self.noise_generator(noise, condition).squeeze(1))


class Sins(_LVCNoiseMixin, DirectSins):
    def __init__(
        self,
        sampling_rate: int,
        block_size: int,
        win_length: int,
        use_mean_filter: bool,
        n_harmonics: int,
        n_mag_noise: int,
        n_mels: int,
        *,
        noise_channels: int,
        hidden_channels: int,
        block_count: int,
        layers_per_block: int,
        convolution_kernel_size: int,
        predictor_hidden_channels: int,
        predictor_kernel_size: int,
        predictor_dropout: float,
        conditioning_upsample_factor: int,
    ) -> None:
        super().__init__(
            sampling_rate,
            block_size,
            win_length,
            use_mean_filter,
            n_harmonics,
            n_mag_noise,
            n_mels,
        )
        self.mel2ctrl = Mel2Control(
            n_mels,
            block_size,
            {
                "amplitudes": n_harmonics,
                "harmonic_phase": win_length // 2 + 1,
                "noise_magnitude": n_mag_noise,
            },
        )
        self._initialize_lvc_noise(
            block_size=block_size,
            n_mag_noise=n_mag_noise,
            noise_channels=noise_channels,
            hidden_channels=hidden_channels,
            block_count=block_count,
            layers_per_block=layers_per_block,
            convolution_kernel_size=convolution_kernel_size,
            predictor_hidden_channels=predictor_hidden_channels,
            predictor_kernel_size=predictor_kernel_size,
            predictor_dropout=predictor_dropout,
            conditioning_upsample_factor=conditioning_upsample_factor,
        )


class CombSub(_LVCNoiseMixin, DirectCombSub):
    def __init__(
        self,
        sampling_rate: int,
        block_size: int,
        win_length: int,
        use_mean_filter: bool,
        n_mag_harmonic: int,
        n_mag_noise: int,
        n_mels: int,
        *,
        noise_channels: int,
        hidden_channels: int,
        block_count: int,
        layers_per_block: int,
        convolution_kernel_size: int,
        predictor_hidden_channels: int,
        predictor_kernel_size: int,
        predictor_dropout: float,
        conditioning_upsample_factor: int,
    ) -> None:
        super().__init__(
            sampling_rate,
            block_size,
            win_length,
            use_mean_filter,
            n_mag_harmonic,
            n_mag_noise,
            n_mels,
        )
        self.mel2ctrl = Mel2Control(
            n_mels,
            block_size,
            {
                "harmonic_magnitude": n_mag_harmonic,
                "harmonic_phase": win_length // 2 + 1,
                "noise_magnitude": n_mag_noise,
            },
        )
        self._initialize_lvc_noise(
            block_size=block_size,
            n_mag_noise=n_mag_noise,
            noise_channels=noise_channels,
            hidden_channels=hidden_channels,
            block_count=block_count,
            layers_per_block=layers_per_block,
            convolution_kernel_size=convolution_kernel_size,
            predictor_hidden_channels=predictor_hidden_channels,
            predictor_kernel_size=predictor_kernel_size,
            predictor_dropout=predictor_dropout,
            conditioning_upsample_factor=conditioning_upsample_factor,
        )
