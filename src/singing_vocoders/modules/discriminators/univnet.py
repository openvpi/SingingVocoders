from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from singing_vocoders.components.contracts import (
    DiscriminatorComponent,
    DiscriminatorOutput,
)
from singing_vocoders.modules.discriminators.common import (
    WindowName,
    build_window,
    validate_resolutions,
)
from singing_vocoders.modules.discriminators.hifigan import (
    MultiPeriodDiscriminator as SharedMultiPeriodDiscriminator,
)

LEAKY_RELU_SLOPE = 0.1
Normalizer = Callable[[nn.Module], nn.Module]


def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def _normalizer(use_spectral_norm: bool) -> Normalizer:
    return spectral_norm if use_spectral_norm else weight_norm


class SpectrogramDiscriminator(nn.Module):
    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 120,
        window_length: int = 600,
        window: WindowName = "hann_window",
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        normalize = _normalizer(use_spectral_norm)
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_length = window_length
        if min(fft_size, hop_size, window_length) <= 0:
            raise ValueError("spectrogram sizes must be positive")
        if window_length > fft_size:
            raise ValueError("window_length must not exceed fft_size")
        self.register_buffer("window", build_window(window, window_length))
        self.layers = nn.ModuleList(
            [
                normalize(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
                normalize(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                normalize(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                normalize(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                normalize(nn.Conv2d(32, 32, kernel_size=3, padding=1)),
            ]
        )
        self.output_projection = normalize(nn.Conv2d(32, 1, kernel_size=3, padding=1))

    def forward(self, audio: Tensor) -> tuple[Tensor, list[Tensor]]:
        hidden = (
            torch.stft(
                audio.squeeze(1),
                self.fft_size,
                self.hop_size,
                self.window_length,
                cast(Tensor, self.window),
                return_complex=True,
            )
            .abs()
            .unsqueeze(1)
        )
        feature_maps: list[Tensor] = []
        for layer in self.layers:
            hidden = F.leaky_relu(layer(hidden), LEAKY_RELU_SLOPE, inplace=True)
            feature_maps.append(hidden)
        hidden = self.output_projection(hidden)
        feature_maps.append(hidden)
        return hidden.flatten(1), feature_maps


class MultiResolutionSpectrogramDiscriminator(DiscriminatorComponent):
    def __init__(
        self,
        fft_sizes: Sequence[int] = (1024, 2048, 512),
        hop_sizes: Sequence[int] = (120, 240, 50),
        window_lengths: Sequence[int] = (600, 1200, 240),
        window: WindowName = "hann_window",
    ) -> None:
        super().__init__()
        resolutions = validate_resolutions(fft_sizes, hop_sizes, window_lengths)
        self.discriminators = nn.ModuleList(
            [
                SpectrogramDiscriminator(fft_size, hop_size, window_length, window)
                for fft_size, hop_size, window_length in resolutions
            ]
        )

    def forward(self, audio: Tensor) -> DiscriminatorOutput:
        outputs = [discriminator(audio) for discriminator in self.discriminators]
        return DiscriminatorOutput(
            [scores for scores, _ in outputs],
            [features for _, features in outputs],
        )


class MultiPeriodDiscriminator(SharedMultiPeriodDiscriminator):
    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11)) -> None:
        super().__init__(periods, _normalizer=weight_norm)
