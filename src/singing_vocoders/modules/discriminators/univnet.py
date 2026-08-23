from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

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

LRELU_SLOPE = 0.1
Normalizer = Callable[[nn.Module], nn.Module]


def _normalizer(use_spectral_norm: bool) -> Normalizer:
    return spectral_norm if use_spectral_norm else weight_norm


class SpecDiscriminator(nn.Module):
    window: Tensor

    def __init__(
        self,
        fft_size: int = 1024,
        shift_size: int = 120,
        win_length: int = 600,
        window: WindowName = "hann_window",
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if min(fft_size, shift_size, win_length) <= 0:
            raise ValueError("spectrogram sizes must be positive")
        if win_length > fft_size:
            raise ValueError("win_length must not exceed fft_size")
        normalize = _normalizer(use_spectral_norm)
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", build_window(window, win_length))
        self.discriminators = nn.ModuleList(
            [
                normalize(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
                normalize(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                normalize(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                normalize(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                normalize(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ]
        )
        self.out = normalize(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y: Tensor) -> tuple[Tensor, list[Tensor]]:
        feature_maps: list[Tensor] = []
        y = torch.stft(
            y.squeeze(1),
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        y = y.unsqueeze(1).abs()
        for discriminator in self.discriminators:
            y = F.leaky_relu(discriminator(y), LRELU_SLOPE, True)
            feature_maps.append(y)
        y = self.out(y)
        feature_maps.append(y)
        return torch.flatten(y, 1, -1), feature_maps


class MultiResSpecDiscriminator(nn.Module):
    def __init__(
        self,
        fft_sizes: Sequence[int] = (1024, 2048, 512),
        hop_sizes: Sequence[int] = (120, 240, 50),
        win_lengths: Sequence[int] = (600, 1200, 240),
        window: WindowName = "hann_window",
    ) -> None:
        super().__init__()
        resolutions = validate_resolutions(fft_sizes, hop_sizes, win_lengths)
        self.discriminators = nn.ModuleList(
            SpecDiscriminator(fft_size, hop_size, win_length, window)
            for fft_size, hop_size, win_length in resolutions
        )

    def forward(self, y: Tensor) -> tuple[list[Tensor], list[list[Tensor]]]:
        outputs = [discriminator(y) for discriminator in self.discriminators]
        return [score for score, _ in outputs], [features for _, features in outputs]


class MultiResolutionSpectrogramDiscriminator(DiscriminatorComponent):
    def __init__(
        self,
        fft_sizes: Sequence[int] = (1024, 2048, 512),
        hop_sizes: Sequence[int] = (120, 240, 50),
        win_lengths: Sequence[int] = (600, 1200, 240),
        window: WindowName = "hann_window",
    ) -> None:
        super().__init__()
        self.backbone = MultiResSpecDiscriminator(
            fft_sizes,
            hop_sizes,
            win_lengths,
            window,
        )

    def forward(self, audio: Tensor) -> DiscriminatorOutput:
        scores, feature_maps = self.backbone(audio)
        return DiscriminatorOutput(scores, feature_maps)


class MultiPeriodDiscriminator(SharedMultiPeriodDiscriminator):
    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11)) -> None:
        super().__init__(periods, _normalizer=cast(Callable[[nn.Module], nn.Module], weight_norm))
