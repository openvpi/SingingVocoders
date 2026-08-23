from __future__ import annotations

from collections.abc import Callable, Sequence

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import spectral_norm, weight_norm

from singing_vocoders.components.contracts import (
    DiscriminatorComponent,
    DiscriminatorOutput,
)

LEAKY_RELU_SLOPE = 0.1
Normalizer = Callable[[nn.Module], nn.Module]


def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        normalizer: Normalizer | None = None,
    ) -> None:
        super().__init__()
        if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
            raise ValueError("period must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.period = period
        normalize = normalizer or (spectral_norm if use_spectral_norm else weight_norm)
        padding = (_same_padding(kernel_size), 0)
        self.convs = nn.ModuleList(
            [
                normalize(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=padding)),
                normalize(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=padding)),
                normalize(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=padding)),
                normalize(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=padding)),
                normalize(nn.Conv2d(1024, 1024, (kernel_size, 1), padding=padding)),
            ]
        )
        self.conv_post = normalize(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, audio: Tensor) -> tuple[Tensor, list[Tensor]]:
        batch_size, channels, samples = audio.shape
        remainder = samples % self.period
        if remainder:
            padding = self.period - remainder
            audio = F.pad(audio, (0, padding), mode="reflect")
            samples += padding
        hidden = audio.reshape(batch_size, channels, samples // self.period, self.period)
        feature_maps: list[Tensor] = []
        for layer in self.convs:
            hidden = F.leaky_relu(layer(hidden), LEAKY_RELU_SLOPE, inplace=True)
            feature_maps.append(hidden)
        hidden = self.conv_post(hidden)
        feature_maps.append(hidden)
        return hidden.flatten(1), feature_maps


class MultiPeriodDiscriminator(DiscriminatorComponent):
    def __init__(
        self,
        periods: Sequence[int] | None = None,
        *,
        _normalizer: Normalizer | None = None,
    ) -> None:
        super().__init__()
        self.periods = list(periods) if periods is not None else [2, 3, 5, 7, 11]
        if not self.periods or any(
            isinstance(period, bool) or not isinstance(period, int) or period <= 0
            for period in self.periods
        ):
            raise ValueError("periods must contain positive integers")
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period, normalizer=_normalizer) for period in self.periods]
        )

    def forward(self, audio: Tensor) -> DiscriminatorOutput:
        outputs = [discriminator(audio) for discriminator in self.discriminators]
        return DiscriminatorOutput(
            [scores for scores, _ in outputs],
            [features for _, features in outputs],
        )


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        normalize = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                normalize(nn.Conv1d(1, 128, 15, padding=7)),
                normalize(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                normalize(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                normalize(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                normalize(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                normalize(nn.Conv1d(1024, 1024, 41, groups=16, padding=20)),
                normalize(nn.Conv1d(1024, 1024, 5, padding=2)),
            ]
        )
        self.conv_post = normalize(nn.Conv1d(1024, 1, 3, padding=1))

    def forward(self, audio: Tensor) -> tuple[Tensor, list[Tensor]]:
        hidden = audio
        feature_maps: list[Tensor] = []
        for layer in self.convs:
            hidden = F.leaky_relu(layer(hidden), LEAKY_RELU_SLOPE, inplace=True)
            feature_maps.append(hidden)
        hidden = self.conv_post(hidden)
        feature_maps.append(hidden)
        return hidden.flatten(1), feature_maps


class MultiScaleDiscriminator(DiscriminatorComponent):
    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()]
        )
        self.meanpools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, audio: Tensor) -> DiscriminatorOutput:
        scores: list[Tensor] = []
        feature_maps: list[list[Tensor]] = []
        hidden = audio
        for index, discriminator in enumerate(self.discriminators):
            if index:
                hidden = self.meanpools[index - 1](hidden)
            discriminator_scores, discriminator_features = discriminator(hidden)
            scores.append(discriminator_scores)
            feature_maps.append(discriminator_features)
        return DiscriminatorOutput(scores, feature_maps)
