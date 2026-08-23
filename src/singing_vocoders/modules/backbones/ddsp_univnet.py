from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from singing_vocoders.modules.backbones.univnet import (
    UnivNetArchitecture,
    UnivNetBackbone,
)
from singing_vocoders.modules.ddsp.vocoder import build_vocoder
from singing_vocoders.modules.lvc.conditioning import SourceDownsampler


class DDSPSource(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        type: Literal["Sins", "CombSub"],
        n_mag_noise: int,
        n_harmonics: int | None,
        n_mag_harmonic: int | None,
    ) -> None:
        super().__init__()
        self.ddsp = build_vocoder(
            type,
            sampling_rate=sample_rate,
            block_size=hop_size,
            win_length=win_size,
            n_mag_noise=n_mag_noise,
            n_mels=num_mels,
            n_harmonics=n_harmonics,
            n_mag_harmonic=n_mag_harmonic,
        )

    def forward(
        self,
        mel: Tensor,
        f0: Tensor,
        infer: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        signal, _, (harmonic, noise) = self.ddsp(mel.transpose(1, 2), f0.unsqueeze(-1), infer=infer)
        return signal.unsqueeze(1), harmonic, noise


class DDSPUnivNetBackbone(UnivNetBackbone):
    def __init__(
        self,
        architecture: UnivNetArchitecture,
        *,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        type: Literal["Sins", "CombSub"],
        n_mag_noise: int,
        n_harmonics: int | None,
        n_mag_harmonic: int | None,
        use_weight_norm: bool,
    ) -> None:
        super().__init__(architecture, source_channels=architecture.hidden_channels)
        self.source_downsampler = SourceDownsampler(
            architecture.hidden_channels,
            architecture.waveform_upsample_rates,
        )
        if use_weight_norm:
            self.apply_weight_norm()
        self.source = DDSPSource(
            sample_rate=sample_rate,
            num_mels=num_mels,
            hop_size=hop_size,
            win_size=win_size,
            type=type,
            n_mag_noise=n_mag_noise,
            n_harmonics=n_harmonics,
            n_mag_harmonic=n_mag_harmonic,
        )

    def forward(
        self,
        noise_input: Tensor,
        mel: Tensor,
        f0: Tensor,
        infer: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ddsp_audio, harmonic, noise = self.source(mel, f0, infer)
        source_features = self.source_downsampler(ddsp_audio)
        audio = self.forward_backbone(noise_input, mel, source_features)
        return audio, ddsp_audio, harmonic, noise
