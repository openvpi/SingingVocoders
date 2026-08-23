from __future__ import annotations

from typing import Literal, cast

from torch import Tensor, nn

from singing_vocoders.modules.ddsp.vocoder import build_vocoder


class DDSPGANBackbone(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        type: Literal["Sins", "CombSub"],
        n_mag_noise: int,
        n_harmonics: int | None = None,
        n_mag_harmonic: int | None = None,
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

    def forward(self, mel: Tensor, f0: Tensor, infer: bool = False) -> Tensor:
        signal, _, _ = self.ddsp(mel.transpose(1, 2), f0.unsqueeze(-1), infer=infer)
        return cast(Tensor, signal.unsqueeze(1))
