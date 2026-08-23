from __future__ import annotations

from typing import Literal, cast

from torch import Tensor, nn

from singing_vocoders.modules.ddsp.lvc import build_lvc_vocoder


class LVCDDSPGANBackbone(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        type: Literal["Sins", "CombSub"],
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
        n_harmonics: int | None = None,
        n_mag_harmonic: int | None = None,
    ) -> None:
        super().__init__()
        self.ddsp = build_lvc_vocoder(
            type,
            sampling_rate=sample_rate,
            block_size=hop_size,
            win_length=win_size,
            n_mag_noise=n_mag_noise,
            n_mels=num_mels,
            lvc_noise_channels=lvc_noise_channels,
            lvc_latent_channels=lvc_latent_channels,
            lvc_block_nums=lvc_block_nums,
            lvc_layers_each_block=lvc_layers_each_block,
            lvc_kernel_size=lvc_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            dropout=dropout,
            up_condition=up_condition,
            n_harmonics=n_harmonics,
            n_mag_harmonic=n_mag_harmonic,
        )

    def forward(self, mel: Tensor, f0: Tensor, infer: bool = False) -> Tensor:
        signal, _, _ = self.ddsp(mel.transpose(1, 2), f0.unsqueeze(-1), infer=infer)
        return cast(Tensor, signal.unsqueeze(1))
