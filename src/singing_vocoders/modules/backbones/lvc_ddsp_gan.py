from __future__ import annotations

from typing import Literal, cast

from torch import Tensor, nn

from singing_vocoders.modules.ddsp.lvc import CombSub, Sins


class LVCDDSPGANBackbone(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        type: Literal["Sins", "CombSub"],
        use_mean_filter: bool,
        n_mag_noise: int,
        noise_channels: int,
        hidden_channels: int,
        block_count: int,
        layers_per_block: int,
        convolution_kernel_size: int,
        kernel_predictor_hidden_channels: int,
        kernel_predictor_kernel_size: int,
        kernel_predictor_dropout: float,
        conditioning_upsample_factor: int,
        n_harmonics: int | None = None,
        n_mag_harmonic: int | None = None,
    ) -> None:
        super().__init__()
        if type == "Sins":
            if n_harmonics is None:
                raise ValueError("Sins requires n_harmonics")
            self.ddsp: Sins | CombSub = Sins(
                sample_rate,
                hop_size,
                win_size,
                use_mean_filter,
                n_harmonics,
                n_mag_noise,
                num_mels,
                noise_channels=noise_channels,
                hidden_channels=hidden_channels,
                block_count=block_count,
                layers_per_block=layers_per_block,
                convolution_kernel_size=convolution_kernel_size,
                predictor_hidden_channels=kernel_predictor_hidden_channels,
                predictor_kernel_size=kernel_predictor_kernel_size,
                predictor_dropout=kernel_predictor_dropout,
                conditioning_upsample_factor=conditioning_upsample_factor,
            )
        elif type == "CombSub":
            if n_mag_harmonic is None:
                raise ValueError("CombSub requires n_mag_harmonic")
            self.ddsp = CombSub(
                sample_rate,
                hop_size,
                win_size,
                use_mean_filter,
                n_mag_harmonic,
                n_mag_noise,
                num_mels,
                noise_channels=noise_channels,
                hidden_channels=hidden_channels,
                block_count=block_count,
                layers_per_block=layers_per_block,
                convolution_kernel_size=convolution_kernel_size,
                predictor_hidden_channels=kernel_predictor_hidden_channels,
                predictor_kernel_size=kernel_predictor_kernel_size,
                predictor_dropout=kernel_predictor_dropout,
                conditioning_upsample_factor=conditioning_upsample_factor,
            )
        else:
            raise ValueError(f"unsupported DDSP type: {type}")

    def forward(self, mel: Tensor, f0: Tensor, infer: bool = False) -> Tensor:
        signal, _, _ = self.ddsp(mel.transpose(1, 2), f0.unsqueeze(-1), infer=infer)
        return cast(Tensor, signal.unsqueeze(1))
