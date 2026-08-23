from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from torch import Tensor, nn

from singing_vocoders.modules.backbones.univnet import (
    UnivNetArchitecture,
    UnivNetBackbone,
)
from singing_vocoders.modules.ddsp.vocoder import build_vocoder
from singing_vocoders.modules.lvc.conditioning import ddsp_down


class DDSP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        model_args_value = config.get("model_args")
        if not isinstance(model_args_value, Mapping):
            raise TypeError("model_args must be a mapping")
        model_args: Mapping[str, Any] = model_args_value
        vocoder_type = model_args.get("type")
        if vocoder_type not in {"Sins", "CombSub"}:
            raise ValueError(f"unsupported DDSP type: {vocoder_type}")
        self.ddsp = build_vocoder(
            vocoder_type,
            sampling_rate=int(config["audio_sample_rate"]),
            block_size=int(config["hop_size"]),
            win_length=int(config["win_size"]),
            n_mag_noise=int(model_args["n_mag_noise"]),
            n_mels=int(config["audio_num_mel_bins"]),
            n_harmonics=(
                int(model_args["n_harmonics"])
                if vocoder_type == "Sins" and model_args.get("n_harmonics") is not None
                else None
            ),
            n_mag_harmonic=(
                int(model_args["n_mag_harmonic"])
                if vocoder_type == "CombSub" and model_args.get("n_mag_harmonic") is not None
                else None
            ),
        )

    def forward(
        self,
        mel: Tensor,
        f0: Tensor,
        infer: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        signal, _, (harmonic, noise) = self.ddsp(
            mel.transpose(1, 2),
            f0.unsqueeze(-1),
            infer=infer,
        )
        return signal.unsqueeze(1), harmonic, noise


class ddspUnivNet(UnivNetBackbone):
    def __init__(self, config: Mapping[str, Any], use_weight_norm: bool = True) -> None:
        nn.Module.__init__(self)
        architecture = UnivNetArchitecture.from_config(config)
        self.ddsp = DDSP(config)
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ddsp_waveform, harmonic, noise = self.ddsp(c, f0, infer=infer)
        spec = self.ddspd(ddsp_waveform)
        output = self.forward_backbone(x, c, spec)
        return output, ddsp_waveform, harmonic, noise


DDSPUnivNetBackbone = ddspUnivNet
DDSPType = Literal["Sins", "CombSub"]
