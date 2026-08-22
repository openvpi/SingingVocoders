from __future__ import annotations

from typing import Any

import torch

from singing_vocoders.components.contracts import (
    Batch,
    DiscriminatorComponent,
    DiscriminatorOutput,
    GeneratorComponent,
    GeneratorOutput,
)
from singing_vocoders.components.factory import ComponentRegistry
from singing_vocoders.models.ddsp_univnet.ddspunivnet import ddspUnivNet as LegacyDDSPUnivNet
from singing_vocoders.models.ddspgan.ddspgan import DDSPgan as LegacyDDSPGenerator
from singing_vocoders.models.HiFivae.models import AttrDict as VAEAttrDict
from singing_vocoders.models.HiFivae.models import HiFivae as LegacyHiFiVAE
from singing_vocoders.models.lvc_ddspgan.lvc_ddspgan import DDSPgan as LegacyLVCDDSPGenerator
from singing_vocoders.models.nsf_HiFigan.models import AttrDict as NSFAttrDict
from singing_vocoders.models.nsf_HiFigan.models import Generator as LegacyNSFGenerator
from singing_vocoders.models.nsf_HiFigan.models import MultiPeriodDiscriminator as LegacyMPD
from singing_vocoders.models.nsf_HiFigan.models import MultiScaleDiscriminator as LegacyMSD
from singing_vocoders.models.nsf_univnet.nsfunivnet import nsfUnivNet as LegacyNSFUnivNet
from singing_vocoders.models.univnet.univnet import UnivNet as LegacyUnivNet
from singing_vocoders.modules.fast_D.discriminator import FastMPD as LegacyFastMPD
from singing_vocoders.modules.fast_D.discriminator import FastMRD as LegacyFastMRD
from singing_vocoders.modules.univ_D.discriminator import MultiPeriodDiscriminator as LegacyUnivMPD
from singing_vocoders.modules.univ_D.discriminator import MultiResSpecDiscriminator as LegacyMRD


def _legacy_config(
    sample_rate: int,
    num_mels: int,
    hop_size: int,
    win_size: int,
    model_args: dict[str, Any],
) -> dict[str, Any]:
    return {
        "audio_sample_rate": sample_rate,
        "audio_num_mel_bins": num_mels,
        "hop_size": hop_size,
        "win_size": win_size,
        "model_args": model_args,
    }


class NSFHiFiGANGenerator(LegacyNSFGenerator, GeneratorComponent):
    def __init__(self, **params: Any) -> None:
        values = dict(params)
        values.setdefault("mini_nsf", False)
        values.setdefault("noise_sigma", 0.0)
        super().__init__(NSFAttrDict(values))

    def forward(self, batch: Batch) -> GeneratorOutput:
        return GeneratorOutput(super().forward(x=batch.mel, f0=batch.f0))


class HiFiVAEGenerator(LegacyHiFiVAE, GeneratorComponent):
    def __init__(self, **params: Any) -> None:
        super().__init__(VAEAttrDict(dict(params)))

    def forward(self, batch: Batch) -> GeneratorOutput:
        audio, latent, mean, log_scale = super().forward(x=batch.audio)
        return GeneratorOutput(
            audio=audio,
            aux={"latent": latent, "mean": mean, "log_scale": log_scale},
        )


class DDSPGenerator(LegacyDDSPGenerator, GeneratorComponent):
    def __init__(
        self,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        **model_args: Any,
    ) -> None:
        super().__init__(_legacy_config(sample_rate, num_mels, hop_size, win_size, model_args))

    def forward(self, batch: Batch) -> GeneratorOutput:
        result = super().forward(mel=batch.mel, f0=batch.f0)
        if isinstance(result, tuple):
            audio, *parts = result
            return GeneratorOutput(
                audio, {f"part_{index}": value for index, value in enumerate(parts)}
            )
        return GeneratorOutput(result)


class LVCDDSPGenerator(LegacyLVCDDSPGenerator, GeneratorComponent):
    def __init__(
        self,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        **model_args: Any,
    ) -> None:
        super().__init__(_legacy_config(sample_rate, num_mels, hop_size, win_size, model_args))

    def forward(self, batch: Batch) -> GeneratorOutput:
        result = super().forward(mel=batch.mel, f0=batch.f0)
        if isinstance(result, tuple):
            audio, *parts = result
            return GeneratorOutput(
                audio, {f"part_{index}": value for index, value in enumerate(parts)}
            )
        return GeneratorOutput(result)


class UnivNetGenerator(LegacyUnivNet, GeneratorComponent):
    def __init__(
        self,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        use_weight_norm: bool = True,
        **model_args: Any,
    ) -> None:
        config = _legacy_config(sample_rate, num_mels, hop_size, win_size, model_args)
        super().__init__(config, use_weight_norm=use_weight_norm)
        self.noise_channels = int(model_args["cond_in_channels"])
        self.upmel = model_args.get("upmel")

    def forward(self, batch: Batch) -> GeneratorOutput:
        frames = batch.mel.shape[-1] * (self.upmel or 1)
        noise = torch.randn(
            batch.mel.shape[0],
            self.noise_channels,
            frames,
            device=batch.mel.device,
            dtype=batch.mel.dtype,
        )
        return GeneratorOutput(super().forward(x=noise, c=batch.mel))


class DDSPUnivNetGenerator(LegacyDDSPUnivNet, GeneratorComponent):
    def __init__(
        self,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        use_weight_norm: bool = True,
        **model_args: Any,
    ) -> None:
        config = _legacy_config(sample_rate, num_mels, hop_size, win_size, model_args)
        super().__init__(config, use_weight_norm=use_weight_norm)
        self.noise_channels = int(model_args["cond_in_channels"])
        self.upmel = model_args.get("upmel")

    def forward(self, batch: Batch) -> GeneratorOutput:
        frames = batch.mel.shape[-1] * (self.upmel or 1)
        noise = torch.randn(
            batch.mel.shape[0],
            self.noise_channels,
            frames,
            device=batch.mel.device,
            dtype=batch.mel.dtype,
        )
        audio, ddsp_audio, harmonic, noise_audio = super().forward(
            x=noise, c=batch.mel, f0=batch.f0
        )
        return GeneratorOutput(
            audio,
            {"ddsp_audio": ddsp_audio, "harmonic": harmonic, "noise": noise_audio},
        )


class NSFUnivNetGenerator(LegacyNSFUnivNet, GeneratorComponent):
    def __init__(
        self,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        use_weight_norm: bool = True,
        **model_args: Any,
    ) -> None:
        config = _legacy_config(sample_rate, num_mels, hop_size, win_size, model_args)
        super().__init__(config, use_weight_norm=use_weight_norm)
        self.noise_channels = int(model_args["cond_in_channels"])
        self.upmel = model_args.get("upmel")

    def forward(self, batch: Batch) -> GeneratorOutput:
        frames = batch.mel.shape[-1] * (self.upmel or 1)
        noise = torch.randn(
            batch.mel.shape[0],
            self.noise_channels,
            frames,
            device=batch.mel.device,
            dtype=batch.mel.dtype,
        )
        audio, nsf_audio = super().forward(x=noise, c=batch.mel, f0=batch.f0)
        return GeneratorOutput(audio, {"nsf_audio": nsf_audio})


class MultiScaleDiscriminator(LegacyMSD, DiscriminatorComponent):
    def forward(self, audio: torch.Tensor) -> DiscriminatorOutput:
        scores, feature_maps = super().forward(audio)
        return DiscriminatorOutput(scores, feature_maps)


class MultiPeriodDiscriminator(LegacyMPD, DiscriminatorComponent):
    def forward(self, audio: torch.Tensor) -> DiscriminatorOutput:
        scores, feature_maps = super().forward(audio)
        return DiscriminatorOutput(scores, feature_maps)


class MultiResolutionDiscriminator(LegacyMRD, DiscriminatorComponent):
    def forward(self, audio: torch.Tensor) -> DiscriminatorOutput:
        scores, feature_maps = super().forward(audio)
        return DiscriminatorOutput(scores, feature_maps)


class UnivMultiPeriodDiscriminator(LegacyUnivMPD, DiscriminatorComponent):
    def forward(self, audio: torch.Tensor) -> DiscriminatorOutput:
        scores, feature_maps = super().forward(audio)
        return DiscriminatorOutput(scores, feature_maps)


class FastMultiResolutionDiscriminator(LegacyFastMRD, DiscriminatorComponent):
    def forward(self, audio: torch.Tensor) -> DiscriminatorOutput:
        scores, feature_maps = super().forward(audio)
        return DiscriminatorOutput(scores, feature_maps)


class FastMultiPeriodDiscriminator(LegacyFastMPD, DiscriminatorComponent):
    def forward(self, audio: torch.Tensor) -> DiscriminatorOutput:
        scores, feature_maps = super().forward(audio)
        return DiscriminatorOutput(scores, feature_maps)


def create_builtin_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.register("nsf_hifigan", NSFHiFiGANGenerator)
    registry.register("hifivae", HiFiVAEGenerator)
    registry.register("ddspgan", DDSPGenerator)
    registry.register("lvc_ddspgan", LVCDDSPGenerator)
    registry.register("univnet", UnivNetGenerator)
    registry.register("ddsp_univnet", DDSPUnivNetGenerator)
    registry.register("nsf_univnet", NSFUnivNetGenerator)
    registry.register("msd", MultiScaleDiscriminator)
    registry.register("mpd", MultiPeriodDiscriminator)
    registry.register("mrd", MultiResolutionDiscriminator)
    registry.register("univ_mpd", UnivMultiPeriodDiscriminator)
    registry.register("fast_mrd", FastMultiResolutionDiscriminator)
    registry.register("fast_mpd", FastMultiPeriodDiscriminator)
    return registry
