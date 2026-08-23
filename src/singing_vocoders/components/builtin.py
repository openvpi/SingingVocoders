from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal, overload

import torch
from torch import Tensor

from singing_vocoders.components.contracts import (
    Batch,
    GeneratorComponent,
    GeneratorOutput,
)
from singing_vocoders.components.factory import ComponentRegistry
from singing_vocoders.modules.backbones.ddsp_gan import DDSPGANBackbone
from singing_vocoders.modules.backbones.ddsp_univnet import DDSPUnivNetBackbone
from singing_vocoders.modules.backbones.hifivae import HiFiVAE as HiFiVAEBackbone
from singing_vocoders.modules.backbones.lvc_ddsp_gan import LVCDDSPGANBackbone
from singing_vocoders.modules.backbones.nsf_hifigan import AttributeDict
from singing_vocoders.modules.backbones.nsf_hifigan import NSFHiFiGAN as NSFHiFiGANBackbone
from singing_vocoders.modules.backbones.nsf_univnet import NSFUnivNet as NSFUnivNetBackbone
from singing_vocoders.modules.backbones.univnet import (
    UnivNet as UnivNetBackbone,
)
from singing_vocoders.modules.backbones.univnet import (
    UnivNetArchitecture,
)
from singing_vocoders.modules.discriminators.fast import (
    FastMultiPeriodDiscriminator,
    FastMultiResolutionDiscriminator,
)
from singing_vocoders.modules.discriminators.hifigan import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from singing_vocoders.modules.discriminators.univnet import (
    MultiPeriodDiscriminator as UnivMultiPeriodDiscriminator,
)
from singing_vocoders.modules.discriminators.univnet import (
    MultiResolutionSpectrogramDiscriminator as MultiResolutionDiscriminator,
)

DDSPType = Literal["Sins", "CombSub"]


def _backbone_config(
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


def _univnet_config(
    *,
    sample_rate: int,
    num_mels: int,
    hop_size: int,
    win_size: int,
    upsample_rates: Sequence[int],
    cond_in_channels: int,
    out_channels: int,
    cg_channels: int,
    num_lvc_blocks: int,
    lvc_kernels: int,
    lvc_hidden_channels: int,
    lvc_conv_size: int,
    dropout: float,
    upmel: int | None,
) -> dict[str, Any]:
    if isinstance(out_channels, bool) or not isinstance(out_channels, int):
        raise TypeError("out_channels must be an integer")
    if out_channels != 1:
        raise ValueError("vocoder generators require out_channels=1")
    return _backbone_config(
        sample_rate,
        num_mels,
        hop_size,
        win_size,
        {
            "upsample_rates": tuple(upsample_rates),
            "cond_in_channels": cond_in_channels,
            "out_channels": out_channels,
            "cg_channels": cg_channels,
            "num_lvc_blocks": num_lvc_blocks,
            "lvc_kernels": lvc_kernels,
            "lvc_hidden_channels": lvc_hidden_channels,
            "lvc_conv_size": lvc_conv_size,
            "dropout": dropout,
            "upmel": upmel,
        },
    )


def _noise_input(batch: Batch, architecture: UnivNetArchitecture) -> Tensor:
    return torch.randn(
        batch.mel.shape[0],
        architecture.cond_in_channels,
        batch.mel.shape[-1] * architecture.upmel_factor,
        device=batch.mel.device,
        dtype=batch.mel.dtype,
    )


class NSFHiFiGANGenerator(NSFHiFiGANBackbone, GeneratorComponent):
    supported_objectives = frozenset({"hifigan", "one_step"})

    def __init__(
        self,
        sampling_rate: int,
        num_mels: int,
        hop_size: int,
        upsample_rates: Sequence[int],
        upsample_kernel_sizes: Sequence[int],
        upsample_initial_channel: int,
        resblock_kernel_sizes: Sequence[int],
        resblock_dilation_sizes: Sequence[Sequence[int]],
        resblock: Literal["1", "2"],
        mini_nsf: bool = False,
        noise_sigma: float | None = 0.0,
    ) -> None:
        rates = tuple(upsample_rates)
        kernels = tuple(upsample_kernel_sizes)
        residual_kernels = tuple(resblock_kernel_sizes)
        residual_dilations = tuple(tuple(group) for group in resblock_dilation_sizes)
        if resblock not in {"1", "2"}:
            raise ValueError("resblock must be '1' or '2'")
        if not rates or len(rates) != len(kernels):
            raise ValueError("upsample rates and kernels must be non-empty and equally sized")
        if mini_nsf and len(rates) < 2:
            raise ValueError("MiniNSF requires at least two upsampling stages")
        if math.prod(rates) != hop_size:
            raise ValueError("product of upsample_rates must equal hop_size")
        if not residual_kernels or len(residual_kernels) != len(residual_dilations):
            raise ValueError(
                "residual kernels and dilation groups must be non-empty and equally sized"
            )
        if any(value <= 0 for value in (*rates, *kernels, *residual_kernels)):
            raise ValueError("NSF architecture sizes must be positive")
        if any(
            kernel < rate or (kernel - rate) % 2
            for rate, kernel in zip(rates, kernels, strict=True)
        ):
            raise ValueError(
                "each upsample kernel must be at least its rate and differ from it by an even value"
            )
        if any(kernel % 2 == 0 for kernel in residual_kernels):
            raise ValueError("residual kernel sizes must be odd")
        if any(
            not group or any(dilation <= 0 for dilation in group) for group in residual_dilations
        ):
            raise ValueError("residual dilation groups must contain positive values")
        expected_dilations = 3 if resblock == "1" else 2
        if any(len(group) != expected_dilations for group in residual_dilations):
            raise ValueError(
                f"resblock {resblock} requires {expected_dilations} dilations per residual block"
            )
        if min(sampling_rate, num_mels, hop_size, upsample_initial_channel) <= 0:
            raise ValueError("NSF architecture sizes must be positive")
        if noise_sigma is not None and noise_sigma < 0:
            raise ValueError("noise_sigma must not be negative")
        values = AttributeDict(
            sampling_rate=sampling_rate,
            num_mels=num_mels,
            hop_size=hop_size,
            upsample_rates=rates,
            upsample_kernel_sizes=kernels,
            upsample_initial_channel=upsample_initial_channel,
            resblock_kernel_sizes=residual_kernels,
            resblock_dilation_sizes=residual_dilations,
            resblock=resblock,
            mini_nsf=mini_nsf,
            noise_sigma=noise_sigma,
        )
        super().__init__(values)

    @overload
    def forward(self, batch: Batch) -> GeneratorOutput: ...

    @overload
    def forward(self, batch: Tensor, f0: Tensor) -> Tensor: ...

    def forward(
        self,
        batch: Batch | Tensor,
        f0: Tensor | None = None,
    ) -> GeneratorOutput | Tensor:
        if isinstance(batch, Batch):
            if f0 is not None:
                raise TypeError("f0 must not be provided with a Batch")
            return GeneratorOutput(super().forward(batch.mel, batch.f0))
        if f0 is None:
            raise TypeError("f0 is required with a mel tensor")
        return super().forward(batch, f0)


class HiFiVAEGenerator(GeneratorComponent):
    supported_objectives = frozenset({"hifivae"})

    def __init__(
        self,
        latent_channels: int,
        hop_size: int,
        upsample_rates: Sequence[int],
        upsample_kernel_sizes: Sequence[int],
        upsample_initial_channel: int,
        resblock_kernel_sizes: Sequence[int],
        resblock_dilation_sizes: Sequence[Sequence[int]],
        resblock: Literal["1", "2"],
    ) -> None:
        super().__init__()
        rates = tuple(upsample_rates)
        if math.prod(rates) != hop_size:
            raise ValueError("product of upsample_rates must equal hop_size")
        self.backbone = HiFiVAEBackbone(
            latent_channels=latent_channels,
            upsample_rates=rates,
            upsample_kernel_sizes=tuple(upsample_kernel_sizes),
            initial_channels=upsample_initial_channel,
            residual_kernel_sizes=tuple(resblock_kernel_sizes),
            residual_dilation_sizes=tuple(tuple(group) for group in resblock_dilation_sizes),
            residual_block=resblock,
        )

    def forward(self, batch: Batch) -> GeneratorOutput:
        audio, latent, mean, log_scale = self.backbone(batch.audio)
        return GeneratorOutput(
            audio=audio,
            aux={"latent": latent, "mean": mean, "log_scale": log_scale},
        )


class DDSPGenerator(GeneratorComponent):
    supported_objectives = frozenset({"ddsp"})

    def __init__(
        self,
        *,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        type: DDSPType,
        n_mag_noise: int,
        n_harmonics: int | None = None,
        n_mag_harmonic: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = DDSPGANBackbone(
            sample_rate=sample_rate,
            num_mels=num_mels,
            hop_size=hop_size,
            win_size=win_size,
            type=type,
            n_mag_noise=n_mag_noise,
            n_harmonics=n_harmonics,
            n_mag_harmonic=n_mag_harmonic,
        )

    def forward(self, batch: Batch) -> GeneratorOutput:
        return GeneratorOutput(self.backbone(batch.mel, batch.f0))


class LVCDDSPGenerator(GeneratorComponent):
    supported_objectives = frozenset({"ddsp"})

    def __init__(
        self,
        *,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        type: DDSPType,
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
        self.backbone = LVCDDSPGANBackbone(
            sample_rate=sample_rate,
            num_mels=num_mels,
            hop_size=hop_size,
            win_size=win_size,
            type=type,
            n_mag_noise=n_mag_noise,
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

    def forward(self, batch: Batch) -> GeneratorOutput:
        return GeneratorOutput(self.backbone(batch.mel, batch.f0))


class UnivNetGenerator(GeneratorComponent):
    supported_objectives = frozenset({"univnet"})

    def __init__(
        self,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        use_weight_norm: bool,
        upsample_rates: Sequence[int],
        cond_in_channels: int,
        out_channels: int,
        cg_channels: int,
        num_lvc_blocks: int,
        lvc_kernels: int,
        lvc_hidden_channels: int,
        lvc_conv_size: int,
        dropout: float,
        upmel: int | None,
    ) -> None:
        config = _univnet_config(
            sample_rate=sample_rate,
            num_mels=num_mels,
            hop_size=hop_size,
            win_size=win_size,
            upsample_rates=upsample_rates,
            cond_in_channels=cond_in_channels,
            out_channels=out_channels,
            cg_channels=cg_channels,
            num_lvc_blocks=num_lvc_blocks,
            lvc_kernels=lvc_kernels,
            lvc_hidden_channels=lvc_hidden_channels,
            lvc_conv_size=lvc_conv_size,
            dropout=dropout,
            upmel=upmel,
        )
        super().__init__()
        self.backbone = UnivNetBackbone(config, use_weight_norm=use_weight_norm)

    def forward(self, batch: Batch) -> GeneratorOutput:
        noise = _noise_input(batch, self.backbone.architecture)
        return GeneratorOutput(self.backbone(noise, batch.mel))


class DDSPUnivNetGenerator(GeneratorComponent):
    supported_objectives = frozenset({"ddsp_univnet"})

    def __init__(
        self,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        use_weight_norm: bool,
        upsample_rates: Sequence[int],
        cond_in_channels: int,
        out_channels: int,
        cg_channels: int,
        num_lvc_blocks: int,
        lvc_kernels: int,
        lvc_hidden_channels: int,
        lvc_conv_size: int,
        dropout: float,
        upmel: int | None,
        type: DDSPType,
        n_mag_noise: int,
        n_harmonics: int | None = None,
        n_mag_harmonic: int | None = None,
    ) -> None:
        config = _univnet_config(
            sample_rate=sample_rate,
            num_mels=num_mels,
            hop_size=hop_size,
            win_size=win_size,
            upsample_rates=upsample_rates,
            cond_in_channels=cond_in_channels,
            out_channels=out_channels,
            cg_channels=cg_channels,
            num_lvc_blocks=num_lvc_blocks,
            lvc_kernels=lvc_kernels,
            lvc_hidden_channels=lvc_hidden_channels,
            lvc_conv_size=lvc_conv_size,
            dropout=dropout,
            upmel=upmel,
        )
        model_args = config["model_args"]
        if not isinstance(model_args, dict):
            raise TypeError("model_args must be a dictionary")
        model_args.update(
            {
                "type": type,
                "n_mag_noise": n_mag_noise,
                "n_harmonics": n_harmonics,
                "n_mag_harmonic": n_mag_harmonic,
            }
        )
        super().__init__()
        self.backbone = DDSPUnivNetBackbone(
            config,
            use_weight_norm=use_weight_norm,
        )

    def forward(self, batch: Batch) -> GeneratorOutput:
        noise_input = _noise_input(batch, self.backbone.architecture)
        audio, ddsp_audio, harmonic, noise = self.backbone(
            noise_input,
            batch.mel,
            batch.f0,
        )
        return GeneratorOutput(
            audio,
            {"ddsp_audio": ddsp_audio, "harmonic": harmonic, "noise": noise},
        )


class NSFUnivNetGenerator(GeneratorComponent):
    supported_objectives = frozenset({"nsf_univnet", "nsf_univnet_msd"})

    def __init__(
        self,
        sample_rate: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        use_weight_norm: bool,
        upsample_rates: Sequence[int],
        cond_in_channels: int,
        out_channels: int,
        cg_channels: int,
        num_lvc_blocks: int,
        lvc_kernels: int,
        lvc_hidden_channels: int,
        lvc_conv_size: int,
        dropout: float,
        upmel: int | None,
    ) -> None:
        config = _univnet_config(
            sample_rate=sample_rate,
            num_mels=num_mels,
            hop_size=hop_size,
            win_size=win_size,
            upsample_rates=upsample_rates,
            cond_in_channels=cond_in_channels,
            out_channels=out_channels,
            cg_channels=cg_channels,
            num_lvc_blocks=num_lvc_blocks,
            lvc_kernels=lvc_kernels,
            lvc_hidden_channels=lvc_hidden_channels,
            lvc_conv_size=lvc_conv_size,
            dropout=dropout,
            upmel=upmel,
        )
        super().__init__()
        self.backbone = NSFUnivNetBackbone(config, use_weight_norm=use_weight_norm)

    def forward(self, batch: Batch) -> GeneratorOutput:
        noise = _noise_input(batch, self.backbone.architecture)
        audio, nsf_audio = self.backbone(
            noise,
            batch.mel,
            batch.f0,
        )
        return GeneratorOutput(audio, {"nsf_audio": nsf_audio})


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
