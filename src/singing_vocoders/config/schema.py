from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentSpec(StrictModel):
    target: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(StrictModel):
    generator: ComponentSpec
    discriminators: dict[str, ComponentSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_discriminator_names(self) -> ModelConfig:
        if any(not name.strip() for name in self.discriminators):
            raise ValueError("discriminator names must not be empty")
        return self


class FeatureConfig(StrictModel):
    sample_rate: PositiveInt = 44_100
    num_mels: PositiveInt = 128
    hop_size: PositiveInt = 512
    n_fft: PositiveInt = 2_048
    win_size: PositiveInt = 2_048
    fmin: float = Field(default=40.0, ge=0)
    fmax: float = Field(default=16_000.0, gt=0)
    f0_min: float = Field(default=65.0, gt=0)
    f0_max: float = Field(default=1_100.0, gt=0)
    pitch_extractor: Literal["parselmouth", "harvest"] = "parselmouth"

    @model_validator(mode="after")
    def validate_ranges(self) -> FeatureConfig:
        if self.fmin >= self.fmax:
            raise ValueError("fmin must be smaller than fmax")
        if self.f0_min >= self.f0_max:
            raise ValueError("f0_min must be smaller than f0_max")
        if self.win_size > self.n_fft:
            raise ValueError("win_size must not exceed n_fft")
        return self


class DataConfig(StrictModel):
    source: Path | None = None
    processed_dir: Path
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    validation_size: int = Field(default=10, ge=1)
    seed: int = 114_514
    crop_frames: PositiveInt = 32
    batch_size: PositiveInt = 3
    num_workers: int = Field(default=4, ge=0)
    prefetch_factor: int = Field(default=2, ge=1)
    pin_memory: bool = True
    persistent_workers: bool = True


class AugmentationConfig(StrictModel):
    volume_enabled: bool = False
    volume_probability: float = Field(default=0.5, ge=0, le=1)
    pc_enabled: bool = False
    pc_probability: float = Field(default=0.4, ge=0, le=1)
    pc_key_range: float = Field(default=12.0, ge=0)


class MultiResolutionSTFTConfig(StrictModel):
    fft_sizes: tuple[PositiveInt, ...]
    hop_sizes: tuple[PositiveInt, ...]
    win_lengths: tuple[PositiveInt, ...]

    @model_validator(mode="after")
    def validate_resolutions(self) -> MultiResolutionSTFTConfig:
        if not self.fft_sizes:
            raise ValueError("at least one STFT resolution is required")
        if not (len(self.fft_sizes) == len(self.hop_sizes) == len(self.win_lengths)):
            raise ValueError("STFT resolution lists must have equal lengths")
        if any(win > fft for fft, win in zip(self.fft_sizes, self.win_lengths, strict=True)):
            raise ValueError("STFT window length must not exceed FFT size")
        return self


def _hifi_stft() -> MultiResolutionSTFTConfig:
    return MultiResolutionSTFTConfig(
        fft_sizes=(2048, 2048, 4096, 1024, 512, 256, 128, 1024, 2048, 512),
        hop_sizes=(512, 240, 480, 100, 50, 25, 12, 120, 240, 50),
        win_lengths=(2048, 1200, 2400, 480, 240, 120, 60, 600, 1200, 240),
    )


def _univ_stft() -> MultiResolutionSTFTConfig:
    return MultiResolutionSTFTConfig(
        fft_sizes=(2048, 2048, 4096, 1024, 512, 256, 128),
        hop_sizes=(512, 240, 480, 100, 50, 25, 12),
        win_lengths=(2048, 1200, 2400, 480, 240, 120, 60),
    )


def _ddsp_stft() -> MultiResolutionSTFTConfig:
    return MultiResolutionSTFTConfig(
        fft_sizes=(2048, 1024, 2048, 512, 1024),
        hop_sizes=(512, 120, 240, 50, 256),
        win_lengths=(2048, 600, 1200, 240, 512),
    )


class HiFiObjectiveConfig(StrictModel):
    name: Literal["hifigan"]
    mel_weight: float = Field(default=45.0, ge=0)
    stft_weight: float = Field(default=0.0, ge=0)
    stft: MultiResolutionSTFTConfig = Field(default_factory=_hifi_stft)
    feature_weight: float = Field(default=1.0, ge=0)
    adversarial_weight: float = Field(default=1.0, ge=0)
    pc_weight: float = Field(default=30.0, ge=0)


class UnivObjectiveConfig(StrictModel):
    name: Literal["univnet", "nsf_univnet", "nsf_univnet_msd"]
    aux_weight: float = Field(default=2.5, ge=0)
    stft: MultiResolutionSTFTConfig = Field(default_factory=_univ_stft)
    feature_weight: float = Field(default=1.0, ge=0)
    adversarial_weight: float = Field(default=1.0, ge=0)


class DDSPObjectiveConfig(StrictModel):
    name: Literal["ddsp", "ddsp_univnet"]
    aux_weight: float = Field(default=45.0, ge=0)
    stft: MultiResolutionSTFTConfig = Field(default_factory=_ddsp_stft)
    ddsp_weight: float = Field(default=2.0, ge=0)
    rss_fft_min: PositiveInt = 256
    rss_fft_max: PositiveInt = 2_048
    rss_scales: PositiveInt = 4
    uv_weight: float = Field(default=1.0, ge=0)
    uv_detach_steps: int = Field(default=2_000, ge=0)
    uv_tolerance: float = Field(default=0.15, ge=0)
    feature_weight: float = Field(default=1.0, ge=0)
    adversarial_weight: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def validate_rss_range(self) -> DDSPObjectiveConfig:
        if self.rss_fft_min >= self.rss_fft_max:
            raise ValueError("rss_fft_min must be smaller than rss_fft_max")
        return self


class HiFiVAEObjectiveConfig(StrictModel):
    name: Literal["hifivae"]
    stft_weight: float = Field(default=2.5, ge=0)
    stft: MultiResolutionSTFTConfig = Field(default_factory=_hifi_stft)
    kl_weight: float = Field(default=0.02, ge=0)
    waveform_weight: float = Field(default=0.0, ge=0)
    feature_weight: float = Field(default=1.0, ge=0)
    adversarial_weight: float = Field(default=1.0, ge=0)


class OneStepObjectiveConfig(StrictModel):
    name: Literal["one_step"]
    mel_weight: float = Field(default=45.0, ge=0)
    adversarial_weight: float = Field(default=1.0, ge=0)
    feature_weight: float = Field(default=1.0, ge=0)
    discriminator_weight: float = Field(default=1.0, ge=0)
    pc_weight: float = Field(default=30.0, ge=0)


ObjectiveConfig = Annotated[
    HiFiObjectiveConfig
    | UnivObjectiveConfig
    | DDSPObjectiveConfig
    | HiFiVAEObjectiveConfig
    | OneStepObjectiveConfig,
    Field(discriminator="name"),
]


class AdamWConfig(StrictModel):
    name: Literal["adamw"] = "adamw"
    lr: float = Field(default=1e-4, gt=0)
    betas: tuple[float, float] = (0.8, 0.99)
    eps: float = Field(default=1e-8, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)


class MuonAdamWConfig(StrictModel):
    name: Literal["muon_adamw"] = "muon_adamw"
    lr: float = Field(default=5e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    momentum: float = Field(default=0.95, ge=0, lt=1)
    nesterov: bool = True
    ns_steps: PositiveInt = 5
    adamw_lr: float = Field(default=5e-4, gt=0)
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = Field(default=1e-8, gt=0)
    adamw_weight_decay: float = Field(default=0.0, ge=0)


OptimizerConfig = Annotated[AdamWConfig | MuonAdamWConfig, Field(discriminator="name")]


class ExponentialSchedulerConfig(StrictModel):
    name: Literal["exponential"] = "exponential"
    gamma: float = Field(gt=0, le=1)


SchedulerConfig = ExponentialSchedulerConfig


class OptimizerSlotConfig(StrictModel):
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig | None = None
    frequency: PositiveInt = 1
    accumulate: PositiveInt = 1
    clip_grad_norm: float | None = Field(default=None, gt=0)


class AlternatingPlanConfig(StrictModel):
    name: Literal["alternating"]
    generator: OptimizerSlotConfig
    discriminator: OptimizerSlotConfig


class WarmupAlternatingPlanConfig(StrictModel):
    name: Literal["warmup_alternating"]
    aux_steps: int = Field(ge=0)
    generator: OptimizerSlotConfig
    discriminator: OptimizerSlotConfig


class JointPlanConfig(StrictModel):
    name: Literal["joint"]
    joint: OptimizerSlotConfig


OptimizationConfig = Annotated[
    AlternatingPlanConfig | WarmupAlternatingPlanConfig | JointPlanConfig,
    Field(discriminator="name"),
]


class RuntimeConfig(StrictModel):
    accelerator: Literal["auto", "cpu", "cuda"] = "auto"
    devices: int | Literal["auto"] = "auto"
    strategy: Literal["auto", "ddp"] = "auto"
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"
    seed: int = 114_514
    max_steps: PositiveInt = 100_000
    validate_every: PositiveInt = 1_000
    log_every: PositiveInt = 100
    checkpoint_every: PositiveInt = 1_000
    keep_last: PositiveInt = 5
    output_dir: Path = Path("experiments")


class ExportConfig(StrictModel):
    pc_aug: bool = False
    mini_nsf: bool = False
    noise_sigma: float = Field(default=0.0, ge=0)


class RecipeConfig(StrictModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    data: DataConfig
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    model: ModelConfig
    objective: ObjectiveConfig
    optimization: OptimizationConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
