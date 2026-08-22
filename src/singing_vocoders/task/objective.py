from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from singing_vocoders.components.contracts import (
    Batch,
    DiscriminatorOutput,
    GeneratorOutput,
)
from singing_vocoders.config.schema import (
    DDSPObjectiveConfig,
    FeatureConfig,
    HiFiObjectiveConfig,
    HiFiVAEObjectiveConfig,
    MultiResolutionSTFTConfig,
    ObjectiveConfig,
    OneStepObjectiveConfig,
    UnivObjectiveConfig,
)
from singing_vocoders.modules.features import LogMelSpectrogram


@dataclass(slots=True)
class LossResult:
    loss: Tensor
    metrics: dict[str, Tensor]


class STFTLoss(nn.Module):
    window: Tensor

    def __init__(self, fft_size: int, hop_size: int, win_length: int) -> None:
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def _magnitude(self, audio: Tensor) -> Tensor:
        minimum_length = self.fft_size // 2 + 1
        if audio.shape[-1] < minimum_length:
            audio = functional.pad(audio, (0, minimum_length - audio.shape[-1]))
        return (
            torch.stft(
                audio.float(),
                self.fft_size,
                hop_length=self.hop_size,
                win_length=self.win_length,
                window=self.window,
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            .abs()
            .clamp_min_(10 ** (-3.5))
        )

    def forward(self, generated: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        generated_magnitude = self._magnitude(generated)
        target_magnitude = self._magnitude(target)
        frames = min(generated_magnitude.shape[-1], target_magnitude.shape[-1])
        generated_magnitude = generated_magnitude[..., :frames]
        target_magnitude = target_magnitude[..., :frames]
        convergence = torch.linalg.vector_norm(
            target_magnitude - generated_magnitude
        ) / torch.linalg.vector_norm(target_magnitude).clamp_min(1e-7)
        magnitude = functional.l1_loss(generated_magnitude.log(), target_magnitude.log())
        return convergence, magnitude


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, config: MultiResolutionSTFTConfig) -> None:
        super().__init__()
        self.losses = nn.ModuleList(
            STFTLoss(fft, hop, win)
            for fft, hop, win in zip(
                config.fft_sizes, config.hop_sizes, config.win_lengths, strict=True
            )
        )

    def forward(self, generated: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if generated.ndim == 3:
            generated = generated.squeeze(1)
        if target.ndim == 3:
            target = target.squeeze(1)
        length = min(generated.shape[-1], target.shape[-1])
        convergence = generated.new_zeros((), dtype=torch.float32)
        magnitude = generated.new_zeros((), dtype=torch.float32)
        for loss in self.losses:
            resolution_convergence, resolution_magnitude = loss(
                generated[..., :length], target[..., :length]
            )
            convergence = convergence + resolution_convergence
            magnitude = magnitude + resolution_magnitude
        return convergence / len(self.losses), magnitude / len(self.losses)


class RandomScaleSpectralLoss(nn.Module):
    def __init__(self, fft_min: int, fft_max: int, scales: int) -> None:
        super().__init__()
        self.fft_min = fft_min
        self.fft_max = fft_max
        self.scales = scales

    @staticmethod
    def _single_scale(generated: Tensor, target: Tensor, fft_size: int) -> Tensor:
        length = min(generated.shape[-1], target.shape[-1])
        generated = generated[..., :length].float()
        target = target[..., :length].float()
        if length < fft_size:
            padding = fft_size - length
            generated = functional.pad(generated, (0, padding))
            target = functional.pad(target, (0, padding))
        window = torch.hann_window(fft_size, device=generated.device, dtype=generated.dtype)
        generated_spectrum = (
            torch.stft(
                generated,
                fft_size,
                hop_length=fft_size,
                win_length=fft_size,
                window=window,
                center=False,
                normalized=True,
                return_complex=True,
            )
            .abs()
            .clamp_min_(1e-7)
        )
        target_spectrum = (
            torch.stft(
                target,
                fft_size,
                hop_length=fft_size,
                win_length=fft_size,
                window=window,
                center=False,
                normalized=True,
                return_complex=True,
            )
            .abs()
            .clamp_min_(1e-7)
        )
        convergence = torch.mean(
            torch.linalg.vector_norm(target_spectrum - generated_spectrum, dim=(-2, -1))
            / torch.linalg.vector_norm(
                target_spectrum + generated_spectrum, dim=(-2, -1)
            ).clamp_min(1e-7)
        )
        magnitude = functional.l1_loss(target_spectrum.log(), generated_spectrum.log())
        return convergence + magnitude

    def forward(self, generated: Tensor, target: Tensor) -> Tensor:
        if generated.ndim == 3:
            generated = generated.squeeze(1)
        if target.ndim == 3:
            target = target.squeeze(1)
        fft_sizes = torch.randint(self.fft_min, self.fft_max, (self.scales,))
        total = generated.new_zeros((), dtype=torch.float32)
        for fft_size in fft_sizes.tolist():
            total = total + self._single_scale(generated, target, fft_size)
        return total / self.scales


class VocoderObjective(nn.Module):
    def __init__(self, config: ObjectiveConfig, features: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.mel = LogMelSpectrogram(features)
        self.stft: MultiResolutionSTFTLoss | None = None
        if hasattr(config, "stft"):
            self.stft = MultiResolutionSTFTLoss(config.stft)
        self.rss: RandomScaleSpectralLoss | None = None
        self.hop_size = features.hop_size
        if isinstance(config, DDSPObjectiveConfig) and config.name == "ddsp_univnet":
            self.rss = RandomScaleSpectralLoss(
                config.rss_fft_min, config.rss_fft_max, config.rss_scales
            )

    @staticmethod
    def _zero(reference: Tensor) -> Tensor:
        return reference.new_zeros(())

    @staticmethod
    def _upsample_frames(signal: Tensor, factor: int) -> Tensor:
        channels_first = signal.permute(0, 2, 1)
        extended = torch.cat((channels_first, channels_first[:, :, -1:]), dim=2)
        result = functional.interpolate(
            extended,
            size=channels_first.shape[-1] * factor + 1,
            mode="linear",
            align_corners=True,
        )
        return result[:, :, :-1].permute(0, 2, 1)

    @staticmethod
    def _paired_audio(batch: Batch, generated: GeneratorOutput) -> tuple[Tensor, Tensor]:
        return (
            generated.aux.get("reconstruction_audio", generated.audio),
            generated.aux.get("reconstruction_target", batch.audio),
        )

    def discriminator_loss(
        self,
        real: dict[str, DiscriminatorOutput],
        fake: dict[str, DiscriminatorOutput],
    ) -> LossResult:
        reference = next(iter(real.values())).scores[0]
        total = self._zero(reference)
        metrics: dict[str, Tensor] = {}
        for name in real:
            real_loss = self._zero(reference)
            fake_loss = self._zero(reference)
            for real_score, fake_score in zip(real[name].scores, fake[name].scores, strict=True):
                real_loss = real_loss + torch.mean((1 - real_score) ** 2)
                fake_loss = fake_loss + torch.mean(fake_score**2)
            metrics[f"loss/discriminator/{name}/real"] = real_loss.detach()
            metrics[f"loss/discriminator/{name}/fake"] = fake_loss.detach()
            total = total + real_loss + fake_loss
        metrics["loss/discriminator"] = total.detach()
        return LossResult(total, metrics)

    @staticmethod
    def _feature_loss(real: DiscriminatorOutput, fake: DiscriminatorOutput) -> Tensor:
        reference = fake.scores[0]
        loss = reference.new_zeros(())
        for real_branch, fake_branch in zip(real.feature_maps, fake.feature_maps, strict=True):
            for real_feature, fake_feature in zip(real_branch, fake_branch, strict=True):
                batch = real_feature.shape[0]
                if fake_feature.shape[0] < batch:
                    raise ValueError(
                        "fake feature batch must not be smaller than the real feature batch"
                    )
                loss = loss + torch.mean(
                    torch.abs(real_feature[:batch].detach() - fake_feature[:batch])
                )
        return loss * 2

    def _stft_loss(self, generated: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if self.stft is None:
            raise RuntimeError("the objective has no STFT loss configured")
        convergence, magnitude = self.stft(generated, target)
        return convergence + magnitude, {
            "loss/aux/stft_convergence": convergence.detach(),
            "loss/aux/stft_magnitude": magnitude.detach(),
        }

    def _ddsp_hybrid_loss(
        self,
        batch: Batch,
        generated: GeneratorOutput,
        logical_step: int,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        config = self.config
        if not isinstance(config, DDSPObjectiveConfig) or self.rss is None:
            raise RuntimeError("DDSP hybrid loss is unavailable")
        signal = generated.aux["ddsp_audio"].squeeze(1)
        harmonic = generated.aux["harmonic"]
        if harmonic.ndim == 3:
            harmonic = harmonic.squeeze(1)
        target = batch.audio.squeeze(1)
        rss = self.rss(signal, target)
        uv_mask = self._upsample_frames(batch.uv.float().unsqueeze(-1), self.hop_size).squeeze(-1)
        length = min(signal.shape[-1], harmonic.shape[-1], uv_mask.shape[-1])
        uv = torch.mean(
            torch.linalg.vector_norm(harmonic[..., :length] * uv_mask[..., :length], dim=1)
            / torch.linalg.vector_norm(
                signal[..., :length] * uv_mask[..., :length], dim=1
            ).clamp_min(1e-5)
        )
        optimized_uv = uv
        if logical_step < config.uv_detach_steps or uv.detach() < config.uv_tolerance:
            optimized_uv = uv.detach()
        loss = rss + config.uv_weight * optimized_uv
        return loss, {"loss/aux/ddsp_rss": rss.detach(), "loss/aux/ddsp_uv": uv.detach()}

    def auxiliary_loss(
        self,
        batch: Batch,
        generated: GeneratorOutput,
        logical_step: int = 0,
    ) -> LossResult:
        generated_audio, target_audio = self._paired_audio(batch, generated)
        config = self.config
        metrics: dict[str, Tensor] = {}
        total = self._zero(generated_audio)

        if isinstance(config, (HiFiObjectiveConfig, OneStepObjectiveConfig)):
            generated_mel = self.mel(generated_audio)
            target_mel = self.mel(target_audio)
            frames = min(generated_mel.shape[-1], target_mel.shape[-1])
            mel_loss = functional.l1_loss(generated_mel[..., :frames], target_mel[..., :frames])
            total = total + mel_loss * config.mel_weight
            metrics["loss/aux/mel"] = mel_loss.detach()
            if isinstance(config, HiFiObjectiveConfig) and config.stft_weight:
                stft_loss, stft_metrics = self._stft_loss(generated_audio, target_audio)
                total = total + stft_loss * config.stft_weight
                metrics.update(stft_metrics)
        elif isinstance(config, UnivObjectiveConfig):
            stft_loss, stft_metrics = self._stft_loss(generated_audio, target_audio)
            total = total + stft_loss * config.aux_weight
            metrics.update(stft_metrics)
        elif isinstance(config, DDSPObjectiveConfig):
            stft_loss, stft_metrics = self._stft_loss(generated_audio, target_audio)
            total = total + stft_loss * config.aux_weight
            metrics.update(stft_metrics)
            if config.name == "ddsp_univnet":
                hybrid, hybrid_metrics = self._ddsp_hybrid_loss(batch, generated, logical_step)
                total = total + hybrid * config.ddsp_weight
                metrics.update(hybrid_metrics)
        elif isinstance(config, HiFiVAEObjectiveConfig):
            stft_loss, stft_metrics = self._stft_loss(generated_audio, target_audio)
            total = total + stft_loss * config.stft_weight
            metrics.update(stft_metrics)
            mean = generated.aux["mean"]
            log_scale = generated.aux["log_scale"]
            kl = (
                0.5
                * (mean.square() + torch.exp(2 * log_scale) - 2 * log_scale - 1).sum(dim=1).mean()
            )
            waveform = functional.l1_loss(generated_audio, target_audio)
            total = total + kl * config.kl_weight + waveform * config.waveform_weight
            metrics["loss/aux/kl"] = kl.detach()
            metrics["loss/aux/waveform"] = waveform.detach()
        else:
            raise TypeError(f"unsupported objective: {type(config).__name__}")

        if "pc_consistency_a" in generated.aux:
            if not isinstance(config, (HiFiObjectiveConfig, OneStepObjectiveConfig)):
                raise TypeError("pitch-consistency output requires a HiFi objective")
            consistency = functional.l1_loss(
                generated.aux["pc_consistency_a"], generated.aux["pc_consistency_b"]
            )
            pc_weight = config.pc_weight
            total = total + consistency * pc_weight
            metrics["loss/aux/pc_consistency"] = consistency.detach()
        metrics["loss/aux"] = total.detach()
        return LossResult(total, metrics)

    def generator_loss(
        self,
        batch: Batch,
        generated: GeneratorOutput,
        real: dict[str, DiscriminatorOutput],
        fake: dict[str, DiscriminatorOutput],
        logical_step: int = 0,
    ) -> LossResult:
        reference = generated.audio
        adversarial = self._zero(reference)
        feature = self._zero(reference)
        metrics: dict[str, Tensor] = {}
        for name in fake:
            branch_adversarial = self._zero(reference)
            for score in fake[name].scores:
                branch_adversarial = branch_adversarial + torch.mean((1 - score) ** 2)
            branch_feature = self._feature_loss(real[name], fake[name])
            use_feature = not (
                isinstance(self.config, UnivObjectiveConfig)
                and self.config.name == "nsf_univnet"
                and name == "mrd"
            )
            adversarial = adversarial + branch_adversarial
            if use_feature:
                feature = feature + branch_feature
            metrics[f"loss/generator/{name}/adversarial"] = branch_adversarial.detach()
            metrics[f"loss/generator/{name}/feature"] = branch_feature.detach()
        auxiliary = self.auxiliary_loss(batch, generated, logical_step)
        total = (
            adversarial * self.config.adversarial_weight
            + feature * self.config.feature_weight
            + auxiliary.loss
        )
        metrics.update(auxiliary.metrics)
        metrics["loss/generator"] = total.detach()
        return LossResult(total, metrics)

    def joint_loss(
        self,
        batch: Batch,
        generated: GeneratorOutput,
        real: dict[str, DiscriminatorOutput],
        fake_detached: dict[str, DiscriminatorOutput],
        fake_connected: dict[str, DiscriminatorOutput],
        logical_step: int = 0,
    ) -> LossResult:
        if not isinstance(self.config, OneStepObjectiveConfig):
            raise TypeError("joint optimization requires the one_step objective")

        reference = generated.audio
        discriminator_loss = self._zero(reference)
        generator_loss = self._zero(reference)
        feature_loss = self._zero(reference)
        metrics: dict[str, Tensor] = {}
        for name in real:
            branch_discriminator = self._zero(reference)
            branch_generator = self._zero(reference)
            for real_score, detached_score, connected_score in zip(
                real[name].scores,
                fake_detached[name].scores,
                fake_connected[name].scores,
                strict=True,
            ):
                branch_discriminator = branch_discriminator + torch.mean((1 - real_score) ** 2)
                branch_discriminator = branch_discriminator + torch.mean(detached_score**2)
                branch_generator = branch_generator + torch.mean((1 - connected_score) ** 2)
            connected_feature = self._feature_loss(real[name], fake_connected[name])
            discriminator_loss = discriminator_loss + branch_discriminator
            generator_loss = generator_loss + branch_generator
            feature_loss = feature_loss + connected_feature
            metrics[f"loss/joint/{name}/discriminator"] = branch_discriminator.detach()
            metrics[f"loss/joint/{name}/generator"] = branch_generator.detach()
            metrics[f"loss/joint/{name}/feature"] = connected_feature.detach()
        auxiliary = self.auxiliary_loss(batch, generated, logical_step)
        total = (
            discriminator_loss * self.config.discriminator_weight
            + generator_loss * self.config.adversarial_weight
            + feature_loss * self.config.feature_weight
            + auxiliary.loss
        )
        metrics.update(auxiliary.metrics)
        metrics["loss/joint"] = total.detach()
        return LossResult(total, metrics)
