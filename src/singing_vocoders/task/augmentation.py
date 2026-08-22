from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from singing_vocoders.components.contracts import Batch, GeneratorOutput
from singing_vocoders.config.schema import RecipeConfig
from singing_vocoders.task.objective import LogMelSpectrogram


class GeneratorStage(nn.Module):
    def __init__(self, generator: nn.Module, config: RecipeConfig) -> None:
        super().__init__()
        self.generator = generator
        self.config = config.augmentation
        self.mel = LogMelSpectrogram(config.data.features)
        params = config.model.generator.params
        if self.config.pc_enabled and not params.get("mini_nsf", False):
            raise ValueError("pitch-consistency augmentation requires a MiniNSF generator")
        sample_rate = config.data.features.sample_rate
        upsample_rates = params.get("upsample_rates", ())
        source_rate = sample_rate
        if params.get("mini_nsf", False) and len(upsample_rates) > 2:
            source_rate /= math.prod(upsample_rates[2:])
        self.max_f0 = source_rate / 2

    @staticmethod
    def _replace_f0(batch: Batch, f0: Tensor) -> Batch:
        return Batch(batch.audio, batch.mel, f0, batch.uv, batch.pe, batch.metadata)

    @staticmethod
    def _slice_with_condition(batch: Batch, mel: Tensor, f0: Tensor, count: int) -> Batch:
        return Batch(
            audio=batch.audio[:count],
            mel=mel,
            f0=f0,
            uv=batch.uv[:count, : f0.shape[-1]],
            pe=batch.pe[:count],
            metadata=batch.metadata[:count],
        )

    def _generate(self, batch: Batch) -> GeneratorOutput:
        output = self.generator(batch)
        if not isinstance(output, GeneratorOutput):
            raise TypeError("generator returned an invalid output")
        return output

    def _pitch_consistency(self, batch: Batch) -> GeneratorOutput:
        count = min(
            batch.audio.shape[0],
            max(1, math.ceil(batch.audio.shape[0] * self.config.pc_probability)),
        )
        selected_f0 = batch.f0[:count]
        key_c = (
            torch.rand(count, 1, device=selected_f0.device, dtype=selected_f0.dtype) * 2 - 1
        ) * self.config.pc_key_range
        shifted_c_f0 = torch.clamp(selected_f0 * torch.exp2(key_c / 12), max=self.max_f0)
        mixed_f0 = torch.cat((shifted_c_f0, batch.f0[count:]), dim=0)
        mixed = self._generate(self._replace_f0(batch, mixed_f0))
        shifted_c = mixed.audio[:count]
        unchanged = mixed.audio[count:]

        shifted_c_mel = self.mel(shifted_c)
        frames = min(shifted_c_mel.shape[-1], selected_f0.shape[-1])
        shifted_back = self._generate(
            self._slice_with_condition(
                batch,
                shifted_c_mel[..., :frames],
                selected_f0[..., :frames],
                count,
            )
        ).audio
        reconstructed = torch.cat((shifted_back, unchanged), dim=0)

        key_a_max = self.config.pc_key_range + key_c.clamp(max=0)
        key_a_min = -self.config.pc_key_range + key_c.clamp(min=0)
        key_a = torch.rand_like(key_c) * (key_a_max - key_a_min) + key_a_min
        shifted_a_f0 = torch.clamp(selected_f0 * torch.exp2(key_a / 12), max=self.max_f0)
        shifted_a = self._generate(
            self._slice_with_condition(batch, batch.mel[:count], shifted_a_f0, count)
        ).audio
        shifted_a_mel = self.mel(shifted_a)
        frames = min(shifted_a_mel.shape[-1], shifted_c_f0.shape[-1])
        shifted_ab = self._generate(
            self._slice_with_condition(
                batch,
                shifted_a_mel[..., :frames],
                shifted_c_f0[..., :frames],
                count,
            )
        ).audio

        length = min(
            batch.audio.shape[-1],
            reconstructed.shape[-1],
            shifted_c.shape[-1],
            shifted_a.shape[-1],
            shifted_ab.shape[-1],
        )
        reconstructed = reconstructed[..., :length]
        shifted_c = shifted_c[..., :length]
        shifted_a = shifted_a[..., :length]
        shifted_ab = shifted_ab[..., :length]
        return GeneratorOutput(
            audio=torch.cat((reconstructed, shifted_c, shifted_a, shifted_ab), dim=0),
            aux={
                "reconstruction_audio": torch.cat((reconstructed, shifted_ab), dim=0),
                "reconstruction_target": torch.cat((batch.audio[..., :length], shifted_c), dim=0),
                "pc_consistency_a": shifted_ab,
                "pc_consistency_b": shifted_c,
            },
        )

    def forward(self, batch: Batch) -> GeneratorOutput:
        if not self.training or not self.config.pc_enabled:
            return self._generate(batch)
        return self._pitch_consistency(batch)
