import torch
from torch import nn

from singing_vocoders.components import Batch, DiscriminatorOutput, GeneratorOutput
from singing_vocoders.config.schema import (
    DDSPObjectiveConfig,
    FeatureConfig,
    HiFiObjectiveConfig,
    OneStepObjectiveConfig,
    RecipeConfig,
)
from singing_vocoders.task.augmentation import GeneratorStage
from singing_vocoders.task.objective import VocoderObjective


def test_gan_losses_are_finite() -> None:
    objective = VocoderObjective(
        HiFiObjectiveConfig(name="hifigan"),
        FeatureConfig(sample_rate=8_000, num_mels=4, hop_size=4, n_fft=16, win_size=16, fmax=3_900),
    )
    audio = torch.randn(2, 1, 64)
    batch = Batch(
        audio=audio,
        mel=objective.mel(audio).detach(),
        f0=torch.ones(2, 16),
        uv=torch.zeros(2, 16, dtype=torch.bool),
        pe=torch.zeros(2, dtype=torch.long),
    )
    generated = GeneratorOutput(audio * 0.5)
    real = {"d": DiscriminatorOutput([torch.ones(2, 3)], [[torch.ones(2, 2)]])}
    fake = {"d": DiscriminatorOutput([torch.zeros(2, 3)], [[torch.zeros(2, 2)]])}
    assert torch.isfinite(objective.discriminator_loss(real, fake).loss)
    assert torch.isfinite(objective.generator_loss(batch, generated, real, fake).loss)


def test_one_step_objective_preserves_difference_formula() -> None:
    objective = VocoderObjective(
        OneStepObjectiveConfig(
            name="one_step", mel_weight=0, feature_weight=0, discriminator_weight=1
        ),
        FeatureConfig(
            sample_rate=8_000,
            num_mels=4,
            hop_size=4,
            n_fft=16,
            win_size=16,
            fmax=3_900,
        ),
    )
    audio = torch.randn(1, 1, 64)
    batch = Batch(
        audio,
        objective.mel(audio),
        torch.ones(1, 16),
        torch.zeros(1, 16, dtype=torch.bool),
        torch.zeros(1, dtype=torch.long),
    )
    generated = GeneratorOutput(audio)
    real = {"d": DiscriminatorOutput([torch.ones(1, 1)], [[torch.zeros(1, 1)]])}
    detached = {"d": DiscriminatorOutput([torch.full((1, 1), 0.2)], [[torch.zeros(1, 1)]])}
    connected = {"d": DiscriminatorOutput([torch.full((1, 1), 0.4)], [[torch.zeros(1, 1)]])}
    result = objective.joint_loss(batch, generated, real, detached, connected)
    assert torch.allclose(result.loss, torch.tensor(-0.24), atol=1e-6)


def test_ddsp_hybrid_uses_uv_and_random_scale_spectral_losses() -> None:
    features = FeatureConfig(
        sample_rate=8_000,
        num_mels=4,
        hop_size=4,
        n_fft=16,
        win_size=16,
        fmax=3_900,
    )
    objective = VocoderObjective(
        DDSPObjectiveConfig.model_validate(
            {
                "name": "ddsp_univnet",
                "aux_weight": 1,
                "stft": {
                    "fft_sizes": [16],
                    "hop_sizes": [4],
                    "win_lengths": [16],
                },
                "ddsp_weight": 1,
                "rss_fft_min": 8,
                "rss_fft_max": 16,
                "rss_scales": 2,
            }
        ),
        features,
    )
    audio = torch.randn(2, 1, 64)
    batch = Batch(
        audio,
        objective.mel(audio),
        torch.ones(2, 16),
        torch.ones(2, 16, dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
    )
    generated_audio = (audio * 0.9).requires_grad_()
    generated = GeneratorOutput(
        generated_audio,
        {
            "ddsp_audio": generated_audio,
            "harmonic": generated_audio.squeeze(1) * 0.5,
        },
    )
    result = objective.auxiliary_loss(batch, generated)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert torch.isfinite(generated_audio.grad).all()


class MelConditionedGenerator(nn.Module):
    def forward(self, batch: Batch) -> GeneratorOutput:
        return GeneratorOutput(batch.mel[:, :1].repeat_interleave(4, dim=-1))


def test_pitch_consistency_stage_builds_composite_fake_and_reconstruction_batches() -> None:
    config = RecipeConfig.model_validate(
        {
            "name": "pc-test",
            "data": {
                "processed_dir": "data/test",
                "features": {
                    "sample_rate": 8_000,
                    "num_mels": 4,
                    "hop_size": 4,
                    "n_fft": 16,
                    "win_size": 16,
                    "fmax": 3_900,
                },
            },
            "augmentation": {
                "pc_enabled": True,
                "pc_probability": 0.5,
                "pc_key_range": 5,
            },
            "model": {
                "generator": {
                    "target": "test",
                    "params": {"mini_nsf": True, "upsample_rates": [2, 2, 2]},
                }
            },
            "objective": {"name": "hifigan"},
            "optimization": {
                "name": "alternating",
                "generator": {"optimizer": {"name": "adamw"}},
                "discriminator": {"optimizer": {"name": "adamw"}},
            },
        }
    )
    stage = GeneratorStage(MelConditionedGenerator(), config)
    batch = Batch(
        audio=torch.randn(2, 1, 64),
        mel=torch.randn(2, 4, 16),
        f0=torch.full((2, 16), 220.0),
        uv=torch.zeros(2, 16, dtype=torch.bool),
        pe=torch.zeros(2, dtype=torch.long),
    )
    output = stage(batch)
    assert output.audio.shape == (5, 1, 64)
    assert output.aux["reconstruction_audio"].shape == (3, 1, 64)
    stage.eval()
    assert stage(batch).audio.shape == (2, 1, 64)
