import pytest
import torch
from torch import nn

from singing_vocoders.components import Batch, DiscriminatorOutput, GeneratorOutput
from singing_vocoders.config.schema import (
    DDSPObjectiveConfig,
    FeatureConfig,
    HiFiObjectiveConfig,
    HiFiVAEObjectiveConfig,
    OneStepObjectiveConfig,
    RecipeConfig,
)
from singing_vocoders.modules.features import LogMelSpectrogram
from singing_vocoders.task.augmentation import GeneratorStage
from singing_vocoders.task.objective import RandomScaleSpectralLoss, STFTLoss, VocoderObjective


def test_stft_loss_preserves_original_spectral_formulas() -> None:
    torch.manual_seed(1234)
    generated = torch.randn(2, 256)
    target = torch.randn(2, 256)
    fft_size = 64
    hop_size = 16
    win_length = 48
    window = torch.hann_window(win_length)

    def magnitude(audio: torch.Tensor) -> torch.Tensor:
        return (
            torch.stft(
                audio,
                fft_size,
                hop_size,
                win_length,
                window,
                return_complex=True,
            )
            .abs()
            .clamp_min(10 ** (-3.5))
            .transpose(2, 1)
        )

    generated_magnitude = magnitude(generated)
    target_magnitude = magnitude(target)
    expected_convergence = torch.norm(target_magnitude - generated_magnitude, p="fro") / torch.norm(
        target_magnitude, p="fro"
    )
    expected_magnitude = torch.nn.functional.l1_loss(
        target_magnitude.log(), generated_magnitude.log()
    )

    convergence, magnitude_loss = STFTLoss(fft_size, hop_size, win_length)(generated, target)
    torch.testing.assert_close(convergence, expected_convergence, atol=0, rtol=0)
    torch.testing.assert_close(magnitude_loss, expected_magnitude, atol=0, rtol=0)


def test_random_scale_spectral_loss_preserves_original_epsilon_and_normalization() -> None:
    torch.manual_seed(4321)
    generated = torch.randn(2, 256)
    target = torch.randn(2, 256)
    fft_size = 32
    window = torch.hann_window(fft_size)

    def spectrum(audio: torch.Tensor) -> torch.Tensor:
        return (
            torch.stft(
                audio,
                fft_size,
                hop_length=fft_size,
                win_length=fft_size,
                window=window,
                center=False,
                normalized=False,
                return_complex=True,
            ).abs()
            / torch.linalg.vector_norm(window)
            + 1e-7
        )

    generated_spectrum = spectrum(generated)
    target_spectrum = spectrum(target)
    expected = torch.mean(
        torch.linalg.vector_norm(target_spectrum - generated_spectrum, dim=(-2, -1))
        / torch.linalg.vector_norm(target_spectrum + generated_spectrum, dim=(-2, -1))
    ) + torch.nn.functional.l1_loss(target_spectrum.log(), generated_spectrum.log())

    actual = RandomScaleSpectralLoss._single_scale(generated, target, fft_size)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


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


def test_log_mel_frame_count_uses_fft_size_when_window_is_shorter() -> None:
    feature_config = FeatureConfig(
        sample_rate=8_000,
        num_mels=4,
        hop_size=4,
        n_fft=16,
        win_size=8,
        fmax=3_900,
    )
    extractor = LogMelSpectrogram(feature_config)

    mel = extractor(torch.randn(2, 1, 64))

    assert mel.shape == (2, 4, 16)
    assert torch.isfinite(mel).all()


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
    assert torch.allclose(result.loss, torch.tensor(0.4), atol=1e-6)


def test_hifivae_kl_treats_encoder_output_as_log_standard_deviation() -> None:
    objective = VocoderObjective(
        HiFiVAEObjectiveConfig(
            name="hifivae",
            stft_weight=0,
            kl_weight=1,
            waveform_weight=0,
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
    audio = torch.zeros(1, 1, 64)
    mean = torch.tensor([[[0.0, 0.5], [1.0, -0.5]]])
    log_scale = torch.tensor([[[0.0, 0.1], [-0.2, 0.3]]])
    batch = Batch(
        audio,
        torch.zeros(1, 4, 16),
        torch.ones(1, 16),
        torch.zeros(1, 16, dtype=torch.bool),
        torch.zeros(1, dtype=torch.long),
    )
    generated = GeneratorOutput(
        audio,
        {"mean": mean, "log_scale": log_scale},
    )
    result = objective.auxiliary_loss(batch, generated)
    expected = (
        0.5 * (mean.square() + torch.exp(2 * log_scale) - 2 * log_scale - 1).sum(dim=1).mean()
    )
    torch.testing.assert_close(result.metrics["loss/aux/kl"], expected)
    torch.testing.assert_close(result.loss, expected)


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


def test_ddsp_uv_loss_preserves_original_additive_epsilon() -> None:
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
                "aux_weight": 0,
                "ddsp_weight": 1,
                "rss_fft_min": 8,
                "rss_fft_max": 16,
                "rss_scales": 1,
                "uv_weight": 1,
                "uv_detach_steps": 0,
                "uv_tolerance": 0,
            }
        ),
        features,
    )

    class ZeroSpectralLoss(RandomScaleSpectralLoss):
        def __init__(self) -> None:
            super().__init__(8, 9, 1)

        def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return generated.new_zeros(())

    objective.rss = ZeroSpectralLoss()
    signal = torch.full((2, 1, 32), 2.0)
    harmonic = torch.ones(2, 32)
    batch = Batch(
        audio=torch.zeros_like(signal),
        mel=torch.zeros(2, 4, 8),
        f0=torch.ones(2, 8),
        uv=torch.ones(2, 8, dtype=torch.bool),
        pe=torch.zeros(2, dtype=torch.long),
    )
    generated = GeneratorOutput(
        signal,
        {"ddsp_audio": signal, "harmonic": harmonic},
    )

    loss, metrics = objective._ddsp_hybrid_loss(batch, generated, logical_step=0)
    expected = torch.mean(
        torch.linalg.vector_norm(harmonic, dim=1)
        / (torch.linalg.vector_norm(signal.squeeze(1), dim=1) + 1e-5)
    )
    torch.testing.assert_close(metrics["loss/aux/ddsp_uv"], expected, atol=0, rtol=0)
    torch.testing.assert_close(loss, expected, atol=0, rtol=0)


class MelConditionedGenerator(nn.Module):
    def forward(self, batch: Batch) -> GeneratorOutput:
        return GeneratorOutput(batch.mel[:, :1].repeat_interleave(4, dim=-1))


class AudioIdentityGenerator(nn.Module):
    def forward(self, batch: Batch) -> GeneratorOutput:
        return GeneratorOutput(batch.audio)


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
                "pc_probability": 1,
                "pc_key_range": 5,
            },
            "model": {
                "generator": {
                    "target": "test",
                    "params": {"mini_nsf": True, "upsample_rates": [2, 2, 2]},
                },
                "discriminators": {"unused": {"target": "unused"}},
            },
            "objective": {"name": "hifigan"},
            "optimization": {
                "name": "alternating",
                "generator": {
                    "optimizers": [{"parameters": "all", "optimizer": {"name": "adamw"}}]
                },
                "discriminator": {
                    "optimizers": [{"parameters": "all", "optimizer": {"name": "adamw"}}]
                },
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
    assert output.audio.shape == (8, 1, 64)
    assert output.aux["reconstruction_audio"].shape == (4, 1, 64)
    stage.eval()
    assert stage(batch).audio.shape == (2, 1, 64)


def test_zero_pitch_consistency_probability_skips_augmentation() -> None:
    config = RecipeConfig.model_validate(
        {
            "name": "pc-disabled-test",
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
            "augmentation": {"pc_enabled": True, "pc_probability": 0},
            "model": {
                "generator": {
                    "target": "test",
                    "params": {"mini_nsf": True, "upsample_rates": [2, 2, 2]},
                },
                "discriminators": {"unused": {"target": "unused"}},
            },
            "objective": {"name": "hifigan"},
            "optimization": {
                "name": "alternating",
                "generator": {
                    "optimizers": [{"parameters": "all", "optimizer": {"name": "adamw"}}]
                },
                "discriminator": {
                    "optimizers": [{"parameters": "all", "optimizer": {"name": "adamw"}}]
                },
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
    assert output.audio.shape == (2, 1, 64)
    assert not output.aux


def test_pitch_consistency_restores_original_order_for_feature_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RecipeConfig.model_validate(
        {
            "name": "pc-order-test",
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
            "augmentation": {"pc_enabled": True, "pc_probability": 0.5},
            "model": {
                "generator": {
                    "target": "test",
                    "params": {"mini_nsf": True, "upsample_rates": [2, 2, 2]},
                },
                "discriminators": {"unused": {"target": "unused"}},
            },
            "objective": {"name": "hifigan"},
            "optimization": {
                "name": "alternating",
                "generator": {
                    "optimizers": [{"parameters": "all", "optimizer": {"name": "adamw"}}]
                },
                "discriminator": {
                    "optimizers": [{"parameters": "all", "optimizer": {"name": "adamw"}}]
                },
            },
        }
    )
    stage = GeneratorStage(AudioIdentityGenerator(), config)
    audio = torch.stack((torch.zeros(1, 64), torch.ones(1, 64)))
    batch = Batch(
        audio=audio,
        mel=torch.randn(2, 4, 16),
        f0=torch.full((2, 16), 220.0),
        uv=torch.zeros(2, 16, dtype=torch.bool),
        pe=torch.zeros(2, dtype=torch.long),
    )

    calls = 0

    def controlled_rand(
        *size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        **_kwargs: object,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor([0.9, 0.1], device=device, dtype=dtype)
        return torch.zeros(size, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "rand", controlled_rand)
    output = stage(batch)
    torch.testing.assert_close(output.audio[:2], batch.audio)
    torch.testing.assert_close(output.aux["reconstruction_target"][:2], batch.audio)
