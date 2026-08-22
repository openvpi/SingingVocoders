from __future__ import annotations

from typing import Any

import pytest
import torch

from singing_vocoders.modules.backbones.nsf_univnet import NSFUnivNet
from singing_vocoders.modules.backbones.univnet import UnivNet, UnivNetArchitecture
from singing_vocoders.modules.lvc.conditioning import (
    SourceDownsampler,
    build_mel_upsampler,
    mel_upsampling_stages,
)


def _tiny_config(*, mel_upsample_factor: int = 8, hop_size: int = 64) -> dict[str, Any]:
    return {
        "audio_sample_rate": 44_100,
        "audio_num_mel_bins": 4,
        "hop_size": hop_size,
        "win_size": 128,
        "model_args": {
            "noise_channels": 2,
            "output_channels": 1,
            "hidden_channels": 4,
            "waveform_upsample_rates": [2, 2, 2],
            "lvc_layers_per_block": 1,
            "lvc_kernel_size": 3,
            "kernel_predictor_hidden_channels": 4,
            "kernel_predictor_kernel_size": 3,
            "kernel_predictor_dropout": 0.0,
            "mel_upsample_factor": mel_upsample_factor,
        },
    }


def test_mel_upsample_factor_eight_uses_three_doubling_stages() -> None:
    upsampler = build_mel_upsampler(8)
    mel = torch.randn(2, 4, 3)
    assert mel_upsampling_stages(8) == 3
    assert len(tuple(upsampler.children())) == 3
    assert upsampler(mel).shape == (2, 4, 24)


@pytest.mark.parametrize("factor", (0, -2, 3, 6))
def test_mel_upsample_factor_rejects_invalid_values(factor: int) -> None:
    with pytest.raises(ValueError, match="positive power of two"):
        mel_upsampling_stages(factor)


def test_univnet_rejects_hop_size_that_does_not_match_upsampling() -> None:
    with pytest.raises(
        ValueError,
        match=r"mel_upsample_factor \* product\(waveform_upsample_rates\)",
    ):
        UnivNetArchitecture.from_config(_tiny_config(hop_size=32))


def test_univnet_mel_upsample_factor_eight_forward_length() -> None:
    model = UnivNet(_tiny_config(), use_weight_norm=False).eval()
    mel = torch.randn(1, 4, 2)
    noise = torch.randn(1, 2, 16)
    with torch.inference_mode():
        audio = model(noise, mel)
    assert audio.shape == (1, 1, 128)
    assert torch.isfinite(audio).all()


def test_source_downsampler_preserves_exact_stage_lengths_for_odd_rates() -> None:
    downsampler = SourceDownsampler(channels=2, rates=(2, 3))
    features = downsampler(torch.randn(1, 1, 12))
    assert [feature.shape[-1] for feature in features] == [4, 12]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for mixed dtypes")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_nsf_univnet_preserves_cuda_source_dtype(dtype: torch.dtype) -> None:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")

    model = NSFUnivNet(_tiny_config(), use_weight_norm=False).cuda().to(dtype).eval()
    mel = torch.randn(1, 4, 2, device="cuda", dtype=dtype)
    noise = torch.randn(1, 2, 16, device="cuda", dtype=dtype)
    f0 = torch.full((1, 2), 220.0, device="cuda", dtype=dtype)
    with torch.inference_mode():
        audio, source = model(noise, mel, f0)

    assert audio.dtype is dtype
    assert source.dtype is dtype
    assert torch.isfinite(audio).all()
    assert torch.isfinite(source).all()
