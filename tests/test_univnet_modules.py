from __future__ import annotations

from typing import Any

import pytest
import torch

from singing_vocoders.modules.backbones.nsf_univnet import nsfUnivNet
from singing_vocoders.modules.backbones.univnet import UnivNet, UnivNetArchitecture
from singing_vocoders.modules.lvc.conditioning import ddsp_down


def _tiny_config(*, upmel: int | None = 2, hop_size: int = 64) -> dict[str, Any]:
    return {
        "audio_sample_rate": 44_100,
        "audio_num_mel_bins": 4,
        "hop_size": hop_size,
        "win_size": 128,
        "model_args": {
            "cond_in_channels": 2,
            "out_channels": 1,
            "cg_channels": 4,
            "upsample_rates": [4, 4, 2],
            "num_lvc_blocks": 1,
            "lvc_kernels": 3,
            "lvc_hidden_channels": 4,
            "lvc_conv_size": 3,
            "dropout": 0.0,
            "upmel": upmel,
        },
    }


def test_univnet_rejects_hop_size_that_does_not_match_upsampling() -> None:
    with pytest.raises(ValueError, match=r"upmel \* product\(upsample_rates\)"):
        UnivNetArchitecture.from_config(_tiny_config(hop_size=32))


@pytest.mark.parametrize("upmel", (0, -2, 3, 8))
def test_univnet_rejects_upmel_values_unsupported_by_original_upsampler(upmel: int) -> None:
    with pytest.raises(ValueError, match="upmel"):
        UnivNetArchitecture.from_config(_tiny_config(upmel=upmel))


def test_univnet_forward_preserves_original_names_and_length() -> None:
    model = UnivNet(_tiny_config(), use_weight_norm=False).eval()
    state_keys = set(model.state_dict())
    assert "upblocke.0.c1.weight" in state_keys
    assert "first_conv.weight" in state_keys
    assert "lvc_blocks.0.convs.0.weight" in state_keys
    assert "last_conv_layers.0.weight" in state_keys

    mel = torch.randn(1, 4, 2)
    noise = torch.randn(1, 2, 4)
    with torch.inference_mode():
        audio = model(noise, mel)
    assert audio.shape == (1, 1, 128)
    assert torch.isfinite(audio).all()


def test_ddsp_down_preserves_original_feature_order_and_names() -> None:
    downsampler = ddsp_down(dims=4, downs=(4, 4, 2))
    state_keys = set(downsampler.state_dict())
    assert "fistpp.weight" in state_keys
    assert "downs.0.c.weight" in state_keys
    assert "ppls.0.weight" in state_keys
    features = downsampler(torch.randn(1, 1, 128))
    assert [feature.shape[-1] for feature in features] == [16, 64, 128]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for mixed dtypes")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_nsf_univnet_preserves_cuda_source_dtype(dtype: torch.dtype) -> None:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")

    model = nsfUnivNet(_tiny_config(), use_weight_norm=False).cuda().to(dtype).eval()
    mel = torch.randn(1, 4, 2, device="cuda", dtype=dtype)
    noise = torch.randn(1, 2, 4, device="cuda", dtype=dtype)
    f0 = torch.full((1, 2), 220.0, device="cuda", dtype=dtype)
    with torch.inference_mode():
        audio, source = model(noise, mel, f0)

    assert audio.dtype is dtype
    assert source.dtype is dtype
    assert torch.isfinite(audio).all()
    assert torch.isfinite(source).all()
