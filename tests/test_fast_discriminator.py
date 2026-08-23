import torch

from singing_vocoders.modules.discriminators.fast import (
    FastMPD,
    FastMRD,
    SoftSignGLU,
)


def test_softsign_glu_matches_original_forward_and_backward() -> None:
    actual_input = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)
    actual = SoftSignGLU()(actual_input)
    values, gates = expected_input.split(4, dim=-1)
    expected = values / (values.abs() + 1) * (gates / (gates.abs() + 1))
    torch.testing.assert_close(actual, expected)

    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual_input.grad, expected_input.grad)


def test_fast_backbones_keep_original_state_names() -> None:
    mpd = FastMPD(periods=(2,), init_channel=4, strides=(2, 2), kernel_size=7)
    mpd_keys = set(mpd.state_dict())
    assert "discriminators.0.pre.weight" in mpd_keys
    assert "discriminators.0.residual_layers.0.net.1.weight" in mpd_keys
    assert "discriminators.0.post.weight" in mpd_keys

    mrd = FastMRD(
        init_channel=4,
        strides=(2, 2),
        fft_sizes=(512,),
        hop_sizes=(128,),
        win_lengths=(512,),
    )
    mrd_keys = set(mrd.state_dict())
    assert "discriminators.0.window" in mrd_keys
    assert "discriminators.0.freq_coords" in mrd_keys
    assert "discriminators.0.pre.weight" in mrd_keys
    assert "discriminators.0.layers.0.net.1.weight" in mrd_keys
    assert "discriminators.0.post.weight" in mrd_keys
