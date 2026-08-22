import torch

from singing_vocoders.modules.ddsp.core import (
    apply_dynamic_window_to_impulse_response,
    crop_and_compensate_delay,
)


def test_dynamic_window_clamps_both_sides_and_has_finite_gradients() -> None:
    impulse_response = torch.ones(1, 1, 8, requires_grad=True)
    half_width = torch.tensor([[[2.0]]], requires_grad=True)
    windowed = apply_dynamic_window_to_impulse_response(impulse_response, half_width)
    torch.testing.assert_close(
        windowed,
        torch.tensor([[[0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0]]]),
    )
    windowed.sum().backward()
    assert impulse_response.grad is not None
    assert half_width.grad is not None
    assert torch.isfinite(impulse_response.grad).all()
    assert torch.isfinite(half_width.grad).all()


def test_delay_crop_handles_zero_end_offset() -> None:
    audio = torch.arange(8).reshape(1, 8)
    cropped = crop_and_compensate_delay(
        audio,
        audio_size=6,
        ir_size=3,
        delay_compensation=2,
    )
    torch.testing.assert_close(cropped, audio[:, 2:])
