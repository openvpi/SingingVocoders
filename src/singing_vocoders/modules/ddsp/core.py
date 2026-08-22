# Portions adapted from Magenta DDSP under Apache-2.0; modified by Team OpenVPI.

from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
    """Calculate final size for efficient FFT.
    Args:
      frame_size: Size of the audio frame.
      ir_size: Size of the convolving impulse response.
      power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
        numbers. TPU requires power of 2, while GPU is more flexible.
    Returns:
      fft_size: Size for efficient FFT.
    """
    convolved_frame_size = ir_size + frame_size - 1
    if power_of_2:
        fft_size = int(2 ** np.ceil(np.log2(convolved_frame_size)))
    else:
        fft_size = convolved_frame_size
    return fft_size


def upsample(signal: Tensor, factor: int) -> Tensor:
    signal = signal.permute(0, 2, 1)
    signal = F.interpolate(
        torch.cat((signal, signal[:, :, -1:]), 2),
        size=signal.shape[-1] * factor + 1,
        mode="linear",
        align_corners=True,
    )
    signal = signal[:, :, :-1]
    return signal.permute(0, 2, 1)


def mean_filter(signal: Tensor, kernel_size: int) -> Tensor:
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    channels_first = signal.transpose(1, 2)
    padded = F.pad(
        channels_first,
        ((kernel_size - 1) // 2, kernel_size // 2),
        mode="reflect",
    )
    kernel = signal.new_ones(channels_first.size(1), 1, kernel_size)
    filtered = F.conv1d(padded, kernel, groups=channels_first.size(1)) / kernel_size
    return filtered.transpose(1, 2)


def remove_above_fmax(
    amplitudes: Tensor, pitch: Tensor, fmax: float, level_start: int = 1
) -> Tensor:
    num_harmonics = amplitudes.shape[-1]
    harmonic_indices = torch.arange(
        level_start,
        num_harmonics + level_start,
        device=pitch.device,
        dtype=pitch.dtype,
    )
    antialias_mask = (pitch * harmonic_indices < fmax).to(amplitudes.dtype) + 1e-7
    return amplitudes * antialias_mask


def crop_and_compensate_delay(
    audio: Tensor,
    audio_size: int,
    ir_size: int,
    padding: str = "same",
    delay_compensation: int = -1,
) -> Tensor:
    """Crop audio output from convolution to compensate for group delay.
    Args:
      audio: Audio after convolution. Tensor of shape [batch, time_steps].
      audio_size: Initial size of the audio before convolution.
      ir_size: Size of the convolving impulse response.
      padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (audio_timesteps). For 'valid' the audio is
        extended to include the tail of the impulse response (audio_timesteps +
        ir_timesteps - 1).
      delay_compensation: Samples to crop from start of output audio to compensate
        for group delay of the impulse response. If delay_compensation < 0 it
        defaults to automatically calculating a constant group delay of the
        windowed linear phase filter from frequency_impulse_response().
    Returns:
      Tensor of cropped and shifted audio.
    Raises:
      ValueError: If padding is not either 'valid' or 'same'.
    """
    if padding == "valid":
        crop_size = ir_size + audio_size - 1
    elif padding == "same":
        crop_size = audio_size
    else:
        raise ValueError(f"Padding must be 'valid' or 'same', instead of {padding}.")

    # Frequency-domain filters here are linear phase, so their group delay is constant.
    total_size = audio.size(-1)
    crop = total_size - crop_size
    start = ir_size // 2 if delay_compensation < 0 else delay_compensation
    end = crop - start
    stop = None if end == 0 else -end
    return audio[:, start:stop]


def fft_convolve(audio: Tensor, impulse_response: Tensor) -> Tensor:
    """Filter audio with frames of time-varying impulse responses.
    Time-varying filter. Given audio [batch, n_samples], and a series of impulse
    responses [batch, n_frames, n_impulse_response], splits the audio into frames,
    applies filters, and then overlap-and-adds audio back together.
    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
    convolution for large impulse response sizes.
    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        impulse_response: Finite impulse response to convolve. Can either be a 2-D
        Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
        ir_frames, ir_size]. A 2-D tensor will apply a single linear
        time-invariant filter to the audio. A 3-D Tensor will apply a linear
        time-varying filter. Automatically chops the audio into equally shaped
        blocks to match ir_frames.
    Returns:
        audio_out: Convolved audio. Tensor of shape
            [batch, audio_timesteps].
    """
    ir_shape = impulse_response.size()
    if len(ir_shape) == 2:
        impulse_response = impulse_response.unsqueeze(1)
        ir_shape = impulse_response.size()

    _, n_ir_frames, ir_size = ir_shape
    _, audio_size = audio.size()

    hop_size = audio_size // n_ir_frames
    frame_size = 2 * hop_size
    audio_frames = F.pad(audio, (hop_size, hop_size)).unfold(1, frame_size, hop_size)

    window = torch.bartlett_window(frame_size, device=audio_frames.device, dtype=audio_frames.dtype)
    audio_frames = audio_frames * window

    fft_size = get_fft_size(frame_size, ir_size, power_of_2=False)
    audio_fft = torch.fft.rfft(audio_frames, fft_size)
    ir_fft = torch.fft.rfft(torch.cat((impulse_response, impulse_response[:, -1:, :]), 1), fft_size)

    audio_ir_fft = torch.multiply(audio_fft, ir_fft)

    audio_frames_out = torch.fft.irfft(audio_ir_fft, fft_size)

    _, n_audio_frames, frame_size = audio_frames_out.size()
    fold = torch.nn.Fold(
        output_size=(1, (n_audio_frames - 1) * hop_size + frame_size),
        kernel_size=(1, frame_size),
        stride=(1, hop_size),
    )
    output_signal = fold(audio_frames_out.transpose(1, 2)).squeeze(1).squeeze(1)

    output_signal = crop_and_compensate_delay(output_signal[:, hop_size:], audio_size, ir_size)
    return output_signal


def apply_window_to_impulse_response(
    impulse_response: Tensor,
    window_size: int = 0,
    causal: bool = False,
) -> Tensor:
    """Apply a window to an impulse response and put in causal form.
    Args:
        impulse_response: A series of impulse responses frames to window, of shape
        [batch, n_frames, ir_size].

        window_size: Size of the window to apply in the time domain. If window_size
        is less than 1, it defaults to the impulse_response size.
        causal: Impulse response input is in causal form (peak in the middle).
    Returns:
        impulse_response: Windowed impulse response in causal form, with last
        dimension cropped to window_size if window_size is greater than 0 and less
        than ir_size.
    """

    if causal:
        impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

    ir_size = impulse_response.size(-1)
    if (window_size <= 0) or (window_size > ir_size):
        window_size = ir_size
    window = torch.hann_window(
        window_size,
        device=impulse_response.device,
        dtype=impulse_response.dtype,
    )

    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = torch.cat([window[half_idx:], window.new_zeros(padding), window[:half_idx]], dim=0)
    else:
        window = window.roll(int(window.size(-1) // 2), -1)

    window = window.unsqueeze(0)
    impulse_response = impulse_response * window

    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = torch.cat(
            [impulse_response[..., first_half_start:], impulse_response[..., :second_half_end]],
            dim=-1,
        )
    else:
        impulse_response = impulse_response.roll(int(impulse_response.size(-1) // 2), -1)

    return impulse_response


def apply_dynamic_window_to_impulse_response(
    impulse_response: Tensor,
    half_width_frames: Tensor,
) -> Tensor:
    ir_size = impulse_response.size(-1)

    positions = torch.arange(
        -(ir_size // 2),
        (ir_size + 1) // 2,
        device=impulse_response.device,
        dtype=impulse_response.dtype,
    )
    normalized = (positions / half_width_frames).clamp(-1, 1)
    window = (1 + torch.cos(np.pi * normalized)) / 2

    impulse_response = impulse_response.roll(int(ir_size // 2), -1)
    impulse_response = impulse_response * window

    return impulse_response


def frequency_impulse_response(
    magnitudes: Tensor,
    hann_window: bool = True,
    half_width_frames: Tensor | None = None,
) -> Tensor:
    impulse_response = torch.fft.irfft(magnitudes)

    if hann_window:
        if half_width_frames is None:
            impulse_response = apply_window_to_impulse_response(impulse_response)
        else:
            impulse_response = apply_dynamic_window_to_impulse_response(
                impulse_response, half_width_frames
            )
    else:
        impulse_response = impulse_response.roll(int(impulse_response.size(-1) // 2), -1)

    return cast(Tensor, impulse_response)


def frequency_filter(
    audio: Tensor,
    magnitudes: Tensor,
    hann_window: bool = True,
    half_width_frames: Tensor | None = None,
) -> Tensor:
    impulse_response = frequency_impulse_response(magnitudes, hann_window, half_width_frames)

    return fft_convolve(audio, impulse_response)
