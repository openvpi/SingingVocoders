from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import torch
from torch import Tensor

WindowName = Literal["bartlett_window", "blackman_window", "hamming_window", "hann_window"]

_WINDOW_FACTORIES: dict[str, Callable[[int], Tensor]] = {
    "bartlett_window": torch.bartlett_window,
    "blackman_window": torch.blackman_window,
    "hamming_window": torch.hamming_window,
    "hann_window": torch.hann_window,
}


def build_window(name: WindowName, length: int) -> Tensor:
    if length <= 0:
        raise ValueError("window length must be positive")
    try:
        factory = _WINDOW_FACTORIES[name]
    except KeyError as error:
        raise ValueError(f"unsupported window function: {name}") from error
    return factory(length)


def validate_resolutions(
    fft_sizes: Sequence[int],
    hop_sizes: Sequence[int],
    window_lengths: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    if not fft_sizes:
        raise ValueError("at least one spectrogram resolution is required")
    if not (len(fft_sizes) == len(hop_sizes) == len(window_lengths)):
        raise ValueError("spectrogram resolution lists must have equal lengths")
    resolutions = tuple(zip(fft_sizes, hop_sizes, window_lengths, strict=True))
    for fft_size, hop_size, window_length in resolutions:
        if min(fft_size, hop_size, window_length) <= 0:
            raise ValueError("spectrogram resolution sizes must be positive")
        if window_length > fft_size:
            raise ValueError("spectrogram window length must not exceed FFT size")
    return resolutions
