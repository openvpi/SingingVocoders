from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as functional
from librosa.filters import mel as librosa_mel
from torch import Tensor, nn

from singing_vocoders.config.schema import FeatureConfig

MAGNITUDE_FLOOR = 1e-5
LOG_MEL_FLOOR = math.log(MAGNITUDE_FLOOR)


class LogMelSpectrogram(nn.Module):
    mel_basis: Tensor
    window: Tensor

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_size = config.hop_size
        self.win_size = config.win_size
        basis = librosa_mel(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.num_mels,
            fmin=config.fmin,
            fmax=config.fmax,
        ).astype(np.float32)
        self.register_buffer("mel_basis", torch.from_numpy(basis), persistent=False)
        self.register_buffer("window", torch.hann_window(config.win_size), persistent=False)

    def forward(self, audio: Tensor) -> Tensor:
        if audio.ndim == 3:
            if audio.shape[1] != 1:
                raise ValueError("audio must be mono")
            audio = audio.squeeze(1)
        if audio.ndim != 2:
            raise ValueError("audio must have shape [batch, samples] or [batch, 1, samples]")
        audio = audio.float()
        left = (self.n_fft - self.hop_size) // 2
        right = (self.n_fft - self.hop_size + 1) // 2
        audio = functional.pad(audio.unsqueeze(1), (left, right), mode="reflect").squeeze(1)
        magnitude = torch.stft(
            audio,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()
        return torch.matmul(self.mel_basis, magnitude).clamp_min_(MAGNITUDE_FLOOR).log_()
