from __future__ import annotations

import numpy as np
import parselmouth
import torch
import torch.nn.functional as torch_functional
from librosa.filters import mel as librosa_mel
from numpy.typing import NDArray

from singing_vocoders.config.schema import FeatureConfig

PITCH_EXTRACTOR_IDS = {"parselmouth": 0, "harvest": 1}


def mel_spectrogram(audio: NDArray[np.float32], config: FeatureConfig) -> NDArray[np.float32]:
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    left = (config.win_size - config.hop_size) // 2
    right = (config.win_size - config.hop_size + 1) // 2
    waveform = torch_functional.pad(waveform.unsqueeze(1), (left, right), mode="reflect").squeeze(1)
    window = torch.hann_window(config.win_size)
    magnitude = torch.stft(
        waveform,
        config.n_fft,
        hop_length=config.hop_size,
        win_length=config.win_size,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    ).abs()
    basis = torch.from_numpy(
        librosa_mel(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.num_mels,
            fmin=config.fmin,
            fmax=config.fmax,
        )
    ).float()
    mel = torch.matmul(basis, magnitude).clamp_min_(1e-5).log_()
    return mel.squeeze(0).transpose(0, 1).numpy().astype(np.float32, copy=False)


def _interpolate_f0(
    f0: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    uv = f0 <= 0
    if uv.all():
        raise ValueError("pitch extractor found no voiced frames")
    if uv.any():
        voiced = np.flatnonzero(~uv)
        f0[uv] = np.interp(np.flatnonzero(uv), voiced, f0[voiced])
    return f0.astype(np.float32, copy=False), uv.astype(np.bool_, copy=False)


def _parselmouth_pitch(
    audio: NDArray[np.float32], frames: int, config: FeatureConfig
) -> NDArray[np.float32]:
    hop = config.hop_size
    left = int(np.ceil(1.5 / config.f0_min * config.sample_rate))
    right = hop * ((len(audio) - 1) // hop + 1) - len(audio) + left + 1
    padded = np.pad(audio, (left, right))
    pitch = parselmouth.Sound(padded, sampling_frequency=config.sample_rate).to_pitch_ac(
        time_step=hop / config.sample_rate,
        voicing_threshold=0.6,
        pitch_floor=config.f0_min,
        pitch_ceiling=config.f0_max,
    )
    f0 = pitch.selected_array["frequency"].astype(np.float32)
    return np.pad(f0, (0, max(0, frames - len(f0))))[:frames]


def _harvest_pitch(
    audio: NDArray[np.float32], frames: int, config: FeatureConfig
) -> NDArray[np.float32]:
    import pyworld

    f0, time_axis = pyworld.harvest(
        audio.astype(np.float64),
        config.sample_rate,
        f0_floor=config.f0_min,
        f0_ceil=config.f0_max,
        frame_period=1_000 * config.hop_size / config.sample_rate,
    )
    f0 = pyworld.stonemask(audio.astype(np.float64), f0, time_axis, config.sample_rate)
    return np.pad(f0.astype(np.float32), (0, max(0, frames - len(f0))))[:frames]


def extract_pitch(
    audio: NDArray[np.float32], frames: int, config: FeatureConfig
) -> tuple[NDArray[np.float32], NDArray[np.bool_], int]:
    if config.pitch_extractor == "parselmouth":
        f0 = _parselmouth_pitch(audio, frames, config)
    elif config.pitch_extractor == "harvest":
        f0 = _harvest_pitch(audio, frames, config)
    else:
        raise ValueError(f"unsupported pitch extractor: {config.pitch_extractor}")
    f0, uv = _interpolate_f0(f0)
    return f0, uv, PITCH_EXTRACTOR_IDS[config.pitch_extractor]
