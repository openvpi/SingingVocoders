import librosa
import numpy as np
import parselmouth
import pyworld as pw
import torch

# from utils.pitch_utils import interp_f0

PITCH_EXTRACTORS_ID_TO_NAME = {
    1: 'parselmouth',
    2: 'harvest',
}
PITCH_EXTRACTORS_NAME_TO_ID = {v: k for k, v in PITCH_EXTRACTORS_ID_TO_NAME.items()}


def norm_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = np.log2(f0 + uv)  # avoid arithmetic error
    f0[uv] = -np.inf
    return f0


def denorm_f0(f0, uv, pitch_padding=None):
    f0 = 2 ** f0
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0, uv)
    if uv.any() and not uv.all():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv


def get_pitch(pe, wav_data, length, hparams, speed=1, interp_uv=False):
    if pe == 'parselmouth':
        return get_pitch_parselmouth(wav_data, length, hparams, speed=speed, interp_uv=interp_uv)
    elif pe == 'harvest':
        return get_pitch_harvest(wav_data, length, hparams, speed=speed, interp_uv=interp_uv)
    else:
        raise ValueError(f" [x] Unknown pitch extractor: {pe}")

   
def get_pitch_parselmouth(wav_data, length, hparams, speed=1, interp_uv=False):
    """

    :param wav_data: [T]
    :param length: Expected number of frames
    :param hparams:
    :param speed: Change the speed
    :param interp_uv: Interpolate unvoiced parts
    :return: f0, uv
    """
    hop_size = int(np.round(hparams['hop_size'] * speed))
    time_step = hop_size / hparams['audio_sample_rate']
    f0_min = hparams['f0_min']
    f0_max = hparams['f0_max']

    l_pad = int(np.ceil(1.5 / f0_min * hparams['audio_sample_rate']))
    r_pad = hop_size * ((len(wav_data) - 1) // hop_size + 1) - len(wav_data) + l_pad + 1
    wav_data = np.pad(wav_data, (l_pad, r_pad))

    # noinspection PyArgumentList
    s = parselmouth.Sound(wav_data, sampling_frequency=hparams['audio_sample_rate']).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max)
    assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
    f0 = s.selected_array['frequency'].astype(np.float32)
    if len(f0) < length:
        f0 = np.pad(f0, (0, length - len(f0)))
    f0 = f0[: length]
    uv = f0 == 0
    if uv.any() and interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv

def get_pitch_harvest(wav_data, length, hparams, speed=1, interp_uv=False):
    hop_size = int(np.round(hparams['hop_size'] * speed))
    time_step = 1000 * hop_size / hparams['audio_sample_rate']
    f0_floor = hparams['f0_min']
    f0_ceil = hparams['f0_max']

    f0, _ = pw.harvest(wav_data.astype(np.float64), hparams['audio_sample_rate'], f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=time_step)
    f0 = f0.astype(np.float32)

    if f0.size < length:
        f0 = np.pad(f0, (0, length - f0.size))
    f0 = f0[:length]
    uv = f0 == 0
    if uv.any() and interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv
