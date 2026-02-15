import pathlib
import json
import click
import torch
import torchaudio
from tqdm import tqdm

from training.nsf_HiFigan_task import nsf_HiFigan, dynamic_range_compression_torch
from utils import get_latest_checkpoint_path
from utils.config_utils import read_full_config, print_config
from utils.wav2F0 import get_pitch
from utils.wav2mel import PitchAdjustableMelSpectrogram


@click.command(help='')
@click.option('--exp_name', required=False, metavar='EXP', help='Name of the experiment')
@click.option('--ckpt_path', required=False, metavar='FILE', help='Path to the checkpoint file')
@click.option('--save_path', required=True, metavar='FILE', help='Path to save the exported checkpoint')
@click.option('--work_dir', required=False, metavar='DIR', help='Working directory containing the experiments')
@click.option('--wav_path', required=True, metavar='DIR', help='Working directory containing the experiments')
@click.option('--key', required=False, metavar='DIR', help='Working directory containing the experiments',default=0)
def export(exp_name, ckpt_path, save_path, work_dir,wav_path,key):
    # print_config(config)
    if exp_name is None and ckpt_path is None:
        raise RuntimeError('Either --exp_name or --ckpt_path should be specified.')
    if ckpt_path is None:
        if work_dir is None:
            work_dir = pathlib.Path(__file__).parent / 'experiments'
        else:
            work_dir = pathlib.Path(work_dir)
        work_dir = work_dir / exp_name
        assert not work_dir.exists() or work_dir.is_dir(), f'Path \'{work_dir}\' is not a directory.'
        ckpt_path = get_latest_checkpoint_path(work_dir)

    config_file = pathlib.Path(ckpt_path).with_name('config.yaml')
    config = read_full_config(config_file)
    temp_dict = torch.load(ckpt_path)['state_dict']
    model=nsf_HiFigan(config)
    model.build_model()
    model.load_state_dict(temp_dict)
    mel_spec_transform = PitchAdjustableMelSpectrogram(sample_rate=config['audio_sample_rate'],
                                                       n_fft=config['fft_size'],
                                                       win_length=config['win_size'],
                                                       hop_length=config['hop_size'],
                                                       f_min=config['fmin'],
                                                       f_max=config['fmax'],
                                                       n_mels=config['audio_num_mel_bins'], )
    audio,sr=torchaudio.load(wav_path)
    if sr!=config['audio_sample_rate']:
        audio=torchaudio.transforms.Resample(audio,sr,config['audio_sample_rate'])
    mel = dynamic_range_compression_torch(mel_spec_transform(audio,key_shift=key))
    f0, uv = get_pitch(audio[0].numpy(), hparams=config, speed=1, interp_uv=True, length=len(mel[0].T))
    f0*=2 ** (key / 12)
    f0=torch.from_numpy(f0).float()[None,:]
    with torch.no_grad():
        aout=model.Gforward(sample={'mel': mel, 'f0': f0, })['audio']
        torchaudio.save(save_path,aout[0],sample_rate=config['audio_sample_rate'])





if __name__ == '__main__':
    export()
