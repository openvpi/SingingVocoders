import itertools
import multiprocessing
import pathlib
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Union

import click
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from utils.config_utils import read_full_config
from utils.wav2F0 import PITCH_EXTRACTORS_NAME_TO_ID, get_pitch
from utils.wav2mel import PitchAdjustableMelSpectrogram


def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def wav2spec(config: dict, source: pathlib.Path, save_path: pathlib.Path) -> Tuple[bool, Union[pathlib.Path, str]]:
    mel_spec_transform = PitchAdjustableMelSpectrogram(
        sample_rate=config['audio_sample_rate'],
        n_fft=config['fft_size'],
        win_length=config['win_size'],
        hop_length=config['hop_size'],
        f_min=config['fmin'],
        f_max=config['fmax'],
        n_mels=config['audio_num_mel_bins'],
    )
    try:
        audio, sr = torchaudio.load(source)
        pe_name = config['pe']
        pe_id = PITCH_EXTRACTORS_NAME_TO_ID[pe_name]
        if sr > config['audio_sample_rate']:
            audio = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=config['audio_sample_rate'],
                lowpass_filter_width=128)(audio)
        elif sr < config['audio_sample_rate']:
            return False, f"Error: sample rate mismatching in \'{source}\' ({sr} != {config['audio_sample_rate']})."
        mel = dynamic_range_compression_torch(mel_spec_transform(audio))
        f0, uv = get_pitch(pe_name, audio.numpy()[0], length=len(mel[0].T), hparams=config, interp_uv=True)
        if f0 is None:
            return False, f"Error: failed to get pitch from \'{source}\'."
        np.savez(save_path, audio=audio[0].numpy(), mel=mel[0].T, f0=f0, uv=uv, pe=pe_id)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return False, f"Error: {e.__class__.__name__}: {e}"
    return True, save_path


@click.command(help='')
@click.option('--config', required=True, metavar='FILE', help='Path to the configuration file')
@click.option('--num_cpu', required=False, metavar='DIR2', help='Number of CPU cores to use')
@click.option('--strx', required=False, metavar='DIR4', help='Whether to use strict path')  # 1 代表开   0代表关
def runx(config, num_cpu, strx):
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    config = pathlib.Path(config)
    config = read_full_config(config)
    # print_config(config)
    if strx is None:
        strx = 1
    else:
        strx = int(strx)
    if strx == 1:
        strx = True
    else:
        strx = False

    in_path_list = config['data_input_path']
    out_path_list = config['data_out_path']
    assert len(in_path_list) == len(out_path_list), 'path list can not match'
    data_filename_set = set()
    for inpath, outpath in tqdm(zip(in_path_list, out_path_list)):
        outlist = preprocess(config=config, input_path=inpath, output_path=outpath, num_cpu=num_cpu, st_path=strx)
        data_filename_set.update(outlist)
    outp = pathlib.Path(config['DataIndexPath'])
    assert not outp.exists() or outp.is_dir(), f'Path \'{outp}\' is not a directory.'
    outp.mkdir(parents=True, exist_ok=True)
    train_name = config['train_set_name']
    val_name = config['valid_set_name']
    val_num = config['val_num']

    val_set = random.sample(tuple(data_filename_set), val_num)
    train_set = data_filename_set - set(val_set)
    with open(outp / train_name, 'w', encoding='utf8') as f:
        [print(p, file=f) for p in sorted(train_set)]
    with open(outp / val_name, 'w', encoding='utf8') as f:
        [print(p, file=f) for p in sorted(val_set)]


def preprocess(config, input_path, output_path, num_cpu, st_path):
    if st_path:
        input_path = pathlib.Path(input_path).resolve()
        output_path = pathlib.Path(output_path).resolve()
    else:
        input_path = pathlib.Path(input_path)
        output_path = pathlib.Path(output_path)

    assert not output_path.exists() or output_path.is_dir(), f'Path \'{output_path}\' is not a directory.'
    output_path.mkdir(parents=True, exist_ok=True)

    if num_cpu is None:
        num_cpu = 5
    else:
        num_cpu = int(num_cpu)

    args = []
    for wav_file in tqdm(
            itertools.chain(input_path.rglob('*.wav'), input_path.rglob('*.flac')),
            desc="Enumerating files", leave=False
    ):
        save_path = output_path / wav_file.relative_to(input_path).with_suffix('.npz')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        args.append((
            config,
            wav_file,
            save_path,
        ))

    filenames = []
    completed = 0
    failed = 0
    try:
        with ProcessPoolExecutor(max_workers=num_cpu) as executor:
            tasks = [
                executor.submit(wav2spec, *a)
                for a in tqdm(args, desc="Submitting tasks", leave=False)
            ]
            with tqdm(as_completed(tasks), desc="Preprocessing", total=len(tasks)) as progress:
                for task in progress:
                    succeeded, result = task.result()
                    if succeeded:
                        result: pathlib.Path
                        filenames.append(result.as_posix())
                        completed += 1
                    else:
                        result: str
                        progress.write(result)
                        failed += 1
                    progress.set_description(
                        "Preprocessing ({} completed, {} failed)".format(completed, failed)
                    )
    except KeyboardInterrupt:
        exit(-1)

    return filenames


if __name__ == '__main__':
    runx()
