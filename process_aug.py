import multiprocessing
import pathlib
import random
import time
from concurrent.futures import ProcessPoolExecutor
from threading import Thread

import click
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from utils.config_utils import read_full_config, print_config
from multiprocessing import Process, Queue

from utils.wav2F0 import get_pitch_parselmouth
from utils.wav2mel import PitchAdjustableMelSpectrogram


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def wav_aug(wav,n=1,sr=44100):
    nsr=int((1/n)*sr)
    return torchaudio.transforms.Resample(orig_freq=sr, new_freq=nsr)(wav)


def wav2spec(warp):
    torch.set_num_threads(1)
    pathslist, Q, config = warp
    Q: Queue
    # pathlib.Path.relative_to
    mel_spec_transform = PitchAdjustableMelSpectrogram(sample_rate=config['audio_sample_rate'],
                                                       n_fft=config['fft_size'],
                                                       win_length=config['win_size'],
                                                       hop_length=config['hop_size'],
                                                       f_min=config['fmin'],
                                                       f_max=config['fmax'],
                                                       n_mels=config['audio_num_mel_bins'], )
    try:
        audio, sr = torchaudio.load(pathslist[0])
        if sr != config['audio_sample_rate']:
            print('error:', str(pathslist[0]))
            return None

        mel = dynamic_range_compression_torch(mel_spec_transform(audio))
        f0, uv = get_pitch_parselmouth(audio.numpy()[0], hparams=config,
                                       interp_uv=True, length=len(mel[0].T))

        if f0 is None:
            print('error:', str(pathslist[0]))
            return None
        pathslist[1].mkdir(parents=True, exist_ok=True)
        np.savez(pathslist[2], audio=audio[0].numpy(), mel=mel[0].T, f0=f0, uv=uv)

        while True:
            if not Q.full():
                Q.put(str(pathslist[2]))
                break
        for i  in range(int(config['aug_num'])):
            audiox=wav_aug(audio,random.uniform(config['aug_min'],config['aug_max']),sr=config['audio_sample_rate'])
            mel = dynamic_range_compression_torch(mel_spec_transform(audiox))
            f0, uv = get_pitch_parselmouth(audiox.numpy()[0], hparams=config,
                                           interp_uv=True, length=len(mel[0].T))
            np.savez(str(pathslist[2])[:-4]+f'_{str(i)}.npz', audio=audiox[0].numpy(), mel=mel[0].T, f0=f0, uv=uv)
            while True:
                if not Q.full():
                    Q.put(str(str(pathslist[2])[:-4]+f'_{str(i)}.npz'))
                    break


    except:
        print('error:', str(pathslist[0]))
        return None


def run_worch(path_list, num_cpu, Q: Queue, config):
    filxlist = []
    for i in tqdm(path_list):
        filxlist.append((i, Q, config))

    with ProcessPoolExecutor(max_workers=num_cpu) as executor:
        list(tqdm(executor.map(wav2spec, filxlist), desc='Preprocessing', total=len(filxlist)))

    while True:
        if not Q.full():
            Q.put('????task_end?????')
            break
        time.sleep(0.1)


@click.command(help='')
@click.option('--config', required=True, metavar='FILE', help='Path to the configuration file')
@click.option('--num_cpu', required=False, metavar='DIR2', help='Directory to save the experiment')
@click.option('--strx', required=False, metavar='DIR4', help='Directory to save the experiment')  # 1 代表开   0代表关
def runx(config, num_cpu, strx):
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
    crash_list = []
    for inpath, outpath in tqdm(zip(in_path_list, out_path_list)):
        outlist = preprocess(config=config, input_path=inpath, output_path=outpath, num_cpu=num_cpu, st_path=strx)
        crash_list.append(outlist)
    outp = pathlib.Path(config['DataIndexPath'])
    assert not outp.exists() or outp.is_dir(), f'Path \'{outp}\' is not a directory.'
    outp.mkdir(parents=True, exist_ok=True)
    trainp = config['train_set_name']
    valp = config['valid_set_name']
    valn = config['val_num']
    outstr = ''
    for i in crash_list:
        outstr = outstr + i
    cpx = outstr.split('\n', valn)
    trainx = cpx[-1]
    valx = cpx[:-1]
    val1 = ''
    for i in valx:
        val1 = val1 + i + '\n'
    with open(outp / trainp, 'w', encoding='utf8') as f:
        f.write(trainx)
    with open(outp / valp, 'w', encoding='utf8') as f:
        f.write(val1)


def preprocess(config, input_path, output_path, num_cpu, st_path):
    # Q=Queue(1000)
    m = multiprocessing.Manager()
    Q = m.Queue(1000)

    if st_path:
        input_path = pathlib.Path(input_path).resolve()
        output_path = pathlib.Path(output_path).resolve()
    else:
        input_path = pathlib.Path(input_path)
        output_path = pathlib.Path(output_path)

    # work_dir = work_dir / crash_data
    # assert not crash_data.exists() or crash_data.is_dir(), f'Path \'{crash_data}\' is not a directory.'
    # crash_data.mkdir(parents=True, exist_ok=True)
    assert not output_path.exists() or output_path.is_dir(), f'Path \'{output_path}\' is not a directory.'
    output_path.mkdir(parents=True, exist_ok=True)

    if num_cpu is None:
        num_cpu = 5
    else:
        num_cpu = int(num_cpu)
    maplist = []
    wav_list = list(input_path.rglob('*.wav'))
    # print(wav_list)
    for i in tqdm(wav_list):
        outpath = output_path / i.relative_to(input_path).parent
        outname = f'{str(i.name)}.npz'

        # print(str(i),str(outpath),str(outname),str(outpath/outname))
        maplist.append((i, outpath, outpath / outname))

    t1 = Thread(target=run_worch, args=(maplist, num_cpu, Q, config))

    t1.start()

    indexlist = ''

    while True:
        if not Q.empty():
            value = Q.get()
            if value == '????task_end?????':
                break
            indexlist = indexlist + value + '\n'

    # if output_name is not None:
    #     opf=crash_data/f'{output_name}.vcidx_crh'
    # else:
    #     opf=crash_data /f'{str(time.strftime("%Y%m%d%H%M%S", time.localtime()))}.vcidx_crh'
    # with open(opf,'w',encoding='utf-8') as f:
    #     f.write(indexlist)
    return indexlist


if __name__ == '__main__':
    runx()
    # preprocess('configs/base_ddspgan.yaml',r'testw','datatt/',None,None,None)
    # preprocess()
