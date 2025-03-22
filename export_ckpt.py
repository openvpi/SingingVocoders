import pathlib
import json
import click
import torch
from tqdm import tqdm

from utils import get_latest_checkpoint_path
from utils.config_utils import read_full_config, print_config


@click.command(help='')
@click.option('--exp_name', required=False, metavar='EXP', help='Name of the experiment')
@click.option('--ckpt_path', required=False, metavar='FILE', help='Path to the checkpoint file')
@click.option('--save_path', required=True, metavar='FILE', help='Path to save the exported checkpoint')
@click.option('--work_dir', required=False, metavar='DIR', help='Working directory containing the experiments')
def export(exp_name, ckpt_path, save_path, work_dir):
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
    
    ckpt = {}
    temp_dict = torch.load(ckpt_path)['state_dict']
    for i in tqdm(temp_dict):
        i: str
        if 'generator.' in i:
            # print(i)
            ckpt[i.replace('generator.', '')] = temp_dict[i]
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'generator': ckpt}, save_path)
    print("Export checkpoint file successfully: ", save_path)
    
    config_file = pathlib.Path(ckpt_path).with_name('config.yaml')
    config = read_full_config(config_file)
    new_config_file = pathlib.Path(save_path).with_name('config.json')
    with open(new_config_file, 'w') as json_file:
        new_config = config['model_args']
        new_config['sampling_rate'] = config['audio_sample_rate']
        new_config['num_mels'] = config['audio_num_mel_bins']
        new_config['hop_size'] = config['hop_size']
        new_config['n_fft'] = config['fft_size']
        new_config['win_size'] = config['win_size']
        new_config['fmin'] = config['fmin']
        new_config['fmax'] = config['fmax']
        if 'pc_aug' not in config.keys():
            new_config['pc_aug'] = False
        else:
            new_config['pc_aug'] = config['pc_aug'] 
        if 'mini_nsf' not in new_config.keys():
            new_config['mini_nsf'] = False
        if 'noise_sigma' not in new_config.keys():
            new_config['noise_sigma'] = 0.0
        
        json_file.write(json.dumps(new_config, indent=1))
        print("Export configuration file successfully: ", new_config_file)


if __name__ == '__main__':
    export()
