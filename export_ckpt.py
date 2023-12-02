import pathlib

import click
import torch
from tqdm import tqdm

from utils import get_latest_checkpoint_path


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

    ckp = {}

    aaa2x = torch.load(ckpt_path)['state_dict']
    for i in tqdm(aaa2x):
        i: str
        if 'generator.' in i:
            # print(i)
            ckp[i.replace('generator.', '')] = aaa2x[i]

    torch.save({'generator': ckp}, save_path)


if __name__ == '__main__':
    export()
