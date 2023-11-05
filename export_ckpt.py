import importlib
import pathlib

import click
import torch
from tqdm import tqdm

from utils import get_latest_checkpoint_path
from utils.config_utils import read_full_config




@click.command(help='Train a SOME model')
@click.option('--exp_name', required=True, metavar='EXP', help='Name of the experiment')
@click.option('--save_path', required=True, metavar='EXP', help='Name of the experiment')
@click.option('--work_dir', required=False, metavar='DIR', help='Directory to save the experiment')
def train( exp_name,save_path, work_dir):

    # print_config(config)
    if work_dir is None:
        work_dir = pathlib.Path(__file__).parent / 'experiments'
    else:
        work_dir = pathlib.Path(work_dir)
    work_dir = work_dir / exp_name
    assert not work_dir.exists() or work_dir.is_dir(), f'Path \'{work_dir}\' is not a directory.'
    work_dir.mkdir(parents=True, exist_ok=True)



    ckp = {}

    aaa2x = torch.load(get_latest_checkpoint_path(work_dir))['state_dict']
    for i in tqdm(aaa2x):
        i: str
        if 'generator.' in i:
            # print(i)
            ckp[i.replace('generator.', '')] = aaa2x[i]

    torch.save({'generator': ckp}, save_path)






if __name__ == '__main__':
    train()