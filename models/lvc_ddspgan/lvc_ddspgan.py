import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d

from modules.lvc_ddsp.vocoder import CombSub, Sins


class DDSPgan(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['model_args']['type'] == 'CombSub':
            self.ddsp = CombSub(
                sampling_rate=config['audio_sample_rate'],
                block_size=config['hop_size'],
                win_length=config['win_size'],
                n_mag_harmonic=config['model_args']['n_mag_harmonic'],
                n_mag_noise=config['model_args']['n_mag_noise'],
                n_mels=config['audio_num_mel_bins'], lvc_noise_channels=config['model_args']['lvc_noise_channels'],
                lvc_latent_channels=config['model_args']['lvc_latent_channels'], lvc_block_nums=config['model_args']['lvc_block_nums'],
                lvc_layers_each_block=config['model_args']['lvc_layers_each_block'],
                lvc_kernel_size=config['model_args']['lvc_kernel_size'],
                kpnet_hidden_channels=config['model_args']['kpnet_hidden_channels'],
                kpnet_conv_size=config['model_args']['kpnet_conv_size'],
                dropout=config['model_args']['dropout'], up_condition=config['model_args']['up_condition'])
        elif config['model_args']['type'] == 'Sins':
            self.ddsp = Sins(
                sampling_rate=config['audio_sample_rate'],
                block_size=config['hop_size'],
                win_length=config['win_size'],
                n_harmonics=config['model_args']['n_harmonics'],
                n_mag_noise=config['model_args']['n_mag_noise'],
                n_mels=config['audio_num_mel_bins'],lvc_noise_channels=config['model_args']['lvc_noise_channels'],
                lvc_latent_channels=config['model_args']['lvc_latent_channels'], lvc_block_nums=config['model_args']['lvc_block_nums'],
                lvc_layers_each_block=config['model_args']['lvc_layers_each_block'],
                lvc_kernel_size=config['model_args']['lvc_kernel_size'],
                kpnet_hidden_channels=config['model_args']['kpnet_hidden_channels'],
                kpnet_conv_size=config['model_args']['kpnet_conv_size'],
                dropout=config['model_args']['dropout'], up_condition=config['model_args']['up_condition'])

    def forward(self, mel, f0, infer=False):
        signal, _, (s_h, s_n) = self.ddsp(mel.transpose(1, 2), torch.unsqueeze(f0, dim=-1), infer=infer)
        return signal.unsqueeze(1)
