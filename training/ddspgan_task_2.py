import logging
import os
import pathlib
import random
import sys
from typing import Dict

import lightning.pytorch as pl
import matplotlib
import numpy as np
import torch.utils.data
from lightning.pytorch.utilities.rank_zero import rank_zero_debug, rank_zero_info, rank_zero_only
from matplotlib import pyplot as plt
from torch import nn

from torch.utils.data import Dataset
from torchmetrics import Metric, MeanMetric

import utils
from models.ddspgan.ddspgan import DDSPgan
from models.nsf_HiFigan.models import Generator, AttrDict, MultiScaleDiscriminator, MultiPeriodDiscriminator
from modules.loss.HiFiloss import HiFiloss
from modules.loss.ddsploss import ddsploss
from training.base_task_gan import GanBaseTask
# from utils.indexed_datasets import IndexedDataset
from utils.training_utils import (
    DsBatchSampler, DsEvalBatchSampler,
    get_latest_checkpoint_path
)
from utils.wav2mel import PitchAdjustableMelSpectrogram
def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 9),dpi=100)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    return fig

class nsf_HiFigan_dataset(Dataset):

    def __init__(self, config: dict, data_dir, infer=False):
        super().__init__()
        self.config = config

        self.data_dir = data_dir if isinstance(data_dir, pathlib.Path) else pathlib.Path(data_dir)
        with open(self.data_dir, 'r', encoding='utf8') as f:
            fills = f.read().strip().split('\n')
        self.data_index = fills
        self.infer = infer
        self.volume_aug = self.config['volume_aug']
        self.volume_aug_prob = self.config['volume_aug_prob'] if not infer else 0

    def __getitem__(self, index):
        data_path = self.data_index[index]
        data = np.load(data_path)

        return {'f0': data['f0'], 'spectrogram': data['mel'], 'audio': data['audio'], 'uv': data['uv']}

    def __len__(self):
        return len(self.data_index)

    def collater(self, minibatch):
        samples_per_frame = self.config['hop_size']
        if self.infer:
            crop_mel_frames = 0
        else:
            crop_mel_frames = self.config['crop_mel_frames']

        for record in minibatch:

            # Filter out records that aren't long enough.
            if len(record['spectrogram']) < crop_mel_frames:
                del record['spectrogram']
                del record['audio']
                del record['f0']
                del record['uv']
                continue

            start = random.randint(0, record['spectrogram'].shape[0] - 1 - crop_mel_frames)
            end = start + crop_mel_frames
            if self.infer:
                record['spectrogram'] = record['spectrogram'].T
                record['f0'] = record['f0']
                record['uv'] = record['uv']
            else:
                record['spectrogram'] = record['spectrogram'][start:end].T
                record['f0'] = record['f0'][start:end]
                record['uv'] = record['uv'][start:end]
            start *= samples_per_frame
            end *= samples_per_frame
            if self.infer:
                cty = (len(record['spectrogram'].T) * samples_per_frame)
                record['audio'] = record['audio'][:cty]
                record['audio'] = np.pad(record['audio'], (
                    0, (len(record['spectrogram'].T) * samples_per_frame) - len(record['audio'])),
                                         mode='constant')
                pass
            else:
                # record['spectrogram'] = record['spectrogram'][start:end].T
                record['audio'] = record['audio'][start:end]
                record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])),
                                         mode='constant')

        if self.volume_aug:
            for record in minibatch:

                if random.random() < self.volume_aug_prob:
                    audio = record['audio']
                    audio_mel = record['spectrogram']
                    max_amp = float(np.max(np.abs(audio))) + 1e-5
                    max_shift = min(3, np.log(1 / max_amp))
                    log_mel_shift = random.uniform(-3, max_shift)
                    # audio *= (10 ** log_mel_shift)
                    audio *= np.exp(log_mel_shift)
                    audio_mel += log_mel_shift
                    audio_mel = torch.clamp(torch.from_numpy(audio_mel), min=np.log(1e-5)).numpy()
                    record['audio'] = audio
                    record['spectrogram'] = audio_mel

        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])

        spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
        f0 = np.stack([record['f0'] for record in minibatch if 'f0' in record])
        uv = np.stack([record['uv'] for record in minibatch if 'uv' in record])
        return {
            'audio': torch.from_numpy(audio).unsqueeze(1),
            'mel': torch.from_numpy(spectrogram), 'f0': torch.from_numpy(f0), 'uv': torch.from_numpy(uv)
        }

class stftlog:
    def __init__(self,
        n_fft=2048,
        win_length=2048,
        hop_length=512,

        center=False,):
        self.hop_length=hop_length
        self.win_size=win_length
        self.n_fft = n_fft
        self.win_size = win_length
        self.center = center
        self.hann_window = {}
    def exc(self,y):


        hann_window_key = f"{y.device}"
        if hann_window_key not in self.hann_window:
            self.hann_window[hann_window_key] = torch.hann_window(
                self.win_size, device=y.device
            )


        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.win_size - self.hop_length) // 2),
                int((self.win_size - self.hop_length+1) // 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_size,
            window=self.hann_window[hann_window_key],
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()
        return spec

class ddspgan_task(GanBaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.TF = PitchAdjustableMelSpectrogram(        f_min=0,
        f_max=None,
        n_mels=256,)
        self.logged_gt_wav = set()
        self.stft=stftlog()

    def build_dataset(self):

        self.train_dataset = nsf_HiFigan_dataset(config=self.config,
                                                 data_dir=pathlib.Path(self.config['DataIndexPath']) / self.config[
                                                     'train_set_name'])
        self.valid_dataset = nsf_HiFigan_dataset(config=self.config,
                                                 data_dir=pathlib.Path(self.config['DataIndexPath']) / self.config[
                                                     'valid_set_name'], infer=True)
    def build_model(self):
        # cfg=self.config['model_args']

        # cfg.update({'sampling_rate':self.config['audio_sample_rate'],'num_mels':self.config['audio_num_mel_bins'],'hop_size':self.config['hop_size']})
        # h=AttrDict(cfg)
        self.generator=DDSPgan(self.config)
        self.discriminator=nn.ModuleDict({'msd':MultiScaleDiscriminator(), 'mpd':MultiPeriodDiscriminator(periods=self.config['model_args']['discriminator_periods'])})

    def build_losses_and_metrics(self):
        self.mix_loss=ddsploss(self.config)

    def Gforward(self, sample, infer=False):
        """
        steps:
            1. run the full model
            2. calculate losses if not infer
        """
        wav=self.generator(mel=sample['mel'], f0=sample['f0'])
        return {'audio':wav}

    def Dforward(self, Goutput):
        msd_out,msd_feature=self.discriminator['msd'](Goutput)
        mpd_out,mpd_feature=self.discriminator['mpd'](Goutput)
        return (msd_out,msd_feature),(mpd_out,mpd_feature)

    # def _training_step(self, sample, batch_idx):
    #     """
    #     :return: total loss: torch.Tensor, loss_log: dict, other_log: dict
    #
    #     """
    #
    #     log_diet = {}
    #     opt_g, opt_d = self.optimizers()
    #     # forward generator start
    #     Goutput = self.Gforward(sample=sample)  #y_g_hat =Goutput
    #     # forward generator start
    #
    #     #forward discriminator start
    #
    #     Dfake = self.Dforward(Goutput=Goutput['audio'].detach()) #y_g_hat =Goutput
    #     Dtrue = self.Dforward(Goutput=sample['audio']) #y =sample['audio']
    #     Dloss, Dlog = self.mix_loss.Dloss(Dfake=Dfake, Dtrue=Dtrue)
    #     log_diet.update(Dlog)
    #     # forward discriminator end
    #     #opt discriminator start
    #     opt_d.zero_grad()  #clean discriminator grad
    #     self.manual_backward(Dloss)
    #     opt_d.step()
    #     # opt discriminator end
    #     # opt generator start
    #     GDfake = self.Dforward(Goutput=Goutput['audio'])
    #     GDtrue=self.Dforward(Goutput=sample['audio'])
    #     GDloss, GDlog = self.mix_loss.GDloss(GDfake=GDfake,GDtrue=GDtrue)
    #     log_diet.update(GDlog)
    #     Auxloss, Auxlog = self.mix_loss.Auxloss(Goutput=Goutput, sample=sample)
    #
    #     log_diet.update(Auxlog)
    #     Gloss=GDloss + Auxloss
    #
    #     opt_g.zero_grad() #clean generator grad
    #     self.manual_backward(Gloss)
    #     opt_g.step()
    #     # opt generator end
    #     return log_diet

    def _validation_step(self, sample, batch_idx):

        wav=self.Gforward(sample)['audio']

        with torch.no_grad():

            # self.TF = self.TF.cpu()
            # mels = torch.log10(torch.clamp(self.TF(wav.squeeze(0).cpu().float()), min=1e-5))
            # GTmels = torch.log10(torch.clamp(self.TF(sample['audio'].squeeze(0).cpu().float()), min=1e-5))
            stfts=self.stft.exc(wav.squeeze(0).cpu().float())
            Gstfts=self.stft.exc(sample['audio'].squeeze(0).cpu().float())
            Gstfts_log10=torch.log10(torch.clamp(Gstfts, min=1e-7))
            Gstfts_log = torch.log(torch.clamp(Gstfts, min=1e-7))
            stfts_log10=torch.log10(torch.clamp(stfts, min=1e-7))
            stfts_log= torch.log(torch.clamp(stfts, min=1e-7))
            # self.plot_mel(batch_idx, GTmels.transpose(1,2), mels.transpose(1,2), name=f'diffmel_{batch_idx}')
            self.plot_mel(batch_idx, Gstfts_log10.transpose(1,2), stfts_log10.transpose(1,2), name=f'HIFImel_{batch_idx}/log10')
            # self.plot_mel(batch_idx, Gstfts_log.transpose(1, 2), stfts_log.transpose(1, 2), name=f'HIFImel_{batch_idx}/log')
            self.logger.experiment.add_audio(f'diff_{batch_idx}_', wav,
                                             sample_rate=self.config['audio_sample_rate'],
                                             global_step=self.global_step)
            if batch_idx not in self.logged_gt_wav:
                # gt_wav = self.vocoder.spec2wav(gt_mel, f0=f0)
                self.logger.experiment.add_audio(f'gt_{batch_idx}_', sample['audio'],
                                                 sample_rate=self.config['audio_sample_rate'],
                                                 global_step=self.global_step)
                self.logged_gt_wav.add(batch_idx)

        return {'l1loss':nn.L1Loss()(wav, sample['audio'])}, 1

    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        name = f'mel_{batch_idx}' if name is None else name
        vmin = self.config['mel_vmin']
        vmax = self.config['mel_vmax']
        spec_cat = torch.cat([(spec_out - spec).abs() + vmin, spec, spec_out], -1)
        self.logger.experiment.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)

