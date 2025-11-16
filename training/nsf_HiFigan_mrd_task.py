import pathlib
import random

import numpy as np
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset

from models.nsf_HiFigan.models import Generator, AttrDict, MultiPeriodDiscriminator
from modules.univ_D.discriminator import MultiResSpecDiscriminator
from modules.loss.univloss import univloss
from training.base_task_gan import GanBaseTask
from utils.wav2F0 import PITCH_EXTRACTORS_ID_TO_NAME, get_pitch
from utils.wav2mel import PitchAdjustableMelSpectrogram


def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 9), dpi=100)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    return fig


def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def wav_aug(wav, hop_size, speed=1):
    orig_freq = int(np.round(hop_size * speed))
    new_freq = hop_size
    resample = torchaudio.transforms.Resample(
        orig_freq=orig_freq,
        new_freq=new_freq,
        lowpass_filter_width=128
    )
    wav_resampled = resample(wav)
    del resample
    return wav_resampled


def get_max_f0_from_config(config: dict):
    model_args = config['model_args']
    if model_args['mini_nsf']:
        source_sr = config['audio_sample_rate'] / int(np.prod(model_args['upsample_rates'][2:]))
    else:
        source_sr = config['audio_sample_rate']
    max_f0 = source_sr / 2
    return max_f0


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
        self.key_aug = self.config.get('key_aug', False)
        self.key_aug_prob = self.config.get('key_aug_prob', 0.5)
        if self.key_aug:
            self.mel_spec_transform = PitchAdjustableMelSpectrogram(
                sample_rate=config['audio_sample_rate'],
                n_fft=config['fft_size'],
                win_length=config['win_size'],
                hop_length=config['hop_size'],
                f_min=config['fmin'],
                f_max=config['fmax'],
                n_mels=config['audio_num_mel_bins'],
            )
        self.max_f0 = get_max_f0_from_config(config)

    def __getitem__(self, index):
        sample = self.get_data(index)
        if sample['f0'].max() >= self.max_f0:
            return self.__getitem__(random.randint(0, len(self) - 1))
        return sample

    def __len__(self):
        return len(self.data_index)

    def get_data(self, index):
        data_path = pathlib.Path(self.data_index[index])
        data = np.load(data_path)
        pe_name = PITCH_EXTRACTORS_ID_TO_NAME[int(data['pe'])]
        if self.infer or not self.key_aug or random.random() > self.key_aug_prob:
            return {'f0': data['f0'], 'spectrogram': data['mel'], 'audio': data['audio']}
        else:
            speed = random.uniform(self.config['aug_min'], self.config['aug_max'])
            crop_mel_frames = int(np.ceil((self.config['crop_mel_frames'] + 4) * speed))
            samples_per_frame = self.config['hop_size']
            crop_wav_samples = crop_mel_frames * samples_per_frame
            if crop_wav_samples >= data['audio'].shape[0]:
                return {'f0': data['f0'], 'spectrogram': data['mel'], 'audio': data['audio']}
            start = random.randint(0, data['audio'].shape[0] - 1 - crop_wav_samples)
            end = start + crop_wav_samples
            audio = data['audio'][start:end]
            audio_aug = wav_aug(torch.from_numpy(audio), self.config["hop_size"], speed=speed)
            mel_aug = dynamic_range_compression_torch(self.mel_spec_transform(audio_aug[None, :]))
            f0, uv = get_pitch(
                pe_name, audio, length=mel_aug.shape[-1], hparams=self.config,
                speed=speed, interp_uv=True
            )
            if f0 is None:
                return {'f0': data['f0'], 'spectrogram': data['mel'], 'audio': data['audio']}
            audio_aug = audio_aug[2 * samples_per_frame: -2 * samples_per_frame].numpy()
            mel_aug = mel_aug[0, :, 2:-2].T.numpy()
            f0_aug = f0[2:-2] * speed
            return {'f0': f0_aug, 'spectrogram': mel_aug, 'audio': audio_aug}

    def collater(self, minibatch):
        samples_per_frame = self.config['hop_size']
        if self.infer:
            crop_mel_frames = 0
        else:
            crop_mel_frames = self.config['crop_mel_frames']

        for record in minibatch:

            # Filter out records that aren't long enough.
            if record['spectrogram'].shape[0] < crop_mel_frames:
                del record['spectrogram']
                del record['audio']
                del record['f0']
                continue
            elif record['spectrogram'].shape[0] == crop_mel_frames:
                start = 0
            else:
                start = random.randint(0, record['spectrogram'].shape[0] - 1 - crop_mel_frames)
            end = start + crop_mel_frames
            if self.infer:
                record['spectrogram'] = record['spectrogram'].T
                record['f0'] = record['f0']
            else:
                record['spectrogram'] = record['spectrogram'][start:end].T
                record['f0'] = record['f0'][start:end]
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
                if record.get('audio') is None:
                    # del record['spectrogram']
                    # del record['audio']
                    # del record['pemel']
                    # del record['uv']
                    continue
                audio = record['audio']
                audio_mel = record['spectrogram']

                if random.random() < self.volume_aug_prob:
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

        return {
            'audio': torch.from_numpy(audio).unsqueeze(1),
            'mel': torch.from_numpy(spectrogram), 'f0': torch.from_numpy(f0),
        }


class stftlog:
    def __init__(self,
                 n_fft=2048,
                 win_length=2048,
                 hop_length=512,
                 center=False, ):
        self.hop_length = hop_length
        self.win_size = win_length
        self.n_fft = n_fft
        self.win_size = win_length
        self.center = center
        self.hann_window = {}

    def exc(self, y):
        hann_window_key = f"{y.device}"
        if hann_window_key not in self.hann_window:
            self.hann_window[hann_window_key] = torch.hann_window(
                self.win_size, device=y.device
            )
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.win_size - self.hop_length) // 2),
                int((self.win_size - self.hop_length + 1) // 2),
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


class nsf_HiFigan(GanBaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.TF = PitchAdjustableMelSpectrogram(
            sample_rate=config['audio_sample_rate'],
            n_fft=config['fft_size'],
            win_length=config['win_size'],
            hop_length=config['hop_size'],
            f_min=config['fmin'],
            f_max=config['fmax'],
            n_mels=config['audio_num_mel_bins'],
        )
        self.pc_aug = self.config.get('pc_aug', False)
        self.pc_aug_rate = self.config.get('pc_aug_rate', 0.5)
        self.pc_aug_key = self.config.get('pc_aug_key', 5)
        self.logged_gt_wav = set()
        self.stft = stftlog()
        self.max_f0 = get_max_f0_from_config(config)

    def build_dataset(self):

        self.train_dataset = nsf_HiFigan_dataset(config=self.config,
                                                 data_dir=pathlib.Path(self.config['DataIndexPath']) / self.config[
                                                     'train_set_name'])
        self.valid_dataset = nsf_HiFigan_dataset(config=self.config,
                                                 data_dir=pathlib.Path(self.config['DataIndexPath']) / self.config[
                                                     'valid_set_name'], infer=True)

    def build_model(self):
        cfg = self.config['model_args']
        cfg.update({
            'sampling_rate': self.config['audio_sample_rate'],
            'num_mels': self.config['audio_num_mel_bins'],
            'hop_size': self.config['hop_size']})
        if 'mini_nsf' not in cfg.keys():
            cfg.update({'mini_nsf': False})
        if 'noise_sigma' not in cfg.keys():
            cfg.update({'noise_sigma': 0.0})
        h = AttrDict(cfg)
        self.generator = Generator(h)
        self.discriminator = nn.ModuleDict({
            'mrd':MultiResSpecDiscriminator(fft_sizes=self.config['model_args'].get('mrd_fft_sizes',[1024, 2048, 512]),
                 hop_sizes=self.config['model_args'].get('mrd_hop_sizes',[120, 240, 50]),
                 win_lengths= self.config['model_args'].get('mrd_win_lengths',[600, 1200, 240]),),
            'mpd': MultiPeriodDiscriminator(periods=cfg['discriminator_periods'])
        })

    def build_losses_and_metrics(self):
        self.mix_loss = univloss(self.config)

    def Gforward(self, sample):
        """
        steps:
            1. run the full model
            2. calculate losses if not infer
        """
        wav = self.generator(x=sample['mel'], f0=sample['f0'])
        return {'audio': wav}

    def G2forward(self, sample, pc_aug_num):
        if pc_aug_num <= 0:
            raise ValueError('pc_aug_num should be greater than 0')
        f0 = sample['f0']
        key_c = (2 * torch.rand(pc_aug_num, device=f0.device).unsqueeze(-1) - 1) * self.pc_aug_key
        # (1) c + (-c) = 0
        f0_shift_c = torch.clip(f0[:pc_aug_num] * 2 ** (key_c / 12), max=self.max_f0)
        wav_mixed = self.generator(x=sample['mel'], f0=torch.cat((f0_shift_c, f0[pc_aug_num:]), dim=0))
        wav_shift_c, wav_shift_0 = wav_mixed[:pc_aug_num], wav_mixed[pc_aug_num:]
        mel_shift_c = self.TF.dynamic_range_compression_torch(self.TF(wav_shift_c.squeeze(1)))
        wav_shift_back = self.generator(x=mel_shift_c, f0=f0[:pc_aug_num])
        wav_ret = torch.cat((wav_shift_back, wav_shift_0), dim=0)
        # (2) a + b = c
        key_a_max = self.pc_aug_key + key_c.clamp(max=0)
        key_a_min = -self.pc_aug_key + key_c.clamp(min=0)
        key_a = (key_a_max - key_a_min) * torch.rand(pc_aug_num, device=f0.device).unsqueeze(-1) + key_a_min
        f0_shift_a = torch.clip(f0[:pc_aug_num] * 2 ** (key_a / 12), max=self.max_f0)
        wav_shift_a = self.generator(x=sample['mel'][:pc_aug_num], f0=f0_shift_a)
        mel_shift_a = self.TF.dynamic_range_compression_torch(self.TF(wav_shift_a.squeeze(1)))
        wav_shift_ab = self.generator(x=mel_shift_a, f0=f0_shift_c)

        return {
            'audio': wav_ret,
            'audio_shift_c': wav_shift_c,
            'audio_shift_a': wav_shift_a,
            'audio_shift_ab': wav_shift_ab
        }

    def Dforward(self, Goutput):
        mrd_out, mrd_feature = self.discriminator['mrd'](Goutput)
        mpd_out, mpd_feature = self.discriminator['mpd'](Goutput)
        return (mrd_out, mrd_feature), (mpd_out, mpd_feature)

    def _training_step(self, sample, batch_idx):
        """
        :return: total loss: torch.Tensor, loss_log: dict, other_log: dict

        """

        log_dict = {}
        opt_g, opt_d = self.optimizers()
        
        # forward generator start
        pc_aug_num = int(np.ceil(sample['audio'].shape[0] * self.pc_aug_rate))
        pc_aug = self.pc_aug and pc_aug_num > 0
        if pc_aug:
            if not self.generator.mini_nsf:
                raise ValueError("PC augmentation is only available for generator with MiniNSF module.")
            Goutput = self.G2forward(sample=sample, pc_aug_num=pc_aug_num)
            audio_fake = torch.cat((
                Goutput['audio'],
                Goutput['audio_shift_c'],
                Goutput['audio_shift_a'],
                Goutput['audio_shift_ab']
            ), dim=0)
        else:
            Goutput = self.Gforward(sample=sample)
            audio_fake = Goutput['audio']
        # forward generator end
        
        # enable grad for discriminator's parameters
        for D in self.discriminator.values():
            for p in D.parameters():
                p.requires_grad = True
                
        # opt discriminator start
        Dfake = self.Dforward(Goutput=audio_fake.detach())  # y_g_hat =Goutput
        Dtrue = self.Dforward(Goutput=sample['audio'])  # y =sample['audio']
        Dloss, Dlog = self.mix_loss.Dloss(Dfake=Dfake, Dtrue=Dtrue)
        log_dict.update(Dlog)

        opt_d.zero_grad()  # clean discriminator grad
        self.manual_backward(Dloss)
        opt_d.step()
        # opt discriminator end

        # disable grad for discriminator's parameters
        for D in self.discriminator.values():
            for p in D.parameters():
                p.requires_grad = False
                
        # opt generator start
        GDfake = self.Dforward(Goutput=audio_fake)
        GDtrue = self.Dforward(Goutput=sample['audio'])
        GDloss, GDlog = self.mix_loss.GDloss(GDfake=GDfake, GDtrue=GDtrue)
        log_dict.update(GDlog)
        if pc_aug:
            pc_wav_loss = F.l1_loss(Goutput['audio_shift_ab'], Goutput['audio_shift_c']) * 30
            sample = {'audio': torch.cat((sample['audio'], Goutput['audio_shift_c']), dim=0)}
            Goutput = {'audio': torch.cat((Goutput['audio'], Goutput['audio_shift_ab']), dim=0)}
            log_dict['pc_wav_loss'] = pc_wav_loss
        else:
            pc_wav_loss = 0
        spec_loss, Auxlog = self.mix_loss.Auxloss(Goutput=Goutput, sample=sample)
        Auxloss = spec_loss + pc_wav_loss
        log_dict.update(Auxlog)
        Gloss = GDloss + Auxloss

        opt_g.zero_grad()  # clean generator grad
        self.manual_backward(Gloss)
        opt_g.step()
        # opt generator end
                
        return log_dict

    def _validation_step(self, sample, batch_idx):

        wav = self.Gforward(sample)['audio']

        with torch.no_grad():

            stfts = self.stft.exc(wav.squeeze(0).cpu().float())
            Gstfts = self.stft.exc(sample['audio'].squeeze(0).cpu().float())

            stfts_log10 = torch.log10(torch.clamp(stfts, min=1e-7))
            Gstfts_log10 = torch.log10(torch.clamp(Gstfts, min=1e-7))

            if self.global_rank == 0:
                self.plot_mel(batch_idx, Gstfts_log10.transpose(1, 2), stfts_log10.transpose(1, 2),
                              name=f'log10stft_{batch_idx}')
                self.logger.experiment.add_audio(f'HIFI_{batch_idx}_', wav,
                                                 sample_rate=self.config['audio_sample_rate'],
                                                 global_step=self.global_step)
                if batch_idx not in self.logged_gt_wav:
                    self.logger.experiment.add_audio(f'gt_{batch_idx}_', sample['audio'],
                                                     sample_rate=self.config['audio_sample_rate'],
                                                     global_step=self.global_step)
                    self.logged_gt_wav.add(batch_idx)

        return {'stft_loss': nn.L1Loss()(Gstfts_log10, stfts_log10)}, 1

    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        name = f'mel_{batch_idx}' if name is None else name
        vmin = self.config['mel_vmin']
        vmax = self.config['mel_vmax']
        spec_cat = torch.cat([(spec_out - spec).abs() + vmin, spec, spec_out], -1)
        self.logger.experiment.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)
