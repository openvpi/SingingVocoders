import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.ddsp.loss import RSSLoss
from modules.loss.stft_loss import warp_stft
from utils.wav2mel import PitchAdjustableMelSpectrogram
def kl_loss(logs, m):
  kl = 0.5 * (m**2 + torch.exp(logs) - logs - 1).sum(dim=1)
  kl = torch.mean(kl)
  return kl

class HiFiloss(nn.Module):
    def __init__(self,config:dict):
        super().__init__()
        self.mel=PitchAdjustableMelSpectrogram( sample_rate=config['audio_sample_rate'],
        n_fft=config['fft_size'],
        win_length=config['win_size'],
        hop_length=config['hop_size'],
        f_min=config['fmin'],
        f_max=config['fmax_for_loss'],
        n_mels=config['audio_num_mel_bins'],)
        self.L1loss=nn.L1Loss()
        self.labauxloss=config.get('lab_aux_loss',45)
        self.lab_kl_loss=config.get('lab_kl_loss',0.02)
        self.lab_wav_loss=config.get('lab_wav_loss',5)
        self.stft = warp_stft({'fft_sizes': config['loss_fft_sizes'], 'hop_sizes': config['loss_hop_sizes'],
                               'win_lengths': config['loss_win_lengths']})
        if config.get('use_rss_loss',False):

            self.loss_rss_func = RSSLoss(fft_min=config['RSSloss_stftmin'], fft_max=config['RSSloss_stftmax'], n_scale=config['RSSloss_stftnum'], device='cuda')
        self.useRSS=config.get('use_rss_loss',False)

    def discriminator_loss(self,disc_real_outputs, disc_generated_outputs):
        loss = 0
        rlosses=0
        glosses=0
        r_losses = []
        g_losses = []

        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss += r_loss + g_loss
            rlosses+=r_loss.item()
            glosses +=g_loss.item()
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, rlosses,glosses,r_losses, g_losses


    def Dloss(self,Dfake, Dtrue):

        (Fmsd_out, _), (Fmpd_out, _)=Dfake
        (Tmsd_out, _), (Tmpd_out, _)=Dtrue
        msdloss, msdrlosses, msdglosses, _, _=self.discriminator_loss(Tmsd_out,Fmsd_out)
        mpdloss, mpdrlosses, mpdglosses, _, _ = self.discriminator_loss(Tmpd_out, Fmpd_out)
        loss=msdloss+mpdloss
        return loss,{'DmsdlossF':msdglosses,'DmsdlossT':msdrlosses,'DmpdlossT':mpdrlosses,'DmpdlossF':mpdglosses}

    def feature_loss(self,fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2

    def GDloss(self,GDfake,GDtrue):
        loss = 0
        gen_losses = []
        msd_losses=0
        mpd_losses = 0
        (msd_out, Fmsd_feature), (mpd_out, Fmpd_feature)=GDfake
        (_, Tmsd_feature), (_, Tmpd_feature) = GDtrue
        for dg in msd_out:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l.item())
            # loss += l
            msd_losses=l+msd_losses

        for dg in mpd_out:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l.item())
            # loss += l
            mpd_losses=l+mpd_losses

        msd_feature_loss=self.feature_loss(Tmsd_feature,Fmsd_feature)
        mpd_feature_loss = self.feature_loss(Tmpd_feature, Fmpd_feature)
        # loss +=msd_feature_loss
        # loss +=mpd_feature_loss
        loss= msd_feature_loss+mpd_feature_loss+mpd_losses+msd_losses
        # (msd_losses, mpd_losses), (msd_feature_loss, mpd_feature_loss), gen_losses
        return loss, {'Gmsdloss':msd_losses,'Gmpdloss':mpd_losses,'Gmsd_feature_loss':msd_feature_loss,'Gmpd_feature_loss':mpd_feature_loss}

    # def Auxloss(self,Goutput, sample):
    #
    #     Gmel=self.mel.dynamic_range_compression_torch(self.mel(Goutput['audio'].squeeze(1)))
    #     # Rmel=sample['mel']
    #     Rmel = self.mel.dynamic_range_compression_torch(self.mel(sample['audio'].squeeze(1)))
    #     loss=self.L1loss(Gmel, Rmel)*self.labauxloss
    #     return loss,{'auxloss':loss}

    def Auxloss(self,Goutput, sample):

        # Gmel=self.mel.dynamic_range_compression_torch(self.mel(Goutput['audio'].squeeze(1)))
        # # Rmel=sample['mel']
        # Rmel = self.mel.dynamic_range_compression_torch(self.mel(sample['audio'].squeeze(1)))
        sc_loss, mag_loss=self.stft.stft(Goutput['audio'].squeeze(1), sample['audio'].squeeze(1))
        klloss=kl_loss(logs=Goutput['lossxxs'][2],m=Goutput['lossxxs'][1])
        wavloss= F.l1_loss(Goutput['audio'], sample['audio'])
        if self.useRSS:
            RSSLoss=self.loss_rss_func(Goutput['audio'].squeeze(1), sample['audio'].squeeze(1))
        else:
            RSSLoss=0

        loss=(sc_loss+ mag_loss+RSSLoss)*self.labauxloss +klloss*self.lab_kl_loss +wavloss*self.lab_wav_loss

        return loss,{'auxloss':loss,'auxloss_sc_loss':sc_loss,'auxloss_mag_loss':mag_loss,'klloss':klloss,'wavloss':wavloss,'RSSLoss':RSSLoss}
    #
