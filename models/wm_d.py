import torch
import torch.nn as nn
import torchaudio
from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker


class PerthDetectorDiscriminator(nn.Module):
    def __init__(self, audio_sr: int, device: str = "cpu", loss_tau: float = 0.75, loss_alpha: float = 0.1, **perth_kwargs):
        super().__init__()
        self.watermarker = PerthImplicitWatermarker(device=device, **perth_kwargs)
        
        self.perth_net = self.watermarker.perth_net
        self.audio_sr = audio_sr
        self.perth_sr = self.perth_net.hp.sample_rate
        
        self.perth_net.eval()
        for param in self.perth_net.parameters():
            param.requires_grad = False
        if self.audio_sr != self.perth_sr:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.audio_sr,
                new_freq=self.perth_sr,
                resampling_method="sinc_interp_hann"
            )
        else:
            self.resampler = nn.Identity()

        self.tau = loss_tau
        self.alpha = loss_alpha

    def forward(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_tensor (torch.Tensor): shape (B, 1, T) or (B, T).
        Returns:
            torch.Tensor: shape (B,).
        """
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(1)
        resampled_audio = self.resampler(audio_tensor)
        magspec, _phase = self.perth_net.ap.signal_to_magphase(resampled_audio)
        wmark_pred_vector = self.perth_net.decoder(magspec)
        wmark_pred_vector = wmark_pred_vector.clamp(0.0, 1.0)
        if wmark_pred_vector.dim() > 1:
            confidence = torch.mean(wmark_pred_vector, dim=list(range(1, wmark_pred_vector.dim())))
        else:
            confidence = wmark_pred_vector

        return confidence

    def loss(self, confidence_scores: torch.Tensor) -> torch.Tensor:
        is_below_tau = (confidence_scores < self.tau)
        strong_term = (self.tau - confidence_scores).pow(2)
        weak_term = self.alpha * (1.0 - confidence_scores).pow(2)
        batch_loss = torch.where(is_below_tau, strong_term, weak_term)
        
        return batch_loss.mean()
