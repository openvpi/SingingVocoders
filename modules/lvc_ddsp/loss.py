import numpy as np

import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F
from .core import upsample

class HybridLoss(nn.Module):
    def __init__(self, block_size, fft_min, fft_max, n_scale, lambda_uv, device):
        super().__init__()
        self.loss_rss_func = RSSLoss(fft_min, fft_max, n_scale, device = device)
        self.loss_uv_func = UVLoss(block_size)
        self.lambda_uv = lambda_uv
        
    def forward(self, signal, s_h, x_true, uv_true, detach_uv=False, uv_tolerance=0.05):
        loss_rss = self.loss_rss_func(signal, x_true)
        loss_uv = self.loss_uv_func(signal, s_h, uv_true)
        if detach_uv or loss_uv < uv_tolerance:
            loss_uv = loss_uv.detach()
        loss = loss_rss + self.lambda_uv * loss_uv
        return loss, (loss_rss.float().item(), loss_uv.float().item())

class UVLoss(nn.Module):
    def __init__(self, block_size, eps = 1e-5):
        super().__init__()
        self.block_size = block_size
        self.eps = eps
        
    def forward(self, signal, s_h, uv_true):
        uv_mask = upsample(uv_true.unsqueeze(-1), self.block_size).squeeze(-1)
        loss = torch.mean(torch.linalg.norm(s_h * uv_mask, dim = 1) / (torch.linalg.norm(signal * uv_mask , dim = 1) + self.eps))
        return loss
        
class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, overlap=0, eps=1e-7):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=1, normalized=True, center=False)
        
    def forward(self, x_true, x_pred):
        S_true = self.spec(x_true) + self.eps
        S_pred = self.spec(x_pred) + self.eps
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        log_term = F.l1_loss(S_true.log(), S_pred.log())

        loss = converge_term + self.alpha * log_term
        return loss
        

class MSSLoss(nn.Module):
    """
    Multi-scale Spectral Loss.
    Usage ::
    mssloss = MSSLoss([2048, 1024, 512, 256], alpha=1.0, overlap=0.75)
    mssloss(y_pred, y_gt)
    input(y_pred, y_gt) : two of torch.tensor w/ shape(batch, 1d-wave)
    output(loss) : torch.tensor(scalar)

    48k: n_ffts=[2048, 1024, 512, 256]
    24k: n_ffts=[1024, 512, 256, 128]
    """

    def __init__(self, n_ffts, alpha=1.0, overlap=0.75, eps=1e-7):
        super().__init__()
        self.losses = nn.ModuleList([SSSLoss(n_fft, alpha, overlap, eps) for n_fft in n_ffts])
        
    def forward(self, x_pred, x_true):
        x_pred = x_pred[..., :x_true.shape[-1]]
        value = 0.
        for loss in self.losses:
            value += loss(x_true, x_pred)
        return value

class RSSLoss(nn.Module):
    '''
    Random-scale Spectral Loss.
    '''
    
    def __init__(self, fft_min, fft_max, n_scale, alpha=1.0, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.fft_min = fft_min
        self.fft_max = fft_max
        self.n_scale = n_scale
        self.lossdict = {}
        for n_fft in range(fft_min, fft_max):
            self.lossdict[n_fft] = SSSLoss(n_fft, alpha, overlap, eps).to(device)
        
    def forward(self, x_pred, x_true):
        value = 0.
        n_ffts = torch.randint(self.fft_min, self.fft_max, (self.n_scale,))
        for n_fft in n_ffts:
            loss_func = self.lossdict[int(n_fft)]
            value += loss_func(x_true, x_pred)
        return value / self.n_scale
            
        
    
