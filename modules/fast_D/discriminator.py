import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def combine_frames(x, n):
    B, L, C = x.shape
    num_groups = L // n
    if num_groups == 0:
        return torch.empty(B, 0, n * C, device=x.device, dtype=x.dtype)
    x = x[:, :num_groups * n, :].reshape(B, num_groups, n * C)
    return x
  
  
class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class ATanGLU(nn.Module):
    # ArcTan-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return out * torch.atan(gate)

        
class LYNXNet2Block(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size=31, dropout=0.):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            ATanGLU(),
            nn.Linear(inner_dim, inner_dim * 2),
            ATanGLU(),
            nn.Linear(inner_dim, dim),
            _dropout
        )

    def forward(self, x):
        norm_x = self.norm(x)
        x = x + self.net(norm_x)
        return x, norm_x


class FastPD(torch.nn.Module):
    def __init__(self, period, init_channel=8, strides=[1, 2, 4, 4, 2], kernel_size=31):
        super(FastPD, self).__init__()
        self.period = period
        self.strides = strides
        self.pre = nn.Linear(1, init_channel)
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=init_channel * np.prod(strides[: i + 1]),
                    expansion_factor=1, 
                    kernel_size=kernel_size,
                    dropout=0
                )
                for i in range(len(strides))
            ]
        )
        self.post_norm = nn.LayerNorm(init_channel * np.prod(strides))
        self.post = nn.Linear(init_channel * np.prod(strides), 1)

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, _, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, 1, t // self.period, self.period)
        x = x.permute(0, 3, 2, 1).reshape(b * self.period, t // self.period, 1)
        
        x = self.pre(x)
        for i, layer in enumerate(self.residual_layers):
            if self.strides[i] > 1:
                x = combine_frames(x, self.strides[i])
            x, norm_x = layer(x)
            if i > 0:
                fmap.append(norm_x.reshape(b, -1))
        x = self.post(self.post_norm(x))
        x = x.reshape(b, -1)

        return x, fmap

     
class FastMPD(torch.nn.Module):
    def __init__(self,periods=None, init_channel=8, strides=[1, 2, 4, 4, 2], kernel_size=31):
        super(FastMPD, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(
                FastPD(period, init_channel=init_channel, strides=strides, kernel_size=kernel_size))

    def forward(self, y,):
        y_d_rs = []
        fmap_rs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs,  fmap_rs
