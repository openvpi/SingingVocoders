import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class SoftSignGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, gate):
        denom_out = out.abs().add(1.0)
        denom_gate = gate.abs().add(1.0)
        out = out / denom_out
        gate = gate / denom_gate
        ctx.save_for_backward(
            out / denom_gate / denom_gate,
            gate / denom_out / denom_out)
        return out * gate

    @staticmethod
    def backward(ctx, grad_output):
        out_d_gate, gate_d_out = ctx.saved_tensors
        grad_out_part = grad_output * gate_d_out
        grad_gate_part = grad_output * out_d_gate
        return grad_out_part, grad_gate_part


class SoftSignGLU(nn.Module):
    # SoftSign-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return SoftSignGLUFunction.apply(out, gate)


class LYNXNet2Block(nn.Module):
    def __init__(self, dim, kernel_size=11, use_dwconv=True):
        super().__init__()
        self.net = nn.Sequential(
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim if use_dwconv else 1),
            Transpose((1, 2)),
            nn.Linear(dim, dim * 2),
            SoftSignGLU(),
            nn.Linear(dim, dim * 2),
            SoftSignGLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        norm_x = F.rms_norm(x, (x.size(-1), ))
        x = x + self.net(norm_x)
        return x, norm_x


class FastPD(torch.nn.Module):
    def __init__(self, period, init_channel=8, strides=[4, 4, 4], kernel_size=11):
        super(FastPD, self).__init__()
        self.period = period
        self.strides = strides
        self.hop_length = self.period * np.prod(self.strides)
        self.pre = nn.Linear(strides[0], init_channel * strides[0])
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=init_channel * np.prod(strides[: i + 1]),
                    kernel_size=kernel_size,
                    use_dwconv=(i > 0),
                )
                for i in range(len(strides))
            ]
        )
        self.post = nn.Linear(init_channel * np.prod(strides), 1)

    def forward(self, x):
        fmap = []

        b, _, t = x.shape
        x = x[:, :, : (t // self.hop_length) * self.hop_length].view(b, -1, self.period)
        x = x.transpose(1, 2).reshape(b * self.period, -1, self.strides[0])

        x = self.pre(x)
        x = F.gelu(x)
        for i, layer in enumerate(self.residual_layers):
            if i > 0 and self.strides[i] > 1:
                x = x.view(b * self.period, -1, x.size(2) * self.strides[i])
            x, norm_x = layer(x)
            if i > 0:
                fmap.append(norm_x.view(b, -1))
        x = F.rms_norm(x, (x.size(-1), ))
        x = self.post(x)
        x = x.view(b, -1)

        return x, fmap


class FastMPD(torch.nn.Module):
    def __init__(self, periods=None, init_channel=8, strides=[4, 4, 4], kernel_size=11):
        super(FastMPD, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(
                FastPD(period, init_channel, strides, kernel_size))

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs,  fmap_rs


class ResBlock(nn.Module):
    def __init__(self, dim, use_dwconv=False):
        super().__init__()
        self.net = nn.Sequential(
            Permute((0, 3, 1, 2)),
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, groups=dim if use_dwconv else 1),
            Permute((0, 2, 3, 1)),
            SoftSignGLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        norm_x = F.rms_norm(x, (x.size(-1), ))
        x = x + self.net(norm_x)
        return x, norm_x


class FastSpecD(nn.Module):
    def __init__(self, init_channel=8, strides=[4, 2, 2], fft_size=1024, shift_size=128, win_length=1024, window="hann_window"):
        super().__init__()
        self.strides = strides
        self.expansion = np.prod(self.strides)
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer('window', getattr(torch, window)(win_length))
        self.register_buffer('freq_coords', torch.linspace(-1, 1, fft_size // 2 + 1).view(-1, 1, 1))
        self.pre = nn.Linear(strides[0], init_channel * strides[0])
        self.freq = nn.Linear(1, init_channel * strides[0])
        self.layers = nn.ModuleList()
        for i in range(len(strides)):
            self.layers.append(ResBlock(init_channel * np.prod(strides[: i + 1]), use_dwconv=False))
        self.post = nn.Linear(init_channel * self.expansion, 1)

    def forward(self, x):
        fmap = []

        x = x.squeeze(1)
        x = torch.stft(x, self.fft_size, self.shift_size, self.win_length, self.window, return_complex=True)
        x = x.abs()
        x.clamp_(min=1e-5).log10_()

        b, f, t = x.shape
        x = x[:, :, :(t // self.expansion) * self.expansion, None]
        x = x.view(b, f, -1, self.strides[0])
        
        x = self.pre(x) + self.freq(self.freq_coords)
        for i, layer in enumerate(self.layers):
            if i > 0:
                x = x.view(b, f, -1, x.size(3) * self.strides[i])
            x, norm_x = layer(x)
            if i > 0:
                fmap.append(norm_x)
        x = F.rms_norm(x, (x.size(-1), ))
        x = self.post(x)

        return x, fmap


class FastMRD(torch.nn.Module):
    def __init__(self, init_channel=8, strides=[4, 2, 2], fft_sizes=[1024, 2048, 512], hop_sizes=[128, 256, 64],
                    win_lengths=[1024, 2048, 512], window="hann_window"):
        super(FastMRD, self).__init__()
        self.discriminators = nn.ModuleList()
        for i in zip(fft_sizes, hop_sizes, win_lengths):
            self.discriminators.append(
               FastSpecD(init_channel, strides, i[0], i[1], i[2], window))

    def forward(self, y,):
        y_d_rs = []

        fmap_rs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)

            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs, fmap_rs