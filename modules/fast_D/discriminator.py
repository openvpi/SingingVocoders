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


class SoftSignGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, gate, decay):
        denom_out = out.abs().mul(decay).add(1.0)
        denom_gate = gate.abs().add(1.0)
        out = out / denom_out
        gate = gate / denom_gate
        out_d_gate = out / denom_gate / denom_gate
        gate_d_out = gate / denom_out / denom_out
        ctx.save_for_backward(out_d_gate, gate_d_out)
        return out * gate

    @staticmethod
    def backward(ctx, grad_output):
        out_d_gate, gate_d_out = ctx.saved_tensors
        grad_out_part = grad_output * gate_d_out
        grad_gate_part = grad_output * out_d_gate
        return grad_out_part, grad_gate_part, None

       
class SoftSignGLU(nn.Module):
    # SoftSign-Applies the gated linear unit function.
    def __init__(self, dim=-1, decay=0.1):
        super().__init__()
        self.dim = dim
        self.decay = decay

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.        
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return SoftSignGLUFunction.apply(out, gate, self.decay)

        
class LYNXNet2Block(nn.Module):
    def __init__(self, dim, expansion_factor=1, kernel_size=31, glu_decay=0.1):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        self.net = nn.Sequential(
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            SoftSignGLU(decay=glu_decay),
            nn.Linear(inner_dim, inner_dim * 2),
            SoftSignGLU(decay=glu_decay),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        norm_x = F.rms_norm(x, (x.size(-1), ))
        x = x + self.net(norm_x)
        return x, norm_x


class FastPD(torch.nn.Module):
    def __init__(self, period, init_channel=8, strides=[4, 4, 4], kernel_size=11, glu_decay=0.1):
        super(FastPD, self).__init__()
        self.period = period
        self.strides = strides
        self.pre = nn.Linear(1, init_channel)
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=init_channel * np.prod(strides[: i + 1]),
                    kernel_size=kernel_size,
                    glu_decay=glu_decay
                )
                for i in range(len(strides))
            ]
        )
        self.post = nn.Linear(init_channel * np.prod(strides), 1)

    def forward(self, x):
        fmap = []

        b, _, t = x.shape
        n = self.period * np.prod(self.strides)
        x = x[:, :, : (t // n) * n].view(b, -1, self.period)
        x = x.transpose(1, 2).reshape(b * self.period, -1, 1)
        
        x = self.pre(x)
        for i, layer in enumerate(self.residual_layers):
            if self.strides[i] > 1:
                x = x.view(b, -1, x.size(2) * self.strides[i])
            x, norm_x = layer(x)
            if i > 0:
                fmap.append(norm_x.view(b, -1))
        x = self.post(F.rms_norm(x, (x.size(-1), )))
        x = x.view(b, -1)

        return x, fmap

     
class FastMPD(torch.nn.Module):
    def __init__(self, periods=None, init_channel=8, strides=[4, 4, 4], kernel_size=11, glu_decay=0.1):
        super(FastMPD, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(
                FastPD(period, init_channel, strides, kernel_size, glu_decay))

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs,  fmap_rs
