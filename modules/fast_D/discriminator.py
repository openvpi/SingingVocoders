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
    def __init__(self, dim, expansion_factor=1, kernel_size=31):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        self.net = nn.Sequential(
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            SoftSignGLU(),
            nn.Linear(inner_dim, inner_dim * 2),
            SoftSignGLU(),
            nn.Linear(inner_dim, dim),
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
        self.pre = nn.Linear(strides[0], init_channel * strides[0])
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=init_channel * np.prod(strides[: i + 1]),
                    kernel_size=kernel_size,
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
        x = x.transpose(1, 2).reshape(b * self.period, -1, self.strides[0])
        
        x = self.pre(x)
        x = F.gelu(x)
        for i, layer in enumerate(self.residual_layers):
            if i > 0 and self.strides[i] > 1:
                x = x.view(b * self.period, -1, x.size(2) * self.strides[i])
            x, norm_x = layer(x)
            if i > 0:
                fmap.append(norm_x.view(b, -1, norm_x.size(2)))
        x = self.post(F.rms_norm(x, (x.size(-1), )))
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
