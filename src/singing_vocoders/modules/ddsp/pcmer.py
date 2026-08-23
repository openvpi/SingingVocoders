# Adapted from yxlllc/pc-ddsp under the MIT License.

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F


class PCmer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        dim_model: int,
        dim_keys: int,
        dim_values: int,
        residual_dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(num_layers)])

    def forward(self, phone: Tensor, mask: Tensor | None = None) -> Tensor:
        for layer in self._layers:
            phone = layer(phone, mask)
        return phone


class _EncoderLayer(nn.Module):
    def __init__(self, parent: PCmer) -> None:
        super().__init__()
        self.conformer = ConformerConvModule(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)
        self.attn = Attention(
            dim=parent.dim_model,
            heads=parent.num_heads,
            dim_head=32,
        )

    def forward(self, phone: Tensor, mask: Tensor | None = None) -> Tensor:
        phone = phone + self.attn(self.norm(phone), mask=mask)
        return phone + cast(Tensor, self.conformer(phone))


def calc_same_padding(kernel_size: int) -> tuple[int, int]:
    pad = kernel_size // 2
    return pad, pad - (kernel_size + 1) % 2


class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


class Transpose(nn.Module):
    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()
        if len(dims) != 2:
            raise ValueError("dims must contain exactly two dimensions")
        self.dims = tuple(dims)

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.dims)


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: int,
        padding: tuple[int, int],
    ) -> None:
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.conv(F.pad(x, self.padding)))


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim: int,
        causal: bool = False,
        expansion_factor: int = 2,
        kernel_size: int = 31,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(
                inner_dim,
                inner_dim,
                kernel_size=kernel_size,
                padding=padding,
            ),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 64,
        conditiondim: int | None = None,
    ) -> None:
        super().__init__()
        conditiondim = dim if conditiondim is None else conditiondim
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(conditiondim, hidden_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(hidden_dim, dim))

    def forward(
        self,
        q: Tensor,
        kv: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        kv = q if kv is None else kv
        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=2)
        q, k, v = (
            cast(Tensor, rearrange(tensor, "b t (h c) -> b h t c", h=self.heads))
            for tensor in (q, k, v)
        )
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        with torch.backends.cuda.sdp_kernel():
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = rearrange(out, "b h t c -> b t (h c)", h=self.heads)
        return cast(Tensor, self.to_out(out))
