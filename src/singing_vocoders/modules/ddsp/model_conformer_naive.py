from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from torch import Tensor, nn

# Adapted from CNChTu/Diffusion-SVC under the MIT License.


class Transpose(nn.Module):
    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()
        if len(dims) != 2:
            raise ValueError("dims must contain exactly two dimensions")
        self.dims = tuple(dims)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.transpose(*self.dims)


def calc_same_padding(kernel_size: int) -> tuple[int, int]:
    pad = kernel_size // 2
    return pad, pad - (kernel_size + 1) % 2


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion_factor: int = 2,
        kernel_size: int = 31,
        dropout: float = 0.0,
        use_norm: bool = False,
        conv_model_type: str = "mode1",
    ) -> None:
        super().__init__()
        if conv_model_type != "mode1":
            raise ValueError(f"unsupported conv_model_type: {conv_model_type}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        inner_dim = dim * expansion_factor
        padding, _ = calc_same_padding(kernel_size)
        self.net = nn.Sequential(
            nn.LayerNorm(dim) if use_norm else nn.Identity(),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                inner_dim,
                inner_dim,
                kernel_size=kernel_size,
                padding=padding,
                groups=inner_dim,
            ),
            nn.PReLU(num_parameters=inner_dim),
            nn.Conv1d(inner_dim, dim, kernel_size=1),
            Transpose((1, 2)),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.net(inputs))


class CFNEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int = 8,
        use_norm: bool = False,
        conv_only: bool = False,
        conv_dropout: float = 0.0,
        atten_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conformer = ConformerConvModule(
            dim_model,
            use_norm=use_norm,
            dropout=conv_dropout,
        )
        self.norm = nn.LayerNorm(dim_model)
        self.attn = (
            None
            if conv_only
            else nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dim_feedforward=dim_model * 4,
                dropout=atten_dropout,
                activation="gelu",
                batch_first=True,
            )
        )

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        hidden = inputs
        if self.attn is not None:
            attention_output = cast(Tensor, self.attn(self.norm(hidden), src_mask=mask))
            hidden = hidden + attention_output
        convolution_output = cast(Tensor, self.conformer(hidden))
        return hidden + convolution_output


class ConformerNaiveEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        dim_model: int,
        use_norm: bool = False,
        conv_only: bool = False,
        conv_dropout: float = 0.0,
        atten_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.encoder_layers = nn.ModuleList(
            [
                CFNEncoderLayer(
                    dim_model,
                    num_heads,
                    use_norm,
                    conv_only,
                    conv_dropout,
                    atten_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        hidden = inputs
        for layer in self.encoder_layers:
            hidden = layer(hidden, mask)
        return hidden
