from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _odd_positive(value: int, name: str) -> int:
    value = _positive(value, name)
    if value % 2 == 0:
        raise ValueError(f"{name} must be odd")
    return value


class KernelPredictor(nn.Module):
    def __init__(
        self,
        cond_channels: int,
        conv_in_channels: int,
        conv_out_channels: int,
        conv_layers: int,
        conv_kernel_size: int = 3,
        kpnet_hidden_channels: int = 64,
        kpnet_conv_size: int = 1,
        kpnet_dropout: float = 0.0,
        kpnet_nonlinear_activation: str = "LeakyReLU",
        kpnet_nonlinear_activation_params: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.conv_in_channels = _positive(conv_in_channels, "conv_in_channels")
        self.conv_out_channels = _positive(conv_out_channels, "conv_out_channels")
        self.conv_kernel_size = _odd_positive(conv_kernel_size, "conv_kernel_size")
        self.conv_layers = _positive(conv_layers, "conv_layers")
        cond_channels = _positive(cond_channels, "cond_channels")
        kpnet_hidden_channels = _positive(kpnet_hidden_channels, "kpnet_hidden_channels")
        kpnet_conv_size = _odd_positive(kpnet_conv_size, "kpnet_conv_size")
        if not 0 <= kpnet_dropout < 1:
            raise ValueError("kpnet_dropout must be in the range [0, 1)")

        activation_type = getattr(nn, kpnet_nonlinear_activation, None)
        if not isinstance(activation_type, type) or not issubclass(activation_type, nn.Module):
            raise ValueError(f"unsupported activation: {kpnet_nonlinear_activation}")
        activation_params = dict(
            {"negative_slope": 0.1}
            if kpnet_nonlinear_activation_params is None
            else kpnet_nonlinear_activation_params
        )

        kernel_channels = (
            self.conv_in_channels
            * self.conv_out_channels
            * self.conv_kernel_size
            * self.conv_layers
        )
        bias_channels = self.conv_out_channels * self.conv_layers
        padding = (kpnet_conv_size - 1) // 2
        self.input_conv = nn.Sequential(
            nn.Conv1d(cond_channels, kpnet_hidden_channels, 5, padding=2, bias=True),
            activation_type(**activation_params),
        )
        self.residual_conv = nn.Sequential(
            nn.Dropout(kpnet_dropout),
            nn.Conv1d(
                kpnet_hidden_channels,
                kpnet_hidden_channels,
                kpnet_conv_size,
                padding=padding,
                bias=True,
            ),
            activation_type(**activation_params),
            nn.Conv1d(
                kpnet_hidden_channels,
                kpnet_hidden_channels,
                kpnet_conv_size,
                padding=padding,
                bias=True,
            ),
        )
        self.kernel_conv = nn.Conv1d(
            kpnet_hidden_channels,
            kernel_channels,
            kpnet_conv_size,
            padding=padding,
            bias=True,
        )
        self.bias_conv = nn.Conv1d(
            kpnet_hidden_channels,
            bias_channels,
            kpnet_conv_size,
            padding=padding,
            bias=True,
        )

    def forward(self, c: Tensor) -> tuple[Tensor, Tensor]:
        batch, _, cond_length = c.shape
        c = self.input_conv(c)
        c = c + self.residual_conv(c)
        kernels = (
            self.kernel_conv(c)
            .contiguous()
            .view(
                batch,
                self.conv_layers,
                self.conv_in_channels,
                self.conv_out_channels,
                self.conv_kernel_size,
                cond_length,
            )
        )
        bias = (
            self.bias_conv(c)
            .contiguous()
            .view(
                batch,
                self.conv_layers,
                self.conv_out_channels,
                cond_length,
            )
        )
        return kernels, bias


class LVCBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        conv_layers: int = 10,
        conv_kernel_size: int = 3,
        cond_hop_length: int = 256,
        kpnet_hidden_channels: int = 64,
        kpnet_conv_size: int = 1,
        kpnet_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_channels = _positive(in_channels, "in_channels")
        self.cond_hop_length = _positive(cond_hop_length, "cond_hop_length")
        self.conv_layers = _positive(conv_layers, "conv_layers")
        self.conv_kernel_size = _odd_positive(conv_kernel_size, "conv_kernel_size")
        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            conv_layers=self.conv_layers,
            conv_kernel_size=self.conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout,
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        in_channels = x.shape[1]
        cond_length = c.shape[2]
        if x.shape[2] != cond_length * self.cond_hop_length:
            raise ValueError(
                f"input length {x.shape[2]} does not match conditioning length "
                f"{cond_length} and hop length {self.cond_hop_length}"
            )
        kernels, bias = self.kernel_predictor(c)
        for index in range(self.conv_layers):
            dilation = 2**index
            x = self.location_variable_convolution(
                x,
                kernels[:, index],
                bias[:, index],
                dilation,
                self.cond_hop_length,
            )
            x = torch.sigmoid(x[:, :in_channels]) * torch.tanh(x[:, in_channels:])
        return x

    @staticmethod
    def location_variable_convolution(
        x: Tensor,
        kernel: Tensor,
        bias: Tensor,
        dilation: int,
        hop_size: int,
    ) -> Tensor:
        batch, _, in_length = x.shape
        _, _, out_channels, kernel_size, kernel_length = kernel.shape
        if in_length != kernel_length * hop_size:
            raise ValueError("input and location-variable kernel lengths do not match")

        padding = dilation * ((kernel_size - 1) // 2)
        x = F.pad(x, (padding, padding), "constant", 0)
        x = x.unfold(2, hop_size + 2 * padding, hop_size)
        if hop_size < dilation:
            x = F.pad(x, (0, dilation), "constant", 0)
        x = x.unfold(3, dilation, dilation)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(3, 4)
        x = x.unfold(4, kernel_size, 1)
        output = torch.einsum("bildsk,biokl->bolsd", x, kernel)
        output = output + bias.unsqueeze(-1).unsqueeze(-1)
        return output.contiguous().view(batch, out_channels, -1)


class LVCNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        inner_channels: int = 8,
        cond_channels: int = 80,
        cond_hop_length: int = 256,
        lvc_block_nums: int = 3,
        lvc_layers_each_block: int = 10,
        lvc_kernel_size: int = 3,
        kpnet_hidden_channels: int = 64,
        kpnet_conv_size: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = _positive(in_channels, "in_channels")
        self.out_channels = _positive(out_channels, "out_channels")
        self.cond_channels = _positive(cond_channels, "cond_channels")
        self.lvc_block_nums = _positive(lvc_block_nums, "lvc_block_nums")
        inner_channels = _positive(inner_channels, "inner_channels")
        self.first_conv = nn.Conv1d(
            self.in_channels,
            inner_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )
        self.lvc_blocks = nn.ModuleList(
            LVCBlock(
                in_channels=inner_channels,
                cond_channels=self.cond_channels,
                conv_layers=lvc_layers_each_block,
                conv_kernel_size=lvc_kernel_size,
                cond_hop_length=cond_hop_length,
                kpnet_hidden_channels=kpnet_hidden_channels,
                kpnet_conv_size=kpnet_conv_size,
                kpnet_dropout=dropout,
            )
            for _ in range(self.lvc_block_nums)
        )
        self.last_conv_layers = nn.ModuleList(
            [
                nn.ReLU(inplace=True),
                nn.Conv1d(inner_channels, inner_channels, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(inner_channels, self.out_channels, kernel_size=1, padding=0, bias=True),
            ]
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.first_conv(x)
        x = self.lvc_blocks[0](x, c)
        for block in self.lvc_blocks[1:]:
            x = x + block(x, c)
        for layer in self.last_conv_layers:
            x = layer(x)
        return x
