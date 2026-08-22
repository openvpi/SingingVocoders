from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _odd_positive(value: int, name: str) -> int:
    value = _positive(value, name)
    if value % 2 == 0:
        raise ValueError(f"{name} must be odd")
    return value


class LocationVariableKernelPredictor(nn.Module):
    def __init__(
        self,
        *,
        conditioning_channels: int,
        input_channels: int,
        output_channels: int,
        convolution_layers: int,
        convolution_kernel_size: int,
        hidden_channels: int,
        predictor_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_channels = _positive(input_channels, "input_channels")
        self.output_channels = _positive(output_channels, "output_channels")
        self.convolution_layers = _positive(convolution_layers, "convolution_layers")
        self.convolution_kernel_size = _odd_positive(
            convolution_kernel_size, "convolution_kernel_size"
        )
        conditioning_channels = _positive(conditioning_channels, "conditioning_channels")
        hidden_channels = _positive(hidden_channels, "hidden_channels")
        predictor_kernel_size = _odd_positive(predictor_kernel_size, "predictor_kernel_size")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        kernel_channels = (
            self.input_channels
            * self.output_channels
            * self.convolution_kernel_size
            * self.convolution_layers
        )
        bias_channels = self.output_channels * self.convolution_layers
        padding = (predictor_kernel_size - 1) // 2
        self.input_projection = nn.Sequential(
            nn.Conv1d(conditioning_channels, hidden_channels, 5, padding=2),
            nn.LeakyReLU(0.1),
        )
        self.residual = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                predictor_kernel_size,
                padding=padding,
            ),
            nn.LeakyReLU(0.1),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                predictor_kernel_size,
                padding=padding,
            ),
        )
        self.kernel_projection = nn.Conv1d(
            hidden_channels,
            kernel_channels,
            predictor_kernel_size,
            padding=padding,
        )
        self.bias_projection = nn.Conv1d(
            hidden_channels,
            bias_channels,
            predictor_kernel_size,
            padding=padding,
        )

    def forward(self, conditioning: Tensor) -> tuple[Tensor, Tensor]:
        if conditioning.ndim != 3:
            raise ValueError("conditioning must have shape [batch, channels, frames]")
        batch_size, _, frame_count = conditioning.shape
        hidden = self.input_projection(conditioning)
        hidden = hidden + self.residual(hidden)
        kernels = self.kernel_projection(hidden).reshape(
            batch_size,
            self.convolution_layers,
            self.input_channels,
            self.output_channels,
            self.convolution_kernel_size,
            frame_count,
        )
        biases = self.bias_projection(hidden).reshape(
            batch_size,
            self.convolution_layers,
            self.output_channels,
            frame_count,
        )
        return kernels, biases


class LocationVariableConvolutionBlock(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        conditioning_channels: int,
        convolution_layers: int,
        convolution_kernel_size: int,
        conditioning_hop_length: int,
        predictor_hidden_channels: int,
        predictor_kernel_size: int,
        predictor_dropout: float,
    ) -> None:
        super().__init__()
        self.channels = _positive(channels, "channels")
        self.convolution_layers = _positive(convolution_layers, "convolution_layers")
        self.convolution_kernel_size = _odd_positive(
            convolution_kernel_size, "convolution_kernel_size"
        )
        self.conditioning_hop_length = _positive(conditioning_hop_length, "conditioning_hop_length")
        self.kernel_predictor = LocationVariableKernelPredictor(
            conditioning_channels=conditioning_channels,
            input_channels=self.channels,
            output_channels=2 * self.channels,
            convolution_layers=self.convolution_layers,
            convolution_kernel_size=self.convolution_kernel_size,
            hidden_channels=predictor_hidden_channels,
            predictor_kernel_size=predictor_kernel_size,
            dropout=predictor_dropout,
        )

    @staticmethod
    def _convolve(
        inputs: Tensor,
        kernel: Tensor,
        bias: Tensor,
        dilation: int,
        hop_length: int,
    ) -> Tensor:
        batch_size, input_channels, input_length = inputs.shape
        (
            kernel_batch,
            kernel_input_channels,
            output_channels,
            kernel_size,
            kernel_frames,
        ) = kernel.shape
        if kernel_batch != batch_size or kernel_input_channels != input_channels:
            raise ValueError("location-variable kernel dimensions do not match the input")
        if input_length != kernel_frames * hop_length:
            raise ValueError("input length does not match kernel frames and hop length")

        padding = dilation * ((kernel_size - 1) // 2)
        unfolded = functional.pad(inputs, (padding, padding))
        unfolded = unfolded.unfold(2, hop_length + 2 * padding, hop_length)
        if hop_length < dilation:
            unfolded = functional.pad(unfolded, (0, dilation))
        unfolded = unfolded.unfold(3, dilation, dilation)[..., :hop_length]
        unfolded = unfolded.transpose(3, 4).unfold(4, kernel_size, 1)
        output = torch.einsum("bildsk,biokl->bolsd", unfolded, kernel)
        output = output + bias.unsqueeze(-1).unsqueeze(-1)
        return output.reshape(batch_size, output_channels, -1)

    def forward(self, inputs: Tensor, conditioning: Tensor) -> Tensor:
        if inputs.shape[-1] != conditioning.shape[-1] * self.conditioning_hop_length:
            raise ValueError("input and conditioning lengths are not aligned")
        kernels, biases = self.kernel_predictor(conditioning)
        hidden = inputs
        for layer_index in range(self.convolution_layers):
            dilation = 2**layer_index
            hidden = self._convolve(
                hidden,
                kernels[:, layer_index],
                biases[:, layer_index],
                dilation,
                self.conditioning_hop_length,
            )
            values, gates = hidden.chunk(2, dim=1)
            hidden = values.sigmoid() * gates.tanh()
        return hidden


class LocationVariableNoiseGenerator(nn.Module):
    def __init__(
        self,
        *,
        noise_channels: int,
        output_channels: int,
        hidden_channels: int,
        conditioning_channels: int,
        conditioning_hop_length: int,
        block_count: int,
        layers_per_block: int,
        convolution_kernel_size: int,
        predictor_hidden_channels: int,
        predictor_kernel_size: int,
        predictor_dropout: float,
    ) -> None:
        super().__init__()
        noise_channels = _positive(noise_channels, "noise_channels")
        output_channels = _positive(output_channels, "output_channels")
        hidden_channels = _positive(hidden_channels, "hidden_channels")
        block_count = _positive(block_count, "block_count")
        self.input_projection = nn.Conv1d(noise_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            LocationVariableConvolutionBlock(
                channels=hidden_channels,
                conditioning_channels=conditioning_channels,
                convolution_layers=layers_per_block,
                convolution_kernel_size=convolution_kernel_size,
                conditioning_hop_length=conditioning_hop_length,
                predictor_hidden_channels=predictor_hidden_channels,
                predictor_kernel_size=predictor_kernel_size,
                predictor_dropout=predictor_dropout,
            )
            for _ in range(block_count)
        )
        self.output_projection = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, output_channels, 1),
        )

    def forward(self, noise: Tensor, conditioning: Tensor) -> Tensor:
        hidden = self.input_projection(noise)
        hidden = self.blocks[0](hidden, conditioning)
        for block in self.blocks[1:]:
            hidden = hidden + block(hidden, conditioning)
        return cast(Tensor, self.output_projection(hidden))
