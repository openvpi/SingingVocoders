from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class KernelPredictor(nn.Module):
    def __init__(
        self,
        *,
        conditioning_channels: int,
        input_channels: int,
        output_channels: int,
        convolution_layers: int,
        convolution_kernel_size: int = 3,
        hidden_channels: int = 64,
        predictor_kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        positive_dimensions = {
            "conditioning_channels": conditioning_channels,
            "input_channels": input_channels,
            "output_channels": output_channels,
            "convolution_layers": convolution_layers,
            "hidden_channels": hidden_channels,
        }
        for name, value in positive_dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if convolution_kernel_size <= 0 or convolution_kernel_size % 2 == 0:
            raise ValueError("convolution_kernel_size must be a positive odd integer")
        if predictor_kernel_size <= 0 or predictor_kernel_size % 2 == 0:
            raise ValueError("predictor_kernel_size must be a positive odd integer")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.convolution_kernel_size = convolution_kernel_size
        self.convolution_layers = convolution_layers

        kernel_channels = (
            input_channels * output_channels * convolution_kernel_size * convolution_layers
        )
        bias_channels = output_channels * convolution_layers
        predictor_padding = (predictor_kernel_size - 1) // 2

        def activation() -> nn.Module:
            return nn.LeakyReLU(negative_slope=0.1)

        self.input_conv = nn.Sequential(
            nn.Conv1d(conditioning_channels, hidden_channels, kernel_size=5, padding=2),
            activation(),
        )
        residual_layers: list[nn.Module] = []
        for _ in range(3):
            residual_layers.append(nn.Dropout(dropout))
            for _ in range(2):
                residual_layers.extend(
                    (
                        nn.Conv1d(
                            hidden_channels,
                            hidden_channels,
                            predictor_kernel_size,
                            padding=predictor_padding,
                        ),
                        activation(),
                    )
                )
        self.residual_conv = nn.Sequential(*residual_layers)
        self.kernel_conv = nn.Conv1d(
            hidden_channels,
            kernel_channels,
            predictor_kernel_size,
            padding=predictor_padding,
        )
        self.bias_conv = nn.Conv1d(
            hidden_channels,
            bias_channels,
            predictor_kernel_size,
            padding=predictor_padding,
        )

    def forward(self, conditioning: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, _, conditioning_length = conditioning.shape
        hidden = self.input_conv(conditioning)
        hidden = hidden + self.residual_conv(hidden)
        kernels = self.kernel_conv(hidden).reshape(
            batch_size,
            self.convolution_layers,
            self.input_channels,
            self.output_channels,
            self.convolution_kernel_size,
            conditioning_length,
        )
        biases = self.bias_conv(hidden).reshape(
            batch_size,
            self.convolution_layers,
            self.output_channels,
            conditioning_length,
        )
        return kernels, biases


class LocationVariableConvolutionBlock(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        conditioning_channels: int,
        upsample_ratio: int,
        convolution_layers: int = 4,
        convolution_kernel_size: int = 3,
        conditioning_hop_length: int = 256,
        predictor_hidden_channels: int = 64,
        predictor_kernel_size: int = 3,
        predictor_dropout: float = 0.0,
        source_channels: int | None = None,
    ) -> None:
        super().__init__()
        if upsample_ratio <= 0:
            raise ValueError("LVC upsample ratio must be positive")
        if conditioning_hop_length <= 0:
            raise ValueError("LVC conditioning hop length must be positive")
        if source_channels is not None and source_channels <= 0:
            raise ValueError("LVC source channels must be positive")

        self.conditioning_hop_length = conditioning_hop_length
        self.input_channels = input_channels
        self.upsample = nn.ConvTranspose1d(
            input_channels,
            input_channels,
            kernel_size=upsample_ratio * 2,
            stride=upsample_ratio,
            padding=upsample_ratio // 2 + upsample_ratio % 2,
            output_padding=upsample_ratio % 2,
        )
        self.source_projection = (
            nn.Conv1d(input_channels + source_channels, input_channels, kernel_size=1)
            if source_channels is not None
            else None
        )
        self.kernel_predictor = KernelPredictor(
            conditioning_channels=conditioning_channels,
            input_channels=input_channels,
            output_channels=2 * input_channels,
            convolution_layers=convolution_layers,
            convolution_kernel_size=convolution_kernel_size,
            hidden_channels=predictor_hidden_channels,
            predictor_kernel_size=predictor_kernel_size,
            dropout=predictor_dropout,
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                input_channels,
                input_channels,
                kernel_size=convolution_kernel_size,
                padding=(3**index) * ((convolution_kernel_size - 1) // 2),
                dilation=3**index,
            )
            for index in range(convolution_layers)
        )

    def forward(
        self,
        inputs: Tensor,
        conditioning: Tensor,
        source: Tensor | None = None,
    ) -> Tensor:
        kernels, biases = self.kernel_predictor(conditioning)
        hidden = self.upsample(functional.leaky_relu(inputs, negative_slope=0.2))

        if self.source_projection is None:
            if source is not None:
                raise ValueError("source features were provided to an unconditioned LVC block")
        else:
            if source is None:
                raise ValueError("source-conditioned LVC block requires source features")
            if source.shape[0] != hidden.shape[0] or source.shape[-1] != hidden.shape[-1]:
                raise ValueError(
                    "source features must match the upsampled LVC batch and time dimensions"
                )
            hidden = self.source_projection(torch.cat((hidden, source), dim=1))

        for index, convolution in enumerate(self.convolutions):
            residual = functional.leaky_relu(hidden, negative_slope=0.2)
            residual = convolution(residual)
            residual = functional.leaky_relu(residual, negative_slope=0.2)
            residual = self._location_variable_convolution(
                residual,
                kernels[:, index],
                biases[:, index],
                hop_size=self.conditioning_hop_length,
            )
            gates, values = residual.chunk(2, dim=1)
            hidden = hidden + gates.sigmoid() * values.tanh()
        return cast(Tensor, hidden)

    @staticmethod
    def _location_variable_convolution(
        inputs: Tensor,
        kernel: Tensor,
        bias: Tensor,
        *,
        hop_size: int,
        dilation: int = 1,
    ) -> Tensor:
        batch_size, input_channels, input_length = inputs.shape
        (
            kernel_batch_size,
            kernel_input_channels,
            output_channels,
            kernel_size,
            kernel_length,
        ) = kernel.shape
        if batch_size != kernel_batch_size or input_channels != kernel_input_channels:
            raise ValueError("LVC input and predicted kernel dimensions do not match")
        if input_length != kernel_length * hop_size:
            raise ValueError(
                "LVC input length must equal conditioning length multiplied by hop size "
                f"({input_length} != {kernel_length} * {hop_size})"
            )

        padding = dilation * ((kernel_size - 1) // 2)
        windows = functional.pad(inputs, (padding, padding))
        windows = windows.unfold(2, hop_size + 2 * padding, hop_size)
        if hop_size < dilation:
            windows = functional.pad(windows, (0, dilation))
        windows = windows.unfold(3, dilation, dilation)
        windows = windows[:, :, :, :, :hop_size].transpose(3, 4)
        windows = windows.unfold(4, kernel_size, 1)

        output = torch.einsum("bildsk,biokl->bolsd", windows, kernel)
        output = output + bias.unsqueeze(-1).unsqueeze(-1)
        return output.reshape(batch_size, output_channels, -1)
