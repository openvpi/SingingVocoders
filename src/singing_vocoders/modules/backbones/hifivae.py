from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

LEAKY_RELU_SLOPE = 0.1


@dataclass(frozen=True)
class _ArchitectureConfig:
    latent_channels: int
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    initial_channels: int
    residual_kernel_sizes: tuple[int, ...]
    residual_dilation_sizes: tuple[tuple[int, ...], ...]
    residual_block: Literal["1", "2"]

    def validate(self) -> None:
        if self.residual_block not in {"1", "2"}:
            raise ValueError(f"unsupported residual block type: {self.residual_block!r}")
        if self.latent_channels <= 0 or self.initial_channels <= 0:
            raise ValueError("channel counts must be positive")
        if not self.upsample_rates:
            raise ValueError("at least one upsampling stage is required")
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("upsampling rates and kernel sizes must have equal lengths")
        for rate, kernel_size in zip(self.upsample_rates, self.upsample_kernel_sizes, strict=True):
            if rate <= 0 or kernel_size < rate or (kernel_size - rate) % 2:
                raise ValueError(
                    "each upsampling kernel must be at least its rate with an even difference"
                )
        channel_divisor = 2 ** len(self.upsample_rates)
        if self.initial_channels % channel_divisor:
            raise ValueError("initial_channels must be divisible by two at every stage")
        if len(self.residual_kernel_sizes) != len(self.residual_dilation_sizes):
            raise ValueError("residual kernel and dilation groups must have equal lengths")
        if not self.residual_kernel_sizes:
            raise ValueError("at least one residual kernel is required")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.residual_kernel_sizes):
            raise ValueError("residual kernels must be positive odd integers")

        required_dilations = 3 if self.residual_block == "1" else 2
        if any(len(group) != required_dilations for group in self.residual_dilation_sizes):
            raise ValueError(
                f"Residual block type {self.residual_block!r} requires "
                f"{required_dilations} dilations per kernel"
            )
        if any(dilation <= 0 for group in self.residual_dilation_sizes for dilation in group):
            raise ValueError("residual dilations must be positive")


def _initialize_convolution(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        with torch.no_grad():
            module.weight.normal_(mean, std)


def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def _normalized_convolution(
    channels: int,
    kernel_size: int,
    dilation: int,
) -> nn.Conv1d:
    return weight_norm(
        nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=_same_padding(kernel_size, dilation),
        )
    )


class _ResidualBlock(nn.Module, ABC):
    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor: ...

    @abstractmethod
    def remove_weight_norm(self) -> None: ...


class PairedResidualBlock(_ResidualBlock):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.dilated_convolutions = nn.ModuleList(
            _normalized_convolution(channels, kernel_size, dilation) for dilation in dilations
        )
        self.standard_convolutions = nn.ModuleList(
            _normalized_convolution(channels, kernel_size, 1) for _ in dilations
        )
        self.dilated_convolutions.apply(_initialize_convolution)
        self.standard_convolutions.apply(_initialize_convolution)

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for dilated_convolution, standard_convolution in zip(
            self.dilated_convolutions,
            self.standard_convolutions,
            strict=True,
        ):
            residual = F.leaky_relu(output, LEAKY_RELU_SLOPE)
            residual = dilated_convolution(residual)
            residual = F.leaky_relu(residual, LEAKY_RELU_SLOPE)
            residual = standard_convolution(residual)
            output = output + residual
        return output

    def remove_weight_norm(self) -> None:
        for convolution in self.dilated_convolutions:
            remove_weight_norm(convolution)
        for convolution in self.standard_convolutions:
            remove_weight_norm(convolution)


class SingleResidualBlock(_ResidualBlock):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 3),
    ) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList(
            _normalized_convolution(channels, kernel_size, dilation) for dilation in dilations
        )
        self.convolutions.apply(_initialize_convolution)

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for convolution in self.convolutions:
            residual = F.leaky_relu(output, LEAKY_RELU_SLOPE)
            residual = convolution(residual)
            output = output + residual
        return output

    def remove_weight_norm(self) -> None:
        for convolution in self.convolutions:
            remove_weight_norm(convolution)


def _residual_block_type(
    config: _ArchitectureConfig,
) -> type[PairedResidualBlock] | type[SingleResidualBlock]:
    return PairedResidualBlock if config.residual_block == "1" else SingleResidualBlock


def _apply_residual_stage(
    inputs: Tensor,
    blocks: nn.ModuleList,
    stage_index: int,
    blocks_per_stage: int,
) -> Tensor:
    offset = stage_index * blocks_per_stage
    first_block = cast(_ResidualBlock, blocks[offset])
    output = first_block.forward(inputs)
    for block_index in range(1, blocks_per_stage):
        block = cast(_ResidualBlock, blocks[offset + block_index])
        output = output + block.forward(inputs)
    return output / blocks_per_stage


class HiFiVAEEncoder(nn.Module):
    def __init__(self, config: _ArchitectureConfig) -> None:
        super().__init__()
        self.blocks_per_stage = len(config.residual_kernel_sizes)
        self.latent_channels = config.latent_channels
        downsampling_stages = len(config.upsample_rates)
        input_channels = config.initial_channels // (2**downsampling_stages)
        self.input_convolution = weight_norm(nn.Conv1d(1, input_channels, 7, padding=3))

        self.downsampling_layers = nn.ModuleList()
        for stage_index, (stride, kernel_size) in enumerate(
            zip(
                reversed(config.upsample_rates),
                reversed(config.upsample_kernel_sizes),
                strict=True,
            )
        ):
            layer_input_channels = config.initial_channels // (
                2 ** (downsampling_stages - stage_index)
            )
            layer_output_channels = config.initial_channels // (
                2 ** (downsampling_stages - stage_index - 1)
            )
            self.downsampling_layers.append(
                weight_norm(
                    nn.Conv1d(
                        layer_input_channels,
                        layer_output_channels,
                        kernel_size,
                        stride,
                        padding=(kernel_size - stride + 1) // 2,
                    )
                )
            )

        residual_block_type = _residual_block_type(config)
        self.residual_blocks = nn.ModuleList()
        for stage_index in range(downsampling_stages):
            channels = config.initial_channels // (2 ** (downsampling_stages - stage_index - 1))
            for kernel_size, dilations in zip(
                config.residual_kernel_sizes,
                config.residual_dilation_sizes,
                strict=True,
            ):
                self.residual_blocks.append(residual_block_type(channels, kernel_size, dilations))

        self.output_convolution = weight_norm(
            nn.Conv1d(config.initial_channels, 2 * config.latent_channels, 7, padding=3)
        )
        self.downsampling_layers.apply(_initialize_convolution)
        self.output_convolution.apply(_initialize_convolution)

    def forward(self, audio: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        output = self.input_convolution(audio)
        for stage_index, downsampling_layer in enumerate(self.downsampling_layers):
            output = F.leaky_relu(output, LEAKY_RELU_SLOPE)
            output = downsampling_layer(output)
            output = _apply_residual_stage(
                output,
                self.residual_blocks,
                stage_index,
                self.blocks_per_stage,
            )

        output = F.leaky_relu(output)
        distribution = self.output_convolution(output)
        mean, log_scale = torch.split(distribution, self.latent_channels, dim=1)
        latent = mean + torch.randn_like(mean) * torch.exp(log_scale)
        return latent, mean, log_scale


class HiFiVAEDecoder(nn.Module):
    def __init__(self, config: _ArchitectureConfig) -> None:
        super().__init__()
        self.blocks_per_stage = len(config.residual_kernel_sizes)
        self.input_convolution = weight_norm(
            nn.Conv1d(config.latent_channels, config.initial_channels, 7, padding=3)
        )

        self.upsampling_layers = nn.ModuleList()
        for stage_index, (stride, kernel_size) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes, strict=True)
        ):
            self.upsampling_layers.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        config.initial_channels // (2**stage_index),
                        config.initial_channels // (2 ** (stage_index + 1)),
                        kernel_size,
                        stride,
                        padding=(kernel_size - stride + 1) // 2,
                    )
                )
            )

        residual_block_type = _residual_block_type(config)
        self.residual_blocks = nn.ModuleList()
        for stage_index in range(len(self.upsampling_layers)):
            channels = config.initial_channels // (2 ** (stage_index + 1))
            for kernel_size, dilations in zip(
                config.residual_kernel_sizes,
                config.residual_dilation_sizes,
                strict=True,
            ):
                self.residual_blocks.append(residual_block_type(channels, kernel_size, dilations))

        output_channels = config.initial_channels // (2 ** len(self.upsampling_layers))
        self.output_convolution = weight_norm(nn.Conv1d(output_channels, 1, 7, padding=3))
        self.upsampling_layers.apply(_initialize_convolution)
        self.output_convolution.apply(_initialize_convolution)

    def forward(self, latent: Tensor) -> Tensor:
        output = self.input_convolution(latent)
        for stage_index, upsampling_layer in enumerate(self.upsampling_layers):
            output = F.leaky_relu(output, LEAKY_RELU_SLOPE)
            output = upsampling_layer(output)
            output = _apply_residual_stage(
                output,
                self.residual_blocks,
                stage_index,
                self.blocks_per_stage,
            )

        output = F.leaky_relu(output)
        return torch.tanh(self.output_convolution(output))

    def remove_weight_norm(self) -> None:
        for upsampling_layer in self.upsampling_layers:
            remove_weight_norm(upsampling_layer)
        for module in self.residual_blocks:
            residual_block = cast(_ResidualBlock, module)
            residual_block.remove_weight_norm()
        remove_weight_norm(self.input_convolution)
        remove_weight_norm(self.output_convolution)


class HiFiVAE(nn.Module):
    def __init__(
        self,
        *,
        latent_channels: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        initial_channels: int,
        residual_kernel_sizes: tuple[int, ...],
        residual_dilation_sizes: tuple[tuple[int, ...], ...],
        residual_block: Literal["1", "2"],
    ) -> None:
        super().__init__()
        architecture = _ArchitectureConfig(
            latent_channels=latent_channels,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            initial_channels=initial_channels,
            residual_kernel_sizes=residual_kernel_sizes,
            residual_dilation_sizes=residual_dilation_sizes,
            residual_block=residual_block,
        )
        architecture.validate()
        self.encoder = HiFiVAEEncoder(architecture)
        self.decoder = HiFiVAEDecoder(architecture)

    def forward(self, audio: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        latent, mean, log_scale = self.encoder(audio)
        reconstructed_audio = self.decoder(latent)
        return reconstructed_audio, latent, mean, log_scale
