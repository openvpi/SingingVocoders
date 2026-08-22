from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch.nn.functional as functional
from torch import Tensor, nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

from singing_vocoders.modules.lvc.conditioning import (
    build_mel_upsampler,
    validate_upsampling_configuration,
)
from singing_vocoders.modules.lvc.univnet import LocationVariableConvolutionBlock


def _required_int(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _required_float(values: Mapping[str, Any], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    return float(value)


def _required_int_sequence(values: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = values.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{key} must be a sequence of integers")
    result = tuple(value)
    if not result or any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise TypeError(f"{key} must contain integers")
    if any(item <= 0 for item in result):
        raise ValueError(f"{key} must contain positive integers")
    return result


@dataclass(frozen=True)
class UnivNetArchitecture:
    noise_channels: int
    output_channels: int
    hidden_channels: int
    conditioning_channels: int
    hop_size: int
    mel_upsample_factor: int
    waveform_upsample_rates: tuple[int, ...]
    lvc_layers_per_block: int
    lvc_kernel_size: int
    predictor_hidden_channels: int
    predictor_kernel_size: int
    predictor_dropout: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> UnivNetArchitecture:
        model_args_value = config.get("model_args")
        if not isinstance(model_args_value, Mapping):
            raise TypeError("model_args must be a mapping")
        model_args: Mapping[str, Any] = model_args_value

        hop_size = _required_int(config, "hop_size")
        waveform_upsample_rates = _required_int_sequence(model_args, "waveform_upsample_rates")
        mel_upsample_factor_value = model_args.get("mel_upsample_factor", 1)
        if mel_upsample_factor_value is None:
            mel_upsample_factor_value = 1
        if isinstance(mel_upsample_factor_value, bool) or not isinstance(
            mel_upsample_factor_value, int
        ):
            raise TypeError("mel_upsample_factor must be an integer")
        mel_upsample_factor = validate_upsampling_configuration(
            hop_size=hop_size,
            mel_upsample_factor=mel_upsample_factor_value,
            waveform_upsample_rates=waveform_upsample_rates,
        )

        dropout = _required_float(model_args, "kernel_predictor_dropout")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        return cls(
            noise_channels=_required_int(model_args, "noise_channels"),
            output_channels=_required_int(model_args, "output_channels"),
            hidden_channels=_required_int(model_args, "hidden_channels"),
            conditioning_channels=_required_int(config, "audio_num_mel_bins"),
            hop_size=hop_size,
            mel_upsample_factor=mel_upsample_factor,
            waveform_upsample_rates=waveform_upsample_rates,
            lvc_layers_per_block=_required_int(model_args, "lvc_layers_per_block"),
            lvc_kernel_size=_required_int(model_args, "lvc_kernel_size"),
            predictor_hidden_channels=_required_int(model_args, "kernel_predictor_hidden_channels"),
            predictor_kernel_size=_required_int(model_args, "kernel_predictor_kernel_size"),
            predictor_dropout=dropout,
        )


class UnivNetBackbone(nn.Module):
    def __init__(
        self,
        architecture: UnivNetArchitecture,
        *,
        source_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.mel_upsampler = build_mel_upsampler(architecture.mel_upsample_factor)
        self.input_projection = nn.Conv1d(
            architecture.noise_channels,
            architecture.hidden_channels,
            kernel_size=7,
            padding=3,
        )

        blocks: list[nn.Module] = []
        conditioning_hop_length = 1
        for upsample_ratio in architecture.waveform_upsample_rates:
            conditioning_hop_length *= upsample_ratio
            blocks.append(
                LocationVariableConvolutionBlock(
                    input_channels=architecture.hidden_channels,
                    conditioning_channels=architecture.conditioning_channels,
                    upsample_ratio=upsample_ratio,
                    convolution_layers=architecture.lvc_layers_per_block,
                    convolution_kernel_size=architecture.lvc_kernel_size,
                    conditioning_hop_length=conditioning_hop_length,
                    predictor_hidden_channels=architecture.predictor_hidden_channels,
                    predictor_kernel_size=architecture.predictor_kernel_size,
                    predictor_dropout=architecture.predictor_dropout,
                    source_channels=source_channels,
                )
            )
        self.lvc_blocks = nn.ModuleList(blocks)
        self.output_projection = nn.Conv1d(
            architecture.hidden_channels,
            architecture.output_channels,
            kernel_size=7,
            padding=3,
        )

    def forward_backbone(
        self,
        noise: Tensor,
        mel: Tensor,
        source_features: Sequence[Tensor] | None = None,
    ) -> Tensor:
        conditioning = self.mel_upsampler(mel)
        if noise.shape[0] != conditioning.shape[0] or noise.shape[-1] != conditioning.shape[-1]:
            raise ValueError("noise must match the upsampled mel batch and time dimensions")
        if source_features is not None and len(source_features) != len(self.lvc_blocks):
            raise ValueError("one source feature tensor is required for each LVC block")

        hidden = self.input_projection(noise)
        for index, block in enumerate(self.lvc_blocks):
            source = None if source_features is None else source_features[index]
            hidden = block(hidden, conditioning, source)
        hidden = self.output_projection(functional.leaky_relu(hidden, negative_slope=0.1))
        return cast(Tensor, hidden.tanh())

    def apply_weight_norm(self) -> None:
        def apply_to_module(module: nn.Module) -> None:
            if isinstance(module, (nn.Conv1d, nn.Conv2d)) and not parametrize.is_parametrized(
                module, "weight"
            ):
                weight_norm(module)

        self.apply(apply_to_module)

    def remove_weight_norm(self) -> None:
        def remove_from_module(module: nn.Module) -> None:
            if isinstance(module, (nn.Conv1d, nn.Conv2d)) and parametrize.is_parametrized(
                module, "weight"
            ):
                parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)

        self.apply(remove_from_module)


class UnivNet(UnivNetBackbone):
    def __init__(self, config: Mapping[str, Any], use_weight_norm: bool = True) -> None:
        architecture = UnivNetArchitecture.from_config(config)
        super().__init__(architecture)
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, noise: Tensor, mel: Tensor) -> Tensor:
        return self.forward_backbone(noise, mel)
