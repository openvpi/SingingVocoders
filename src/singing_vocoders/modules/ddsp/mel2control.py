# Adapted from yxlllc/pc-ddsp under the MIT License.

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import weight_norm

from singing_vocoders.modules.ddsp.pcmer import PCmer


def split_to_dict(tensor: Tensor, tensor_splits: Mapping[str, int]) -> dict[str, Tensor]:
    labels = tuple(tensor_splits)
    sizes = list(tensor_splits.values())
    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors, strict=True))


class Mel2Control(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_splits: Mapping[str, int],
    ) -> None:
        super().__init__()
        if not output_splits or any(size < 1 for size in output_splits.values()):
            raise ValueError("output_splits must contain positive sizes")
        self.output_splits = dict(output_splits)
        self.phase_embed = nn.Linear(1, 256)
        self.stack = nn.Sequential(
            nn.Conv1d(input_channel, 256, 3, 1, 1),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
        )
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
        )
        self.norm = nn.LayerNorm(256)
        self.n_out = sum(output_splits.values())
        self.dense_out = weight_norm(nn.Linear(256, self.n_out))

    def forward(self, mel: Tensor, phase: Tensor) -> dict[str, Tensor]:
        hidden = self.stack(mel.transpose(1, 2)).transpose(1, 2)
        hidden = hidden + self.phase_embed(phase / np.pi)
        hidden = self.decoder(hidden)
        hidden = self.norm(hidden)
        return split_to_dict(self.dense_out(hidden), self.output_splits)
