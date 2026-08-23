from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from torch import Tensor, nn


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        output, gate = x.chunk(2, dim=self.dim)
        return output * gate.sigmoid()


class downblock(nn.Module):
    def __init__(self, down: int, indim: int, outdim: int) -> None:
        super().__init__()
        if isinstance(down, bool) or not isinstance(down, int) or down <= 0:
            raise ValueError("down must be a positive integer")
        self.c = nn.Conv1d(
            indim,
            outdim * 2,
            kernel_size=down * 2,
            stride=down,
            padding=down // 2,
        )
        self.act = GLU(1)
        self.out = nn.Conv1d(outdim, outdim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.act1(self.out(self.act(self.c(x)))))


class ddsp_down(nn.Module):
    def __init__(self, dims: int, downs: Sequence[int]) -> None:
        super().__init__()
        if isinstance(dims, bool) or not isinstance(dims, int) or dims <= 0:
            raise ValueError("dims must be a positive integer")
        reversed_downs = tuple(reversed(downs))
        if not reversed_downs or any(
            isinstance(down, bool) or not isinstance(down, int) or down <= 0
            for down in reversed_downs
        ):
            raise ValueError("downs must contain positive integers")

        down_layers: list[nn.Module] = []
        projections: list[nn.Module] = []
        self.fistpp = nn.Conv1d(1, dims, kernel_size=1)
        for index, down in enumerate(reversed_downs[:-1]):
            if index == 0:
                down_layers.append(downblock(down, 1, dims))
                projections.append(nn.Conv1d(dims, dims, kernel_size=1))
            else:
                down_layers.append(downblock(down, dims * index, dims * (index + 1)))
                projections.append(nn.Conv1d(dims * (index + 1), dims, kernel_size=1))
        self.downs = nn.ModuleList(down_layers)
        self.ppls = nn.ModuleList(projections)

    def forward(self, x: Tensor) -> list[Tensor]:
        spec = [self.fistpp(x)]
        for down_layer, projection in zip(self.downs, self.ppls, strict=True):
            x = down_layer(x)
            spec.append(projection(x))
        spec.reverse()
        return spec
