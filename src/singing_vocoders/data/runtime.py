from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Sampler

from singing_vocoders.config.schema import AugmentationConfig, DataConfig
from singing_vocoders.data.dataset import VocoderDataset, collate_batches
from singing_vocoders.data.integrity import validate_processed_dataset


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    root: Path
    fingerprint: str
    train_count: int
    validation_count: int


@dataclass(frozen=True, slots=True)
class VocoderLoaders:
    train: DataLoader[Any]
    valid: DataLoader[Any]
    metadata: DatasetMetadata


class EpochShuffleSampler(Sampler[tuple[int, int]]):
    def __init__(
        self,
        size: int,
        seed: int,
        *,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if size < 1:
            raise ValueError("sampler size must be positive")
        if num_replicas < 1:
            raise ValueError("sampler replica count must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("sampler rank is outside the replica range")
        self.size = size
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = (size + num_replicas - 1) // num_replicas
        self.total_size = self.num_samples * num_replicas
        self.epoch = 0

    def __iter__(self) -> Iterator[tuple[int, int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        padding = self.total_size - len(indices)
        if padding:
            repeats = (padding + len(indices) - 1) // len(indices)
            indices.extend((indices * repeats)[:padding])
        for logical_position in range(self.rank, self.total_size, self.num_replicas):
            yield indices[logical_position], logical_position

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must not be negative")
        self.epoch = epoch

    def state_dict(self) -> dict[str, int]:
        return {
            "format_version": 1,
            "size": self.size,
            "seed": self.seed,
            "num_replicas": self.num_replicas,
            "epoch": self.epoch,
        }

    def validate_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format_version") != 1:
            raise ValueError("unsupported sampler state version")
        if (
            state.get("size") != self.size
            or state.get("seed") != self.seed
            or state.get("num_replicas") != self.num_replicas
        ):
            raise ValueError("sampler configuration does not match")
        epoch = state.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("sampler epoch must be a non-negative integer")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.validate_state_dict(state)
        self.set_epoch(int(state["epoch"]))


class DistributedEvaluationSampler(Sampler[tuple[int, int, float]]):
    def __init__(self, size: int, *, num_replicas: int = 1, rank: int = 0) -> None:
        if size < 1:
            raise ValueError("sampler size must be positive")
        if num_replicas < 1:
            raise ValueError("sampler replica count must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("sampler rank is outside the replica range")
        self.size = size
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = (size + num_replicas - 1) // num_replicas
        self.total_size = self.num_samples * num_replicas

    def __iter__(self) -> Iterator[tuple[int, int, float]]:
        indices = list(range(self.size))
        padding = self.total_size - self.size
        if padding:
            repeats = (padding + self.size - 1) // self.size
            indices.extend((indices * repeats)[:padding])
        for logical_position in range(self.rank, self.total_size, self.num_replicas):
            weight = 1.0 if logical_position < self.size else 0.0
            yield indices[logical_position], logical_position, weight

    def __len__(self) -> int:
        return self.num_samples


def load_dataset_metadata(config: DataConfig) -> DatasetMetadata:
    root = config.processed_dir.resolve()
    value = validate_processed_dataset(root, expected_features=config.features).metadata
    fingerprint = value.get("fingerprint")
    counts = value.get("counts")
    if not isinstance(fingerprint, str) or not isinstance(counts, dict):
        raise ValueError("processed dataset metadata is incomplete")
    train_count = counts.get("train")
    validation_count = counts.get("valid")
    if not isinstance(train_count, int) or not isinstance(validation_count, int):
        raise ValueError("processed dataset counts are invalid")
    if value.get("seed") != config.seed:
        raise ValueError("processed dataset split seed does not match the recipe")
    if validation_count != config.validation_size:
        raise ValueError("processed dataset validation size does not match the recipe")
    return DatasetMetadata(root, fingerprint, train_count, validation_count)


def build_dataloaders(
    config: DataConfig,
    augmentation: AugmentationConfig,
    *,
    num_replicas: int = 1,
    rank: int = 0,
) -> VocoderLoaders:
    metadata = load_dataset_metadata(config)
    train_dataset = VocoderDataset(
        metadata.root,
        "train",
        crop_frames=config.crop_frames,
        hop_size=config.features.hop_size,
        seed=config.seed,
        augmentation=augmentation,
        verify_item_hashes=False,
    )
    valid_dataset = VocoderDataset(
        metadata.root,
        "valid",
        crop_frames=config.crop_frames,
        hop_size=config.features.hop_size,
        seed=config.seed,
        verify_item_hashes=False,
    )
    common: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "collate_fn": collate_batches,
    }
    if config.num_workers:
        common["prefetch_factor"] = config.prefetch_factor
        common["persistent_workers"] = config.persistent_workers
    train_loader = DataLoader(
        train_dataset,
        sampler=EpochShuffleSampler(
            len(train_dataset),
            config.seed,
            num_replicas=num_replicas,
            rank=rank,
        ),
        generator=torch.Generator().manual_seed(config.seed),
        drop_last=True,
        **common,
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler=DistributedEvaluationSampler(
            len(valid_dataset),
            num_replicas=num_replicas,
            rank=rank,
        ),
        generator=torch.Generator().manual_seed(config.seed + 1),
        drop_last=False,
        **common,
    )
    return VocoderLoaders(
        cast(DataLoader[Any], train_loader), cast(DataLoader[Any], valid_loader), metadata
    )
