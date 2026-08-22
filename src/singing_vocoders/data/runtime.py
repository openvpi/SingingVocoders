from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Sampler

from singing_vocoders.config.schema import AugmentationConfig, DataConfig
from singing_vocoders.data.dataset import VocoderDataset, collate_batches


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


class EpochShuffleSampler(Sampler[int]):
    def __init__(self, size: int, seed: int) -> None:
        if size < 1:
            raise ValueError("sampler size must be positive")
        self.size = size
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(self.size, generator=generator).tolist()

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must not be negative")
        self.epoch = epoch

    def state_dict(self) -> dict[str, int]:
        return {"format_version": 1, "size": self.size, "seed": self.seed, "epoch": self.epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format_version") != 1:
            raise ValueError("unsupported sampler state version")
        if state.get("size") != self.size or state.get("seed") != self.seed:
            raise ValueError("sampler configuration does not match")
        self.set_epoch(int(state["epoch"]))


def load_dataset_metadata(config: DataConfig) -> DatasetMetadata:
    root = config.processed_dir.resolve()
    metadata_path = root / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError(f"unsupported dataset metadata: {metadata_path}")
    expected_features = config.features.model_dump(mode="json")
    if value.get("features") != expected_features:
        raise ValueError("processed dataset feature configuration does not match the recipe")
    fingerprint = value.get("fingerprint")
    counts = value.get("counts")
    if not isinstance(fingerprint, str) or not isinstance(counts, dict):
        raise ValueError("processed dataset metadata is incomplete")
    train_count = counts.get("train")
    validation_count = counts.get("valid")
    if not isinstance(train_count, int) or not isinstance(validation_count, int):
        raise ValueError("processed dataset counts are invalid")
    train_index = (root / "train.txt").read_text(encoding="utf-8").splitlines()
    valid_index = (root / "valid.txt").read_text(encoding="utf-8").splitlines()
    if len(train_index) != train_count or len(valid_index) != validation_count:
        raise ValueError("processed dataset indices do not match dataset metadata")
    return DatasetMetadata(root, fingerprint, train_count, validation_count)


def build_dataloaders(config: DataConfig, augmentation: AugmentationConfig) -> VocoderLoaders:
    metadata = load_dataset_metadata(config)
    train_dataset = VocoderDataset(
        metadata.root,
        "train",
        crop_frames=config.crop_frames,
        hop_size=config.features.hop_size,
        seed=config.seed,
        augmentation=augmentation,
    )
    valid_dataset = VocoderDataset(
        metadata.root,
        "valid",
        crop_frames=config.crop_frames,
        hop_size=config.features.hop_size,
        seed=config.seed,
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
        sampler=EpochShuffleSampler(len(train_dataset), config.seed),
        drop_last=True,
        **common,
    )
    valid_loader = DataLoader(valid_dataset, shuffle=False, drop_last=False, **common)
    return VocoderLoaders(
        cast(DataLoader[Any], train_loader), cast(DataLoader[Any], valid_loader), metadata
    )
