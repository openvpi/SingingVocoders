from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from singing_vocoders.components.contracts import Batch
from singing_vocoders.config.schema import AugmentationConfig, FeatureConfig
from singing_vocoders.data.integrity import (
    DATASET_FORMAT_VERSION,
    load_validated_item,
)
from singing_vocoders.modules.features import LOG_MEL_FLOOR


class VocoderDataset(Dataset[Batch]):
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "valid"],
        *,
        crop_frames: int,
        hop_size: int,
        seed: int,
        augmentation: AugmentationConfig | None = None,
        verify_item_hashes: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.split = split
        self.crop_frames = crop_frames
        self.hop_size = hop_size
        self.seed = seed
        self.augmentation = augmentation or AugmentationConfig()
        self.verify_item_hashes = verify_item_hashes
        self._epoch = torch.zeros((), dtype=torch.int64)
        cast(Any, self._epoch).share_memory_()
        metadata_path = self.root / "dataset.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        format_version = self.metadata.get("format_version")
        if format_version != DATASET_FORMAT_VERSION:
            raise ValueError(f"unsupported dataset format: {self.metadata.get('format_version')}")
        index_path = self.root / f"{split}.txt"
        self.items = [line for line in index_path.read_text(encoding="utf-8").splitlines() if line]
        self._records: dict[str, dict[str, Any]] = {}
        self._verified_hashes: set[str] = set()
        self._features = FeatureConfig.model_validate(self.metadata.get("features"))
        if self.hop_size != self._features.hop_size:
            raise ValueError("dataset hop size does not match the requested crop alignment")
        manifest_path = self.root / "manifest.jsonl"
        records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self._records = {
            str(record["item"]): record for record in records if isinstance(record, dict)
        }
        missing = [relative for relative in self.items if relative not in self._records]
        if missing:
            raise ValueError(f"dataset index item is missing from the manifest: {missing[0]}")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must not be negative")
        self._epoch.fill_(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def _random_fraction(
        self,
        sample_index: int,
        logical_position: int,
        item: str,
        stream: str,
    ) -> float:
        payload = (
            f"{self.seed}:{int(self._epoch.item())}:{sample_index}:"
            f"{logical_position}:{item}:{stream}"
        ).encode()
        value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
        return value / (2**64 - 1)

    def _crop_start(
        self,
        sample_index: int,
        logical_position: int,
        item: str,
        available_frames: int,
    ) -> int:
        choices = max(1, available_frames - self.crop_frames + 1)
        return (
            int(self._random_fraction(sample_index, logical_position, item, "crop") * choices)
            % choices
        )

    def _augment_volume(
        self,
        sample_index: int,
        logical_position: int,
        relative: str,
        audio: NDArray[np.float32],
        mel: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        config = self.augmentation
        if (
            self.split != "train"
            or not config.volume_enabled
            or self._random_fraction(
                sample_index,
                logical_position,
                relative,
                "volume_enabled",
            )
            >= config.volume_probability
        ):
            return audio, mel, 0.0
        maximum = float(np.max(np.abs(audio))) + 1e-5
        upper = min(3.0, float(np.log(1.0 / maximum)))
        fraction = self._random_fraction(
            sample_index,
            logical_position,
            relative,
            "volume_gain",
        )
        log_gain = -3.0 + fraction * (upper + 3.0)
        augmented_audio = (audio * np.exp(log_gain)).astype(np.float32, copy=False)
        augmented_mel = (mel + log_gain).astype(np.float32, copy=False)
        return augmented_audio, augmented_mel, log_gain

    def __getitem__(self, index: int | tuple[int, int] | tuple[int, int, float]) -> Batch:
        sample_weight = 1.0
        if isinstance(index, tuple):
            if len(index) not in {2, 3}:
                raise ValueError("dataset sample key must contain an index and logical position")
            sample_index, logical_position = index[:2]
            if (
                not isinstance(sample_index, int)
                or isinstance(sample_index, bool)
                or not isinstance(logical_position, int)
                or isinstance(logical_position, bool)
                or logical_position < 0
            ):
                raise ValueError("dataset sample key must contain an index and logical position")
            if len(index) == 3:
                sample_weight = float(index[2])
                if sample_weight not in {0.0, 1.0}:
                    raise ValueError("dataset sample weight must be zero or one")
        else:
            sample_index = index
            logical_position = index
        relative = self.items[sample_index]
        values = load_validated_item(
            self.root,
            self._records[relative],
            self._features,
            verify_hash=self.verify_item_hashes and relative not in self._verified_hashes,
        )
        if self.verify_item_hashes:
            self._verified_hashes.add(relative)
        mel = values["mel"].astype(np.float32, copy=False)
        f0 = values["f0"].astype(np.float32, copy=False)
        uv = values["uv"].astype(np.bool_, copy=False)
        audio = values["audio"].astype(np.float32, copy=False)
        pe = np.asarray(values["pe"], dtype=np.int64)
        frames = int(mel.shape[0])
        audio, mel, log_gain = self._augment_volume(
            sample_index,
            logical_position,
            relative,
            audio,
            mel,
        )
        start = (
            0
            if self.split == "valid"
            else self._crop_start(sample_index, logical_position, relative, frames)
        )
        end = start + self.crop_frames
        mel_crop = mel[start:end]
        f0_crop = f0[start:end]
        uv_crop = uv[start:end]
        sample_start = start * self.hop_size
        sample_end = end * self.hop_size
        audio_crop = audio[sample_start:sample_end]
        missing_frames = max(0, self.crop_frames - len(mel_crop))
        missing_samples = max(0, self.crop_frames * self.hop_size - len(audio_crop))
        if missing_frames:
            mel_crop = np.pad(
                mel_crop,
                ((0, missing_frames), (0, 0)),
                constant_values=LOG_MEL_FLOOR,
            )
            f0_crop = np.pad(f0_crop, (0, missing_frames))
            uv_crop = np.pad(uv_crop, (0, missing_frames), constant_values=True)
        if missing_samples:
            audio_crop = np.pad(audio_crop, (0, missing_samples))
        metadata: dict[str, Any] = {
            "item": relative,
            "logical_position": logical_position,
            "validation_weight": sample_weight,
            "start_frame": start,
            "volume_log_gain": log_gain,
        }
        return Batch(
            audio=torch.from_numpy(audio_crop.copy()).unsqueeze(0),
            mel=torch.from_numpy(mel_crop.copy()).transpose(0, 1),
            f0=torch.from_numpy(f0_crop.copy()),
            uv=torch.from_numpy(uv_crop.copy()),
            pe=torch.as_tensor(pe, dtype=torch.long),
            metadata=[metadata],
        )


def collate_batches(samples: list[Batch]) -> Batch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    return Batch(
        audio=torch.stack([sample.audio for sample in samples]),
        mel=torch.stack([sample.mel for sample in samples]),
        f0=torch.stack([sample.f0 for sample in samples]),
        uv=torch.stack([sample.uv for sample in samples]),
        pe=torch.stack([sample.pe for sample in samples]),
        metadata=[entry for sample in samples for entry in sample.metadata],
    )
