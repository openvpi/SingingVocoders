from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from singing_vocoders.components.contracts import Batch
from singing_vocoders.config.schema import AugmentationConfig


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
    ) -> None:
        self.root = Path(root).resolve()
        self.split = split
        self.crop_frames = crop_frames
        self.hop_size = hop_size
        self.seed = seed
        self.augmentation = augmentation or AugmentationConfig()
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        metadata_path = self.root / "dataset.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != 1:
            raise ValueError(f"unsupported dataset format: {self.metadata.get('format_version')}")
        index_path = self.root / f"{split}.txt"
        self.items = [line for line in index_path.read_text(encoding="utf-8").splitlines() if line]

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must not be negative")
        self._epoch.fill_(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def _random_fraction(self, index: int, item: str, stream: str) -> float:
        payload = f"{self.seed}:{int(self._epoch.item())}:{index}:{item}:{stream}".encode()
        value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
        return value / (2**64 - 1)

    def _crop_start(self, index: int, item: str, available_frames: int) -> int:
        choices = max(1, available_frames - self.crop_frames + 1)
        return int(self._random_fraction(index, item, "crop") * choices) % choices

    def _validate_item(
        self,
        relative: str,
        audio: np.ndarray,
        mel: np.ndarray,
        f0: np.ndarray,
        uv: np.ndarray,
        pe: np.ndarray,
    ) -> None:
        frames = mel.shape[0] if mel.ndim == 2 else -1
        if audio.ndim != 1 or mel.ndim != 2 or f0.ndim != 1 or uv.ndim != 1 or pe.ndim != 0:
            raise ValueError(f"invalid array ranks in {relative}")
        if len(f0) != frames or len(uv) != frames or len(audio) != frames * self.hop_size:
            raise ValueError(f"unaligned arrays in {relative}")
        if any(not np.isfinite(array).all() for array in (audio, mel, f0)):
            raise ValueError(f"non-finite arrays in {relative}")

    def _augment_volume(
        self, index: int, relative: str, audio: np.ndarray, mel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        config = self.augmentation
        if (
            self.split != "train"
            or not config.volume_enabled
            or self._random_fraction(index, relative, "volume_enabled") >= config.volume_probability
        ):
            return audio, mel, 0.0
        maximum = float(np.max(np.abs(audio))) + 1e-5
        upper = min(3.0, float(np.log(1.0 / maximum)))
        fraction = self._random_fraction(index, relative, "volume_gain")
        log_gain = -3.0 + fraction * (upper + 3.0)
        augmented_audio = (audio * np.exp(log_gain)).astype(np.float32, copy=False)
        augmented_mel = (mel + log_gain).astype(np.float32, copy=False)
        return augmented_audio, augmented_mel, log_gain

    def __getitem__(self, index: int) -> Batch:
        relative = self.items[index]
        with np.load(self.root / relative, allow_pickle=False) as item:
            mel = item["mel"].astype(np.float32, copy=False)
            f0 = item["f0"].astype(np.float32, copy=False)
            uv = item["uv"].astype(np.bool_, copy=False)
            audio = item["audio"].astype(np.float32, copy=False)
            pe = np.asarray(item["pe"], dtype=np.int64)
        self._validate_item(relative, audio, mel, f0, uv, pe)
        frames = int(mel.shape[0])
        audio, mel, log_gain = self._augment_volume(index, relative, audio, mel)
        start = 0 if self.split == "valid" else self._crop_start(index, relative, frames)
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
            mel_crop = np.pad(mel_crop, ((0, missing_frames), (0, 0)))
            f0_crop = np.pad(f0_crop, (0, missing_frames))
            uv_crop = np.pad(uv_crop, (0, missing_frames), constant_values=True)
        if missing_samples:
            audio_crop = np.pad(audio_crop, (0, missing_samples))
        metadata: dict[str, Any] = {
            "item": relative,
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
