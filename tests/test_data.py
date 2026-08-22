from pathlib import Path

import numpy as np
import pytest
import soundfile

from singing_vocoders.config.schema import AugmentationConfig, FeatureConfig
from singing_vocoders.data.dataset import VocoderDataset, collate_batches
from singing_vocoders.data.preprocess import preprocess_dataset


def _write_dataset(root: Path, frames: int) -> None:
    (root / "items").mkdir(parents=True)
    np.savez(
        root / "items/example.npz",
        audio=np.arange(frames * 4, dtype=np.float32),
        mel=np.ones((frames, 2), dtype=np.float32),
        f0=np.full(frames, 220, dtype=np.float32),
        uv=np.zeros(frames, dtype=np.bool_),
        pe=np.asarray(0, dtype=np.int64),
    )
    (root / "dataset.json").write_text('{"format_version": 1}', encoding="utf-8")
    (root / "train.txt").write_text("items/example.npz\n", encoding="utf-8")
    (root / "valid.txt").write_text("items/example.npz\n", encoding="utf-8")


def test_exact_length_crop_is_valid(tmp_path: Path) -> None:
    _write_dataset(tmp_path, 8)
    dataset = VocoderDataset(tmp_path, "train", crop_frames=8, hop_size=4, seed=1)
    batch = dataset[0]
    assert batch.audio.shape == (1, 32)
    assert batch.mel.shape == (2, 8)


def test_short_items_are_padded_and_collated(tmp_path: Path) -> None:
    _write_dataset(tmp_path, 3)
    dataset = VocoderDataset(tmp_path, "train", crop_frames=5, hop_size=4, seed=1)
    batch = collate_batches([dataset[0], dataset[0]])
    assert batch.audio.shape == (2, 1, 20)
    assert batch.mel.shape == (2, 2, 5)
    assert batch.uv[:, -1].all()


def test_crop_and_volume_augmentation_are_epoch_deterministic(tmp_path: Path) -> None:
    _write_dataset(tmp_path, 32)
    augmentation = AugmentationConfig(volume_enabled=True, volume_probability=1)
    dataset = VocoderDataset(
        tmp_path,
        "train",
        crop_frames=8,
        hop_size=4,
        seed=114514,
        augmentation=augmentation,
    )
    first = dataset[0]
    repeated = dataset[0]
    assert first.metadata == repeated.metadata
    assert np.isclose(first.metadata[0]["volume_log_gain"], repeated.metadata[0]["volume_log_gain"])
    dataset.set_epoch(1)
    next_epoch = dataset[0]
    assert next_epoch.metadata != first.metadata


def test_corrupt_alignment_is_rejected_without_sample_replacement(tmp_path: Path) -> None:
    _write_dataset(tmp_path, 8)
    with np.load(tmp_path / "items/example.npz", allow_pickle=False) as item:
        values = {name: item[name] for name in item.files}
    values["audio"] = values["audio"][:-1]
    np.savez(tmp_path / "items/example.npz", **values)
    dataset = VocoderDataset(tmp_path, "train", crop_frames=8, hop_size=4, seed=1)
    with pytest.raises(ValueError, match="unaligned arrays"):
        dataset[0]


def test_preprocessing_is_reproducible_across_pcm_widths_and_sample_rates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index, (sample_rate, subtype) in enumerate(
        ((22_050, "PCM_16"), (44_100, "PCM_24"), (44_100, "PCM_16"))
    ):
        time = np.arange(sample_rate // 4, dtype=np.float32) / sample_rate
        waveform = 0.2 * np.sin(2 * np.pi * (220 + index * 20) * time)
        audio = np.stack((waveform, waveform * 0.8), axis=1) if index == 0 else waveform
        soundfile.write(source / f"item-{index}.wav", audio, sample_rate, subtype=subtype)
    features = FeatureConfig()
    first = preprocess_dataset(source, tmp_path / "first", features, validation_size=1)
    second = preprocess_dataset(source, tmp_path / "second", features, validation_size=1)
    assert (first.source_count, first.train_count, first.validation_count) == (3, 2, 1)
    assert first.fingerprint == second.fingerprint
    assert (first.output_dir / "manifest.jsonl").read_text() == (
        second.output_dir / "manifest.jsonl"
    ).read_text()
    for path in (first.output_dir / "items").glob("*.npz"):
        with np.load(path, allow_pickle=False) as item:
            assert set(item.files) == {"audio", "mel", "f0", "uv", "pe"}
            assert len(item["audio"]) == len(item["mel"]) * features.hop_size
            assert all(np.isfinite(item[name]).all() for name in ("audio", "mel", "f0"))
