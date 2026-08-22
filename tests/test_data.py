import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from singing_vocoders.config.schema import AugmentationConfig, DataConfig, FeatureConfig
from singing_vocoders.data.dataset import VocoderDataset, collate_batches
from singing_vocoders.data.features import feature_implementation, mel_spectrogram
from singing_vocoders.data.integrity import dataset_fingerprint, sha256_file
from singing_vocoders.data.preprocess import preprocess_dataset
from singing_vocoders.data.runtime import (
    DistributedEvaluationSampler,
    EpochShuffleSampler,
    build_dataloaders,
    load_dataset_metadata,
)
from singing_vocoders.modules.features import LOG_MEL_FLOOR


def _write_dataset(root: Path, frames: int) -> None:
    (root / "items").mkdir(parents=True)
    features = FeatureConfig(
        sample_rate=8_000,
        num_mels=2,
        hop_size=4,
        n_fft=16,
        win_size=16,
        fmax=3_900,
    )
    records: list[dict[str, object]] = []
    for split in ("train", "valid"):
        relative = f"items/{split}.npz"
        np.savez(
            root / relative,
            audio=np.arange(frames * 4, dtype=np.float32),
            mel=np.ones((frames, 2), dtype=np.float32),
            f0=np.full(frames, 220, dtype=np.float32),
            uv=np.zeros(frames, dtype=np.bool_),
            pe=np.asarray(0, dtype=np.int64),
        )
        records.append(
            {
                "source": f"{split}.wav",
                "item": relative,
                "item_sha256": sha256_file(root / relative),
                "samples": frames * 4,
                "frames": frames,
                "split": split,
            }
        )
        (root / f"{split}.txt").write_text(relative + "\n", encoding="utf-8")
    feature_values = features.model_dump(mode="json")
    implementation = feature_implementation(features)
    metadata = {
        "format_version": 2,
        "fingerprint": dataset_fingerprint(records, feature_values, implementation),
        "features": feature_values,
        "feature_implementation": implementation,
        "seed": 114_514,
        "counts": {"all": 2, "train": 1, "valid": 1},
    }
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (root / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")


def _write_audio_sources(root: Path, count: int = 3) -> list[Path]:
    root.mkdir()
    paths: list[Path] = []
    for index in range(count):
        sample_rate = 22_050 if index == 0 else 44_100
        subtype = "PCM_24" if index == 1 else "PCM_16"
        time = np.arange(sample_rate // 4, dtype=np.float32) / sample_rate
        waveform = 0.2 * np.sin(2 * np.pi * (220 + index * 20) * time)
        audio = np.stack((waveform, waveform * 0.8), axis=1) if index == 0 else waveform
        path = root / f"item-{index}.wav"
        soundfile.write(path, audio, sample_rate, subtype=subtype)
        paths.append(path)
    return paths


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
    assert np.allclose(batch.mel[:, :, -2:].numpy(), LOG_MEL_FLOOR)


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
    with np.load(tmp_path / "items/train.npz", allow_pickle=False) as item:
        values = {name: item[name] for name in item.files}
    values["audio"] = values["audio"][:-1]
    np.savez(tmp_path / "items/train.npz", **values)
    dataset = VocoderDataset(
        tmp_path,
        "train",
        crop_frames=8,
        hop_size=4,
        seed=1,
        verify_item_hashes=False,
    )
    with pytest.raises(ValueError, match="arrays are unaligned"):
        dataset[0]


def test_epoch_sampler_is_deterministic_and_stateful() -> None:
    sampler = EpochShuffleSampler(20, seed=114514)
    first = list(sampler)
    assert first == list(EpochShuffleSampler(20, seed=114514))
    sampler.set_epoch(1)
    second = list(sampler)
    assert second != first
    restored = EpochShuffleSampler(20, seed=114514)
    restored.load_state_dict(sampler.state_dict())
    assert list(restored) == second


def test_epoch_sampler_partitions_ranks_and_distinguishes_padded_positions() -> None:
    rank_zero = list(EpochShuffleSampler(5, seed=114514, num_replicas=2, rank=0))
    rank_one = list(EpochShuffleSampler(5, seed=114514, num_replicas=2, rank=1))
    combined = sorted(rank_zero + rank_one, key=lambda item: item[1])

    assert [position for _, position in combined] == list(range(6))
    assert len({index for index, _ in combined}) == 5
    repeated_index = next(
        index for index, _ in combined if sum(i == index for i, _ in combined) == 2
    )
    repeated_positions = [position for index, position in combined if index == repeated_index]
    assert len(set(repeated_positions)) == 2


def test_evaluation_sampler_partitions_without_counting_padding() -> None:
    rank_zero = list(DistributedEvaluationSampler(5, num_replicas=2, rank=0))
    rank_one = list(DistributedEvaluationSampler(5, num_replicas=2, rank=1))
    combined = sorted(rank_zero + rank_one, key=lambda item: item[1])

    assert [position for _, position, _ in combined] == list(range(6))
    assert sum(weight for _, _, weight in combined) == 5
    assert combined[-1][2] == 0


def test_dataset_randomness_includes_logical_position(tmp_path: Path) -> None:
    _write_dataset(tmp_path, 32)
    dataset = VocoderDataset(tmp_path, "train", crop_frames=8, hop_size=4, seed=114514)

    first = dataset[(0, 1)]
    repeated = dataset[(0, 1)]
    different_position = dataset[(0, 2)]

    assert first.metadata == repeated.metadata
    assert first.metadata[0]["logical_position"] == 1
    assert different_position.metadata[0]["logical_position"] == 2


@pytest.mark.parametrize("epoch", [None, -1, True, "1"])
def test_epoch_sampler_rejects_invalid_restored_epoch(epoch: object) -> None:
    sampler = EpochShuffleSampler(5, seed=1)
    state: dict[str, object] = sampler.state_dict()
    state["epoch"] = epoch
    with pytest.raises(ValueError, match="epoch"):
        sampler.validate_state_dict(state)


def test_preprocessing_is_reproducible_across_pcm_widths_and_sample_rates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_audio_sources(source)
    features = FeatureConfig()
    first = preprocess_dataset(source, tmp_path / "first", features, validation_size=1)
    second = preprocess_dataset(source, tmp_path / "second", features, validation_size=1)
    assert (first.source_count, first.train_count, first.validation_count) == (3, 2, 1)
    assert first.fingerprint == second.fingerprint
    assert (first.output_dir / "manifest.jsonl").read_text() == (
        second.output_dir / "manifest.jsonl"
    ).read_text()
    metadata = json.loads((first.output_dir / "dataset.json").read_text(encoding="utf-8"))
    assert metadata["format_version"] == 2
    assert metadata["feature_implementation"]["version"] == 3
    manifest_lines = (first.output_dir / "manifest.jsonl").read_text().splitlines()
    manifest = [json.loads(line) for line in manifest_lines]
    assert all(
        record["item_sha256"] == sha256_file(first.output_dir / record["item"])
        for record in manifest
    )
    assert {record["source_channels"] for record in manifest} == {1, 2}
    assert {record["source_subtype"] for record in manifest} == {"PCM_16", "PCM_24"}
    assert all(record["source_frames"] > 0 for record in manifest)
    assert all(record["source_format"] == "WAV" for record in manifest)
    for path in (first.output_dir / "items").glob("*.npz"):
        with np.load(path, allow_pickle=False) as item:
            assert set(item.files) == {"audio", "mel", "f0", "uv", "pe"}
            assert len(item["audio"]) == len(item["mel"]) * features.hop_size
            assert all(np.isfinite(item[name]).all() for name in ("audio", "mel", "f0"))
            np.testing.assert_allclose(item["mel"], mel_spectrogram(item["audio"], features))

    stale_marker = first.output_dir / "stale-marker"
    stale_marker.write_text("old", encoding="utf-8")
    replaced = preprocess_dataset(source, first.output_dir, features, validation_size=1)
    assert replaced.fingerprint == first.fingerprint
    assert not stale_marker.exists()
    assert (replaced.output_dir / "manifest.jsonl").read_text() == (
        second.output_dir / "manifest.jsonl"
    ).read_text()

    data_config = DataConfig(
        processed_dir=first.output_dir,
        features=features,
        validation_size=1,
        crop_frames=4,
        batch_size=1,
        num_workers=0,
    )
    loaders = build_dataloaders(data_config, AugmentationConfig())
    torch.manual_seed(1234)
    expected = torch.rand(4)
    torch.manual_seed(1234)
    iter(loaders.train)
    assert torch.equal(torch.rand(4), expected)


def test_preprocess_rejects_output_inside_source_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    paths = _write_audio_sources(source, count=2)
    output = source / "processed"

    with pytest.raises(ValueError, match="must not contain one another"):
        preprocess_dataset(source, output, FeatureConfig(), validation_size=1)

    assert all(path.is_file() for path in paths)
    assert not output.exists()


@pytest.mark.parametrize("precreate_file", [False, True])
def test_preprocess_refuses_to_replace_unowned_directory(
    tmp_path: Path, precreate_file: bool
) -> None:
    source = tmp_path / "source"
    _write_audio_sources(source, count=2)
    output = tmp_path / "unrelated"
    output.mkdir()
    marker = output / "keep.ckpt"
    if precreate_file:
        marker.write_bytes(b"preserve")

    with pytest.raises(FileExistsError, match="not a valid processed dataset"):
        preprocess_dataset(source, output, FeatureConfig(), validation_size=1)

    assert output.is_dir()
    if precreate_file:
        assert marker.read_bytes() == b"preserve"


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_preprocess_rejects_junction_output_without_touching_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_audio_sources(source, count=2)
    target = tmp_path / "important"
    target.mkdir()
    marker = target / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    output = tmp_path / "output-link"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(output), str(target)],
        check=False,
        capture_output=True,
    )
    if created.returncode != 0:
        pytest.skip("junction creation is unavailable")
    try:
        with pytest.raises(ValueError, match="reparse point"):
            preprocess_dataset(source, output, FeatureConfig(), validation_size=1)
        assert marker.read_text(encoding="utf-8") == "keep"
        assert not (target / "dataset.json").exists()
    finally:
        output.rmdir()


def test_failed_preprocess_preserves_published_dataset_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source_paths = _write_audio_sources(source, count=2)
    output = tmp_path / "published"
    features = FeatureConfig()
    published = preprocess_dataset(source, output, features, validation_size=1)
    metadata_before = (published.output_dir / "dataset.json").read_bytes()

    import singing_vocoders.data.preprocess as preprocess_module

    def fail_item(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("injected preprocessing failure")

    monkeypatch.setattr(preprocess_module, "_process_one", fail_item)
    with pytest.raises(RuntimeError, match="failed to preprocess 2 of 2 files"):
        preprocess_dataset(source, output, features, validation_size=1)

    assert (output / "dataset.json").read_bytes() == metadata_before
    assert all(path.is_file() for path in source_paths)
    assert not list(tmp_path.glob(".published.staging-*"))
    assert not list(tmp_path.glob(".published.backup-*"))
    report = json.loads(
        (tmp_path / ".published.preprocess-failures.json").read_text(encoding="utf-8")
    )
    assert len(report["failures"]) == 2
    assert all("injected preprocessing failure" in item["error"] for item in report["failures"])


def test_failed_publication_rolls_back_the_previous_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _write_audio_sources(source, count=2)
    output = tmp_path / "published"
    features = FeatureConfig()
    preprocess_dataset(source, output, features, validation_size=1)
    metadata_before = (output / "dataset.json").read_bytes()

    import singing_vocoders.data.preprocess as preprocess_module

    replace = preprocess_module.os.replace

    def fail_staging_publish(source_path: str | Path, destination_path: str | Path) -> None:
        source_name = Path(source_path).name
        if source_name.startswith(".published.staging-") and Path(destination_path) == output:
            raise OSError("injected publication failure")
        replace(source_path, destination_path)

    monkeypatch.setattr(preprocess_module.os, "replace", fail_staging_publish)
    with pytest.raises(OSError, match="injected publication failure"):
        preprocess_dataset(source, output, features, validation_size=1)

    assert (output / "dataset.json").read_bytes() == metadata_before
    assert not list(tmp_path.glob(".published.staging-*"))
    assert not list(tmp_path.glob(".published.backup-*"))


def test_tampered_npz_is_rejected_when_dataset_metadata_is_loaded(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_audio_sources(source, count=2)
    output = tmp_path / "processed"
    features = FeatureConfig()
    preprocess_dataset(source, output, features, validation_size=1)
    record = json.loads((output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    item_path = output / record["item"]
    with np.load(item_path, allow_pickle=False) as item:
        arrays = {name: item[name] for name in item.files}
    arrays["audio"] = arrays["audio"].copy()
    arrays["audio"][0] += 0.25
    np.savez(item_path, **arrays)

    config = DataConfig(processed_dir=output, features=features)
    with pytest.raises(ValueError, match="content hash does not match"):
        load_dataset_metadata(config)
