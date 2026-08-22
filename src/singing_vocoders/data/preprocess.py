from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile

from singing_vocoders.config.schema import FeatureConfig
from singing_vocoders.data.features import extract_pitch, mel_spectrogram

DATASET_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class PreprocessSummary:
    source_count: int
    train_count: int
    validation_count: int
    fingerprint: str
    output_dir: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_relative(path: Path, root: Path) -> str:
    return unicodedata.normalize("NFC", path.relative_to(root).as_posix())


def _process_one(
    source_root: str,
    source_path: str,
    output_root: str,
    feature_values: dict[str, Any],
) -> dict[str, Any]:
    root = Path(source_root)
    source = Path(source_path)
    output = Path(output_root)
    config = FeatureConfig.model_validate(feature_values)
    relative = _normalized_relative(source, root)
    destination = output / "items" / Path(relative).with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)

    info = soundfile.info(source)
    audio, sample_rate = soundfile.read(source, dtype="float32", always_2d=True)
    audio = np.mean(audio, axis=1, dtype=np.float32)
    if sample_rate != config.sample_rate:
        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=config.sample_rate, res_type="soxr_hq"
        ).astype(np.float32, copy=False)
    audio = np.clip(audio, -1.0, 1.0)
    mel = mel_spectrogram(audio, config)
    frames = int(mel.shape[0])
    if frames < 1:
        raise ValueError("audio is too short to produce a mel frame")
    aligned_samples = frames * config.hop_size
    audio = np.pad(audio[:aligned_samples], (0, max(0, aligned_samples - len(audio))))
    f0, uv, pe = extract_pitch(audio, frames, config)
    arrays = (audio, mel, f0, uv)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("preprocessed arrays contain non-finite values")

    temporary = destination.with_suffix(".npz.tmp.npz")
    np.savez(
        temporary,
        audio=audio.astype(np.float32, copy=False),
        mel=mel,
        f0=f0,
        uv=uv,
        pe=np.asarray(pe, dtype=np.int64),
    )
    os.replace(temporary, destination)
    return {
        "source": relative,
        "item": destination.relative_to(output).as_posix(),
        "source_sha256": _sha256_file(source),
        "source_sample_rate": int(sample_rate),
        "source_subtype": info.subtype,
        "samples": int(aligned_samples),
        "frames": frames,
    }


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _dataset_fingerprint(records: list[dict[str, Any]], features: dict[str, Any]) -> str:
    payload = json.dumps(
        {"format_version": DATASET_FORMAT_VERSION, "features": features, "items": records},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preprocess_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    features: FeatureConfig,
    *,
    validation_size: int = 10,
    seed: int = 114_514,
    workers: int = 1,
) -> PreprocessSummary:
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if workers < 0:
        raise ValueError("workers must not be negative")
    paths = sorted(
        (path for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: _normalized_relative(path, source),
    )
    if not paths:
        raise ValueError(f"source directory contains no WAV files: {source}")
    if validation_size >= len(paths):
        raise ValueError("validation_size must be smaller than the number of source files")
    output.mkdir(parents=True, exist_ok=True)
    feature_values = features.model_dump(mode="json")
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    if workers <= 1:
        for path in paths:
            try:
                records.append(_process_one(str(source), str(path), str(output), feature_values))
            except Exception as error:
                failures.append({"source": str(path), "error": repr(error)})
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_one, str(source), str(path), str(output), feature_values
                ): path
                for path in paths
            }
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except Exception as error:
                    failures.append({"source": str(futures[future]), "error": repr(error)})

    if failures:
        _write_text_atomic(
            output / "failures.json", json.dumps(failures, ensure_ascii=False, indent=2)
        )
        raise RuntimeError(f"failed to preprocess {len(failures)} of {len(paths)} files")
    records.sort(key=lambda record: record["source"])
    ranked = sorted(
        records,
        key=lambda record: hashlib.sha256(f"{seed}:{record['source']}".encode()).digest(),
    )
    validation_sources = {record["source"] for record in ranked[:validation_size]}
    for record in records:
        record["split"] = "valid" if record["source"] in validation_sources else "train"
    fingerprint = _dataset_fingerprint(records, feature_values)
    manifest = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    train = "".join(record["item"] + "\n" for record in records if record["split"] == "train")
    valid = "".join(record["item"] + "\n" for record in records if record["split"] == "valid")
    metadata = {
        "format_version": DATASET_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "source": str(source),
        "features": feature_values,
        "seed": seed,
        "counts": {
            "all": len(records),
            "train": len(records) - validation_size,
            "valid": validation_size,
        },
    }
    _write_text_atomic(output / "manifest.jsonl", manifest)
    _write_text_atomic(output / "train.txt", train)
    _write_text_atomic(output / "valid.txt", valid)
    _write_text_atomic(output / "dataset.json", json.dumps(metadata, sort_keys=True, indent=2))
    failure_path = output / "failures.json"
    if failure_path.exists():
        failure_path.unlink()
    return PreprocessSummary(
        len(records), len(records) - validation_size, validation_size, fingerprint, output
    )
