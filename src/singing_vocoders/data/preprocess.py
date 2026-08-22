from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import unicodedata
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile

from singing_vocoders.config.schema import FeatureConfig
from singing_vocoders.data.features import extract_pitch, feature_implementation, mel_spectrogram
from singing_vocoders.data.integrity import (
    DATASET_FORMAT_VERSION,
    dataset_fingerprint,
    sha256_file,
    validate_processed_dataset,
)


@dataclass(frozen=True, slots=True)
class PreprocessSummary:
    source_count: int
    train_count: int
    validation_count: int
    fingerprint: str
    output_dir: Path


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
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
    frames = len(audio) // config.hop_size
    if frames < 1:
        raise ValueError("audio is shorter than one feature frame")
    aligned_samples = frames * config.hop_size
    audio = audio[:aligned_samples]
    mel = mel_spectrogram(audio, config)
    if mel.shape[0] != frames:
        raise ValueError("mel extractor returned an unexpected frame count")
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
        "source_sha256": sha256_file(source),
        "item_sha256": sha256_file(destination),
        "source_sample_rate": int(sample_rate),
        "source_channels": int(info.channels),
        "source_frames": int(info.frames),
        "source_format": info.format,
        "source_subtype": info.subtype,
        "samples": int(aligned_samples),
        "frames": frames,
    }


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(value, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _generated_sibling(output: Path, purpose: str) -> Path:
    parent = output.parent.resolve()
    candidate = (parent / f".{output.name}.{purpose}-{uuid.uuid4().hex}").resolve()
    if candidate.parent != parent or not candidate.name.startswith(f".{output.name}.{purpose}-"):
        raise RuntimeError("failed to create a safe preprocessing work path")
    return candidate


def _remove_generated_tree(path: Path, output: Path, purpose: str) -> None:
    resolved = path.resolve()
    parent = output.parent.resolve()
    if resolved.parent != parent or not resolved.name.startswith(f".{output.name}.{purpose}-"):
        raise RuntimeError(f"refusing to remove an unsafe preprocessing path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _existing_dataset_fingerprint(output: Path) -> str | None:
    if not output.exists():
        return None
    if not output.is_dir():
        raise NotADirectoryError(output)
    try:
        validated = validate_processed_dataset(output)
    except Exception as error:
        raise FileExistsError(
            "refusing to replace an existing directory that is not a valid processed dataset: "
            f"{output}"
        ) from error
    fingerprint = validated.metadata.get("fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError(f"processed dataset fingerprint is invalid: {output}")
    return fingerprint


def _publish_staging(
    staging: Path,
    output: Path,
    expected_existing_fingerprint: str | None,
) -> None:
    if output.is_symlink():
        raise ValueError(f"output directory must not be a symbolic link: {output}")
    current_fingerprint = _existing_dataset_fingerprint(output)
    if expected_existing_fingerprint is None:
        if current_fingerprint is not None:
            raise FileExistsError(f"output directory appeared during preprocessing: {output}")
        os.replace(staging, output)
        return
    if current_fingerprint != expected_existing_fingerprint:
        raise RuntimeError(f"output dataset changed during preprocessing: {output}")

    backup = _generated_sibling(output, "backup")
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        os.replace(backup, output)
        raise
    with suppress(OSError):
        _remove_generated_tree(backup, output, "backup")


def _reject_reparse_path(path: Path) -> None:
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for candidate in (path, *path.parents):
        try:
            status = os.lstat(candidate)
        except FileNotFoundError:
            continue
        attributes = getattr(status, "st_file_attributes", 0)
        if stat.S_ISLNK(status.st_mode) or attributes & reparse_attribute:
            raise ValueError(f"output path must not traverse a link or reparse point: {candidate}")


def _failure_report_path(output: Path) -> Path:
    return output.parent / f".{output.name}.preprocess-failures.json"


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
    lexical_output = Path(output_dir).absolute()
    _reject_reparse_path(lexical_output)
    output = lexical_output.resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if output == output.parent:
        raise ValueError("output directory must not be a filesystem root")
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("source and output directories must not contain one another")
    if workers < 0:
        raise ValueError("workers must not be negative")
    if validation_size < 1:
        raise ValueError("validation_size must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_fingerprint = _existing_dataset_fingerprint(output)
    paths = sorted(
        (path for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: _normalized_relative(path, source),
    )
    if not paths:
        raise ValueError(f"source directory contains no WAV files: {source}")
    if validation_size >= len(paths):
        raise ValueError("validation_size must be smaller than the number of source files")
    relative_items = [
        str(Path(_normalized_relative(path, source)).with_suffix(".npz")) for path in paths
    ]
    if len({item.casefold() for item in relative_items}) != len(relative_items):
        raise ValueError("source paths collide after conversion to NPZ item paths")

    staging = _generated_sibling(output, "staging")
    staging.mkdir()
    report_path = _failure_report_path(output)
    feature_values = features.model_dump(mode="json")
    implementation = feature_implementation(features)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    try:
        if workers <= 1:
            for path in paths:
                try:
                    records.append(
                        _process_one(str(source), str(path), str(staging), feature_values)
                    )
                except Exception as error:
                    failures.append(
                        {"source": _normalized_relative(path, source), "error": repr(error)}
                    )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _process_one, str(source), str(path), str(staging), feature_values
                    ): path
                    for path in paths
                }
                for future in as_completed(futures):
                    try:
                        records.append(future.result())
                    except Exception as error:
                        failures.append(
                            {
                                "source": _normalized_relative(futures[future], source),
                                "error": repr(error),
                            }
                        )

        if failures:
            raise RuntimeError(f"failed to preprocess {len(failures)} of {len(paths)} files")
        records.sort(key=lambda record: record["source"])
        ranked = sorted(
            records,
            key=lambda record: hashlib.sha256(f"{seed}:{record['source']}".encode()).digest(),
        )
        validation_sources = {record["source"] for record in ranked[:validation_size]}
        for record in records:
            record["split"] = "valid" if record["source"] in validation_sources else "train"
        fingerprint = dataset_fingerprint(records, feature_values, implementation)
        manifest = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        train = "".join(record["item"] + "\n" for record in records if record["split"] == "train")
        valid = "".join(record["item"] + "\n" for record in records if record["split"] == "valid")
        metadata = {
            "format_version": DATASET_FORMAT_VERSION,
            "fingerprint": fingerprint,
            "source": str(source),
            "features": feature_values,
            "feature_implementation": implementation,
            "seed": seed,
            "counts": {
                "all": len(records),
                "train": len(records) - validation_size,
                "valid": validation_size,
            },
        }
        _write_text_atomic(staging / "manifest.jsonl", manifest)
        _write_text_atomic(staging / "train.txt", train)
        _write_text_atomic(staging / "valid.txt", valid)
        _write_text_atomic(staging / "dataset.json", json.dumps(metadata, sort_keys=True, indent=2))
        validated = validate_processed_dataset(staging, expected_features=features)
        if validated.metadata["fingerprint"] != fingerprint:
            raise ValueError("staged dataset fingerprint changed during validation")
        _publish_staging(staging, output, existing_fingerprint)
    except BaseException as error:
        if not failures:
            failures.append({"source": "<dataset>", "error": repr(error)})
        _write_text_atomic(
            report_path,
            json.dumps(
                {
                    "format_version": 1,
                    "source": str(source),
                    "output": str(output),
                    "failures": failures,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        raise
    finally:
        if staging.exists():
            _remove_generated_tree(staging, output, "staging")

    report_path.unlink(missing_ok=True)
    return PreprocessSummary(
        len(records), len(records) - validation_size, validation_size, fingerprint, output
    )
