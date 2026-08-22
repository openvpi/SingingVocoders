from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from singing_vocoders.config.schema import FeatureConfig
from singing_vocoders.data.features import feature_implementation

DATASET_FORMAT_VERSION = 2
ITEM_FIELDS = frozenset({"audio", "mel", "f0", "uv", "pe"})


@dataclass(frozen=True, slots=True)
class ValidatedDataset:
    metadata: dict[str, Any]
    records: tuple[dict[str, Any], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(
    records: list[dict[str, Any]],
    features: dict[str, Any],
    implementation: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "format_version": DATASET_FORMAT_VERSION,
            "features": features,
            "feature_implementation": implementation,
            "items": records,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_indices(root: Path) -> tuple[list[str], list[str]]:
    indices: list[list[str]] = []
    for split in ("train", "valid"):
        path = root / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        values = path.read_text(encoding="utf-8").splitlines()
        if any(not value for value in values):
            raise ValueError(f"dataset index contains an empty path: {path}")
        indices.append(values)
    return indices[0], indices[1]


def _resolve_item(root: Path, relative: str) -> Path:
    pure_path = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or pure_path.is_absolute()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise ValueError(f"invalid dataset item path: {relative!r}")
    path = (root / Path(*pure_path.parts)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"dataset item escapes its root: {relative!r}")
    return path


def load_validated_item(
    root: Path,
    record: dict[str, Any],
    features: FeatureConfig,
    *,
    verify_hash: bool = True,
) -> dict[str, NDArray[Any]]:
    relative = record.get("item")
    if not isinstance(relative, str):
        raise ValueError("dataset manifest item path is invalid")
    path = _resolve_item(root, relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = record.get("item_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"dataset item hash is invalid: {relative}")
    if verify_hash and sha256_file(path) != expected_hash:
        raise ValueError(f"dataset item content hash does not match: {relative}")

    try:
        with np.load(path, allow_pickle=False) as item:
            if set(item.files) != ITEM_FIELDS:
                raise ValueError(f"dataset item fields are invalid: {relative}")
            arrays = {name: item[name] for name in ITEM_FIELDS}
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, ValueError) and str(error).startswith("dataset item fields"):
            raise
        raise ValueError(f"cannot read dataset item: {relative}") from error

    audio = arrays["audio"]
    mel = arrays["mel"]
    f0 = arrays["f0"]
    uv = arrays["uv"]
    pitch_extractor = arrays["pe"]
    frames = mel.shape[0] if mel.ndim == 2 else -1
    expected_frames = record.get("frames")
    expected_samples = record.get("samples")
    if (
        audio.ndim != 1
        or mel.ndim != 2
        or mel.shape[1] != features.num_mels
        or f0.ndim != 1
        or uv.ndim != 1
        or pitch_extractor.ndim != 0
    ):
        raise ValueError(f"dataset item array shapes are invalid: {relative}")
    if (
        not isinstance(expected_frames, int)
        or expected_frames < 1
        or not isinstance(expected_samples, int)
        or expected_samples < 1
        or frames != expected_frames
        or len(audio) != expected_samples
        or len(audio) != frames * features.hop_size
        or len(f0) != frames
        or len(uv) != frames
    ):
        raise ValueError(f"dataset item arrays are unaligned: {relative}")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError(f"dataset item contains non-finite values: {relative}")
    return arrays


def validate_processed_dataset(
    root: str | Path,
    *,
    expected_features: FeatureConfig | None = None,
) -> ValidatedDataset:
    dataset_root = Path(root).resolve()
    metadata_path = dataset_root / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = _load_json_object(metadata_path)
    format_version = metadata.get("format_version")
    raw_features = metadata.get("features")
    if not isinstance(raw_features, dict):
        raise ValueError("processed dataset feature configuration is missing")
    features = FeatureConfig.model_validate(raw_features)
    if expected_features is not None and features != expected_features:
        raise ValueError("processed dataset feature configuration does not match the recipe")
    if format_version != DATASET_FORMAT_VERSION:
        raise ValueError(f"unsupported dataset format: {format_version}")

    implementation = feature_implementation(features)
    if metadata.get("feature_implementation") != implementation:
        raise ValueError("processed dataset feature implementation does not match this release")
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid dataset manifest line {line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"dataset manifest line {line_number} is not an object")
        records.append(record)
    if not records:
        raise ValueError("dataset manifest is empty")
    sources = [record.get("source") for record in records]
    items = [record.get("item") for record in records]
    if any(not isinstance(value, str) or not value for value in (*sources, *items)):
        raise ValueError("dataset manifest contains invalid source or item paths")
    source_paths = cast(list[str], sources)
    item_paths = cast(list[str], items)
    if source_paths != sorted(source_paths) or len(set(source_paths)) != len(source_paths):
        raise ValueError("dataset manifest sources must be unique and sorted")
    if len(set(item_paths)) != len(item_paths):
        raise ValueError("dataset manifest item paths must be unique")

    counts = metadata.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("processed dataset counts are invalid")
    train_records = [record for record in records if record.get("split") == "train"]
    valid_records = [record for record in records if record.get("split") == "valid"]
    if len(train_records) + len(valid_records) != len(records):
        raise ValueError("dataset manifest contains an invalid split")
    if counts != {
        "all": len(records),
        "train": len(train_records),
        "valid": len(valid_records),
    }:
        raise ValueError("dataset manifest counts do not match dataset metadata")
    train_index, valid_index = _read_indices(dataset_root)
    if train_index != [str(record["item"]) for record in train_records] or valid_index != [
        str(record["item"]) for record in valid_records
    ]:
        raise ValueError("dataset indices do not match the manifest")

    expected_fingerprint = dataset_fingerprint(records, raw_features, implementation)
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError("processed dataset fingerprint does not match its manifest")
    for record in records:
        load_validated_item(dataset_root, record, features)
    return ValidatedDataset(metadata, tuple(records))
