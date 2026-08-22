from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
import torch
import torch.distributed
from torch import Tensor

from singing_vocoders.engine.engine import FabricGanEngine
from singing_vocoders.task.task import VocoderTask

CHECKPOINT_FORMAT_VERSION = 1
_STEP_PATTERN = re.compile(r"step-(\d{8,})\.ckpt")
_MUTABLE_RESUME_FIELDS = {"max_steps", "log_every", "validate_every"}


@runtime_checkable
class StatefulSampler(Protocol):
    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class LegacyLoadReport:
    path: Path
    scope: Literal["generator", "all"]
    key_count: int
    schema_hash: str


def _capture_rng() -> dict[str, Any]:
    numpy_state = cast(tuple[Any, Any, Any, Any, Any], np.random.get_state())
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state().tolist(),
        "cuda": (
            [state.tolist() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _capture_rank_rng(engine: FabricGanEngine) -> list[dict[str, Any]]:
    local = _capture_rng()
    if engine.fabric.world_size == 1:
        return [local]
    if not torch.distributed.is_initialized():
        raise RuntimeError("distributed RNG capture requires an initialized process group")
    states: list[dict[str, Any] | None] = [None] * engine.fabric.world_size
    torch.distributed.all_gather_object(states, local)
    if any(state is None for state in states):
        raise RuntimeError("failed to gather RNG state from every rank")
    return cast(list[dict[str, Any]], states)


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(
            [torch.tensor(item, dtype=torch.uint8) for item in state["cuda"]]
        )


def _schema_hash(state: dict[str, Tensor]) -> str:
    entries = "\n".join(
        f"{name}:{','.join(map(str, tensor.shape))}:{tensor.dtype}"
        for name, tensor in sorted(state.items())
    )
    return hashlib.sha256(entries.encode()).hexdigest()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _resume_fingerprint(config: dict[str, Any]) -> str:
    normalized = copy.deepcopy(config)
    runtime = normalized.get("runtime")
    if isinstance(runtime, dict):
        for field in _MUTABLE_RESUME_FIELDS:
            runtime.pop(field, None)
    return _fingerprint(normalized)


def _tensor_state(value: Any) -> dict[str, Tensor]:
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(tensor, Tensor) for name, tensor in value.items()
    ):
        raise TypeError("checkpoint state must be a string-to-tensor mapping")
    return cast(dict[str, Tensor], value)


def _extract_legacy_state(checkpoint: Any) -> tuple[dict[str, Tensor], str]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint root must be a mapping")
    if "state_dict" in checkpoint:
        return _tensor_state(checkpoint["state_dict"]), "state_dict"
    if "generator" in checkpoint:
        generator = _tensor_state(checkpoint["generator"])
        return {f"generator.{name}": tensor for name, tensor in generator.items()}, "generator"
    if "model" in checkpoint:
        return _tensor_state(checkpoint["model"]), "model"
    return _tensor_state(checkpoint), "raw"


def _validate_state_schema(expected: dict[str, Tensor], actual: dict[str, Tensor]) -> None:
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    if missing or unexpected:
        raise RuntimeError(
            f"state name mismatch: missing={missing or None}, unexpected={unexpected or None}"
        )
    mismatched = [
        f"{name}: expected {tuple(expected[name].shape)}, found {tuple(actual[name].shape)}"
        for name in expected
        if expected[name].shape != actual[name].shape
    ]
    if mismatched:
        raise RuntimeError("state shape mismatch: " + "; ".join(mismatched))


def load_legacy_weights(
    task: VocoderTask,
    path: str | Path,
    *,
    scope: Literal["generator", "all"] = "all",
) -> LegacyLoadReport:
    checkpoint_path = Path(path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    state, _ = _extract_legacy_state(checkpoint)
    if scope == "generator":
        prefixed = {
            name.removeprefix("generator."): tensor
            for name, tensor in state.items()
            if name.startswith("generator.")
        }
        if prefixed:
            selected = prefixed
        elif not any(name.startswith("discriminator.") for name in state):
            selected = state
        else:
            raise ValueError("checkpoint contains no generator weights")
        _validate_state_schema(task.generator.state_dict(), selected)
        task.generator.load_state_dict(selected, strict=True)
        schema_state = selected
    else:
        selected = {
            name: tensor
            for name, tensor in state.items()
            if name.startswith(("generator.", "discriminator."))
        }
        if not selected:
            raise ValueError("checkpoint contains no complete task weights")
        _validate_state_schema(task.state_dict(), selected)
        task.load_state_dict(selected, strict=True)
        schema_state = selected
    return LegacyLoadReport(checkpoint_path, scope, len(schema_state), _schema_hash(schema_state))


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="meta", mmap=True, weights_only=True)
    state, container = _extract_legacy_state(checkpoint)
    return {
        "path": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "container": container,
        "format_version": checkpoint.get("format_version")
        if isinstance(checkpoint, dict)
        else None,
        "keys": len(state),
        "parameters": sum(tensor.numel() for tensor in state.values()),
        "schema_hash": _schema_hash(state),
        "dtypes": sorted({str(tensor.dtype) for tensor in state.values()}),
    }


def _sampler_identity(sampler: object | None) -> str | None:
    if sampler is None:
        return None
    sampler_type = type(sampler)
    return f"{sampler_type.__module__}.{sampler_type.__qualname__}"


def _capture_sampler(sampler: object | None, engine: FabricGanEngine) -> dict[str, Any]:
    state = sampler.state_dict() if isinstance(sampler, StatefulSampler) else None
    return {
        "identity": _sampler_identity(sampler),
        "state": state,
        "epoch": engine.state.epoch,
        "batches_in_epoch": engine.state.batches_in_epoch,
    }


def _restore_sampler(
    sampler: object | None, saved: dict[str, Any], engine: FabricGanEngine
) -> None:
    if saved.get("identity") != _sampler_identity(sampler):
        raise ValueError("checkpoint sampler type does not match")
    if (
        saved.get("epoch") != engine.state.epoch
        or saved.get("batches_in_epoch") != engine.state.batches_in_epoch
    ):
        raise ValueError("checkpoint sampler position does not match engine state")
    sampler_state = saved.get("state")
    if sampler_state is not None:
        if not isinstance(sampler, StatefulSampler):
            raise ValueError("checkpoint requires a stateful sampler")
        sampler.load_state_dict(sampler_state)


class CheckpointManager:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        config: dict[str, Any],
        config_fingerprint: str,
        data_fingerprint: str,
        keep_last: int,
    ) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be positive")
        self.run_dir = Path(run_dir).resolve()
        self.config = config
        self.config_fingerprint = config_fingerprint
        self.resume_fingerprint = _resume_fingerprint(config)
        self.data_fingerprint = data_fingerprint
        self.keep_last = keep_last

    def path_for_step(self, step: int) -> Path:
        return self.run_dir / f"step-{step:08d}.ckpt"

    def latest(self) -> Path | None:
        candidates: list[tuple[int, Path]] = []
        if not self.run_dir.is_dir():
            return None
        for path in self.run_dir.iterdir():
            match = _STEP_PATTERN.fullmatch(path.name)
            if match and path.is_file():
                candidates.append((int(match.group(1)), path))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def save(self, engine: FabricGanEngine, sampler: object | None = None) -> Path:
        if not engine.accumulation_boundary():
            raise RuntimeError("checkpoints may only be saved at an accumulation boundary")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        destination = self.path_for_step(engine.state.logical_step)
        if destination.exists():
            raise FileExistsError(destination)
        temporary = destination.with_suffix(".ckpt.tmp")
        state: dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model": engine.task,
            "engine_state": engine.state,
            "precision": engine.fabric.strategy.precision,
            "rng_by_rank": _capture_rank_rng(engine),
            "sampler": _capture_sampler(sampler, engine),
            "config": self.config,
            "config_fingerprint": self.config_fingerprint,
            "resume_fingerprint": self.resume_fingerprint,
            "data_fingerprint": self.data_fingerprint,
        }
        for name, slot in engine.slots.items():
            state[f"optimizer_{name}"] = slot.optimizer
            if slot.scheduler is not None:
                state[f"scheduler_{name}"] = slot.scheduler
        engine.fabric.save(temporary, state)
        engine.fabric.barrier()
        if engine.fabric.global_rank == 0:
            os.replace(temporary, destination)
            self._prune()
        engine.fabric.barrier()
        return destination

    def load(
        self,
        engine: FabricGanEngine,
        path: str | Path,
        sampler: object | None = None,
    ) -> None:
        checkpoint_path = Path(path).resolve()
        metadata = torch.load(checkpoint_path, map_location="meta", mmap=True, weights_only=True)
        if not isinstance(metadata, dict):
            raise TypeError("checkpoint root must be a mapping")
        if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format: {metadata.get('format_version')}")
        saved_config = metadata.get("config")
        if not isinstance(saved_config, dict):
            raise TypeError("checkpoint contains no resolved configuration")
        if metadata.get("resume_fingerprint") != _resume_fingerprint(saved_config):
            raise ValueError("checkpoint resume fingerprint is corrupt")
        if metadata.get("resume_fingerprint") != self.resume_fingerprint:
            raise ValueError("checkpoint model, data, or optimization configuration does not match")
        if metadata.get("data_fingerprint") != self.data_fingerprint:
            raise ValueError("checkpoint data fingerprint does not match")
        state: dict[str, Any] = {
            "model": engine.task,
            "engine_state": engine.state,
            "precision": engine.fabric.strategy.precision,
        }
        for name, slot in engine.slots.items():
            state[f"optimizer_{name}"] = slot.optimizer
            if slot.scheduler is not None:
                state[f"scheduler_{name}"] = slot.scheduler
        remainder = engine.fabric.load(checkpoint_path, state, strict=True, weights_only=True)
        rng_by_rank = remainder.pop("rng_by_rank")
        if len(rng_by_rank) != engine.fabric.world_size:
            raise ValueError("checkpoint world size does not match")
        _restore_rng(rng_by_rank[engine.fabric.global_rank])
        _restore_sampler(sampler, remainder.pop("sampler"), engine)

    def _prune(self) -> None:
        checkpoints = []
        for path in self.run_dir.iterdir():
            match = _STEP_PATTERN.fullmatch(path.name)
            if match and path.is_file():
                checkpoints.append((int(match.group(1)), path))
        for _, path in sorted(checkpoints)[: -self.keep_last]:
            path.unlink()
