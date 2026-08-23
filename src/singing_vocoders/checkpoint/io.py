from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
import torch
import torch.distributed
from torch import Tensor

from singing_vocoders.config.schema import WarmupAlternatingPlanConfig
from singing_vocoders.engine.engine import FabricGanEngine
from singing_vocoders.engine.muon import Muon
from singing_vocoders.task.task import VocoderTask

CHECKPOINT_FORMAT_VERSION = 1
_STEP_PATTERN = re.compile(r"step-(\d{8,})\.ckpt")
_MUTABLE_RESUME_FIELDS = {"max_steps", "log_every", "validate_every"}


@runtime_checkable
class StatefulSampler(Protocol):
    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


@runtime_checkable
class ValidatingStatefulSampler(StatefulSampler, Protocol):
    def validate_state_dict(self, state: dict[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class LegacyLoadReport:
    path: Path
    scope: Literal["generator", "all"]
    key_count: int
    schema_hash: str


def _capture_rng(*, include_cuda: bool = True) -> dict[str, Any]:
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
            if include_cuda and torch.cuda.is_available()
            else []
        ),
    }


def _capture_rank_rng(engine: FabricGanEngine) -> list[dict[str, Any]]:
    local = _capture_rng(include_cuda=engine.fabric.device.type == "cuda")
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


def _validate_rng(state: object, *, expected_cuda_states: int) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise TypeError("checkpoint RNG state must be a mapping")
    required = {"python", "numpy", "torch", "cuda"}
    if set(state) != required:
        raise ValueError("checkpoint RNG state fields are invalid")
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, dict):
        raise TypeError("checkpoint NumPy RNG state must be a mapping")
    try:
        random.Random().setstate(state["python"])
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(
            (
                numpy_state["algorithm"],
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch_probe = torch.Generator()
        torch_probe.set_state(torch.tensor(state["torch"], dtype=torch.uint8))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("checkpoint RNG state is invalid") from error
    cuda_states = state["cuda"]
    if not isinstance(cuda_states, list) or any(not isinstance(item, list) for item in cuda_states):
        raise TypeError("checkpoint CUDA RNG state must be a list of byte lists")
    if len(cuda_states) != expected_cuda_states:
        raise ValueError("checkpoint CUDA RNG device count does not match")
    try:
        for device_index, item in enumerate(cuda_states):
            cuda_probe = torch.Generator(device=f"cuda:{device_index}")
            cuda_probe.set_state(torch.tensor(item, dtype=torch.uint8))
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("checkpoint CUDA RNG state is invalid") from error
    return cast(dict[str, Any], state)


@contextmanager
def preserve_rng_state() -> Iterator[None]:
    state = _capture_rng()
    try:
        yield
    finally:
        _restore_rng(state)


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


def _validate_state_schema(
    expected: dict[str, Tensor],
    actual: dict[str, Tensor],
    *,
    exact: bool = False,
) -> None:
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
    if exact:
        incompatible = [
            f"{name}: expected {expected[name].dtype}/{expected[name].layout}, "
            f"found {actual[name].dtype}/{actual[name].layout}"
            for name in expected
            if expected[name].dtype != actual[name].dtype
            or expected[name].layout != actual[name].layout
        ]
        if incompatible:
            raise RuntimeError("state tensor schema mismatch: " + "; ".join(incompatible))


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
        "state_elements": sum(tensor.numel() for tensor in state.values()),
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


def _preflight_sampler(
    sampler: object | None,
    saved: object,
    saved_engine_state: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(saved, dict):
        raise TypeError("checkpoint sampler state must be a mapping")
    if saved.get("identity") != _sampler_identity(sampler):
        raise ValueError("checkpoint sampler type does not match")
    if saved.get("epoch") != saved_engine_state.get("epoch") or saved.get(
        "batches_in_epoch"
    ) != saved_engine_state.get("batches_in_epoch"):
        raise ValueError("checkpoint sampler position does not match engine state")
    sampler_state = saved.get("state")
    if sampler_state is not None:
        if not isinstance(sampler_state, dict):
            raise TypeError("checkpoint sampler payload must be a mapping")
        if not isinstance(sampler, StatefulSampler):
            raise ValueError("checkpoint requires a stateful sampler")
        if isinstance(sampler, ValidatingStatefulSampler):
            sampler.validate_state_dict(sampler_state)
    return cast(dict[str, Any], saved)


def _validate_precision_state(expected: dict[str, Any], saved: dict[str, Any]) -> None:
    if set(saved) != set(expected):
        raise ValueError("checkpoint precision state fields do not match")
    if not expected:
        return
    required = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
    if set(expected) != required:
        raise ValueError("unsupported precision state format")
    for field in ("growth_factor", "backoff_factor", "growth_interval"):
        if saved[field] != expected[field]:
            raise ValueError(f"checkpoint precision option {field} does not match")
    scale = saved["scale"]
    growth_tracker = saved["_growth_tracker"]
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("checkpoint precision scale must be finite and positive")
    if (
        not isinstance(growth_tracker, int)
        or isinstance(growth_tracker, bool)
        or growth_tracker < 0
        or growth_tracker >= int(saved["growth_interval"])
    ):
        raise ValueError("checkpoint precision growth tracker is invalid")


def _validate_torch_optimizer_state(
    expected: dict[str, Any],
    saved: dict[str, Any],
    parameter_groups: list[list[Tensor]],
    label: str,
    *,
    state_kind: Literal["adamw", "muon"],
    allow_scheduled_lr: bool,
    successful_updates: int,
) -> None:
    if set(saved) != set(expected) or set(expected) != {"state", "param_groups"}:
        raise ValueError(f"checkpoint {label} optimizer fields do not match")
    expected_groups = expected["param_groups"]
    saved_groups = saved["param_groups"]
    saved_state = saved["state"]
    if not isinstance(expected_groups, list) or not isinstance(saved_groups, list):
        raise TypeError(f"checkpoint {label} parameter groups must be lists")
    if len(saved_groups) != len(expected_groups):
        raise ValueError(f"checkpoint {label} parameter-group count does not match")
    if len(parameter_groups) != len(expected_groups):
        raise ValueError(f"live {label} optimizer parameter-group count is invalid")
    if not isinstance(saved_state, dict):
        raise TypeError(f"checkpoint {label} optimizer state must be a mapping")

    parameter_ids: list[int] = []
    parameters_by_id: dict[int, Tensor] = {}
    options_by_id: dict[int, dict[str, Any]] = {}
    for group_index, (expected_group, saved_group, parameters) in enumerate(
        zip(expected_groups, saved_groups, parameter_groups, strict=True)
    ):
        if not isinstance(expected_group, dict) or not isinstance(saved_group, dict):
            raise TypeError(f"checkpoint {label} parameter group must be a mapping")
        if set(saved_group) != set(expected_group):
            raise ValueError(f"checkpoint {label} parameter-group fields do not match")
        expected_parameters = expected_group.get("params")
        saved_parameters = saved_group.get("params")
        if not isinstance(expected_parameters, list) or not isinstance(saved_parameters, list):
            raise TypeError(f"checkpoint {label} parameter IDs must be lists")
        if len(saved_parameters) != len(expected_parameters):
            raise ValueError(
                f"checkpoint {label} parameter-group {group_index} size does not match"
            )
        if len(parameters) != len(expected_parameters):
            raise ValueError(f"live {label} parameter-group {group_index} size is invalid")
        if not all(isinstance(parameter_id, int) for parameter_id in saved_parameters):
            raise TypeError(f"checkpoint {label} parameter IDs must be integers")
        if saved_parameters != expected_parameters:
            raise ValueError(f"checkpoint {label} parameter order does not match")
        for field, expected_value in expected_group.items():
            if field == "params" or (allow_scheduled_lr and field == "lr"):
                continue
            if saved_group[field] != expected_value:
                raise ValueError(
                    f"checkpoint {label} parameter-group option {field} does not match"
                )
        parameter_ids.extend(saved_parameters)
        parameters_by_id.update(zip(saved_parameters, parameters, strict=True))
        options_by_id.update((parameter_id, saved_group) for parameter_id in saved_parameters)
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError(f"checkpoint {label} contains duplicate parameter IDs")
    if any(
        not isinstance(parameter_id, int) or parameter_id not in parameter_ids
        for parameter_id in saved_state
    ):
        raise ValueError(f"checkpoint {label} state references an unknown parameter")
    for parameter_id, state in saved_state.items():
        if not isinstance(state, dict):
            raise TypeError(f"checkpoint {label} per-parameter state must be a mapping")
        if state_kind == "adamw":
            required_fields = {"step", "exp_avg", "exp_avg_sq"}
            if options_by_id[parameter_id].get("amsgrad") is True:
                required_fields.add("max_exp_avg_sq")
        elif state_kind == "muon":
            required_fields = {"momentum_buffer"}
        if set(state) != required_fields:
            raise ValueError(f"checkpoint {label} optimizer state fields do not match")
        parameter = parameters_by_id[parameter_id]
        for field, value in state.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"checkpoint {label} optimizer state {field} must be a tensor")
            if field == "step":
                if value.ndim != 0 or value.dtype != torch.float32:
                    raise ValueError(f"checkpoint {label} optimizer step must be a float32 scalar")
                step = float(value.item())
                if (
                    not math.isfinite(step)
                    or step < 1
                    or not step.is_integer()
                    or step > successful_updates
                ):
                    raise ValueError(f"checkpoint {label} optimizer step is invalid")
            elif value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(
                    f"checkpoint {label} optimizer state {field} does not match its parameter"
                )


def _parameter_groups(optimizer: object) -> list[list[Tensor]]:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list):
        raise TypeError("live optimizer parameter groups are invalid")
    result: list[list[Tensor]] = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise TypeError("live optimizer parameter group is invalid")
        parameters = group["params"]
        if not all(isinstance(parameter, Tensor) for parameter in parameters):
            raise TypeError("live optimizer parameters must be tensors")
        result.append(cast(list[Tensor], parameters))
    return result


def _validate_optimizer_state(
    optimizer: object,
    saved: object,
    label: str,
    *,
    allow_scheduled_lr: bool,
    successful_updates: int,
) -> None:
    if not isinstance(saved, dict):
        raise TypeError(f"checkpoint optimizer state is missing for {label}")
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    state_dict = getattr(optimizer, "state_dict", None)
    if not callable(state_dict):
        raise TypeError(f"live optimizer state is unavailable for {label}")
    expected = cast(dict[str, Any], state_dict())
    if set(expected) != {"state", "param_groups"}:
        raise ValueError(f"checkpoint {label} optimizer format does not match")
    if isinstance(raw_optimizer, Muon):
        state_kind: Literal["adamw", "muon"] = "muon"
    elif isinstance(raw_optimizer, torch.optim.AdamW):
        state_kind = "adamw"
    else:
        raise TypeError(f"unsupported live optimizer for {label}: {type(raw_optimizer).__name__}")
    _validate_torch_optimizer_state(
        expected,
        saved,
        _parameter_groups(raw_optimizer),
        label,
        state_kind=state_kind,
        allow_scheduled_lr=allow_scheduled_lr,
        successful_updates=successful_updates,
    )


def _outer_optimizer_groups(saved: dict[str, Any], label: str) -> list[dict[str, Any]]:
    if not isinstance(saved.get("param_groups"), list):
        raise TypeError(f"checkpoint {label} optimizer parameter groups are invalid")
    groups = saved["param_groups"]
    if not all(isinstance(group, dict) for group in groups):
        raise TypeError(f"checkpoint {label} optimizer parameter groups are invalid")
    return cast(list[dict[str, Any]], groups)


def _validate_scheduler_state(
    expected: dict[str, Any],
    saved: dict[str, Any],
    optimizer_state: dict[str, Any],
    successful_updates: int,
    label: str,
) -> None:
    if set(saved) != set(expected):
        raise ValueError(f"checkpoint {label} scheduler fields do not match")
    runtime_fields = {"last_epoch", "_step_count", "_last_lr"}
    for field, value in expected.items():
        if field not in runtime_fields and saved[field] != value:
            raise ValueError(f"checkpoint {label} scheduler option {field} does not match")
    if saved.get("last_epoch") != successful_updates:
        raise ValueError(f"checkpoint {label} scheduler step does not match engine state")
    if saved.get("_step_count") != successful_updates + 1:
        raise ValueError(f"checkpoint {label} scheduler call count does not match engine state")
    last_lrs = saved.get("_last_lr")
    base_lrs = expected.get("base_lrs")
    gamma = expected.get("gamma")
    if (
        not isinstance(last_lrs, list)
        or not isinstance(base_lrs, list)
        or not isinstance(gamma, (int, float))
        or len(last_lrs) != len(base_lrs)
    ):
        raise ValueError(f"checkpoint {label} scheduler learning rates are invalid")
    calculated = [float(base_lr) * float(gamma) ** successful_updates for base_lr in base_lrs]
    if any(
        not math.isclose(float(actual), expected_lr, rel_tol=1e-14, abs_tol=0.0)
        for actual, expected_lr in zip(last_lrs, calculated, strict=True)
    ):
        raise ValueError(f"checkpoint {label} scheduler learning rates do not match")
    groups = _outer_optimizer_groups(optimizer_state, label)
    if len(groups) != len(last_lrs) or any(
        not math.isclose(
            float(group.get("lr", float("nan"))),
            float(lr),
            rel_tol=1e-14,
            abs_tol=0.0,
        )
        for group, lr in zip(groups, last_lrs, strict=True)
    ):
        raise ValueError(f"checkpoint {label} optimizer and scheduler learning rates do not match")


def _validate_engine_state(saved: dict[str, Any], engine: FabricGanEngine) -> None:
    required = {"format_version", "logical_step", "epoch", "batches_in_epoch", "optimizers"}
    if set(saved) != required or saved.get("format_version") != 1:
        raise ValueError("checkpoint engine state fields are invalid")
    for field in ("logical_step", "epoch", "batches_in_epoch"):
        if not isinstance(saved[field], int) or isinstance(saved[field], bool) or saved[field] < 0:
            raise ValueError(f"checkpoint engine {field} must be a non-negative integer")
    optimizers = saved["optimizers"]
    if not isinstance(optimizers, dict) or set(optimizers) != set(engine.optimizers):
        raise ValueError("checkpoint optimizers do not match")
    logical_step = saved["logical_step"]
    if saved["epoch"] > logical_step or saved["batches_in_epoch"] > logical_step:
        raise ValueError("checkpoint epoch position exceeds its logical step")
    plan = engine.raw_task.config.optimization
    for name, optimizer in engine.optimizers.items():
        phase = engine.phases[optimizer.phase]
        counters = optimizers[name]
        if not isinstance(counters, dict) or set(counters) != {
            "attempted_updates",
            "successful_updates",
            "micro_steps",
        }:
            raise ValueError(f"checkpoint optimizer counters are invalid for {name}")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counters.values()
        ):
            raise ValueError(f"checkpoint optimizer counters must be non-negative for {name}")
        if counters["successful_updates"] > counters["attempted_updates"]:
            raise ValueError(f"checkpoint successful updates exceed attempts for {name}")
        start = (
            plan.aux_steps
            if isinstance(plan, WarmupAlternatingPlanConfig) and optimizer.phase == "discriminator"
            else 0
        )
        first_step = ((start + phase.config.frequency - 1) // phase.config.frequency) * (
            phase.config.frequency
        )
        expected_micro_steps = (
            0
            if first_step >= logical_step
            else (logical_step - 1 - first_step) // phase.config.frequency + 1
        )
        if counters["micro_steps"] != expected_micro_steps:
            raise ValueError(f"checkpoint optimizer micro-step count does not match for {name}")
        if counters["attempted_updates"] != expected_micro_steps // phase.config.accumulate:
            raise ValueError(f"checkpoint optimizer attempt count does not match for {name}")
        if counters["micro_steps"] % phase.config.accumulate != 0:
            raise ValueError(f"checkpoint optimizer {name} is not at an accumulation boundary")


def _preflight_restore(
    metadata: dict[str, Any],
    engine: FabricGanEngine,
    sampler: object | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_state = _tensor_state(metadata.get("model"))
    _validate_state_schema(engine.task.state_dict(), model_state, exact=True)

    saved_engine_state = metadata.get("engine_state")
    if not isinstance(saved_engine_state, dict):
        raise TypeError("checkpoint engine state must be a mapping")
    _validate_engine_state(saved_engine_state, engine)
    for name, optimizer in engine.optimizers.items():
        optimizer_state = metadata.get(f"optimizer_{name}")
        optimizer_counters = cast(dict[str, Any], saved_engine_state["optimizers"])[name]
        _validate_optimizer_state(
            optimizer.optimizer,
            optimizer_state,
            name,
            allow_scheduled_lr=optimizer.scheduler is not None,
            successful_updates=optimizer_counters["successful_updates"],
        )
        scheduler_key = f"scheduler_{name}"
        if optimizer.scheduler is not None:
            scheduler_state = metadata.get(scheduler_key)
            if not isinstance(scheduler_state, dict):
                raise TypeError(f"checkpoint scheduler state is missing for {name}")
            _validate_scheduler_state(
                optimizer.scheduler.state_dict(),
                scheduler_state,
                cast(dict[str, Any], optimizer_state),
                optimizer_counters["successful_updates"],
                name,
            )
        if optimizer.scheduler is None and scheduler_key in metadata:
            raise ValueError(f"checkpoint has an unexpected scheduler for {name}")
    precision_state = metadata.get("precision")
    if not isinstance(precision_state, dict):
        raise TypeError("checkpoint precision state must be a mapping")
    _validate_precision_state(
        engine.fabric.strategy.precision.state_dict(),
        precision_state,
    )

    rng_value = metadata.get("rng_by_rank")
    if not isinstance(rng_value, list):
        raise TypeError("checkpoint rank RNG state must be a list")
    if len(rng_value) != engine.fabric.world_size:
        raise ValueError("checkpoint world size does not match")
    expected_cuda_states = torch.cuda.device_count() if engine.fabric.device.type == "cuda" else 0
    rng_by_rank = [
        _validate_rng(state, expected_cuda_states=expected_cuda_states) for state in rng_value
    ]
    saved_sampler = _preflight_sampler(sampler, metadata.get("sampler"), saved_engine_state)
    return rng_by_rank, saved_sampler


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
        for name, optimizer in engine.optimizers.items():
            state[f"optimizer_{name}"] = optimizer.optimizer
            if optimizer.scheduler is not None:
                state[f"scheduler_{name}"] = optimizer.scheduler
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
        metadata = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
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
        rng_by_rank, saved_sampler = _preflight_restore(metadata, engine, sampler)
        state: dict[str, Any] = {
            "model": engine.task,
            "engine_state": engine.state,
            "precision": engine.fabric.strategy.precision,
        }
        for name, optimizer in engine.optimizers.items():
            state[f"optimizer_{name}"] = optimizer.optimizer
            if optimizer.scheduler is not None:
                state[f"scheduler_{name}"] = optimizer.scheduler
        engine.fabric.load(checkpoint_path, state, strict=True, weights_only=True)
        _restore_rng(rng_by_rank[engine.fabric.global_rank])
        _restore_sampler(sampler, saved_sampler, engine)

    def _prune(self) -> None:
        checkpoints = []
        for path in self.run_dir.iterdir():
            match = _STEP_PATTERN.fullmatch(path.name)
            if match and path.is_file():
                checkpoints.append((int(match.group(1)), path))
        for _, path in sorted(checkpoints)[: -self.keep_last]:
            path.unlink()
