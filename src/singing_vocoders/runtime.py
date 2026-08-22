from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import torch
from lightning_fabric import Fabric
from lightning_fabric.loggers.tensorboard import TensorBoardLogger
from omegaconf import OmegaConf

from singing_vocoders.checkpoint import (
    CheckpointManager,
    export_diffsinger,
    inspect_checkpoint,
    load_legacy_weights,
)
from singing_vocoders.components import Batch
from singing_vocoders.config import LoadedConfig
from singing_vocoders.config.schema import RecipeConfig
from singing_vocoders.data import VocoderLoaders, build_dataloaders
from singing_vocoders.engine import FabricGanEngine
from singing_vocoders.task import VocoderTask


def choose_run_dir(config: RecipeConfig, requested: str | Path | None = None) -> Path:
    if requested is not None:
        candidate = Path(requested).resolve()
    else:
        root = config.runtime.output_dir.resolve()
        recipe_root = (root / config.name).resolve()
        if not recipe_root.is_relative_to(root):
            raise ValueError("recipe name escapes the configured output directory")
        identifier = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        candidate = recipe_root / identifier
    if candidate.exists():
        raise FileExistsError(candidate)
    return candidate


def create_fabric(config: RecipeConfig, run_dir: Path | None = None) -> Fabric:
    logger = (
        TensorBoardLogger(root_dir=run_dir, name="logs", version="", default_hp_metric=False)
        if run_dir is not None
        else None
    )
    return Fabric(
        accelerator=config.runtime.accelerator,
        devices=config.runtime.devices,
        strategy=config.runtime.strategy,
        precision=config.runtime.precision,
        loggers=logger,
    )


def _write_resolved_config(path: Path, resolved: dict[str, Any]) -> None:
    temporary = path.with_suffix(".yaml.tmp")
    OmegaConf.save(OmegaConf.create(resolved), temporary)
    os.replace(temporary, path)


def _setup_loaders(fabric: Fabric, config: RecipeConfig) -> tuple[Any, Any, VocoderLoaders]:
    loaders = build_dataloaders(config.data, config.augmentation)
    configured = fabric.setup_dataloaders(
        loaders.train,
        loaders.valid,
        use_distributed_sampler=True,
        move_to_device=False,
    )
    if not isinstance(configured, list) or len(configured) != 2:
        raise RuntimeError("Fabric returned invalid training loaders")
    return configured[0], configured[1], loaders


def _checkpoint_manager(
    run_dir: Path, loaded: LoadedConfig, data_fingerprint: str
) -> CheckpointManager:
    return CheckpointManager(
        run_dir,
        config=loaded.resolved,
        config_fingerprint=loaded.fingerprint,
        data_fingerprint=data_fingerprint,
        keep_last=loaded.config.runtime.keep_last,
    )


def _load_for_training(
    engine: FabricGanEngine,
    manager: CheckpointManager,
    sampler: object | None,
    *,
    resume: Path | None,
    warm_start: Path | None,
    warm_start_scope: Literal["generator", "all"],
) -> None:
    if resume is not None:
        manager.load(engine, resume, sampler)
    elif warm_start is not None:
        load_legacy_weights(engine.raw_task, warm_start, scope=warm_start_scope)


def _log_validation_audio(
    engine: FabricGanEngine, loader: Any, *, sample_rate: int, step: int
) -> None:
    batch = next(iter(loader), None)
    if batch is None:
        raise ValueError("validation loader is empty")
    if not isinstance(batch, Batch):
        raise TypeError("validation loader returned an invalid batch")
    generated = engine.predict(batch)
    if engine.fabric.global_rank != 0 or engine.fabric.logger is None:
        return
    writer = cast(Any, engine.fabric.logger).experiment
    writer.add_audio(
        "validation/generated",
        generated.audio[0].detach().float().cpu().clamp(-1, 1),
        step,
        sample_rate,
    )
    writer.add_audio(
        "validation/reference",
        batch.audio[0].detach().float().cpu().clamp(-1, 1),
        step,
        sample_rate,
    )


def train_worker(
    fabric: Fabric,
    loaded: LoadedConfig,
    run_dir: Path,
    resume: Path | None,
    warm_start: Path | None,
    warm_start_scope: Literal["generator", "all"],
) -> str | None:
    config = loaded.config
    fabric.seed_everything(config.runtime.seed, workers=True)
    if fabric.global_rank == 0:
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_resolved_config(run_dir / "config.yaml", loaded.resolved)
    fabric.barrier()
    train_loader, valid_loader, loaders = _setup_loaders(fabric, config)
    engine = FabricGanEngine(fabric, VocoderTask(config))
    manager = _checkpoint_manager(run_dir, loaded, loaders.metadata.fingerprint)
    sampler = getattr(train_loader, "sampler", None)
    _load_for_training(
        engine,
        manager,
        sampler,
        resume=resume,
        warm_start=warm_start,
        warm_start_scope=warm_start_scope,
    )
    pending_checkpoint = False
    saved_at: set[int] = set()

    def on_step(current: FabricGanEngine, metrics: dict[str, torch.Tensor]) -> None:
        nonlocal pending_checkpoint
        step = current.state.logical_step
        if step % config.runtime.validate_every == 0:
            validation = current.validate(valid_loader)
            current.fabric.log_dict(validation, step=step)
            _log_validation_audio(
                current, valid_loader, sample_rate=config.data.features.sample_rate, step=step
            )
        if step % config.runtime.checkpoint_every == 0:
            pending_checkpoint = True
        if pending_checkpoint and current.accumulation_boundary():
            manager.save(current, sampler)
            saved_at.add(step)
            pending_checkpoint = False

    engine.fit(
        train_loader,
        max_steps=config.runtime.max_steps,
        log_every=config.runtime.log_every,
        on_step=on_step,
    )
    step = engine.state.logical_step
    if engine.accumulation_boundary() and step not in saved_at:
        manager.save(engine, sampler)
        saved_at.add(step)
    final_checkpoint = manager.path_for_step(step) if step in saved_at else None
    fabric.barrier()
    export_path: Path | None = None
    if fabric.global_rank == 0 and final_checkpoint is not None:
        export_path, _ = export_diffsinger(final_checkpoint, run_dir / "export" / "model.pt")
    fabric.barrier()
    if fabric.global_rank == 0:
        result = {
            "run_dir": str(run_dir),
            "logical_step": step,
            "checkpoint": str(final_checkpoint) if final_checkpoint is not None else None,
            "export": str(export_path) if export_path is not None else None,
        }
        fabric.print(json.dumps(result, ensure_ascii=False, indent=2))
        return str(final_checkpoint) if final_checkpoint is not None else None
    return None


def validate_worker(
    fabric: Fabric,
    loaded: LoadedConfig,
    checkpoint: Path,
    legacy_scope: Literal["generator", "all"],
    max_batches: int | None,
) -> dict[str, float] | None:
    config = loaded.config
    fabric.seed_everything(config.runtime.seed, workers=True)
    train_loader, valid_loader, loaders = _setup_loaders(fabric, config)
    engine = FabricGanEngine(fabric, VocoderTask(config))
    report = inspect_checkpoint(checkpoint)
    if report["format_version"] == 1:
        manager = _checkpoint_manager(Path.cwd(), loaded, loaders.metadata.fingerprint)
        manager.load(engine, checkpoint, getattr(train_loader, "sampler", None))
    else:
        load_legacy_weights(engine.raw_task, checkpoint, scope=legacy_scope)
    metrics = engine.validate(valid_loader, max_batches=max_batches)
    result = {name: float(value.detach().cpu()) for name, value in metrics.items()}
    if fabric.global_rank == 0:
        fabric.print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    return None
