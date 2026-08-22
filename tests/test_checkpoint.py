from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from lightning_fabric import Fabric
from torch import nn

from singing_vocoders.checkpoint import (
    CheckpointManager,
    export_diffsinger,
    inspect_checkpoint,
    load_legacy_weights,
)
from singing_vocoders.components import (
    Batch,
    ComponentRegistry,
    DiscriminatorComponent,
    DiscriminatorOutput,
    GeneratorComponent,
    GeneratorOutput,
)
from singing_vocoders.config.schema import RecipeConfig
from singing_vocoders.engine import FabricGanEngine
from singing_vocoders.task import VocoderTask


class ScaleGenerator(GeneratorComponent):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, batch: Batch) -> GeneratorOutput:
        return GeneratorOutput(batch.audio * self.scale)


class TinyDiscriminator(DiscriminatorComponent):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 3, padding=1)

    def forward(self, audio: torch.Tensor) -> DiscriminatorOutput:
        feature = self.conv(audio)
        return DiscriminatorOutput([feature.flatten(1)], [[feature]])


class StatefulTestSampler:
    def __init__(self) -> None:
        self.position = 3

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = state["position"]


def _config(accumulate: int = 1) -> RecipeConfig:
    return RecipeConfig.model_validate(
        {
            "name": "checkpoint-test",
            "data": {
                "processed_dir": "data/test",
                "features": {
                    "sample_rate": 8_000,
                    "num_mels": 4,
                    "hop_size": 4,
                    "n_fft": 16,
                    "win_size": 16,
                    "fmax": 3_900,
                },
            },
            "model": {
                "generator": {"target": "generator"},
                "discriminators": {"tiny": {"target": "discriminator"}},
            },
            "objective": {"name": "hifigan", "mel_weight": 1},
            "optimization": {
                "name": "alternating",
                "generator": {
                    "optimizer": {"name": "adamw", "lr": 0.001},
                    "accumulate": accumulate,
                },
                "discriminator": {
                    "optimizer": {"name": "adamw", "lr": 0.001},
                    "accumulate": accumulate,
                },
            },
        }
    )


def _engine(config: RecipeConfig | None = None) -> FabricGanEngine:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    return FabricGanEngine(
        Fabric(accelerator="cpu", devices=1), VocoderTask(config or _config(), registry)
    )


def _batch() -> Batch:
    return Batch(
        audio=torch.randn(2, 1, 64),
        mel=torch.randn(2, 4, 16),
        f0=torch.ones(2, 16),
        uv=torch.zeros(2, 16, dtype=torch.bool),
        pe=torch.zeros(2, dtype=torch.long),
    )


def _manager(path: Path, config: RecipeConfig, keep_last: int = 2) -> CheckpointManager:
    resolved = config.model_dump(mode="json")
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return CheckpointManager(
        path,
        config=resolved,
        config_fingerprint=hashlib.sha256(payload.encode()).hexdigest(),
        data_fingerprint="dataset-1",
        keep_last=keep_last,
    )


def test_checkpoint_restores_full_training_state_rng_and_sampler(tmp_path: Path) -> None:
    config = _config()
    engine = _engine(config)
    engine.train_step(_batch())
    sampler = StatefulTestSampler()
    manager = _manager(tmp_path, config)
    torch.manual_seed(123)
    path = manager.save(engine, sampler)
    expected_random = torch.rand(4)
    expected_scale = engine.raw_task.generator.scale.detach().clone()
    engine.raw_task.generator.scale.data.fill_(99)
    engine.state.logical_step = 99
    engine.slots["generator"].optimizer.state.clear()
    sampler.position = 99

    manager.load(engine, path, sampler)

    assert engine.state.logical_step == 1
    assert engine.state.optimizers["generator"].successful_updates == 1
    assert engine.slots["generator"].optimizer.state
    assert torch.equal(engine.raw_task.generator.scale, expected_scale)
    assert sampler.position == 3
    assert torch.equal(torch.rand(4), expected_random)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["format_version"] == 1
    assert "precision" in checkpoint
    assert "optimizer_generator" in checkpoint


def test_checkpoint_resumes_at_the_next_batch_in_the_same_epoch(tmp_path: Path) -> None:
    config = _config()
    batches = [_batch(), _batch(), _batch()]
    engine = _engine(config)
    engine.fit(batches, max_steps=1, log_every=1)
    assert engine.state.epoch == 0
    assert engine.state.batches_in_epoch == 1
    manager = _manager(tmp_path, config)
    path = manager.save(engine)

    resumed = _engine(config)
    manager.load(resumed, path)
    resumed.fit(batches, max_steps=2, log_every=1)

    assert resumed.state.logical_step == 2
    assert resumed.state.epoch == 0
    assert resumed.state.batches_in_epoch == 2


def test_resume_allows_only_selected_runtime_frequency_changes(tmp_path: Path) -> None:
    config = _config()
    engine = _engine(config)
    engine.train_step(_batch())
    path = _manager(tmp_path, config).save(engine)
    allowed = config.model_dump(mode="json")
    allowed["runtime"]["max_steps"] = 200_000
    allowed["runtime"]["log_every"] = 1
    allowed["runtime"]["validate_every"] = 2
    CheckpointManager(
        tmp_path,
        config=allowed,
        config_fingerprint="changed",
        data_fingerprint="dataset-1",
        keep_last=2,
    ).load(_engine(RecipeConfig.model_validate(allowed)), path)

    rejected = copy.deepcopy(allowed)
    rejected["runtime"]["checkpoint_every"] = 2
    with pytest.raises(ValueError, match="configuration does not match"):
        CheckpointManager(
            tmp_path,
            config=rejected,
            config_fingerprint="changed-again",
            data_fingerprint="dataset-1",
            keep_last=2,
        ).load(_engine(RecipeConfig.model_validate(rejected)), path)


def test_checkpoint_requires_all_accumulation_boundaries(tmp_path: Path) -> None:
    config = _config(accumulate=2)
    engine = _engine(config)
    engine.train_step(_batch())
    with pytest.raises(RuntimeError, match="accumulation boundary"):
        _manager(tmp_path, config).save(engine)


def test_legacy_import_accepts_raw_generator_and_rejects_shape_mismatch(
    tmp_path: Path,
) -> None:
    engine = _engine()
    original = {
        name: tensor.detach().clone()
        for name, tensor in engine.raw_task.generator.state_dict().items()
    }
    path = tmp_path / "generator.pt"
    torch.save(original, path)
    engine.raw_task.generator.scale.data.fill_(7)
    report = load_legacy_weights(engine.raw_task, path, scope="generator")
    assert report.key_count == 1
    assert torch.equal(engine.raw_task.generator.scale, original["scale"])

    broken = tmp_path / "broken.pt"
    torch.save({"scale": torch.ones(2)}, broken)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        load_legacy_weights(engine.raw_task, broken, scope="generator")


def test_checkpoint_inspection_and_diffsinger_export(tmp_path: Path) -> None:
    config = _config()
    engine = _engine(config)
    engine.train_step(_batch())
    path = _manager(tmp_path / "run", config).save(engine)
    report = inspect_checkpoint(path)
    assert report["format_version"] == 1
    assert report["keys"] == len(engine.raw_task.state_dict())

    export_path, config_path = export_diffsinger(path, tmp_path / "export" / "model.pt")
    exported = torch.load(export_path, map_location="cpu", weights_only=True)["generator"]
    engine.raw_task.generator.load_state_dict(exported, strict=True)
    exported_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert exported_config["sampling_rate"] == 8_000
    assert exported_config["num_mels"] == 4
    assert exported_config["pc_aug"] is False
