from __future__ import annotations

import copy
import gc
import hashlib
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
from lightning_fabric import Fabric
from lightning_fabric.plugins import MixedPrecision
from torch import nn
from torch.utils.data import DataLoader

from singing_vocoders.checkpoint import (
    CheckpointManager,
    export_diffsinger,
    inspect_checkpoint,
    load_legacy_weights,
    preserve_rng_state,
)
from singing_vocoders.components import (
    Batch,
    ComponentRegistry,
    DiscriminatorComponent,
    DiscriminatorOutput,
    GeneratorComponent,
    GeneratorOutput,
)
from singing_vocoders.components.builtin import create_builtin_registry
from singing_vocoders.config import load_config
from singing_vocoders.config.schema import RecipeConfig
from singing_vocoders.engine import FabricGanEngine
from singing_vocoders.task import VocoderTask


class ScaleGenerator(GeneratorComponent):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, batch: Batch) -> GeneratorOutput:
        return GeneratorOutput(batch.audio * self.scale)


class RandomScaleGenerator(GeneratorComponent):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, batch: Batch) -> GeneratorOutput:
        random_scale = torch.rand((), device=batch.audio.device)
        return GeneratorOutput(batch.audio * self.scale * random_scale)


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

    def validate_state_dict(self, state: dict[str, int]) -> None:
        if not isinstance(state.get("position"), int):
            raise ValueError("invalid sampler position")

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.validate_state_dict(state)
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


def _random_engine(config: RecipeConfig) -> FabricGanEngine:
    registry = ComponentRegistry()
    registry.register("generator", RandomScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    return FabricGanEngine(Fabric(accelerator="cpu", devices=1), VocoderTask(config, registry))


def _mixed_engine(config: RecipeConfig) -> FabricGanEngine:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    scaler = torch.GradScaler("cuda", init_scale=8.0)
    fabric = Fabric(
        accelerator="cuda",
        devices=1,
        plugins=MixedPrecision("16-mixed", "cuda", scaler=scaler),
    )
    return FabricGanEngine(fabric, VocoderTask(config, registry))


def _independent_loader(batches: list[Batch]) -> DataLoader[Batch]:
    return DataLoader(
        batches,
        batch_size=None,
        collate_fn=lambda batch: batch,
        generator=torch.Generator().manual_seed(99),
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


def _seed_rngs(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _draw_rngs() -> tuple[float, float, torch.Tensor, torch.Tensor | None]:
    cuda = torch.rand(4, device="cuda") if torch.cuda.is_available() else None
    return random.random(), float(np.random.random()), torch.rand(4), cuda


def _assert_rng_draws_equal(
    actual: tuple[float, float, torch.Tensor, torch.Tensor | None],
    expected: tuple[float, float, torch.Tensor, torch.Tensor | None],
) -> None:
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    if actual[3] is None or expected[3] is None:
        assert actual[3] is expected[3]
    else:
        assert torch.equal(actual[3], expected[3])


@pytest.mark.parametrize("raises", [False, True], ids=["normal", "exception"])
def test_preserve_rng_state_restores_all_generators(raises: bool) -> None:
    _seed_rngs(1234)
    expected = _draw_rngs()
    _seed_rngs(1234)

    with pytest.raises(RuntimeError) if raises else nullcontext(), preserve_rng_state():
        _draw_rngs()
        if raises:
            raise RuntimeError("validation failed")

    _assert_rng_draws_equal(_draw_rngs(), expected)


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


def test_mid_epoch_resume_matches_uninterrupted_stochastic_training(tmp_path: Path) -> None:
    config = _config()
    batches = [_batch(), _batch(), _batch()]

    torch.manual_seed(1234)
    uninterrupted = _random_engine(config)
    uninterrupted.fit(_independent_loader(batches), max_steps=2, log_every=1)

    torch.manual_seed(1234)
    first_stage = _random_engine(config)
    first_stage.fit(_independent_loader(batches), max_steps=1, log_every=1)
    manager = _manager(tmp_path, config)
    checkpoint_path = manager.save(first_stage)

    torch.manual_seed(9876)
    resumed = _random_engine(config)
    manager.load(resumed, checkpoint_path)
    resumed.fit(_independent_loader(batches), max_steps=2, log_every=1)

    assert resumed.state.state_dict() == uninterrupted.state.state_dict()
    for name, expected in uninterrupted.raw_task.state_dict().items():
        torch.testing.assert_close(resumed.raw_task.state_dict()[name], expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "corruption",
    [
        "world-size",
        "sampler-identity",
        "cuda-rng",
        "model-shape",
        "model-dtype",
        "optimizer-parameter-group",
        "optimizer-parameter-order",
        "optimizer-group-option",
        "optimizer-state-shape",
        "optimizer-step-nan",
        "optimizer-step-zero",
        "optimizer-step-negative",
        "optimizer-step-fractional",
        "optimizer-step-dtype",
        "engine-logical-step",
        "engine-attempt-count",
        "engine-micro-steps",
        "engine-batches-in-epoch",
    ],
)
def test_checkpoint_preflight_rejects_before_mutating_engine(
    tmp_path: Path, corruption: str
) -> None:
    config = _config()
    source_engine = _engine(config)
    source_engine.train_step(_batch())
    source_sampler = StatefulTestSampler()
    source_path = _manager(tmp_path / "source", config).save(source_engine, source_sampler)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    if corruption == "world-size":
        checkpoint["rng_by_rank"].append(copy.deepcopy(checkpoint["rng_by_rank"][0]))
    elif corruption == "sampler-identity":
        checkpoint["sampler"]["identity"] = "invalid.Sampler"
    elif corruption == "cuda-rng":
        checkpoint["rng_by_rank"][0]["cuda"] = [[]]
    elif corruption == "model-shape":
        checkpoint["model"]["generator.scale"] = torch.ones(2)
    elif corruption == "model-dtype":
        checkpoint["model"]["generator.scale"] = torch.ones((), dtype=torch.int64)
    elif corruption == "optimizer-parameter-group":
        checkpoint["optimizer_generator"]["param_groups"][0]["params"] = []
    elif corruption == "optimizer-parameter-order":
        parameters = checkpoint["optimizer_discriminator"]["param_groups"][0]["params"]
        checkpoint["optimizer_discriminator"]["param_groups"][0]["params"] = list(
            reversed(parameters)
        )
    elif corruption == "optimizer-group-option":
        checkpoint["optimizer_generator"]["param_groups"][0]["lr"] = 123.0
    elif corruption == "optimizer-state-shape":
        generator_state = checkpoint["optimizer_generator"]["state"]
        parameter_id = next(iter(generator_state))
        generator_state[parameter_id]["exp_avg"] = torch.ones(2)
    elif corruption == "engine-logical-step":
        checkpoint["engine_state"]["logical_step"] = 999
    elif corruption == "engine-attempt-count":
        checkpoint["engine_state"]["optimizers"]["generator"]["attempted_updates"] = 998
    elif corruption == "engine-micro-steps":
        checkpoint["engine_state"]["optimizers"]["generator"]["micro_steps"] = 998
    elif corruption == "engine-batches-in-epoch":
        checkpoint["engine_state"]["batches_in_epoch"] = 999
        checkpoint["sampler"]["batches_in_epoch"] = 999
    else:
        generator_state = checkpoint["optimizer_generator"]["state"]
        parameter_id = next(iter(generator_state))
        if corruption == "optimizer-step-nan":
            step = torch.tensor(float("nan"))
        elif corruption == "optimizer-step-zero":
            step = torch.tensor(0.0)
        elif corruption == "optimizer-step-negative":
            step = torch.tensor(-1.0)
        elif corruption == "optimizer-step-fractional":
            step = torch.tensor(0.5)
        else:
            step = torch.tensor(1, dtype=torch.int64)
        generator_state[parameter_id]["step"] = step
    corrupted_path = tmp_path / f"{corruption}.ckpt"
    torch.save(checkpoint, corrupted_path)

    target = _engine(config)
    target.raw_task.generator.scale.data.fill_(7)
    target.state.logical_step = 37
    target_sampler = StatefulTestSampler()
    target_sampler.position = 41
    expected_model = copy.deepcopy(target.raw_task.state_dict())
    expected_optimizer = copy.deepcopy(target.slots["generator"].optimizer.state_dict())

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _manager(tmp_path / "target", config).load(target, corrupted_path, target_sampler)

    assert target.state.logical_step == 37
    assert target_sampler.position == 41
    assert all(
        torch.equal(value, expected_model[name])
        for name, value in target.raw_task.state_dict().items()
    )
    assert target.slots["generator"].optimizer.state_dict() == expected_optimizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GradScaler state")
def test_checkpoint_preflight_rejects_invalid_precision_before_mutation(tmp_path: Path) -> None:
    config = _config()
    source = _mixed_engine(config)
    source.train_step(_batch().to("cuda"))
    path = _manager(tmp_path / "source", config).save(source)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["precision"]["scale"] = -123.0
    corrupted = tmp_path / "invalid-precision.ckpt"
    torch.save(checkpoint, corrupted)

    target = _mixed_engine(config)
    target.raw_task.generator.scale.data.fill_(7)
    target.state.logical_step = 37
    expected_model = copy.deepcopy(target.raw_task.state_dict())
    expected_precision = copy.deepcopy(target.fabric.strategy.precision.state_dict())

    with pytest.raises(ValueError, match="precision scale"):
        _manager(tmp_path / "target", config).load(target, corrupted)

    assert target.state.logical_step == 37
    assert all(
        torch.equal(value, expected_model[name])
        for name, value in target.raw_task.state_dict().items()
    )
    assert target.fabric.strategy.precision.state_dict() == expected_precision


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

    with pytest.raises(ValueError, match="only supported for the nsf_hifigan"):
        export_diffsinger(path, tmp_path / "export" / "model.pt")


def test_diffsinger_export_uses_embedded_config_and_strict_nsf_schema(tmp_path: Path) -> None:
    recipe_path = (
        Path(__file__).parents[1] / "src" / "singing_vocoders" / "recipes" / "pc-nsf-hifigan.yaml"
    )
    loaded = load_config(recipe_path)
    registry = create_builtin_registry()
    generator = registry.build(loaded.config.model.generator, GeneratorComponent)
    raw_state = generator.state_dict()
    legacy_source = tmp_path / "legacy-generator.pt"
    torch.save(raw_state, legacy_source)
    versioned_source = tmp_path / "versioned.ckpt"
    torch.save(
        {
            "format_version": 1,
            "model": {f"generator.{name}": value for name, value in raw_state.items()},
            "config": loaded.resolved,
        },
        versioned_source,
    )
    del generator, raw_state
    gc.collect()

    legacy_output, legacy_config_path = export_diffsinger(
        legacy_source,
        tmp_path / "legacy-export" / "model.pt",
        resolved_config=loaded.resolved,
    )
    versioned_output, versioned_config_path = export_diffsinger(
        versioned_source,
        tmp_path / "versioned-export" / "model.pt",
    )
    with pytest.raises(ValueError, match="must use their embedded configuration"):
        export_diffsinger(
            versioned_source,
            tmp_path / "wrong-export" / "model.pt",
            resolved_config=_config().model_dump(mode="json"),
        )

    exported = torch.load(versioned_output, map_location="cpu", weights_only=True)["generator"]
    restored = registry.build(loaded.config.model.generator, GeneratorComponent)
    restored.load_state_dict(exported, strict=True)
    assert torch.load(legacy_output, map_location="cpu", weights_only=True)["generator"].keys() == (
        exported.keys()
    )
    legacy_config = json.loads(legacy_config_path.read_text(encoding="utf-8"))
    versioned_config = json.loads(versioned_config_path.read_text(encoding="utf-8"))
    assert legacy_config == versioned_config
    assert versioned_config["sampling_rate"] == 44_100
    assert versioned_config["num_mels"] == 128
    assert versioned_config["hop_size"] == 512
    assert versioned_config["n_fft"] == 2_048
    assert versioned_config["win_size"] == 2_048
    assert versioned_config["fmin"] == 40
    assert versioned_config["fmax"] == 16_000
    assert versioned_config["pc_aug"] is True
    assert versioned_config["mini_nsf"] is True
    assert versioned_config["noise_sigma"] == 0

    import singing_vocoders.checkpoint.export as export_module

    replace = export_module.os.replace

    def fail_config_publish(source: str | Path, destination: str | Path) -> None:
        if Path(destination).name == "config.json":
            raise OSError("injected config publication failure")
        replace(source, destination)

    rollback_output = tmp_path / "rollback-export" / "model.pt"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(export_module.os, "replace", fail_config_publish)
        with pytest.raises(OSError, match="injected config publication failure"):
            export_diffsinger(versioned_source, rollback_output)
    assert not rollback_output.exists()
    assert not rollback_output.with_name("config.json").exists()


def test_diffsinger_export_preserves_disabled_noise_sigma(tmp_path: Path) -> None:
    recipe_path = (
        Path(__file__).parents[1] / "src" / "singing_vocoders" / "recipes" / "nsf-hifigan.yaml"
    )
    loaded = load_config(recipe_path)
    resolved = copy.deepcopy(loaded.resolved)
    resolved["model"]["generator"]["params"]["noise_sigma"] = None
    generator = create_builtin_registry().build(
        RecipeConfig.model_validate(resolved).model.generator,
        GeneratorComponent,
    )
    source = tmp_path / "versioned.ckpt"
    torch.save(
        {
            "format_version": 1,
            "model": {f"generator.{name}": value for name, value in generator.state_dict().items()},
            "config": resolved,
        },
        source,
    )

    _, config_path = export_diffsinger(source, tmp_path / "export" / "model.pt")

    exported_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert exported_config["noise_sigma"] is None
