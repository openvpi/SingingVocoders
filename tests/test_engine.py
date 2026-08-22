import copy
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from lightning_fabric import Fabric
from lightning_fabric.plugins import MixedPrecision
from torch import nn
from torch.nn.utils import spectral_norm

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
from singing_vocoders.engine.muon import MuonAdamW
from singing_vocoders.task import VocoderTask
from singing_vocoders.task.objective import LossResult


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


class SpectralTinyDiscriminator(DiscriminatorComponent):
    def __init__(self) -> None:
        super().__init__()
        self.conv = spectral_norm(nn.Conv1d(1, 1, 3, padding=1))

    def forward(self, audio: torch.Tensor) -> DiscriminatorOutput:
        feature = self.conv(audio)
        return DiscriminatorOutput([feature.flatten(1)], [[feature]])


class NonfiniteGradientDiscriminator(DiscriminatorComponent):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.scale.register_hook(lambda gradient: gradient * float("nan"))

    def forward(self, audio: torch.Tensor) -> DiscriminatorOutput:
        feature = audio * self.scale
        return DiscriminatorOutput([feature.flatten(1)], [[feature]])


class BatchValueObjective(nn.Module):
    def auxiliary_loss(
        self,
        batch: Batch,
        _generated: GeneratorOutput,
        _logical_step: int,
    ) -> LossResult:
        return LossResult(batch.audio[:, 0, 0].mean(), {})


def _config(
    accumulate: int = 1,
    *,
    discriminator_accumulate: int | None = None,
    plan: str = "alternating",
    aux_steps: int = 0,
    generator_frequency: int = 1,
    discriminator_frequency: int = 1,
) -> RecipeConfig:
    optimization: dict[str, object]
    if plan == "joint":
        optimization = {
            "name": "joint",
            "joint": {
                "optimizer": {"name": "adamw", "lr": 0.001},
                "accumulate": accumulate,
                "frequency": generator_frequency,
            },
        }
    else:
        optimization = {
            "name": plan,
            "generator": {
                "optimizer": {"name": "adamw", "lr": 0.001},
                "accumulate": accumulate,
                "frequency": generator_frequency,
            },
            "discriminator": {
                "optimizer": {"name": "adamw", "lr": 0.001},
                "accumulate": (
                    accumulate if discriminator_accumulate is None else discriminator_accumulate
                ),
                "frequency": discriminator_frequency,
            },
        }
        if plan == "warmup_alternating":
            optimization["aux_steps"] = aux_steps
    return RecipeConfig.model_validate(
        {
            "name": "test",
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
            "objective": (
                {
                    "name": "one_step",
                    "mel_weight": 0,
                    "feature_weight": 0,
                }
                if plan == "joint"
                else {"name": "hifigan", "mel_weight": 1}
            ),
            "optimization": optimization,
        }
    )


def _batch() -> Batch:
    audio = torch.randn(2, 1, 64)
    return Batch(
        audio=audio,
        mel=torch.randn(2, 4, 16),
        f0=torch.ones(2, 16),
        uv=torch.zeros(2, 16, dtype=torch.bool),
        pe=torch.zeros(2, dtype=torch.long),
    )


def test_alternating_plan_counts_one_logical_step_per_batch() -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    engine = FabricGanEngine(Fabric(accelerator="cpu", devices=1), VocoderTask(_config(), registry))
    metrics = engine.train_step(_batch())
    assert engine.state.logical_step == 1
    assert engine.state.optimizers["generator"].successful_updates == 1
    assert engine.state.optimizers["discriminator"].successful_updates == 1
    assert torch.isfinite(metrics["loss/generator"])


def test_independent_accumulation_defers_optimizer_updates() -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    engine = FabricGanEngine(
        Fabric(accelerator="cpu", devices=1), VocoderTask(_config(accumulate=2), registry)
    )
    engine.train_step(_batch())
    assert engine.state.optimizers["generator"].attempted_updates == 0
    engine.train_step(_batch())
    assert engine.state.optimizers["generator"].successful_updates == 1


def test_accumulation_no_sync_covers_forward_and_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    engine = FabricGanEngine(
        Fabric(accelerator="cpu", devices=1), VocoderTask(_config(accumulate=2), registry)
    )
    assert engine.discriminator is not None
    active: list[str] = []
    generator_forward_contexts: list[tuple[str, ...]] = []
    discriminator_forward_contexts: list[tuple[str, ...]] = []
    backward_contexts: list[tuple[str, ...]] = []

    @contextmanager
    def recording_no_sync(module: nn.Module, *, enabled: bool = True):
        name = next(name for name, slot in engine.slots.items() if slot.module is module)
        if enabled:
            active.append(name)
        try:
            yield
        finally:
            if enabled:
                active.remove(name)

    original_generate = engine._generate

    def recording_generate(batch: Batch) -> GeneratorOutput:
        generator_forward_contexts.append(tuple(active))
        return original_generate(batch)

    original_discriminator_forward = engine._raw_discriminator.forward

    def recording_discriminator_forward(
        real_audio: torch.Tensor, fake_audio: torch.Tensor
    ) -> object:
        discriminator_forward_contexts.append(tuple(active))
        return original_discriminator_forward(real_audio, fake_audio)

    original_backward = engine.fabric.backward

    def recording_backward(loss: torch.Tensor) -> None:
        backward_contexts.append(tuple(active))
        original_backward(loss)

    monkeypatch.setattr(engine.fabric, "no_backward_sync", recording_no_sync)
    monkeypatch.setattr(engine, "_generate", recording_generate)
    monkeypatch.setattr(engine._raw_discriminator, "forward", recording_discriminator_forward)
    monkeypatch.setattr(engine.fabric, "backward", recording_backward)

    engine.train_step(_batch())

    assert generator_forward_contexts == [("generator",)]
    assert discriminator_forward_contexts == [("generator", "discriminator")]
    assert backward_contexts == [
        ("generator", "discriminator"),
        ("generator",),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FP16 GradScaler")
def test_fp16_scale_growth_rescales_other_optimizer_pending_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    task = VocoderTask(_config(accumulate=2, discriminator_accumulate=1), registry)
    scaler = torch.GradScaler("cuda", init_scale=8, growth_interval=1, growth_factor=2)
    fabric = Fabric(
        accelerator="cuda",
        devices=1,
        plugins=MixedPrecision("16-mixed", "cuda", scaler=scaler),
    )
    engine = FabricGanEngine(fabric, task)
    batch = _batch().to("cuda")
    engine.train_step(batch)
    parameter = task.generator.scale
    assert parameter.grad is not None
    pending_gradient = parameter.grad.detach().clone()
    assert scaler.get_scale() == 16

    original = engine._run_generator_phase
    observed = False

    def inspect_pending_gradient(
        batch: Batch,
        generated: GeneratorOutput,
        slot: Any,
    ) -> dict[str, torch.Tensor]:
        nonlocal observed
        observed = True
        assert scaler.get_scale() == 32
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, pending_gradient * 2)
        return original(batch, generated, slot)

    monkeypatch.setattr(engine, "_run_generator_phase", inspect_pending_gradient)
    engine.train_step(batch)
    assert observed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FP16 GradScaler")
def test_fp16_scaled_gradient_overflow_backs_off_without_aborting() -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    task = VocoderTask(_config(), registry)
    scaler = torch.GradScaler("cuda", init_scale=65_536, growth_interval=2_000)
    fabric = Fabric(
        accelerator="cuda",
        devices=1,
        plugins=MixedPrecision("16-mixed", "cuda", scaler=scaler),
    )
    engine = FabricGanEngine(fabric, task)
    slot = engine.slots["generator"]
    batch = _batch().to("cuda")
    parameter = task.generator.scale
    before = parameter.detach().clone()

    with engine._optimization_context(slot):
        succeeded = engine._backward_and_maybe_step(
            slot,
            parameter * 1e35,
            "generator",
            batch,
        )

    assert not succeeded
    assert scaler.get_scale() == 32_768
    assert slot.state.attempted_updates == 1
    assert slot.state.successful_updates == 0
    assert torch.equal(parameter, before)
    assert parameter.grad is None

    with engine._optimization_context(slot):
        succeeded = engine._backward_and_maybe_step(slot, parameter, "generator", batch)

    assert succeeded
    assert slot.state.attempted_updates == 2
    assert slot.state.successful_updates == 1
    assert not torch.equal(parameter, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FP16 GradScaler")
def test_fp16_scale_backoff_rescales_other_optimizer_pending_gradients() -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    task = VocoderTask(_config(accumulate=2, discriminator_accumulate=1), registry)
    scaler = torch.GradScaler("cuda", init_scale=8, growth_interval=2_000)
    fabric = Fabric(
        accelerator="cuda",
        devices=1,
        plugins=MixedPrecision("16-mixed", "cuda", scaler=scaler),
    )
    engine = FabricGanEngine(fabric, task)
    batch = _batch().to("cuda")
    engine.train_step(batch)
    generator_parameter = task.generator.scale
    assert generator_parameter.grad is not None
    pending_gradient = generator_parameter.grad.detach().clone()

    discriminator_slot = engine.slots["discriminator"]
    discriminator_parameter = next(task.discriminator.parameters())
    with engine._optimization_context(discriminator_slot):
        succeeded = engine._backward_and_maybe_step(
            discriminator_slot,
            discriminator_parameter.flatten()[0] * 1e38,
            "discriminator",
            batch,
        )

    assert not succeeded
    assert scaler.get_scale() == 4
    assert generator_parameter.grad is not None
    torch.testing.assert_close(generator_parameter.grad, pending_gradient * 0.5)


class AutocastRecordingGenerator(ScaleGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 1)
        self.autocast_contexts: list[bool] = []

    def forward(self, batch: Batch) -> GeneratorOutput:
        self.autocast_contexts.append(torch.is_autocast_enabled("cpu"))
        return GeneratorOutput(self.conv(batch.audio) * self.scale)


class AutocastRecordingDiscriminator(TinyDiscriminator):
    def __init__(self) -> None:
        super().__init__()
        self.autocast_contexts: list[bool] = []

    def forward(self, audio: torch.Tensor) -> DiscriminatorOutput:
        self.autocast_contexts.append(torch.is_autocast_enabled("cpu"))
        return super().forward(audio)


def test_mixed_precision_covers_raw_discriminator_and_joint_prediction_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ComponentRegistry()
    registry.register("generator", AutocastRecordingGenerator)
    registry.register("discriminator", AutocastRecordingDiscriminator)
    task = VocoderTask(_config(), registry)
    engine = FabricGanEngine(Fabric(accelerator="cpu", devices=1, precision="bf16-mixed"), task)
    observed_generator_dtypes: list[torch.dtype] = []
    original_generator_loss = task.objective.generator_loss

    def record_generator_loss(
        batch: Batch,
        generated: GeneratorOutput,
        real: dict[str, DiscriminatorOutput],
        fake: dict[str, DiscriminatorOutput],
        logical_step: int = 0,
    ) -> LossResult:
        observed_generator_dtypes.extend(
            tensor.dtype
            for outputs in (real, fake)
            for output in outputs.values()
            for branch in ([*output.scores], *output.feature_maps)
            for tensor in (branch if isinstance(branch, list) else [branch])
        )
        return original_generator_loss(batch, generated, real, fake, logical_step)

    monkeypatch.setattr(task.objective, "generator_loss", record_generator_loss)
    engine.train_step(_batch())
    discriminator = task.discriminator["tiny"]
    assert discriminator.autocast_contexts
    assert all(discriminator.autocast_contexts)
    assert observed_generator_dtypes
    assert set(observed_generator_dtypes) == {torch.float32}

    joint_task = VocoderTask(_config(plan="joint"), registry)
    joint_engine = FabricGanEngine(
        Fabric(accelerator="cpu", devices=1, precision="bf16-mixed"), joint_task
    )
    joint_output = joint_engine.predict(_batch())
    assert joint_task.generator.autocast_contexts == [True]
    assert joint_output.audio.dtype == torch.float32


class ScalarGenerator(GeneratorComponent):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.5))

    def forward(self, _batch: Batch) -> GeneratorOutput:
        return GeneratorOutput(self.value)


class ScalarDiscriminator(DiscriminatorComponent):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.5))

    def forward(self, _audio: torch.Tensor) -> DiscriminatorOutput:
        return DiscriminatorOutput([self.value], [[self.value]])


def test_component_tensor_contract_errors_are_diagnostic() -> None:
    generator_registry = ComponentRegistry()
    generator_registry.register("generator", ScalarGenerator)
    generator_registry.register("discriminator", TinyDiscriminator)
    generator_engine = FabricGanEngine(
        Fabric(accelerator="cpu", devices=1),
        VocoderTask(_config(), generator_registry),
    )
    with pytest.raises(ValueError, match="generator audio must have shape"):
        generator_engine.predict(_batch())

    discriminator_registry = ComponentRegistry()
    discriminator_registry.register("generator", ScaleGenerator)
    discriminator_registry.register("discriminator", ScalarDiscriminator)
    discriminator_engine = FabricGanEngine(
        Fabric(accelerator="cpu", devices=1),
        VocoderTask(_config(), discriminator_registry),
    )
    with pytest.raises(ValueError, match="scores violate the tensor contract"):
        discriminator_engine.train_step(_batch())


def test_warmup_crosses_into_adversarial_training() -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    engine = FabricGanEngine(
        Fabric(accelerator="cpu", devices=1),
        VocoderTask(_config(plan="warmup_alternating", aux_steps=1), registry),
    )
    engine.train_step(_batch())
    assert engine.state.optimizers["generator"].successful_updates == 1
    assert engine.state.optimizers["discriminator"].successful_updates == 0
    engine.train_step(_batch())
    assert engine.state.optimizers["generator"].successful_updates == 2
    assert engine.state.optimizers["discriminator"].successful_updates == 1


def test_joint_plan_updates_generator_and_discriminator_together() -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    task = VocoderTask(_config(plan="joint"), registry)
    generator_before = task.generator.scale.detach().clone()
    discriminator_before = task.discriminator["tiny"].conv.weight.detach().clone()
    engine = FabricGanEngine(Fabric(accelerator="cpu", devices=1), task)
    engine.train_step(_batch())
    assert engine.state.optimizers["joint"].successful_updates == 1
    assert not torch.equal(task.generator.scale, generator_before)
    assert not torch.equal(task.discriminator["tiny"].conv.weight, discriminator_before)


def test_joint_generator_path_cannot_leak_into_spectral_norm_discriminator_gradients() -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", SpectralTinyDiscriminator)
    task = VocoderTask(_config(accumulate=2, plan="joint"), registry)
    reference = copy.deepcopy(task)
    batch = _batch()

    engine = FabricGanEngine(Fabric(accelerator="cpu", devices=1), task)
    engine.train_step(batch)

    generated = reference.generator(batch)
    discriminator = reference.discriminator["tiny"]
    real = {"tiny": discriminator(batch.audio)}
    fake_detached = {"tiny": discriminator(generated.audio.detach())}
    with torch.no_grad():
        discriminator(generated.audio)
    reference.objective.discriminator_loss(real, fake_detached).loss.backward()

    actual_gradient = task.discriminator["tiny"].conv.weight_orig.grad
    expected_gradient = reference.discriminator["tiny"].conv.weight_orig.grad
    assert actual_gradient is not None
    assert expected_gradient is not None
    torch.testing.assert_close(actual_gradient, expected_gradient / 2, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize(
    ("plan", "aux_steps"),
    (("joint", 0), ("warmup_alternating", 3)),
)
def test_optimization_frequency_is_honored_in_joint_and_warmup_phases(
    plan: str, aux_steps: int
) -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    task = VocoderTask(_config(plan=plan, aux_steps=aux_steps, generator_frequency=2), registry)
    generator_before = task.generator.scale.detach().clone()
    engine = FabricGanEngine(Fabric(accelerator="cpu", devices=1), task)
    engine.state.logical_step = 1
    engine.train_step(_batch())
    slot = "joint" if plan == "joint" else "generator"
    assert engine.state.optimizers[slot].attempted_updates == 0
    assert torch.equal(task.generator.scale, generator_before)


def test_phase_frequency_keeps_other_component_isolated() -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    task = VocoderTask(_config(generator_frequency=1, discriminator_frequency=2), registry)
    engine = FabricGanEngine(Fabric(accelerator="cpu", devices=1), task)
    engine.state.logical_step = 1
    discriminator_before = task.discriminator["tiny"].conv.weight.detach().clone()
    engine.train_step(_batch())
    assert engine.state.optimizers["generator"].successful_updates == 1
    assert engine.state.optimizers["discriminator"].successful_updates == 0
    assert torch.equal(task.discriminator["tiny"].conv.weight, discriminator_before)
    assert all(parameter.grad is None for parameter in task.discriminator.parameters())


def test_nonfinite_loss_reports_phase_and_item() -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    engine = FabricGanEngine(Fabric(accelerator="cpu", devices=1), VocoderTask(_config(), registry))
    batch = _batch()
    batch.audio.fill_(float("nan"))
    batch.metadata = [{"item": "broken.npz"}]
    with pytest.raises(FloatingPointError, match=r"discriminator.*broken\.npz"):
        engine.train_step(batch)


def test_nonfinite_gradient_stops_on_first_microbatch_and_reports_component() -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    registry.register("nonfinite_discriminator", NonfiniteGradientDiscriminator)
    config = _config(accumulate=2)
    raw_config = config.model_dump(mode="python")
    raw_config["model"]["discriminators"]["broken"] = {"target": "nonfinite_discriminator"}
    engine = FabricGanEngine(
        Fabric(accelerator="cpu", devices=1),
        VocoderTask(RecipeConfig.model_validate(raw_config), registry),
    )
    batch = _batch()
    batch.metadata = [{"item": "broken.npz"}]

    with pytest.raises(
        FloatingPointError,
        match=r"discriminator\.broken.*scale.*broken\.npz",
    ):
        engine.train_step(batch)

    assert engine.state.optimizers["discriminator"].micro_steps == 0


def test_validation_restores_training_mode_after_an_error() -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    engine = FabricGanEngine(Fabric(accelerator="cpu", devices=1), VocoderTask(_config(), registry))
    engine.raw_task.train()
    assert engine.generator is not None
    engine.generator.train()
    with pytest.raises(ValueError, match="validation loader is empty"):
        engine.validate([])
    assert engine.raw_task.training
    assert engine.generator.training


def test_validation_weights_partial_batches_by_sample_count() -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    engine = FabricGanEngine(Fabric(accelerator="cpu", devices=1), VocoderTask(_config(), registry))
    engine.raw_task.objective = BatchValueObjective()
    full = _batch()
    full.audio = torch.ones(3, 1, 64)
    partial = _batch()
    partial.audio = torch.full((1, 1, 64), 9.0)

    result = engine.validate([full, partial])

    torch.testing.assert_close(result["validation/loss"], torch.tensor(3.0))


def test_validation_excludes_distributed_padding_samples() -> None:
    registry = ComponentRegistry()
    registry.register("generator", ScaleGenerator)
    registry.register("discriminator", TinyDiscriminator)
    engine = FabricGanEngine(Fabric(accelerator="cpu", devices=1), VocoderTask(_config(), registry))
    engine.raw_task.objective = BatchValueObjective()
    expected_batch = _batch()
    expected_batch.audio = torch.ones(1, 1, 64)
    expected_batch.metadata = [{"validation_weight": 1.0}]
    padded_batch = _batch()
    padded_batch.audio = torch.cat((torch.ones(1, 1, 64), torch.full((1, 1, 64), 99.0)))
    padded_batch.metadata = [
        {"validation_weight": 1.0},
        {"validation_weight": 0.0},
    ]

    expected = engine.validate([expected_batch])
    actual = engine.validate([padded_batch])

    torch.testing.assert_close(actual["validation/loss"], expected["validation/loss"])


class PartitionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 3, 3)
        self.linear = nn.Linear(3, 2)
        self.embedding = nn.Embedding(4, 3)
        self.weight_g = nn.Parameter(torch.ones(2, 1))


def test_muon_adamw_partition_and_state_roundtrip() -> None:
    model = PartitionModel()
    optimizer = MuonAdamW(
        model,
        lr=0.01,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=2,
        adamw_lr=0.001,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_weight_decay=0,
    )
    muon_ids = {id(parameter) for parameter in optimizer.muon_parameters}
    adamw_ids = {id(parameter) for parameter in optimizer.adamw_parameters}
    assert id(model.linear.weight) in muon_ids
    assert id(model.embedding.weight) in muon_ids
    assert id(model.weight_g) in muon_ids
    assert id(model.conv.weight) in adamw_ids
    assert not muon_ids & adamw_ids
    assert muon_ids | adamw_ids == {id(parameter) for parameter in model.parameters()}
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    state = optimizer.state_dict()

    restored_model = PartitionModel()
    restored = MuonAdamW(
        restored_model,
        lr=0.01,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=2,
        adamw_lr=0.001,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_weight_decay=0,
    )
    restored.load_state_dict(state)
    assert restored.state_dict()["muon"]["state"]
    assert restored.state_dict()["adamw"]["state"]
