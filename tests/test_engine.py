import pytest
import torch
from lightning_fabric import Fabric
from torch import nn

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


def _config(
    accumulate: int = 1,
    *,
    plan: str = "alternating",
    aux_steps: int = 0,
    generator_frequency: int = 1,
    discriminator_frequency: int = 1,
) -> RecipeConfig:
    optimization: dict[str, object]
    if plan == "joint":
        optimization = {
            "name": "joint",
            "joint": {"optimizer": {"name": "adamw", "lr": 0.001}},
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
                "accumulate": accumulate,
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
            "objective": {"name": "hifigan", "mel_weight": 1},
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


class PartitionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 3, 3)
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
    assert id(model.conv.weight) in muon_ids
    assert id(model.embedding.weight) in adamw_ids
    assert id(model.weight_g) in adamw_ids
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
