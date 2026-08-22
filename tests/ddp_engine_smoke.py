from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed
from lightning_fabric import Fabric
from lightning_fabric.utilities.seed import seed_everything
from torch import nn

from singing_vocoders.checkpoint import CheckpointManager
from singing_vocoders.components import (
    Batch,
    ComponentRegistry,
    DiscriminatorComponent,
    DiscriminatorOutput,
    GeneratorComponent,
    GeneratorOutput,
)
from singing_vocoders.config.schema import RecipeConfig
from singing_vocoders.data.runtime import EpochShuffleSampler
from singing_vocoders.engine import FabricGanEngine
from singing_vocoders.task import VocoderTask


class Generator(GeneratorComponent):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, batch: Batch) -> GeneratorOutput:
        return GeneratorOutput(batch.audio * self.scale)


class Discriminator(DiscriminatorComponent):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 3, padding=1)

    def forward(self, audio: torch.Tensor) -> DiscriminatorOutput:
        feature = self.conv(audio)
        return DiscriminatorOutput([feature.flatten(1)], [[feature]])


def config() -> RecipeConfig:
    return RecipeConfig.model_validate(
        {
            "name": "ddp-test",
            "data": {
                "processed_dir": "data/test",
                "features": {
                    "sample_rate": 8000,
                    "num_mels": 4,
                    "hop_size": 4,
                    "n_fft": 16,
                    "win_size": 16,
                    "fmax": 3900,
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
                    "accumulate": 2,
                },
                "discriminator": {
                    "optimizer": {"name": "adamw", "lr": 0.001},
                    "accumulate": 2,
                },
            },
        }
    )


def worker(fabric: Fabric, marker: Path) -> None:
    seed_everything(1234, workers=True)
    registry = ComponentRegistry()
    registry.register("generator", Generator)
    registry.register("discriminator", Discriminator)
    engine = FabricGanEngine(fabric, VocoderTask(config(), registry))
    for _ in range(2):
        audio = torch.randn(2, 1, 64) + fabric.global_rank
        batch = Batch(
            audio=audio,
            mel=torch.randn(2, 4, 16),
            f0=torch.ones(2, 16),
            uv=torch.zeros(2, 16, dtype=torch.bool),
            pe=torch.zeros(2, dtype=torch.long),
        )
        engine.train_step(batch)
    checksum = sum(parameter.detach().sum() for parameter in engine.raw_task.parameters())
    gathered = fabric.all_gather(checksum)
    if not torch.allclose(gathered, gathered[0].expand_as(gathered)):
        raise RuntimeError("DDP parameters diverged")
    if torch.distributed.get_backend() != "gloo":
        raise RuntimeError("CPU DDP did not select Gloo")
    if engine.state.logical_step != 2:
        raise RuntimeError("logical step diverged")
    if any(state.successful_updates != 1 for state in engine.state.optimizers.values()):
        raise RuntimeError("optimizer update counters diverged")
    sampler = EpochShuffleSampler(
        5,
        1234,
        num_replicas=fabric.world_size,
        rank=fabric.global_rank,
    )
    sampler.set_epoch(engine.state.epoch)
    local_samples = list(sampler)
    gathered_samples: list[list[tuple[int, int]] | None] = [None] * fabric.world_size
    torch.distributed.all_gather_object(gathered_samples, local_samples)
    combined = sorted(
        (sample for rank_samples in gathered_samples for sample in rank_samples or []),
        key=lambda sample: sample[1],
    )
    if [position for _, position in combined] != list(range(6)):
        raise RuntimeError("distributed sampler logical positions diverged")
    if len({index for index, _ in combined}) != 5:
        raise RuntimeError("distributed sampler partition diverged")
    checkpoint_dir = marker.parent / "checkpoint"
    manager = CheckpointManager(
        checkpoint_dir,
        config=config().model_dump(mode="json"),
        config_fingerprint="ddp-test",
        data_fingerprint="ddp-data",
        keep_last=1,
    )
    checkpoint = manager.save(engine, sampler)
    restored = FabricGanEngine(fabric, VocoderTask(config(), registry))
    restored_sampler = EpochShuffleSampler(
        5,
        1234,
        num_replicas=fabric.world_size,
        rank=fabric.global_rank,
    )
    manager.load(restored, checkpoint, restored_sampler)
    if restored.state.state_dict() != engine.state.state_dict():
        raise RuntimeError("DDP checkpoint state diverged")
    if restored_sampler.state_dict() != sampler.state_dict():
        raise RuntimeError("DDP sampler state diverged")
    fabric.barrier()
    if fabric.global_rank == 0:
        if [path.name for path in checkpoint_dir.glob("*.ckpt")] != ["step-00000002.ckpt"]:
            raise RuntimeError("DDP checkpoint writer or naming is invalid")
        marker.write_text("ok\n", encoding="utf-8")
    fabric.barrier()


def main(marker: Path) -> None:
    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp")
    fabric.launch(worker, marker)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
