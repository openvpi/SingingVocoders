from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from lightning_fabric import Fabric
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from singing_vocoders.components.contracts import (
    Batch,
    DiscriminatorOutput,
    GeneratorOutput,
)
from singing_vocoders.config.schema import (
    AlternatingPlanConfig,
    JointPlanConfig,
    OptimizerSlotConfig,
    WarmupAlternatingPlanConfig,
)
from singing_vocoders.engine.optim import build_optimizer
from singing_vocoders.task.augmentation import GeneratorStage
from singing_vocoders.task.task import VocoderTask


@dataclass(slots=True)
class OptimizerState:
    attempted_updates: int = 0
    successful_updates: int = 0
    micro_steps: int = 0


class EngineState:
    __slots__ = ("batches_in_epoch", "epoch", "format_version", "logical_step", "optimizers")

    def __init__(self) -> None:
        self.format_version = 1
        self.logical_step = 0
        self.epoch = 0
        self.batches_in_epoch = 0
        self.optimizers: dict[str, OptimizerState] = {}

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "logical_step": self.logical_step,
            "epoch": self.epoch,
            "batches_in_epoch": self.batches_in_epoch,
            "optimizers": {name: asdict(state) for name, state in self.optimizers.items()},
        }

    def load_state_dict(self, value: dict[str, Any]) -> None:
        if value.get("format_version") != 1:
            raise ValueError(f"unsupported engine state version: {value.get('format_version')}")
        self.logical_step = int(value["logical_step"])
        self.epoch = int(value["epoch"])
        self.batches_in_epoch = int(value.get("batches_in_epoch", 0))
        restored: dict[str, OptimizerState] = {}
        for name, slot in value["optimizers"].items():
            state = self.optimizers.get(name, OptimizerState())
            state.attempted_updates = int(slot["attempted_updates"])
            state.successful_updates = int(slot["successful_updates"])
            state.micro_steps = int(slot["micro_steps"])
            restored[name] = state
        self.optimizers.clear()
        self.optimizers.update(restored)


@dataclass(slots=True)
class OptimizerRuntime:
    name: str
    config: OptimizerSlotConfig
    optimizer: Optimizer
    scheduler: LRScheduler | None
    state: OptimizerState
    module: nn.Module | None = None

    @property
    def boundary(self) -> bool:
        return (self.state.micro_steps + 1) % self.config.accumulate == 0

    @property
    def cycle_start(self) -> bool:
        return self.state.micro_steps % self.config.accumulate == 0


@dataclass(slots=True)
class JointForwardOutput:
    generated: GeneratorOutput
    real: dict[str, DiscriminatorOutput]
    fake_detached: dict[str, DiscriminatorOutput]
    fake_connected: dict[str, DiscriminatorOutput]


class _DiscriminatorEnsemble(nn.Module):
    def __init__(self, components: nn.ModuleDict) -> None:
        super().__init__()
        self.components = components

    def discriminate(self, audio: Tensor) -> dict[str, DiscriminatorOutput]:
        return {
            name: cast(DiscriminatorOutput, discriminator(audio))
            for name, discriminator in self.components.items()
        }

    def forward(
        self, real_audio: Tensor, fake_audio: Tensor
    ) -> tuple[dict[str, DiscriminatorOutput], dict[str, DiscriminatorOutput]]:
        return self.discriminate(real_audio), self.discriminate(fake_audio)


class _JointTrainingModule(nn.Module):
    def __init__(self, task: VocoderTask) -> None:
        super().__init__()
        self.generator = GeneratorStage(task.generator, task.config)
        self.discriminator = _DiscriminatorEnsemble(task.discriminator)

    def forward(self, batch: Batch) -> JointForwardOutput:
        generated = self.generator(batch)
        if not isinstance(generated, GeneratorOutput):
            raise TypeError("generator returned an invalid output")
        real = self.discriminator.discriminate(batch.audio)
        fake_detached = self.discriminator.discriminate(generated.audio.detach())
        fake_connected = self.discriminator.discriminate(generated.audio)
        return JointForwardOutput(generated, real, fake_detached, fake_connected)


class FabricGanEngine:
    def __init__(self, fabric: Fabric, task: VocoderTask) -> None:
        strategy_name = type(fabric.strategy).__name__
        if strategy_name in {"DeepSpeedStrategy", "FSDPStrategy"}:
            raise NotImplementedError(f"{strategy_name} is not supported by the GAN engine")
        self.fabric = fabric
        self.raw_task = task
        self.task = task
        self.state = EngineState()
        self.slots = self._build_optimizers(task)
        self._raw_discriminator = _DiscriminatorEnsemble(task.discriminator)
        self.generator: nn.Module | None = None
        self.discriminator: nn.Module | None = None
        self.joint_module: nn.Module | None = None
        self._setup_modules(task)
        task.objective.to(fabric.device)

    def _build_optimizers(self, task: VocoderTask) -> dict[str, OptimizerRuntime]:
        plan = task.config.optimization
        definitions: list[tuple[str, OptimizerSlotConfig, nn.Module]]
        if isinstance(plan, (AlternatingPlanConfig, WarmupAlternatingPlanConfig)):
            definitions = [
                ("generator", plan.generator, task.generator),
                ("discriminator", plan.discriminator, task.discriminator),
            ]
        elif isinstance(plan, JointPlanConfig):
            definitions = [("joint", plan.joint, task)]
        else:
            raise TypeError(f"unsupported optimization plan: {type(plan).__name__}")
        slots: dict[str, OptimizerRuntime] = {}
        for name, config, module in definitions:
            built = build_optimizer(config, module)
            state = OptimizerState()
            self.state.optimizers[name] = state
            slots[name] = OptimizerRuntime(
                name=name,
                config=config,
                optimizer=built.optimizer,
                scheduler=built.scheduler,
                state=state,
            )
        return slots

    def _setup_one(self, module: nn.Module, slot: OptimizerRuntime) -> nn.Module:
        setup = self.fabric.setup(module, slot.optimizer)
        if not isinstance(setup, tuple):
            raise RuntimeError("Fabric did not return the configured optimizer")
        configured_module, configured_optimizer = setup
        slot.optimizer = cast(Optimizer, configured_optimizer)
        slot.module = configured_module
        return cast(nn.Module, configured_module)

    def _setup_modules(self, task: VocoderTask) -> None:
        plan = task.config.optimization
        if isinstance(plan, JointPlanConfig):
            self.joint_module = self._setup_one(_JointTrainingModule(task), self.slots["joint"])
            return
        self.generator = self._setup_one(
            GeneratorStage(task.generator, task.config), self.slots["generator"]
        )
        self.discriminator = self._setup_one(self._raw_discriminator, self.slots["discriminator"])

    @contextmanager
    def _frozen_discriminators(self) -> Iterator[None]:
        parameters = list(self.raw_task.discriminator.parameters())
        original = [parameter.requires_grad for parameter in parameters]
        for parameter in parameters:
            parameter.requires_grad_(False)
        try:
            yield
        finally:
            for parameter, requires_grad in zip(parameters, original, strict=True):
                parameter.requires_grad_(requires_grad)

    def _generate(self, batch: Batch) -> GeneratorOutput:
        module = self.raw_task.generator if self.generator is None else self.generator
        output = module(batch)
        if not isinstance(output, GeneratorOutput):
            raise TypeError("generator returned an invalid output")
        return output

    def _raw_discriminate(self, audio: Tensor) -> dict[str, DiscriminatorOutput]:
        return self._raw_discriminator.discriminate(audio)

    def _scaler_scale(self) -> float | None:
        scaler = getattr(self.fabric.strategy.precision, "scaler", None)
        return None if scaler is None else float(scaler.get_scale())

    @staticmethod
    def _ensure_finite_loss(loss: Tensor, phase: str, batch: Batch) -> None:
        if not torch.isfinite(loss.detach()).all():
            item = batch.metadata[0].get("item", "unknown") if batch.metadata else "unknown"
            raise FloatingPointError(f"non-finite {phase} loss at item {item}")

    @staticmethod
    def _ensure_finite_gradients(optimizer: Optimizer, phase: str) -> None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"non-finite gradient in {phase}")

    def _backward_and_maybe_step(
        self, slot: OptimizerRuntime, loss: Tensor, phase: str, batch: Batch
    ) -> bool:
        self._ensure_finite_loss(loss, phase, batch)
        if slot.module is None:
            raise RuntimeError(f"optimizer module is not configured: {slot.name}")
        if slot.cycle_start:
            slot.optimizer.zero_grad(set_to_none=True)
        should_step = slot.boundary
        with self.fabric.no_backward_sync(cast(Any, slot.module), enabled=not should_step):
            self.fabric.backward(loss / slot.config.accumulate)
        slot.state.micro_steps += 1
        if not should_step:
            return False
        slot.state.attempted_updates += 1
        if slot.config.clip_grad_norm is not None:
            self.fabric.clip_gradients(
                slot.module,
                slot.optimizer,
                max_norm=slot.config.clip_grad_norm,
                error_if_nonfinite=True,
            )
        else:
            self._ensure_finite_gradients(slot.optimizer, phase)
        scale_before = self._scaler_scale()
        slot.optimizer.step()
        scale_after = self._scaler_scale()
        succeeded = scale_before is None or scale_after is None or scale_after >= scale_before
        if succeeded:
            slot.state.successful_updates += 1
            if slot.scheduler is not None:
                slot.scheduler.step()
        slot.optimizer.zero_grad(set_to_none=True)
        return succeeded

    def _run_generator_phase(
        self, batch: Batch, generated: GeneratorOutput, slot: OptimizerRuntime
    ) -> dict[str, Tensor]:
        with self._frozen_discriminators():
            with torch.no_grad():
                real = self._raw_discriminate(batch.audio)
            fake = self._raw_discriminate(generated.audio)
            result = self.raw_task.objective.generator_loss(
                batch, generated, real, fake, self.state.logical_step
            )
        self._backward_and_maybe_step(slot, result.loss, "generator", batch)
        return result.metrics

    def _run_discriminator_phase(
        self, batch: Batch, generated: GeneratorOutput, slot: OptimizerRuntime
    ) -> dict[str, Tensor]:
        if self.discriminator is None:
            raise RuntimeError("separate discriminator is unavailable for a joint plan")
        output = self.discriminator(batch.audio, generated.audio.detach())
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("discriminator ensemble returned an invalid output")
        real, fake = output
        result = self.raw_task.objective.discriminator_loss(real, fake)
        self._backward_and_maybe_step(slot, result.loss, "discriminator", batch)
        return result.metrics

    def _run_joint_phase(self, batch: Batch) -> tuple[GeneratorOutput, dict[str, Tensor]]:
        if self.joint_module is None:
            raise RuntimeError("joint module is unavailable for an alternating plan")
        output = self.joint_module(batch)
        if not isinstance(output, JointForwardOutput):
            raise TypeError("joint module returned an invalid output")
        result = self.raw_task.objective.joint_loss(
            batch,
            output.generated,
            output.real,
            output.fake_detached,
            output.fake_connected,
            self.state.logical_step,
        )
        self._backward_and_maybe_step(self.slots["joint"], result.loss, "joint", batch)
        return output.generated, result.metrics

    def train_step(self, batch: Batch) -> dict[str, Tensor]:
        plan = self.raw_task.config.optimization
        metrics: dict[str, Tensor]
        if isinstance(plan, JointPlanConfig):
            generated, metrics = self._run_joint_phase(batch)
        else:
            generated = self._generate(batch)
            metrics = {}
            if (
                isinstance(plan, WarmupAlternatingPlanConfig)
                and self.state.logical_step < plan.aux_steps
            ):
                result = self.raw_task.objective.auxiliary_loss(
                    batch, generated, self.state.logical_step
                )
                self._backward_and_maybe_step(
                    self.slots["generator"], result.loss, "generator_aux", batch
                )
                metrics.update(result.metrics)
            else:
                discriminator = self.slots["discriminator"]
                generator = self.slots["generator"]
                if self.state.logical_step % discriminator.config.frequency == 0:
                    metrics.update(self._run_discriminator_phase(batch, generated, discriminator))
                if self.state.logical_step % generator.config.frequency == 0:
                    metrics.update(self._run_generator_phase(batch, generated, generator))
        self.state.logical_step += 1
        metrics["steps/logical"] = generated.audio.new_tensor(float(self.state.logical_step))
        for name, slot in self.slots.items():
            metrics[f"steps/{name}/attempted"] = generated.audio.new_tensor(
                float(slot.state.attempted_updates)
            )
            metrics[f"steps/{name}/successful"] = generated.audio.new_tensor(
                float(slot.state.successful_updates)
            )
        return metrics

    @torch.no_grad()
    def validate(self, loader: Any, max_batches: int | None = None) -> dict[str, Tensor]:
        self.raw_task.eval()
        if self.generator is not None:
            self.generator.eval()
        if self.joint_module is not None:
            self.joint_module.eval()
        total: Tensor | None = None
        count = 0
        for batch in loader:
            batch = batch.to(self.fabric.device, non_blocking=True)
            generated = self._generate(batch)
            loss = self.raw_task.objective.auxiliary_loss(
                batch, generated, self.state.logical_step
            ).loss.detach()
            total = loss if total is None else total + loss
            count += 1
            if max_batches is not None and count >= max_batches:
                break
        self.raw_task.train()
        if self.generator is not None:
            self.generator.train()
        if self.joint_module is not None:
            self.joint_module.train()
        if total is None:
            raise ValueError("validation loader is empty")
        result = cast(Tensor, self.fabric.all_reduce(total / count, reduce_op="mean"))
        return {"validation/loss": result}

    @torch.no_grad()
    def predict(self, batch: Batch) -> GeneratorOutput:
        was_training = self.raw_task.training
        self.raw_task.eval()
        if self.generator is not None:
            self.generator.eval()
        if self.joint_module is not None:
            self.joint_module.eval()
        output = self._generate(batch.to(self.fabric.device, non_blocking=True))
        if was_training:
            self.raw_task.train()
            if self.generator is not None:
                self.generator.train()
            if self.joint_module is not None:
                self.joint_module.train()
        return output

    def fit(
        self,
        loader: Any,
        *,
        max_steps: int,
        log_every: int,
        on_step: Callable[[FabricGanEngine, dict[str, Tensor]], None] | None = None,
    ) -> None:
        self.raw_task.train()
        while self.state.logical_step < max_steps:
            dataset = getattr(loader, "dataset", None)
            if dataset is not None and hasattr(dataset, "set_epoch"):
                dataset.set_epoch(self.state.epoch)
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.state.epoch)
            had_batches = False
            completed_epoch = True
            resume_position = self.state.batches_in_epoch
            for batch_index, batch in enumerate(loader):
                had_batches = True
                if batch_index < resume_position:
                    continue
                batch = batch.to(self.fabric.device, non_blocking=True)
                metrics = self.train_step(batch)
                self.state.batches_in_epoch += 1
                if self.state.logical_step % log_every == 0:
                    self.fabric.log_dict(metrics, step=self.state.logical_step)
                if on_step is not None:
                    on_step(self, metrics)
                if self.state.logical_step >= max_steps:
                    completed_epoch = False
                    break
            if not had_batches:
                raise ValueError("training loader is empty")
            if completed_epoch:
                self.state.epoch += 1
                self.state.batches_in_epoch = 0

    def accumulation_boundary(self) -> bool:
        return all(
            slot.state.micro_steps % slot.config.accumulate == 0 for slot in self.slots.values()
        )
