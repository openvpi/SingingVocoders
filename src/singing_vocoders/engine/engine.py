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
from singing_vocoders.engine.optim import build_optimizers
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
    phase: str
    optimizer: Optimizer
    scheduler: LRScheduler | None
    state: OptimizerState


@dataclass(slots=True)
class OptimizationPhaseRuntime:
    name: str
    config: OptimizerSlotConfig
    optimizers: list[OptimizerRuntime]
    module: nn.Module | None = None

    @property
    def boundary(self) -> bool:
        return (self.micro_steps + 1) % self.config.accumulate == 0

    @property
    def cycle_start(self) -> bool:
        return self.micro_steps % self.config.accumulate == 0

    @property
    def micro_steps(self) -> int:
        values = {optimizer.state.micro_steps for optimizer in self.optimizers}
        if len(values) != 1:
            raise RuntimeError(f"optimizer routes are out of sync in phase {self.name}")
        return next(iter(values))

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.optimizer.zero_grad(set_to_none=True)


@contextmanager
def _temporarily_frozen(module: nn.Module) -> Iterator[None]:
    parameters = list(module.parameters())
    original = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in zip(parameters, original, strict=True):
            parameter.requires_grad_(requires_grad)


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
        outputs: dict[str, DiscriminatorOutput] = {}
        for name, discriminator in self.components.items():
            output = discriminator(audio)
            if not isinstance(output, DiscriminatorOutput):
                raise TypeError(f"discriminator {name} returned an invalid output")
            if not output.scores or len(output.scores) != len(output.feature_maps):
                raise ValueError(
                    f"discriminator {name} must return non-empty aligned scores and feature maps"
                )
            if any(
                score.ndim < 1 or not score.is_floating_point() or score.device != audio.device
                for score in output.scores
            ):
                raise ValueError(f"discriminator {name} scores violate the tensor contract")
            if any(score.shape[0] != audio.shape[0] for score in output.scores):
                raise ValueError(f"discriminator {name} score batch does not match its audio input")
            if any(
                not branch
                or any(
                    feature.ndim < 1
                    or not feature.is_floating_point()
                    or feature.device != audio.device
                    for feature in branch
                )
                for branch in output.feature_maps
            ):
                raise ValueError(f"discriminator {name} feature maps violate the tensor contract")
            if any(
                feature.shape[0] != audio.shape[0]
                for branch in output.feature_maps
                for feature in branch
            ):
                raise ValueError(f"discriminator {name} feature maps do not match its audio input")
            outputs[name] = output
        return outputs

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
        with _temporarily_frozen(self.discriminator):
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
        self.optimizers: dict[str, OptimizerRuntime] = {}
        self.phases = self._build_optimizers(task)
        self._raw_discriminator = _DiscriminatorEnsemble(task.discriminator)
        self.generator: nn.Module | None = None
        self.discriminator: nn.Module | None = None
        self.joint_module: nn.Module | None = None
        self._joint_generator: GeneratorStage | None = None
        self._setup_modules(task)
        task.objective.to(fabric.device)

    def _build_optimizers(self, task: VocoderTask) -> dict[str, OptimizationPhaseRuntime]:
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
        phases: dict[str, OptimizationPhaseRuntime] = {}
        for phase_name, config, module in definitions:
            built_optimizers = build_optimizers(config, module)
            route_runtimes: list[OptimizerRuntime] = []
            multiple_routes = len(built_optimizers) > 1
            for built in built_optimizers:
                name = f"{phase_name}.{built.route.parameters}" if multiple_routes else phase_name
                state = OptimizerState()
                self.state.optimizers[name] = state
                runtime = OptimizerRuntime(
                    name=name,
                    phase=phase_name,
                    optimizer=built.optimizer,
                    scheduler=built.scheduler,
                    state=state,
                )
                route_runtimes.append(runtime)
                self.optimizers[name] = runtime
            phases[phase_name] = OptimizationPhaseRuntime(
                name=phase_name,
                config=config,
                optimizers=route_runtimes,
            )
        return phases

    def _setup_one(self, module: nn.Module, phase: OptimizationPhaseRuntime) -> nn.Module:
        setup = self.fabric.setup(
            module,
            *(optimizer.optimizer for optimizer in phase.optimizers),
        )
        if not isinstance(setup, tuple):
            raise RuntimeError("Fabric did not return the configured optimizers")
        configured_module, *configured_optimizers = setup
        if len(configured_optimizers) != len(phase.optimizers):
            raise RuntimeError("Fabric returned an unexpected optimizer count")
        for runtime, configured_optimizer in zip(
            phase.optimizers,
            configured_optimizers,
            strict=True,
        ):
            runtime.optimizer = cast(Optimizer, configured_optimizer)
        phase.module = configured_module
        return cast(nn.Module, configured_module)

    def _setup_modules(self, task: VocoderTask) -> None:
        plan = task.config.optimization
        if isinstance(plan, JointPlanConfig):
            joint_module = _JointTrainingModule(task)
            self._joint_generator = joint_module.generator
            self.joint_module = self._setup_one(joint_module, self.phases["joint"])
            return
        self.generator = self._setup_one(
            GeneratorStage(task.generator, task.config), self.phases["generator"]
        )
        self.discriminator = self._setup_one(
            self._raw_discriminator,
            self.phases["discriminator"],
        )

    @contextmanager
    def _frozen_discriminators(self) -> Iterator[None]:
        with _temporarily_frozen(self.raw_task.discriminator):
            yield

    @contextmanager
    def _evaluation_mode(self) -> Iterator[None]:
        modules = [
            module
            for module in (self.raw_task, self.generator, self.joint_module)
            if module is not None
        ]
        training_modes = [module.training for module in modules]
        for module in modules:
            module.eval()
        try:
            yield
        finally:
            for module, training in zip(modules, training_modes, strict=True):
                module.train(training)

    def _generate(self, batch: Batch) -> GeneratorOutput:
        if self.generator is None:
            if self._joint_generator is None:
                raise RuntimeError("joint generator stage is unavailable")
            with self.fabric.autocast():
                output = self._joint_generator(batch)
            output = GeneratorOutput(
                output.audio.float(),
                {
                    name: value.float() if value.is_floating_point() else value
                    for name, value in output.aux.items()
                },
            )
        else:
            output = self.generator(batch)
        if not isinstance(output, GeneratorOutput):
            raise TypeError("generator returned an invalid output")
        return output

    def _raw_discriminate(self, audio: Tensor) -> dict[str, DiscriminatorOutput]:
        with self.fabric.autocast():
            outputs = self._raw_discriminator.discriminate(audio)
        return {
            name: DiscriminatorOutput(
                [score.float() for score in output.scores],
                [[feature.float() for feature in branch] for branch in output.feature_maps],
            )
            for name, output in outputs.items()
        }

    def _scaler_scale(self) -> float | None:
        scaler = getattr(self.fabric.strategy.precision, "scaler", None)
        return None if scaler is None else float(scaler.get_scale())

    def _rescale_pending_gradients(
        self,
        stepped_phase: OptimizationPhaseRuntime,
        scale_before: float | None,
        scale_after: float | None,
    ) -> None:
        if (
            scale_before is None
            or scale_after is None
            or scale_before == scale_after
            or scale_before <= 0
            or scale_after <= 0
        ):
            return
        factor = scale_after / scale_before
        for phase in self.phases.values():
            if phase is stepped_phase or phase.micro_steps % phase.config.accumulate == 0:
                continue
            for optimizer in phase.optimizers:
                for parameter_group in optimizer.optimizer.param_groups:
                    for parameter in parameter_group["params"]:
                        if parameter.grad is not None:
                            parameter.grad.mul_(factor)

    @contextmanager
    def _optimization_context(self, phase: OptimizationPhaseRuntime) -> Iterator[None]:
        if phase.module is None:
            raise RuntimeError(f"optimization phase module is not configured: {phase.name}")
        if phase.cycle_start:
            phase.zero_grad()
        with self.fabric.no_backward_sync(cast(Any, phase.module), enabled=not phase.boundary):
            yield

    @staticmethod
    def _ensure_finite_loss(loss: Tensor, phase: str, batch: Batch) -> None:
        if not torch.isfinite(loss.detach()).all():
            item = batch.metadata[0].get("item", "unknown") if batch.metadata else "unknown"
            raise FloatingPointError(
                f"non-finite loss in {phase} component objective at item {item}"
            )

    @staticmethod
    def _gradient_component(phase: OptimizationPhaseRuntime, parameter_name: str) -> str:
        parts = parameter_name.split(".")
        if "components" in parts:
            index = parts.index("components")
            if index + 1 < len(parts):
                return f"discriminator.{parts[index + 1]}"
        if "generator" in parts or phase.name == "generator":
            return "generator"
        return phase.name

    @classmethod
    def _ensure_finite_gradients(
        cls,
        phase_runtime: OptimizationPhaseRuntime,
        phase: str,
        batch: Batch,
    ) -> None:
        if phase_runtime.module is None:
            raise RuntimeError(f"optimization phase module is not configured: {phase_runtime.name}")
        for parameter_name, parameter in phase_runtime.module.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            if torch.isfinite(values).all():
                continue
            item = batch.metadata[0].get("item", "unknown") if batch.metadata else "unknown"
            component = cls._gradient_component(phase_runtime, parameter_name)
            raise FloatingPointError(
                f"non-finite gradient in {phase} component {component} "
                f"parameter {parameter_name} at item {item}"
            )

    @staticmethod
    def _raw_optimizer(optimizer: OptimizerRuntime) -> Optimizer:
        return cast(Optimizer, getattr(optimizer.optimizer, "optimizer", optimizer.optimizer))

    @staticmethod
    def _gradients_are_finite(phase: OptimizationPhaseRuntime) -> bool:
        if phase.module is None:
            raise RuntimeError(f"optimization phase module is not configured: {phase.name}")
        for parameter in phase.module.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            if not torch.isfinite(values).all():
                return False
        return True

    def _backward_and_maybe_step(
        self,
        phase_runtime: OptimizationPhaseRuntime,
        loss: Tensor,
        phase: str,
        batch: Batch,
    ) -> bool:
        self._ensure_finite_loss(loss, phase, batch)
        if phase_runtime.module is None:
            raise RuntimeError(f"optimization phase module is not configured: {phase_runtime.name}")
        should_step = phase_runtime.boundary
        self.fabric.backward(loss / phase_runtime.config.accumulate)
        uses_gradient_scaler = self._scaler_scale() is not None
        if not uses_gradient_scaler:
            self._ensure_finite_gradients(phase_runtime, phase, batch)
        for optimizer in phase_runtime.optimizers:
            optimizer.state.micro_steps += 1
        if not should_step:
            return False
        for optimizer in phase_runtime.optimizers:
            optimizer.state.attempted_updates += 1

        if uses_gradient_scaler:
            precision = self.fabric.strategy.precision
            scale_before = self._scaler_scale()
            for optimizer in phase_runtime.optimizers:
                precision.unscale_gradients(self._raw_optimizer(optimizer))
            succeeded = self._gradients_are_finite(phase_runtime)
            if succeeded:
                torch.nn.utils.clip_grad_norm_(
                    phase_runtime.module.parameters(),
                    max_norm=(
                        phase_runtime.config.clip_grad_norm
                        if phase_runtime.config.clip_grad_norm is not None
                        else float("inf")
                    ),
                    error_if_nonfinite=True,
                )
                for optimizer in phase_runtime.optimizers:
                    self._raw_optimizer(optimizer).step()
            scaler = getattr(precision, "scaler", None)
            if scaler is None:
                raise RuntimeError("mixed precision scaler disappeared during optimizer step")
            scaler.update()
            scale_after = self._scaler_scale()
            self._rescale_pending_gradients(phase_runtime, scale_before, scale_after)
        else:
            succeeded = True
            if phase_runtime.config.clip_grad_norm is not None:
                self.fabric.clip_gradients(
                    phase_runtime.module,
                    phase_runtime.optimizers[0].optimizer,
                    max_norm=phase_runtime.config.clip_grad_norm,
                    error_if_nonfinite=True,
                )
            for optimizer in phase_runtime.optimizers:
                optimizer.optimizer.step()

        if succeeded:
            for optimizer in phase_runtime.optimizers:
                optimizer.state.successful_updates += 1
                if optimizer.scheduler is not None:
                    optimizer.scheduler.step()
        phase_runtime.zero_grad()
        return succeeded

    def _run_generator_phase(
        self, batch: Batch, generated: GeneratorOutput, phase: OptimizationPhaseRuntime
    ) -> dict[str, Tensor]:
        with self._frozen_discriminators():
            with torch.no_grad():
                real = self._raw_discriminate(batch.audio)
            fake = self._raw_discriminate(generated.audio)
            result = self.raw_task.objective.generator_loss(
                batch, generated, real, fake, self.state.logical_step
            )
        self._backward_and_maybe_step(phase, result.loss, "generator", batch)
        return result.metrics

    def _run_discriminator_phase(
        self, batch: Batch, generated: GeneratorOutput, phase: OptimizationPhaseRuntime
    ) -> dict[str, Tensor]:
        if self.discriminator is None:
            raise RuntimeError("separate discriminator is unavailable for a joint plan")
        with self._optimization_context(phase):
            output = self.discriminator(batch.audio, generated.audio.detach())
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError("discriminator ensemble returned an invalid output")
            real, fake = output
            result = self.raw_task.objective.discriminator_loss(real, fake)
            self._backward_and_maybe_step(phase, result.loss, "discriminator", batch)
        return result.metrics

    def _run_joint_phase(self, batch: Batch) -> tuple[GeneratorOutput, dict[str, Tensor]]:
        if self.joint_module is None:
            raise RuntimeError("joint module is unavailable for an alternating plan")
        phase = self.phases["joint"]
        with self._optimization_context(phase):
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
            self._backward_and_maybe_step(phase, result.loss, "joint", batch)
        return output.generated, result.metrics

    def train_step(self, batch: Batch) -> dict[str, Tensor]:
        plan = self.raw_task.config.optimization
        metrics: dict[str, Tensor]
        if isinstance(plan, JointPlanConfig):
            joint = self.phases["joint"]
            if self.state.logical_step % joint.config.frequency == 0:
                generated, metrics = self._run_joint_phase(batch)
            else:
                with torch.no_grad():
                    generated = self._generate(batch)
                metrics = {}
        else:
            metrics = {}
            generator = self.phases["generator"]
            discriminator = self.phases["discriminator"]
            if (
                isinstance(plan, WarmupAlternatingPlanConfig)
                and self.state.logical_step < plan.aux_steps
            ):
                if self.state.logical_step % generator.config.frequency == 0:
                    with self._optimization_context(generator):
                        generated = self._generate(batch)
                        result = self.raw_task.objective.auxiliary_loss(
                            batch, generated, self.state.logical_step
                        )
                        self._backward_and_maybe_step(
                            generator, result.loss, "generator_aux", batch
                        )
                    metrics.update(result.metrics)
                else:
                    with torch.no_grad():
                        generated = self._generate(batch)
            else:
                run_discriminator = self.state.logical_step % discriminator.config.frequency == 0
                run_generator = self.state.logical_step % generator.config.frequency == 0
                if run_generator:
                    with self._optimization_context(generator):
                        generated = self._generate(batch)
                        if run_discriminator:
                            metrics.update(
                                self._run_discriminator_phase(batch, generated, discriminator)
                            )
                        metrics.update(self._run_generator_phase(batch, generated, generator))
                else:
                    with torch.no_grad():
                        generated = self._generate(batch)
                    if run_discriminator:
                        metrics.update(
                            self._run_discriminator_phase(batch, generated, discriminator)
                        )
        self.state.logical_step += 1
        metrics["steps/logical"] = generated.audio.new_tensor(float(self.state.logical_step))
        for name, optimizer in self.optimizers.items():
            metrics[f"steps/{name}/attempted"] = generated.audio.new_tensor(
                float(optimizer.state.attempted_updates)
            )
            metrics[f"steps/{name}/successful"] = generated.audio.new_tensor(
                float(optimizer.state.successful_updates)
            )
        return metrics

    @torch.no_grad()
    def validate(self, loader: Any, max_batches: int | None = None) -> dict[str, Tensor]:
        weighted_total: Tensor | None = None
        sample_count = 0.0
        batch_count = 0
        with self._evaluation_mode():
            for batch in loader:
                batch = batch.to(self.fabric.device, non_blocking=True)
                generated = self._generate(batch)
                batch_size = batch.audio.shape[0]
                if len(batch.metadata) not in {0, batch_size}:
                    raise ValueError("validation metadata count does not match the batch")
                if weighted_total is None:
                    weighted_total = generated.audio.new_zeros(())
                for index in range(batch_size):
                    weight = (
                        float(batch.metadata[index].get("validation_weight", 1.0))
                        if batch.metadata
                        else 1.0
                    )
                    if weight not in {0.0, 1.0}:
                        raise ValueError("validation sample weight must be zero or one")
                    if weight == 0:
                        continue
                    sample_batch = Batch(
                        audio=batch.audio[index : index + 1],
                        mel=batch.mel[index : index + 1],
                        f0=batch.f0[index : index + 1],
                        uv=batch.uv[index : index + 1],
                        pe=batch.pe[index : index + 1],
                        metadata=batch.metadata[index : index + 1] if batch.metadata else [],
                    )
                    sample_generated = GeneratorOutput(
                        audio=generated.audio[index : index + 1],
                        aux={
                            name: value[index : index + 1]
                            if value.ndim > 0 and value.shape[0] == batch_size
                            else value
                            for name, value in generated.aux.items()
                        },
                    )
                    loss = self.raw_task.objective.auxiliary_loss(
                        sample_batch,
                        sample_generated,
                        self.state.logical_step,
                    ).loss.detach()
                    weighted_total = weighted_total + loss * weight
                    sample_count += weight
                batch_count += 1
                if max_batches is not None and batch_count >= max_batches:
                    break
        if weighted_total is None:
            raise ValueError("validation loader is empty")
        totals = torch.stack((weighted_total, weighted_total.new_tensor(float(sample_count))))
        reduced = cast(Tensor, self.fabric.all_reduce(totals, reduce_op="sum"))
        result = reduced[0] / reduced[1]
        return {"validation/loss": result}

    @torch.no_grad()
    def predict(self, batch: Batch) -> GeneratorOutput:
        with self._evaluation_mode():
            return self._generate(batch.to(self.fabric.device, non_blocking=True))

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
            phase.micro_steps % phase.config.accumulate == 0 for phase in self.phases.values()
        )
