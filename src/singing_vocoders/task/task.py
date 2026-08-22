from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from singing_vocoders.components.builtin import create_builtin_registry
from singing_vocoders.components.contracts import (
    Batch,
    DiscriminatorComponent,
    DiscriminatorOutput,
    GeneratorComponent,
    GeneratorOutput,
)
from singing_vocoders.components.factory import ComponentRegistry
from singing_vocoders.config.schema import RecipeConfig
from singing_vocoders.task.objective import VocoderObjective


class VocoderTask(nn.Module):
    def __init__(self, config: RecipeConfig, registry: ComponentRegistry | None = None) -> None:
        super().__init__()
        self.config = config
        component_registry = registry or create_builtin_registry()
        self.generator = component_registry.build(config.model.generator, GeneratorComponent)
        self.discriminator = nn.ModuleDict(
            {
                name: component_registry.build(spec, DiscriminatorComponent)
                for name, spec in config.model.discriminators.items()
            }
        )
        self.objective = VocoderObjective(config.objective, config.data.features)

    def generate(self, batch: Batch) -> GeneratorOutput:
        output = self.generator(batch)
        if not isinstance(output, GeneratorOutput):
            raise TypeError("generator returned an invalid output")
        return output

    def discriminate(self, audio: Tensor) -> dict[str, DiscriminatorOutput]:
        output: dict[str, DiscriminatorOutput] = {}
        for name, discriminator in self.discriminator.items():
            result = discriminator(audio)
            if not isinstance(result, DiscriminatorOutput):
                raise TypeError(f"discriminator {name} returned an invalid output")
            output[name] = result
        return output

    def forward(
        self,
        operation: Literal["generate", "discriminate"],
        *,
        batch: Batch | None = None,
        audio: Tensor | None = None,
    ) -> GeneratorOutput | dict[str, DiscriminatorOutput]:
        if operation == "generate":
            if batch is None:
                raise ValueError("generate requires batch")
            return self.generate(batch)
        if operation == "discriminate":
            if audio is None:
                raise ValueError("discriminate requires audio")
            return self.discriminate(audio)
        raise ValueError(f"unknown task operation: {operation}")

    def generator_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.generator.parameters() if parameter.requires_grad]

    def discriminator_parameters(self) -> list[nn.Parameter]:
        return [
            parameter for parameter in self.discriminator.parameters() if parameter.requires_grad
        ]
