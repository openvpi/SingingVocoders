import pytest
import torch

from singing_vocoders.components import (
    Batch,
    ComponentRegistry,
    GeneratorComponent,
    GeneratorOutput,
)
from singing_vocoders.config.schema import ComponentSpec


class ExampleGenerator(GeneratorComponent):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, batch: Batch) -> GeneratorOutput:
        return GeneratorOutput(batch.audio)


class FlexibleGenerator(GeneratorComponent):
    def __init__(self, channels: int, **options: object) -> None:
        super().__init__()
        self.channels = channels
        self.options = options

    def forward(self, batch: Batch) -> GeneratorOutput:
        return GeneratorOutput(batch.audio)


def test_component_factory_checks_constructor_signature() -> None:
    registry = ComponentRegistry()
    registry.register("test_generator", ExampleGenerator)
    component = registry.build(
        ComponentSpec(target="test_generator", params={"channels": 8}), GeneratorComponent
    )
    assert isinstance(component, ExampleGenerator)
    with pytest.raises(TypeError, match="invalid parameters"):
        registry.build(
            ComponentSpec(target="test_generator", params={"channels": 8, "ignored": True}),
            GeneratorComponent,
        )


def test_component_factory_supports_dotted_targets_and_explicit_kwargs() -> None:
    registry = ComponentRegistry()
    component = registry.build(
        ComponentSpec(
            target=f"{__name__}.FlexibleGenerator",
            params={"channels": 8, "experiment": {"variant": "wide"}},
        ),
        GeneratorComponent,
    )
    assert isinstance(component, FlexibleGenerator)
    assert component.options == {"experiment": {"variant": "wide"}}


def test_batch_moves_all_tensor_fields() -> None:
    batch = Batch(*(torch.zeros(1) for _ in range(5)))
    moved = batch.to("cpu")
    assert all(
        tensor.device.type == "cpu"
        for tensor in (moved.audio, moved.mel, moved.f0, moved.uv, moved.pe)
    )
