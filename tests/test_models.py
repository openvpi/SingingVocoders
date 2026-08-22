import gc
from pathlib import Path

import pytest
import torch

from singing_vocoders.components import (
    Batch,
    DiscriminatorComponent,
    GeneratorComponent,
)
from singing_vocoders.components.builtin import (
    NSFHiFiGANGenerator,
    create_builtin_registry,
)
from singing_vocoders.config import load_config
from singing_vocoders.models.nsf_HiFigan.models import (
    AttrDict,
)
from singing_vocoders.models.nsf_HiFigan.models import (
    Generator as LegacyNSFGenerator,
)

RECIPE_DIR = Path(__file__).parents[1] / "src" / "singing_vocoders" / "recipes"


def test_every_recipe_constructs_its_reflective_components() -> None:
    registry = create_builtin_registry()
    recipes = sorted(path for path in RECIPE_DIR.glob("*.yaml") if not path.name.startswith("_"))
    for path in recipes:
        config = load_config(path).config
        generator = registry.build(config.model.generator, GeneratorComponent)
        discriminators = [
            registry.build(spec, DiscriminatorComponent)
            for spec in config.model.discriminators.values()
        ]
        assert isinstance(generator, GeneratorComponent)
        assert all(isinstance(item, DiscriminatorComponent) for item in discriminators)
        del generator, discriminators
        gc.collect()


@pytest.mark.parametrize(
    ("recipe", "num_mels"),
    (("pc-nsf-hifigan.yaml", 128), ("nsf-hifigan.yaml", 128), ("nsf-hifigan.yaml", 144)),
)
def test_nsf_adapter_preserves_legacy_forward(recipe: str, num_mels: int) -> None:
    params = dict(load_config(RECIPE_DIR / recipe).config.model.generator.params)
    params["num_mels"] = num_mels
    torch.manual_seed(1234)
    legacy = LegacyNSFGenerator(AttrDict(params)).eval()
    adapter = NSFHiFiGANGenerator(**params).eval()
    adapter.load_state_dict(legacy.state_dict(), strict=True)
    assert {name: tuple(tensor.shape) for name, tensor in adapter.state_dict().items()} == {
        name: tuple(tensor.shape) for name, tensor in legacy.state_dict().items()
    }

    mel = torch.randn(1, num_mels, 2)
    f0 = torch.full((1, 2), 220.0)
    batch = Batch(
        audio=torch.zeros(1, 1, 1024),
        mel=mel,
        f0=f0,
        uv=torch.zeros_like(f0, dtype=torch.bool),
        pe=torch.zeros(1, dtype=torch.long),
    )
    with torch.inference_mode():
        torch.manual_seed(5678)
        expected = legacy(x=mel, f0=f0)
        torch.manual_seed(5678)
        actual = adapter(batch).audio
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
