from pathlib import Path

import pytest
import torch

from singing_vocoders.components import Batch, GeneratorComponent
from singing_vocoders.components.builtin import create_builtin_registry
from singing_vocoders.config import load_config
from singing_vocoders.modules.ddsp.pcmer import PCmer

RECIPE_DIR = Path(__file__).parents[1] / "src" / "singing_vocoders" / "recipes"


@pytest.mark.parametrize("recipe", ("ddspgan", "lvc-ddspgan", "ddsp-univnet"))
def test_ddsp_recipes_retain_pcmer_control_decoder(recipe: str) -> None:
    config = load_config(RECIPE_DIR / f"{recipe}.yaml").config
    generator = create_builtin_registry().build(config.model.generator, GeneratorComponent)
    assert any(isinstance(module, PCmer) for module in generator.modules())


def test_lvc_ddsp_recipe_uses_original_parameter_names() -> None:
    config = load_config(RECIPE_DIR / "lvc-ddspgan.yaml").config
    params = config.model.generator.params
    assert params["lvc_noise_channels"] == 8
    assert params["lvc_latent_channels"] == 8
    assert params["lvc_block_nums"] == 3
    assert params["lvc_layers_each_block"] == 10
    assert params["lvc_kernel_size"] == 7
    assert params["kpnet_hidden_channels"] == 128
    assert params["kpnet_conv_size"] == 3
    assert params["up_condition"] == 2
    assert "conditioning_upsample_factor" not in params


@pytest.mark.parametrize("recipe", ("ddspgan", "lvc-ddspgan", "ddsp-univnet"))
def test_ddsp_recipe_generator_forward_is_finite_and_length_aligned(recipe: str) -> None:
    config = load_config(RECIPE_DIR / f"{recipe}.yaml").config
    generator = create_builtin_registry().build(config.model.generator, GeneratorComponent).eval()
    frames = 4
    hop_size = config.data.features.hop_size
    f0 = torch.full((1, frames), 220.0)
    batch = Batch(
        audio=torch.zeros(1, 1, frames * hop_size),
        mel=torch.randn(1, config.data.features.num_mels, frames),
        f0=f0,
        uv=torch.zeros_like(f0, dtype=torch.bool),
        pe=torch.zeros(1, dtype=torch.long),
    )
    with torch.inference_mode():
        torch.manual_seed(114514)
        output = generator(batch)
    assert output.audio.shape == batch.audio.shape
    assert torch.isfinite(output.audio).all()
