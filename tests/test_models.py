import gc
from pathlib import Path

import pytest
import torch

from singing_vocoders.components import (
    Batch,
    DiscriminatorComponent,
    DiscriminatorOutput,
    GeneratorComponent,
)
from singing_vocoders.components.builtin import (
    NSFHiFiGANGenerator,
    create_builtin_registry,
)
from singing_vocoders.config import load_config
from singing_vocoders.config.schema import ComponentSpec
from singing_vocoders.modules.backbones.nsf_hifigan import (
    AttributeDict,
)
from singing_vocoders.modules.backbones.nsf_hifigan import (
    NSFHiFiGAN as LegacyNSFGenerator,
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


def test_hifigan_discriminator_state_keys_remain_legacy_compatible() -> None:
    registry = create_builtin_registry()
    mpd = registry.build(ComponentSpec(target="mpd"), DiscriminatorComponent)
    expected_mpd = {
        f"discriminators.{discriminator}.{layer}.{parameter}"
        for discriminator in range(5)
        for layer in (*[f"convs.{index}" for index in range(5)], "conv_post")
        for parameter in ("bias", "weight_g", "weight_v")
    }
    assert set(mpd.state_dict()) == expected_mpd

    msd = registry.build(ComponentSpec(target="msd"), DiscriminatorComponent)
    expected_msd = {
        f"discriminators.0.{layer}.{parameter}"
        for layer in (*[f"convs.{index}" for index in range(7)], "conv_post")
        for parameter in ("bias", "weight_orig", "weight_u", "weight_v")
    }
    expected_msd.update(
        f"discriminators.{discriminator}.{layer}.{parameter}"
        for discriminator in (1, 2)
        for layer in (*[f"convs.{index}" for index in range(7)], "conv_post")
        for parameter in ("bias", "weight_g", "weight_v")
    )
    assert set(msd.state_dict()) == expected_msd


@pytest.mark.parametrize(
    ("target", "params"),
    (
        ("msd", {}),
        ("mpd", {"periods": [2, 3]}),
        (
            "mrd",
            {"fft_sizes": [512], "hop_sizes": [128], "win_lengths": [512]},
        ),
        ("univ_mpd", {"periods": [2, 3]}),
        (
            "fast_mrd",
            {
                "init_channel": 4,
                "strides": [2, 2],
                "fft_sizes": [512],
                "hop_sizes": [128],
                "win_lengths": [512],
            },
        ),
        (
            "fast_mpd",
            {"periods": [2, 3], "init_channel": 4, "strides": [2, 2], "kernel_size": 7},
        ),
    ),
)
def test_builtin_discriminators_return_typed_outputs(
    target: str,
    params: dict[str, object],
) -> None:
    discriminator = create_builtin_registry().build(
        ComponentSpec(target=target, params=params),
        DiscriminatorComponent,
    )
    with torch.inference_mode():
        output = discriminator(torch.randn(1, 1, 4096))
    assert isinstance(output, DiscriminatorOutput)
    assert output.scores
    assert len(output.scores) == len(output.feature_maps)
    assert all(torch.isfinite(score).all() for score in output.scores)


def test_builtin_components_reject_unknown_parameters() -> None:
    with pytest.raises(TypeError, match="invalid parameters"):
        create_builtin_registry().build(
            ComponentSpec(target="mpd", params={"ignored": True}),
            DiscriminatorComponent,
        )


@pytest.mark.parametrize(
    ("target", "params", "message"),
    (
        ("mpd", {"periods": []}, "periods"),
        ("univ_mpd", {"periods": [0]}, "periods"),
        ("fast_mpd", {"periods": [2], "kernel_size": 4}, "kernel_size"),
        ("mrd", {"fft_sizes": [], "hop_sizes": [], "win_lengths": []}, "resolution"),
        (
            "fast_mrd",
            {"fft_sizes": [128], "hop_sizes": [32], "win_lengths": [256]},
            "must not exceed",
        ),
    ),
)
def test_builtin_discriminators_reject_invalid_architectures(
    target: str, params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_builtin_registry().build(
            ComponentSpec(target=target, params=params),
            DiscriminatorComponent,
        )


def test_hifivae_recipe_names_its_latent_dimension_and_validates_hop_size() -> None:
    params = dict(load_config(RECIPE_DIR / "hifivae.yaml").config.model.generator.params)
    assert params["latent_channels"] == 192
    assert "num_mels" not in params
    params["hop_size"] = 256
    with pytest.raises(ValueError, match="product of upsample_rates"):
        create_builtin_registry().build(
            ComponentSpec(target="hifivae", params=params),
            GeneratorComponent,
        )


@pytest.mark.parametrize(
    ("recipe", "num_mels"),
    (("pc-nsf-hifigan.yaml", 128), ("nsf-hifigan.yaml", 128), ("nsf-hifigan.yaml", 144)),
)
def test_nsf_adapter_preserves_legacy_forward(recipe: str, num_mels: int) -> None:
    params = dict(load_config(RECIPE_DIR / recipe).config.model.generator.params)
    params["num_mels"] = num_mels
    torch.manual_seed(1234)
    legacy = LegacyNSFGenerator(AttributeDict(params)).eval()
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
