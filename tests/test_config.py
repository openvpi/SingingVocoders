from pathlib import Path

import pytest
from pydantic import ValidationError

from singing_vocoders.config.loader import load_config
from singing_vocoders.recipes import list_recipes, resolve_recipe

BASE = """
schema_version: 1
name: base
data:
  processed_dir: data/test
model:
  generator: {target: test_generator, params: {channels: 8}}
  discriminators: {test: {target: test_discriminator}}
objective: {name: hifigan}
optimization:
  name: alternating
  generator: {optimizers: [{parameters: all, optimizer: {name: adamw}}]}
  discriminator: {optimizers: [{parameters: all, optimizer: {name: adamw}}]}
"""


def test_relative_bases_and_dotlist_override(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(BASE, encoding="utf-8")
    (tmp_path / "recipe.yaml").write_text(
        "bases: [base.yaml]\nname: child\ndata: {crop_frames: 16}\n", encoding="utf-8"
    )
    loaded = load_config(tmp_path / "recipe.yaml", ("data.batch_size=7",))
    assert loaded.config.name == "child"
    assert loaded.config.data.crop_frames == 16
    assert loaded.config.data.batch_size == 7
    assert len(loaded.fingerprint) == 64


def test_cycle_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("bases: [b.yaml]\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("bases: [a.yaml]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cyclic config inheritance"):
        load_config(tmp_path / "a.yaml")


def test_unknown_outer_field_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "recipe.yaml").write_text(BASE + "unknown: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(tmp_path / "recipe.yaml")


def test_unresolved_interpolation_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "recipe.yaml").write_text(
        BASE.replace("data/test", "${missing.path}"), encoding="utf-8"
    )
    with pytest.raises(Exception, match="missing"):
        load_config(tmp_path / "recipe.yaml")


def test_all_builtin_recipes_validate() -> None:
    canonical = {
        "ddsp-univnet",
        "ddspgan",
        "fast-nsf-hifigan",
        "hifivae",
        "lvc-ddspgan",
        "nsf-hifigan",
        "nsf-univnet-msd",
        "nsf-univnet",
        "one-step-nsf-hifigan",
        "pc-nsf-hifigan",
        "univnet",
    }
    historical = {
        "historical/base-ddspgan",
        "historical/base-hifi-one",
        "historical/base-hifi",
        "historical/ddsp-univnet",
        "historical/ft-hifigan",
        "historical/hifivae",
        "historical/lvc-base-ddspgan",
        "historical/nsf-hifigan-fast",
        "historical/nsf-univnet",
        "historical/univnet",
    }
    names = set(list_recipes())
    assert names == canonical | historical
    loaded = {name: load_config(resolve_recipe(name)).config for name in names}
    assert {config.name for config in loaded.values()} == names
    assert {loaded[name].data.features.num_mels for name in canonical} == {128}
    assert loaded["historical/ft-hifigan"].data.batch_size == 10
    assert loaded["historical/nsf-hifigan-fast"].data.validation_size == 5
    fast_optimization = loaded["historical/nsf-hifigan-fast"].optimization
    assert fast_optimization.name == "alternating"
    assert [route.optimizer.name for route in fast_optimization.generator.optimizers] == [
        "muon",
        "adamw",
    ]
    assert fast_optimization.generator.optimizers[1].optimizer.betas == (0.9, 0.999)
    assert loaded["historical/hifivae"].data.features.num_mels == 192
    assert loaded["historical/base-ddspgan"].data.features.pitch_extractor == "parselmouth"
    univnet_params = loaded["univnet"].model.generator.params
    assert univnet_params["cond_in_channels"] == 128
    assert univnet_params["cg_channels"] == 32
    assert univnet_params["upsample_rates"] == [8, 8, 4]
    assert univnet_params["upmel"] == 2
    assert "noise_channels" not in univnet_params


def test_builtin_recipe_resolution_rejects_internal_and_escaping_paths() -> None:
    assert resolve_recipe("historical/base-hifi").name == "base-hifi.yaml"
    with pytest.raises(FileNotFoundError, match="unknown recipe"):
        resolve_recipe("historical/_base")
    with pytest.raises(FileNotFoundError, match="unknown recipe"):
        resolve_recipe("../outside")


@pytest.mark.parametrize(
    (
        "name",
        "target",
        "batch_size",
        "crop_frames",
        "validation_size",
        "max_steps",
        "validate_every",
    ),
    [
        ("historical/base-hifi", "nsf_hifigan", 3, 20, 1, 100_000, 1_000),
        ("historical/base-hifi-one", "nsf_hifigan", 3, 20, 1, 100_000, 1_000),
        ("historical/ft-hifigan", "nsf_hifigan", 10, 32, 10, 100_000, 2_000),
        ("historical/nsf-hifigan-fast", "nsf_hifigan", 40, 48, 5, 1_000_000, 2_000),
        ("historical/base-ddspgan", "ddspgan", 3, 20, 1, 100_000, 1_000),
        ("historical/lvc-base-ddspgan", "lvc_ddspgan", 3, 20, 1, 800_000, 1_000),
        ("historical/ddsp-univnet", "ddsp_univnet", 10, 20, 1, 800_000, 8_000),
        ("historical/nsf-univnet", "nsf_univnet", 10, 20, 1, 800_000, 8_000),
        ("historical/univnet", "univnet", 10, 20, 1, 800_000, 8_000),
        ("historical/hifivae", "hifivae", 6, 22, 1, 5_000_000, 4_000),
    ],
)
def test_historical_recipes_preserve_effective_training_values(
    name: str,
    target: str,
    batch_size: int,
    crop_frames: int,
    validation_size: int,
    max_steps: int,
    validate_every: int,
) -> None:
    config = load_config(resolve_recipe(name)).config
    assert config.model.generator.target == target
    assert config.data.batch_size == batch_size
    assert config.data.crop_frames == crop_frames
    assert config.data.validation_size == validation_size
    assert config.runtime.max_steps == max_steps
    assert config.runtime.validate_every == validate_every


@pytest.mark.parametrize(
    ("objective", "optimization"),
    [
        (
            "{name: one_step}",
            "{name: alternating, generator: {optimizers: [{parameters: all, optimizer: "
            "{name: adamw}}]}, discriminator: {optimizers: [{parameters: all, optimizer: "
            "{name: adamw}}]}}",
        ),
        (
            "{name: hifigan}",
            "{name: joint, joint: {optimizers: [{parameters: all, optimizer: {name: adamw}}]}}",
        ),
    ],
)
def test_joint_plan_and_one_step_objective_must_be_paired(
    tmp_path: Path, objective: str, optimization: str
) -> None:
    recipe = BASE.replace("{name: hifigan}", objective).replace(
        "name: alternating\n  generator: {optimizers: [{parameters: all, optimizer: "
        "{name: adamw}}]}\n  discriminator: {optimizers: [{parameters: all, optimizer: "
        "{name: adamw}}]}",
        optimization,
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(recipe, encoding="utf-8")
    with pytest.raises(ValidationError, match="joint optimization and one_step"):
        load_config(path)


def test_warmup_must_end_at_generator_accumulation_boundary(tmp_path: Path) -> None:
    recipe = BASE.replace(
        "name: alternating\n  generator: {optimizers: [{parameters: all, optimizer: "
        "{name: adamw}}]}",
        "name: warmup_alternating\n  aux_steps: 1\n  generator: "
        "{optimizers: [{parameters: all, optimizer: {name: adamw}}], accumulate: 2}",
    )
    path = tmp_path / "invalid-warmup.yaml"
    path.write_text(recipe, encoding="utf-8")

    with pytest.raises(ValidationError, match="accumulation boundary"):
        load_config(path)


def test_hifivae_hop_size_tracks_data_features() -> None:
    recipe = Path(__file__).parents[1] / "src" / "singing_vocoders" / "recipes" / "hifivae.yaml"
    with pytest.raises(ValidationError, match="upsample rates"):
        load_config(recipe, ("data.features.hop_size=256",))


@pytest.mark.parametrize(
    ("recipe", "override", "message"),
    [
        ("univnet", "model.generator.params.out_channels=2", "out_channels=1"),
        (
            "nsf-hifigan",
            "model.generator.params.upsample_kernel_sizes=[3,16,4,4,4]",
            "upsample kernel",
        ),
        (
            "nsf-hifigan",
            "model.generator.params.resblock_dilation_sizes=[[1,3,5,7],[1,3,5],[1,3,5]]",
            "requires 3 dilations",
        ),
        ("nsf-hifigan", "model.discriminators.mpd.params.periods=[2.5]", "positive integers"),
    ],
)
def test_builtin_component_semantics_fail_during_validation(
    recipe: str, override: str, message: str
) -> None:
    recipe_path = (
        Path(__file__).parents[1] / "src" / "singing_vocoders" / "recipes" / f"{recipe}.yaml"
    )
    with pytest.raises((TypeError, ValueError), match=message):
        loaded = load_config(recipe_path, (override,))
        from singing_vocoders.components.builtin import create_builtin_registry
        from singing_vocoders.task.task import VocoderTask

        VocoderTask(loaded.config, create_builtin_registry())
