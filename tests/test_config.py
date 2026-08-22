from pathlib import Path

import pytest
from pydantic import ValidationError

from singing_vocoders.config.loader import load_config

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
  generator: {optimizer: {name: adamw}}
  discriminator: {optimizer: {name: adamw}}
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
    recipe_dir = Path(__file__).parents[1] / "src" / "singing_vocoders" / "recipes"
    recipes = sorted(path for path in recipe_dir.glob("*.yaml") if not path.name.startswith("_"))
    assert len(recipes) == 11
    loaded = [load_config(path).config for path in recipes]
    assert {config.name for config in loaded} == {path.stem for path in recipes}
    assert {config.data.features.num_mels for config in loaded} == {128}


@pytest.mark.parametrize(
    ("objective", "optimization"),
    [
        (
            "{name: one_step}",
            "{name: alternating, generator: {optimizer: {name: adamw}}, "
            "discriminator: {optimizer: {name: adamw}}}",
        ),
        ("{name: hifigan}", "{name: joint, joint: {optimizer: {name: adamw}}}"),
    ],
)
def test_joint_plan_and_one_step_objective_must_be_paired(
    tmp_path: Path, objective: str, optimization: str
) -> None:
    recipe = BASE.replace("{name: hifigan}", objective).replace(
        "name: alternating\n  generator: {optimizer: {name: adamw}}\n  "
        "discriminator: {optimizer: {name: adamw}}",
        optimization,
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(recipe, encoding="utf-8")
    with pytest.raises(ValidationError, match="joint optimization and one_step"):
        load_config(path)


def test_warmup_must_end_at_generator_accumulation_boundary(tmp_path: Path) -> None:
    recipe = BASE.replace(
        "name: alternating\n  generator: {optimizer: {name: adamw}}",
        "name: warmup_alternating\n  aux_steps: 1\n  generator: "
        "{optimizer: {name: adamw}, accumulate: 2}",
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
        ("univnet", "model.generator.params.output_channels=2", "output_channels=1"),
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
