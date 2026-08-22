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
    assert {load_config(path).config.name for path in recipes} == {path.stem for path in recipes}
