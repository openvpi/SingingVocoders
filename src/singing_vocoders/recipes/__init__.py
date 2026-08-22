"""Built-in training recipes."""

from pathlib import Path

RECIPE_ROOT = Path(__file__).resolve().parent


def list_recipes() -> tuple[str, ...]:
    return tuple(
        path.stem for path in sorted(RECIPE_ROOT.glob("*.yaml")) if not path.name.startswith("_")
    )


def resolve_recipe(value: str | Path) -> Path:
    requested = Path(value)
    if requested.is_file():
        return requested.resolve()
    name = requested.name
    if requested.parent != Path("."):
        raise FileNotFoundError(requested.resolve())
    candidate = RECIPE_ROOT / (name if name.endswith(".yaml") else f"{name}.yaml")
    if not candidate.is_file() or candidate.name.startswith("_"):
        raise FileNotFoundError(f"unknown recipe or configuration path: {value}")
    return candidate


__all__ = ["list_recipes", "resolve_recipe"]
