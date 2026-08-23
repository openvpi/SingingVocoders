"""Built-in training recipes."""

from pathlib import Path

RECIPE_ROOT = Path(__file__).resolve().parent


def list_recipes() -> tuple[str, ...]:
    return tuple(
        path.relative_to(RECIPE_ROOT).with_suffix("").as_posix()
        for path in sorted(RECIPE_ROOT.rglob("*.yaml"))
        if not path.name.startswith("_")
    )


def resolve_recipe(value: str | Path) -> Path:
    requested = Path(value)
    if requested.is_file():
        return requested.resolve()
    if requested.is_absolute():
        raise FileNotFoundError(requested)
    relative = requested if requested.suffix == ".yaml" else requested.with_suffix(".yaml")
    candidate = (RECIPE_ROOT / relative).resolve()
    if (
        not candidate.is_relative_to(RECIPE_ROOT)
        or not candidate.is_file()
        or candidate.name.startswith("_")
    ):
        raise FileNotFoundError(f"unknown recipe or configuration path: {value}")
    return candidate


__all__ = ["list_recipes", "resolve_recipe"]
