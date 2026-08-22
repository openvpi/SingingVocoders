from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import click

from singing_vocoders import __version__
from singing_vocoders.checkpoint import export_diffsinger, inspect_checkpoint
from singing_vocoders.config import LoadedConfig, load_config
from singing_vocoders.data import preprocess_dataset
from singing_vocoders.recipes import list_recipes, resolve_recipe
from singing_vocoders.runtime import (
    choose_run_dir,
    create_fabric,
    train_worker,
    validate_worker,
)
from singing_vocoders.task import VocoderTask


def _load(recipe: str, overrides: tuple[str, ...]) -> LoadedConfig:
    return load_config(resolve_recipe(recipe), overrides)


def _scope(value: str) -> Literal["generator", "all"]:
    return cast(Literal["generator", "all"], value)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__)
def main() -> None:
    """Train, validate, inspect, and export singing voice vocoders."""


@main.group("recipes")
def recipes_group() -> None:
    """Discover built-in training recipes."""


@recipes_group.command("list")
def recipes_list() -> None:
    """List built-in recipe names."""
    click.echo("\n".join(list_recipes()))


@main.group("config")
def config_group() -> None:
    """Inspect and validate resolved configurations."""


@config_group.command("validate")
@click.argument("recipe")
@click.option("--set", "overrides", multiple=True, metavar="KEY=VALUE")
def config_validate(recipe: str, overrides: tuple[str, ...]) -> None:
    """Resolve RECIPE and validate its full component graph."""
    loaded = _load(recipe, overrides)
    task = VocoderTask(loaded.config)
    result = {
        "name": loaded.config.name,
        "path": str(loaded.path),
        "fingerprint": loaded.fingerprint,
        "generator_parameters": sum(parameter.numel() for parameter in task.generator.parameters()),
        "discriminator_parameters": sum(
            parameter.numel() for parameter in task.discriminator.parameters()
        ),
    }
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


@main.command("preprocess")
@click.argument("recipe")
@click.option("--set", "overrides", multiple=True, metavar="KEY=VALUE")
@click.option(
    "--source", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None
)
@click.option("--output", type=click.Path(path_type=Path, file_okay=False), default=None)
@click.option("--workers", type=click.IntRange(min=0), default=None)
def preprocess_command(
    recipe: str,
    overrides: tuple[str, ...],
    source: Path | None,
    output: Path | None,
    workers: int | None,
) -> None:
    """Preprocess WAV files according to RECIPE."""
    loaded = _load(recipe, overrides)
    data = loaded.config.data
    source_dir = source or data.source
    if source_dir is None:
        raise click.UsageError("provide --source or set data.source in the recipe")
    summary = preprocess_dataset(
        source_dir,
        output or data.processed_dir,
        data.features,
        validation_size=data.validation_size,
        seed=data.seed,
        workers=data.num_workers if workers is None else workers,
    )
    click.echo(
        json.dumps(
            {
                "source_count": summary.source_count,
                "train_count": summary.train_count,
                "validation_count": summary.validation_count,
                "fingerprint": summary.fingerprint,
                "output_dir": str(summary.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@main.command("train")
@click.argument("recipe")
@click.option("--set", "overrides", multiple=True, metavar="KEY=VALUE")
@click.option("--run-dir", type=click.Path(path_type=Path, file_okay=False), default=None)
@click.option("--resume", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--warm-start", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option(
    "--warm-start-scope",
    type=click.Choice(("generator", "all"), case_sensitive=True),
    default="all",
    show_default=True,
)
def train_command(
    recipe: str,
    overrides: tuple[str, ...],
    run_dir: Path | None,
    resume: Path | None,
    warm_start: Path | None,
    warm_start_scope: str,
) -> None:
    """Train RECIPE in a new run directory."""
    if resume is not None and warm_start is not None:
        raise click.UsageError("--resume and --warm-start are mutually exclusive")
    loaded = _load(recipe, overrides)
    destination = choose_run_dir(loaded.config, run_dir)
    fabric = create_fabric(loaded.config, destination)
    fabric.launch(
        cast(Any, train_worker),
        loaded,
        destination,
        resume.resolve() if resume is not None else None,
        warm_start.resolve() if warm_start is not None else None,
        _scope(warm_start_scope),
    )


@main.command("validate")
@click.argument("recipe")
@click.argument("checkpoint", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--set", "overrides", multiple=True, metavar="KEY=VALUE")
@click.option(
    "--legacy-scope",
    type=click.Choice(("generator", "all"), case_sensitive=True),
    default="generator",
    show_default=True,
)
@click.option("--max-batches", type=click.IntRange(min=1), default=None)
def validate_command(
    recipe: str,
    checkpoint: Path,
    overrides: tuple[str, ...],
    legacy_scope: str,
    max_batches: int | None,
) -> None:
    """Validate CHECKPOINT on the validation split for RECIPE."""
    loaded = _load(recipe, overrides)
    fabric = create_fabric(loaded.config)
    fabric.launch(
        cast(Any, validate_worker),
        loaded,
        checkpoint.resolve(),
        _scope(legacy_scope),
        max_batches,
    )


@main.group("checkpoint")
def checkpoint_group() -> None:
    """Inspect versioned and legacy checkpoints."""


@checkpoint_group.command("inspect")
@click.argument("path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
def checkpoint_inspect(path: Path) -> None:
    """Report checkpoint container and tensor schema metadata."""
    click.echo(json.dumps(inspect_checkpoint(path), ensure_ascii=False, indent=2))


@main.command("export")
@click.argument("checkpoint", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.argument("output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--recipe", default=None, help="Recipe required only for a legacy checkpoint.")
@click.option("--set", "overrides", multiple=True, metavar="KEY=VALUE")
def export_command(
    checkpoint: Path,
    output: Path,
    recipe: str | None,
    overrides: tuple[str, ...],
) -> None:
    """Export an NSF CHECKPOINT and its DiffSinger configuration."""
    report = inspect_checkpoint(checkpoint)
    if report["format_version"] == 1:
        if recipe is not None or overrides:
            raise click.UsageError("versioned checkpoints use their embedded recipe")
        resolved_config = None
    else:
        if recipe is None:
            raise click.UsageError("--recipe is required for a legacy checkpoint")
        resolved_config = _load(recipe, overrides).resolved
    checkpoint_path, config_path = export_diffsinger(
        checkpoint, output, resolved_config=resolved_config
    )
    click.echo(
        json.dumps(
            {"checkpoint": str(checkpoint_path), "config": str(config_path)},
            ensure_ascii=False,
            indent=2,
        )
    )
