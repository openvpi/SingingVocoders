from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from singing_vocoders.checkpoint.io import _extract_legacy_state, _validate_state_schema
from singing_vocoders.components import GeneratorComponent
from singing_vocoders.components.builtin import create_builtin_registry
from singing_vocoders.config.schema import RecipeConfig


def export_diffsinger(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    resolved_config: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    source = Path(checkpoint_path).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = torch.load(source, map_location="cpu", mmap=True, weights_only=True)
    is_versioned = isinstance(checkpoint, dict) and checkpoint.get("format_version") == 1
    if is_versioned:
        if resolved_config is not None:
            raise ValueError("versioned checkpoints must use their embedded configuration")
        embedded_config = checkpoint.get("config")
        if not isinstance(embedded_config, dict):
            raise ValueError("versioned checkpoint contains no resolved configuration")
        resolved_config = embedded_config
    elif resolved_config is None:
        raise ValueError("a resolved config is required for legacy checkpoints")

    config = RecipeConfig.model_validate(resolved_config)
    if config.model.generator.target != "nsf_hifigan":
        raise ValueError("DiffSinger export is only supported for the nsf_hifigan generator")
    state, _ = _extract_legacy_state(checkpoint)
    generator = {
        name.removeprefix("generator."): tensor
        for name, tensor in state.items()
        if name.startswith("generator.")
    }
    if not generator and not any(name.startswith("discriminator.") for name in state):
        generator = state
    if not generator:
        raise ValueError("checkpoint contains no generator weights")
    expected_generator = create_builtin_registry().build(config.model.generator, GeneratorComponent)
    _validate_state_schema(expected_generator.state_dict(), generator)

    features = config.data.features
    generator_params = dict(config.model.generator.params)
    noise_sigma = generator_params.get("noise_sigma", 0.0)
    model_args = {
        **generator_params,
        "sampling_rate": features.sample_rate,
        "num_mels": features.num_mels,
        "hop_size": features.hop_size,
        "n_fft": features.n_fft,
        "win_size": features.win_size,
        "fmin": features.fmin,
        "fmax": features.fmax,
        "pc_aug": config.augmentation.pc_enabled,
        "mini_nsf": bool(generator_params.get("mini_nsf", False)),
        "noise_sigma": None if noise_sigma is None else float(noise_sigma),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    config_path = destination.with_name("config.json")
    if config_path.exists():
        raise FileExistsError(config_path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    config_temporary = config_path.with_suffix(".json.tmp")
    model_published = False
    try:
        torch.save({"generator": generator}, temporary)
        config_temporary.write_text(
            json.dumps(model_args, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, destination)
        model_published = True
        os.replace(config_temporary, config_path)
    except BaseException:
        if model_published:
            destination.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
        config_temporary.unlink(missing_ok=True)
    return destination, config_path
