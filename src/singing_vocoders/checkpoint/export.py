from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from singing_vocoders.checkpoint.io import _extract_legacy_state


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
    state, _ = _extract_legacy_state(checkpoint)
    generator = {
        name.removeprefix("generator."): tensor
        for name, tensor in state.items()
        if name.startswith("generator.")
    }
    if not generator:
        raise ValueError("checkpoint contains no generator weights")
    if resolved_config is None:
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
            raise ValueError("a resolved config is required for legacy checkpoints")
        resolved_config = checkpoint["config"]
    features = resolved_config["data"]["features"]
    generator_params = dict(resolved_config["model"]["generator"]["params"])
    export_config = resolved_config.get("export", {})
    model_args = {
        **generator_params,
        "sampling_rate": features["sample_rate"],
        "num_mels": features["num_mels"],
        "hop_size": features["hop_size"],
        "n_fft": features["n_fft"],
        "win_size": features["win_size"],
        "fmin": features["fmin"],
        "fmax": features["fmax"],
        "pc_aug": export_config.get("pc_aug", False),
        "mini_nsf": export_config.get("mini_nsf", generator_params.get("mini_nsf", False)),
        "noise_sigma": export_config.get("noise_sigma", generator_params.get("noise_sigma", 0.0)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({"generator": generator}, temporary)
    os.replace(temporary, destination)
    config_path = destination.with_name("config.json")
    config_temporary = config_path.with_suffix(".json.tmp")
    config_temporary.write_text(
        json.dumps(model_args, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    os.replace(config_temporary, config_path)
    return destination, config_path
