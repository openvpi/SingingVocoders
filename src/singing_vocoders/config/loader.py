from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from singing_vocoders.config.schema import RecipeConfig


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    path: Path
    config: RecipeConfig
    resolved: dict[str, Any]
    fingerprint: str


def _load_tree(path: Path, stack: tuple[Path, ...]) -> DictConfig:
    resolved_path = path.resolve()
    if resolved_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f"cyclic config inheritance: {chain}")
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)

    current = OmegaConf.load(resolved_path)
    if not isinstance(current, DictConfig):
        raise TypeError(f"configuration root must be a mapping: {resolved_path}")
    base_values = current.pop("bases", [])
    if isinstance(base_values, str) or not isinstance(base_values, (list, ListConfig)):
        raise TypeError(f"bases must be a list in {resolved_path}")

    merged = OmegaConf.create()
    next_stack = (*stack, resolved_path)
    for base_value in base_values:
        if not isinstance(base_value, str):
            raise TypeError(f"base path must be a string in {resolved_path}")
        base_path = (resolved_path.parent / base_value).resolve()
        merged = cast(DictConfig, OmegaConf.merge(merged, _load_tree(base_path, next_stack)))
    return cast(DictConfig, OmegaConf.merge(merged, current))


def _fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path, overrides: tuple[str, ...] = ()) -> LoadedConfig:
    config_path = Path(path).resolve()
    merged = _load_tree(config_path, ())
    if overrides:
        merged = cast(DictConfig, OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides))))
    OmegaConf.resolve(merged)
    value = OmegaConf.to_container(merged, resolve=True, enum_to_str=True)
    if not isinstance(value, dict):
        raise TypeError("resolved configuration root must be a mapping")
    config = RecipeConfig.model_validate(value)
    normalized = config.model_dump(mode="json")
    return LoadedConfig(config_path, config, normalized, _fingerprint(normalized))
