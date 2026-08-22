from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from typing import TypeVar

from singing_vocoders.components.contracts import DiscriminatorComponent, GeneratorComponent
from singing_vocoders.config.schema import ComponentSpec

ComponentT = TypeVar("ComponentT", GeneratorComponent, DiscriminatorComponent)


class ComponentRegistry:
    def __init__(self) -> None:
        self._targets: dict[str, type[GeneratorComponent] | type[DiscriminatorComponent]] = {}

    def register(
        self,
        name: str,
        component: type[GeneratorComponent] | type[DiscriminatorComponent],
    ) -> None:
        if not name or "." in name:
            raise ValueError("component aliases must be non-empty and must not contain dots")
        if name in self._targets:
            raise ValueError(f"component alias is already registered: {name}")
        self._targets[name] = component

    def aliases(self) -> Mapping[str, type[GeneratorComponent] | type[DiscriminatorComponent]]:
        return dict(self._targets)

    def resolve(self, target: str) -> type[GeneratorComponent] | type[DiscriminatorComponent]:
        if target in self._targets:
            return self._targets[target]
        module_name, separator, attribute = target.rpartition(".")
        if not separator:
            raise LookupError(f"unknown component alias: {target}")
        module = importlib.import_module(module_name)
        candidate = getattr(module, attribute)
        if not inspect.isclass(candidate):
            raise TypeError(f"component target must resolve to a class: {target}")
        return candidate

    def build(self, spec: ComponentSpec, expected: type[ComponentT]) -> ComponentT:
        component_type = self.resolve(spec.target)
        if not issubclass(component_type, expected):
            raise TypeError(
                f"{spec.target} must implement {expected.__module__}.{expected.__qualname__}"
            )
        try:
            inspect.signature(component_type).bind(**spec.params)
        except TypeError as error:
            raise TypeError(f"invalid parameters for {spec.target}: {error}") from error
        return component_type(**spec.params)
