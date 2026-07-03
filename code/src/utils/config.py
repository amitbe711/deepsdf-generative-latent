"""Minimal YAML config loader with attribute access and dict-style overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class AttrDict(dict):
    """A dict whose keys are also accessible as attributes, recursively."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def merge(self, overrides: dict[str, Any]) -> "AttrDict":
        """Return a deep copy with ``overrides`` applied (supports dotted keys)."""
        out = AttrDict(copy.deepcopy(dict(self)))
        for dotted_key, value in overrides.items():
            node: Any = out
            parts = dotted_key.split(".")
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    node[part] = AttrDict()
                node = node[part]
            node[parts[-1]] = AttrDict._wrap(value)
        return out


def load_config(path: str | Path) -> AttrDict:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AttrDict(raw)
