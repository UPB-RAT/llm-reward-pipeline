"""Load a list of prompt strings from an external file.

Supported formats:
- ``.py``  : a Python module exposing a ``TEST_PROMPTS`` (or ``PROMPTS`` /
             ``PROMPT_LIST``) list of strings.
- ``.json``: a JSON array of strings.
- other    : treated as plain text, one prompt per non-empty line.

``.py`` and ``.json`` support multi-line prompts; the plain-text fallback does
not (a single prompt spanning several lines would be split).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROMPT_ATTRIBUTES = ("TEST_PROMPTS", "PROMPTS", "PROMPT_LIST")


def load_prompts(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    if path.suffix == ".py":
        return _load_from_module(path)
    if path.suffix == ".json":
        return _load_from_json(path)
    return _load_from_text(path)


def _load_from_module(path: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("_external_prompts", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ValueError(f"Failed to import prompts module '{path}': {e}") from e

    for name in PROMPT_ATTRIBUTES:
        value = getattr(module, name, None)
        if isinstance(value, list) and value and all(isinstance(p, str) for p in value):
            return list(value)

    raise ValueError(
        f"No prompt list found in '{path}'. Expected a module attribute named "
        f"{'/'.join(PROMPT_ATTRIBUTES)} containing a list of strings."
    )


def _load_from_json(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data or not all(isinstance(p, str) for p in data):
        raise ValueError(f"JSON file '{path}' must be a non-empty array of strings")
    return data


def _load_from_text(path: Path) -> list[str]:
    prompts = [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"No prompts found in '{path}'")
    return prompts
