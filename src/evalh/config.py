"""Config loading with shell-style ${VAR:-default} expansion.

YAML has no env interpolation. We support exactly one form so a config file
can say `provider: ${EVAL_PROVIDER:-stub}` and stay offline by default.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def expand_env(text: str) -> str:
    def sub(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        value = os.environ.get(name)
        if value:
            return value
        if default is None:
            raise KeyError(f"environment variable {name} is not set and has no default")
        return default

    return _ENV.sub(sub, text)


def load_config(path: Path | str) -> dict[str, Any]:
    return yaml.safe_load(expand_env(Path(path).read_text()))


def fixtures_dir(config: dict[str, Any]) -> Path:
    """Base fixture dir, or a named variant under it (EVAL_FIXTURES_VARIANT).

    Variants model "same model name, different behavior": a provider moving an
    alias to a new snapshot without telling you. See `make demo-regression`.
    """
    base = Path(config["fixtures"])
    variant = os.environ.get("EVAL_FIXTURES_VARIANT", "")
    return base / variant if variant else base
