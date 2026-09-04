from __future__ import annotations

"""Configuration loading: packaged defaults, user YAML, and CLI overrides."""

import importlib.resources
from pathlib import Path

import yaml

_PACKAGED_STYLES = ("default", "en")


def builtin_defaults() -> dict:
    """Load AlwaysWhisper's packaged default configuration (data/default_config.yaml)."""
    text = (
        importlib.resources.files("alwayswhisper") / "data" / "default_config.yaml"
    ).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto `base`; override wins.

    Only dict-vs-dict pairs are merged key-by-key; a list or scalar in
    `override` fully replaces the corresponding value in `base` rather than
    combining with it. This is a deliberate departure from the origin
    repo's project config, which was all-or-nothing: the mere presence of a
    project config.yaml meant packaged defaults were not consulted for ANY
    key, not just the ones the project file actually set. Returns a new
    dict; `base` and `override` are not mutated.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: str | Path | None = None, overrides: dict | None = None
) -> dict:
    """Build the effective config: packaged defaults <- user YAML <- overrides.

    `config_path`, if given, must exist -- a typo'd --config path fails
    loudly rather than silently falling back to defaults. `overrides` is
    typically built from CLI flags the user actually passed (so unset flags
    don't clobber the user's YAML).
    """
    config = builtin_defaults()

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        user_config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config = deep_merge(config, user_config)

    if overrides:
        config = deep_merge(config, overrides)

    return config


def resolve_style_path(value: str | None) -> Path:
    """Resolve a caption style reference to a concrete file path.

    None -> the packaged default style. A bare name matching a packaged
    style ("default", "en") -> that packaged file. Anything else is treated
    as a filesystem path (raises FileNotFoundError if it doesn't exist).
    """
    styles_dir = importlib.resources.files("alwayswhisper") / "data" / "styles"

    if value is None:
        return Path(str(styles_dir / "default.yaml"))

    if value in _PACKAGED_STYLES:
        return Path(str(styles_dir / f"{value}.yaml"))

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(
            f"Caption style not found: {value!r} (not a packaged style "
            f"{_PACKAGED_STYLES!r} and not an existing file path)"
        )
    return path
