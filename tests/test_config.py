"""Tests for alwayswhisper.config: packaged defaults, YAML/override layering,
and caption style path resolution.
"""

import pytest

from alwayswhisper.config import (
    builtin_defaults,
    deep_merge,
    load_config,
    resolve_style_path,
)


class TestBuiltinDefaults:
    def test_loads_packaged_default_config(self):
        defaults = builtin_defaults()

        assert isinstance(defaults, dict)
        assert defaults["transcribe"]["backend"] == "faster-whisper"
        assert defaults["transcribe"]["model"] == "large-v3"
        assert defaults["qa"]["enabled"] is True

    def test_returns_a_fresh_dict_each_call(self):
        # Mutating one call's result must not leak into the next.
        first = builtin_defaults()
        first["transcribe"]["backend"] = "mutated"

        second = builtin_defaults()

        assert second["transcribe"]["backend"] == "faster-whisper"


class TestDeepMerge:
    def test_nested_dicts_merge_key_by_key(self):
        base = {"a": {"x": 1, "y": 2}, "b": 10}
        override = {"a": {"y": 99}}

        assert deep_merge(base, override) == {"a": {"x": 1, "y": 99}, "b": 10}

    def test_override_wins_on_scalar_conflict(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_list_in_override_replaces_rather_than_merges(self):
        assert deep_merge({"a": [1, 2, 3]}, {"a": [4]}) == {"a": [4]}

    def test_dict_in_override_replaces_scalar_in_base(self):
        assert deep_merge({"a": 1}, {"a": {"nested": True}}) == {"a": {"nested": True}}

    def test_scalar_in_override_replaces_dict_in_base(self):
        assert deep_merge({"a": {"nested": True}}, {"a": None}) == {"a": None}

    def test_keys_only_in_base_are_kept(self):
        assert deep_merge({"a": 1, "b": 2}, {"a": 99}) == {"a": 99, "b": 2}

    def test_keys_only_in_override_are_added(self):
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}

        deep_merge(base, override)

        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}

    def test_deeply_nested_merge(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}

        assert deep_merge(base, override) == {"a": {"b": {"c": 99, "d": 2}}}


class TestLoadConfig:
    def test_no_args_returns_builtin_defaults(self):
        assert load_config() == builtin_defaults()

    def test_user_yaml_overrides_defaults(self, tmp_path):
        user_yaml = tmp_path / "config.yaml"
        user_yaml.write_text("transcribe:\n  model: small\n", encoding="utf-8")

        config = load_config(config_path=user_yaml)

        assert config["transcribe"]["model"] == "small"
        # Untouched sibling keys still come from the packaged defaults.
        assert config["transcribe"]["backend"] == "faster-whisper"

    def test_missing_config_path_raises_clear_error(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"

        with pytest.raises(FileNotFoundError, match="does_not_exist.yaml"):
            load_config(config_path=missing)

    def test_overrides_apply_on_top_of_user_yaml(self, tmp_path):
        user_yaml = tmp_path / "config.yaml"
        user_yaml.write_text(
            "transcribe:\n  model: small\n  language: en\n", encoding="utf-8"
        )

        config = load_config(
            config_path=user_yaml,
            overrides={"transcribe": {"model": "large-v3"}},
        )

        assert config["transcribe"]["model"] == "large-v3"  # overrides win
        assert config["transcribe"]["language"] == "en"  # user yaml preserved
        assert config["transcribe"]["backend"] == "faster-whisper"  # defaults preserved

    def test_overrides_without_user_yaml(self):
        config = load_config(overrides={"qa": {"enabled": False}})

        assert config["qa"]["enabled"] is False
        assert config["qa"]["samples"] == 5  # default preserved

    def test_accepts_str_path(self, tmp_path):
        user_yaml = tmp_path / "config.yaml"
        user_yaml.write_text("transcribe:\n  model: small\n", encoding="utf-8")

        config = load_config(config_path=str(user_yaml))

        assert config["transcribe"]["model"] == "small"

    def test_empty_user_yaml_is_fine(self, tmp_path):
        user_yaml = tmp_path / "config.yaml"
        user_yaml.write_text("", encoding="utf-8")

        assert load_config(config_path=user_yaml) == builtin_defaults()


class TestResolveStylePath:
    def test_none_resolves_to_packaged_default(self):
        path = resolve_style_path(None)

        assert path.exists()
        assert path.name == "default.yaml"

    def test_bare_name_default_resolves_to_packaged_file(self):
        path = resolve_style_path("default")

        assert path.exists()
        assert path.name == "default.yaml"

    def test_bare_name_en_resolves_to_packaged_file(self):
        path = resolve_style_path("en")

        assert path.exists()
        assert path.name == "en.yaml"

    def test_filesystem_path_is_used_when_it_exists(self, tmp_path):
        custom = tmp_path / "my_style.yaml"
        custom.write_text("text:\n  font_size: 12\n", encoding="utf-8")

        path = resolve_style_path(str(custom))

        assert path == custom

    def test_nonexistent_filesystem_path_raises(self, tmp_path):
        missing = tmp_path / "nope.yaml"

        with pytest.raises(FileNotFoundError):
            resolve_style_path(str(missing))
