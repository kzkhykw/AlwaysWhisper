"""Folder setup must preserve user choices and avoid unattended prompts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alwayswhisper.live import transcript_settings as settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SETTINGS_PATH", tmp_path / "config/live.json")
    monkeypatch.setattr(settings, "DEFAULT_TRANSCRIPTS", tmp_path / "Documents/AlwaysWhisper")
    monkeypatch.setattr(settings, "LEGACY_TRANSCRIPTS", tmp_path / "old transcripts")
    monkeypatch.setattr(settings, "sys", SimpleNamespace(
        stdin=SimpleNamespace(isatty=lambda: True),
        stdout=SimpleNamespace(isatty=lambda: True), stderr=__import__('sys').stderr))


def test_first_choice_is_remembered_without_prompting_again(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert settings.resolve_transcript_directory() == settings.DEFAULT_TRANSCRIPTS
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("unexpected second prompt"))
    assert settings.resolve_transcript_directory() == settings.DEFAULT_TRANSCRIPTS
    assert json.loads(settings.SETTINGS_PATH.read_text())["transcript_dir"] == str(settings.DEFAULT_TRANSCRIPTS)


def test_existing_transcripts_remain_in_place(monkeypatch):
    settings.LEGACY_TRANSCRIPTS.mkdir()
    existing = settings.LEGACY_TRANSCRIPTS / "2026-09-01.md"
    existing.write_text("existing notes")
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert settings.resolve_transcript_directory() == settings.LEGACY_TRANSCRIPTS
    assert existing.read_text() == "existing notes"


def test_reselect_and_one_run_override(monkeypatch, tmp_path):
    settings._save_directory(settings.DEFAULT_TRANSCRIPTS)
    chosen = tmp_path / "日本語 notes"
    monkeypatch.setattr("builtins.input", lambda _: f'"{chosen}"')
    assert settings.resolve_transcript_directory(choose=True) == chosen
    assert settings.resolve_transcript_directory(tmp_path / "one run") == tmp_path / "one run"
    assert settings.resolve_transcript_directory() == chosen


def test_noninteractive_uses_legacy_without_remembering(monkeypatch):
    monkeypatch.setattr(settings.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("must not consume stdin"))
    assert settings.resolve_transcript_directory() == settings.LEGACY_TRANSCRIPTS
    assert not settings.SETTINGS_PATH.exists()
    with pytest.raises(RuntimeError, match="requires a terminal"):
        settings.resolve_transcript_directory(choose=True)
    settings._save_directory(settings.DEFAULT_TRANSCRIPTS)
    assert settings.resolve_transcript_directory() == settings.DEFAULT_TRANSCRIPTS


def test_bad_folder_reprompts_before_saving(monkeypatch, tmp_path):
    occupied = tmp_path / "file"
    occupied.write_text("do not overwrite")
    answers = iter([str(occupied), ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert settings.resolve_transcript_directory() == settings.DEFAULT_TRANSCRIPTS
    assert occupied.read_text() == "do not overwrite"


@pytest.mark.parametrize("failure", [EOFError, KeyboardInterrupt])
def test_cancel_does_not_remember_choice(monkeypatch, failure):
    def cancel(_):
        raise failure
    monkeypatch.setattr("builtins.input", cancel)
    with pytest.raises(RuntimeError, match="cancelled"):
        settings.resolve_transcript_directory()
    assert not settings.SETTINGS_PATH.exists()
    assert not settings.DEFAULT_TRANSCRIPTS.exists()


def test_corrupt_settings_can_be_repaired(monkeypatch):
    settings.SETTINGS_PATH.parent.mkdir()
    settings.SETTINGS_PATH.write_text("{invalid")
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert settings.resolve_transcript_directory() == settings.DEFAULT_TRANSCRIPTS
    assert json.loads(settings.SETTINGS_PATH.read_text())["transcript_dir"]


def test_unwritable_saved_folder_does_not_silently_relocate(tmp_path):
    occupied = tmp_path / "file"
    occupied.write_text("original")
    settings._save_directory(occupied)
    with pytest.raises(RuntimeError, match="Cannot configure transcript folder"):
        settings.resolve_transcript_directory()
    assert not settings.DEFAULT_TRANSCRIPTS.exists()


@pytest.mark.parametrize("answer, expected", [("1", "ja"), ("2", "en")])
def test_language_choice_is_remembered_and_preserves_folder(monkeypatch, answer, expected):
    settings._save_directory(settings.DEFAULT_TRANSCRIPTS)
    monkeypatch.setattr("builtins.input", lambda _: answer)
    assert settings.resolve_ui_language() == expected
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("unexpected language prompt"))
    assert settings.resolve_ui_language() == expected
    assert settings.resolve_transcript_directory() == settings.DEFAULT_TRANSCRIPTS
    settings._save_directory(settings.LEGACY_TRANSCRIPTS)
    assert settings.resolve_ui_language() == expected


def test_language_flag_changes_and_remembers_without_prompt(monkeypatch):
    settings._save_settings(ui_language="ja", transcript_dir=str(settings.DEFAULT_TRANSCRIPTS))
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("explicit language must not prompt"))
    assert settings.resolve_ui_language("en") == "en"
    assert settings.resolve_ui_language() == "en"
    assert settings._load_directory() == settings.DEFAULT_TRANSCRIPTS


def test_language_menu_retries_and_localizes_following_folder_setup(monkeypatch, capsys):
    answers = iter(["invalid", "1", ""])
    prompts = []
    def reply(prompt):
        prompts.append(prompt)
        return next(answers)
    monkeypatch.setattr("builtins.input", reply)
    language = settings.resolve_ui_language()
    settings.resolve_transcript_directory(ui_language=language)
    assert "選択 / Choose" in prompts[0]
    assert "保存先" in prompts[-1]
    out = capsys.readouterr().out
    assert "Please enter 1 or 2" in out
    assert "保存先を記憶しました" in out
    assert "Suggested folder" not in out
    data = json.loads(settings.SETTINGS_PATH.read_text())
    assert data["ui_language"] == "ja" and data["transcript_dir"]


@pytest.mark.parametrize("locale, expected", [("ja_JP.UTF-8", "ja"), ("en_US.UTF-8", "en")])
def test_language_noninteractive_fallback_does_not_persist(monkeypatch, locale, expected):
    monkeypatch.setattr(settings.sys.stdin, "isatty", lambda: False)
    monkeypatch.setenv("LC_ALL", locale)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("must not prompt"))
    assert settings.resolve_ui_language() == expected
    assert not settings.SETTINGS_PATH.exists()


def test_language_enter_uses_suggested_locale(monkeypatch):
    monkeypatch.setenv("LC_ALL", "ja_JP.UTF-8")
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert settings.resolve_ui_language() == "ja"


@pytest.mark.parametrize("failure", [EOFError, KeyboardInterrupt])
def test_language_cancellation_preserves_settings(monkeypatch, failure):
    settings._save_directory(settings.DEFAULT_TRANSCRIPTS)
    before = settings.SETTINGS_PATH.read_bytes()
    def cancel(_):
        raise failure
    monkeypatch.setattr("builtins.input", cancel)
    with pytest.raises(RuntimeError, match="cancelled"):
        settings.resolve_ui_language()
    assert settings.SETTINGS_PATH.read_bytes() == before
