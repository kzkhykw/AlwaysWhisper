"""First-run transcript folder selection, independent of audio dependencies."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

from .terminal_ui import message


LEGACY_TRANSCRIPTS = Path.home() / "Library/Application Support/AlwaysWhisper/transcripts"
DEFAULT_TRANSCRIPTS = Path.home() / "Documents/AlwaysWhisper"
SETTINGS_PATH = Path.home() / ".config/alwayswhisper/live.json"
COMMAND = "alwayswhisper live"
SHOW_SYNC_HINT = False


def _load_settings():
    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("expected a settings object")
        return settings
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        print(f"Could not read settings / 設定を読み込めません: {exc}", file=sys.stderr)
        return {}


def _load_directory():
    value = _load_settings().get("transcript_dir")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid transcript_dir")
    return Path(value).expanduser().resolve()


def _prepare_directory(path):
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    # Fail before starting audio/model work, including for an existing read-only folder.
    with tempfile.TemporaryFile(dir=path):
        pass
    return path


def _save_settings(**changes):
    settings = _load_settings()
    settings.update(changes)
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=SETTINGS_PATH.parent, delete=False) as file:
            temp_path = Path(file.name)
            json.dump(settings, file, ensure_ascii=False)
            file.write("\n")
        os.replace(temp_path, SETTINGS_PATH)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _save_directory(path):
    _save_settings(transcript_dir=str(path))


def _environment_language():
    for key in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(key)
        if value:
            return "ja" if value.lower().startswith("ja") else "en"
    return "en"


def resolve_ui_language(explicit=None):
    """Remember an explicit/interactive choice; never prompt in unattended runs."""
    if explicit is not None and explicit not in ("ja", "en"):
        raise ValueError("UI language must be ja or en")
    saved = _load_settings().get("ui_language")
    if explicit is None and saved in ("ja", "en"):
        return saved
    language = explicit
    if language is None:
        suggested = _environment_language()
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return suggested  # no setting written; ask on the next interactive run
        default = "1" if suggested == "ja" else "2"
        print("\nAlwaysWhisper — 表示言語 / Display language")
        print("  1. 日本語\n  2. English")
        print("文字起こしの認識言語は変わりません。 / This does not change recognition language.")
        while language is None:
            try:
                answer = input(f"選択 / Choose [1/2, Enter = {default}]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                raise RuntimeError("設定を中止しました。録音は開始していません。 / Setup cancelled. No recording was started.") from None
            language = {"1": "ja", "ja": "ja", "2": "en", "en": "en", "": suggested}.get(answer)
            if language is None:
                print("1 または 2 を入力してください。 / Please enter 1 or 2.")
    try:
        _save_settings(ui_language=language)
    except OSError as exc:
        raise RuntimeError(f"表示言語を保存できません / Cannot save display language: {exc}") from None
    print(message(language, "language_saved"), flush=True)
    return language


def resolve_transcript_directory(explicit=None, *, choose=False, ui_language="en"):
    """Explicit flags are per-run overrides; only interactive choices are remembered."""
    try:
        if explicit is not None:
            return _prepare_directory(explicit)
        saved = _load_directory()
        if saved is not None and not choose:
            return _prepare_directory(saved)

        interactive = sys.stdin.isatty() and sys.stdout.isatty()
        if not interactive:
            if choose:
                raise RuntimeError(message(ui_language, "folder_terminal"))
            # Preserve unattended installations; do not consume stdin or mark setup done.
            return _prepare_directory(LEGACY_TRANSCRIPTS)

        has_legacy = LEGACY_TRANSCRIPTS.is_dir() and any(
            path.suffix in {".md", ".jsonl"} for path in LEGACY_TRANSCRIPTS.iterdir())
        suggested = saved or (LEGACY_TRANSCRIPTS if has_legacy else DEFAULT_TRANSCRIPTS)
        print(message(ui_language, "folder_title"))
        print(message(ui_language, "folder_intro"))
        if has_legacy and saved is None:
            print(message(ui_language, "existing"))
        print(message(ui_language, "suggested", path=suggested))
        print(message(ui_language, "folder_enter"))
        print(message(ui_language, "remember_folder"))
        if SHOW_SYNC_HINT:
            print(message(ui_language, "sync_hint"))
        print(message(ui_language, "change_folder", command=COMMAND), flush=True)
        while True:
            try:
                answer = input(message(ui_language, "folder_input")).strip()
            except (EOFError, KeyboardInterrupt):
                raise RuntimeError(message(ui_language, "folder_cancelled")) from None
            # Accept quoted paths copied from a shell without interpreting shell syntax.
            if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in "\"'":
                answer = answer[1:-1]
            try:
                directory = _prepare_directory(Path(answer) if answer else suggested)
            except (OSError, ValueError, RuntimeError) as exc:
                print(message(ui_language, "folder_invalid", error=exc))
                continue
            _save_directory(directory)
            print(message(ui_language, "folder_saved", path=directory), flush=True)
            return directory
    except (OSError, ValueError) as exc:
        raise RuntimeError(message(ui_language, "folder_error", error=exc)) from None
