"""Tests for alwayswhisper.__main__: the `python -m alwayswhisper` entrypoint.

Uses runpy instead of a real subprocess so this stays hermetic and fast,
matching the rest of the suite (no real ffmpeg/Whisper/MoviePy call).
"""

import runpy


def test_python_dash_m_invokes_cli_main(monkeypatch):
    called = {}
    monkeypatch.setattr("alwayswhisper.cli.main", lambda: called.setdefault("ran", True))

    runpy.run_module("alwayswhisper", run_name="__main__")

    assert called.get("ran") is True
