"""Public live command defaults, placement persistence and resource cleanup."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from alwayswhisper.cli import build_parser
from alwayswhisper.live import cli, caption_overlay as overlay, live_transcriber


def test_live_defaults_and_legacy_mode():
    args = build_parser().parse_args(["live"])
    assert args.captions is True
    assert args.captions_position is None  # first-run island, then saved selection
    args = build_parser().parse_args(["live", "--captions-position", "free"])
    assert args.captions_position == "free"
    assert not build_parser().parse_args(["live", "--no-captions"]).captions


def test_first_run_island_and_saved_free_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(overlay, "_require_appkit", lambda: None)
    monkeypatch.setattr(overlay, "_OverlayController", lambda queue, **kwargs: kwargs)
    path = tmp_path / "settings.json"
    assert overlay.attach_timer_based_overlay(None, settings_path=path)["position"] == "dynamic-island"
    overlay.save_geometry(path, overlay.OverlayGeometry(position="free"))
    assert overlay.attach_timer_based_overlay(None, settings_path=path)["position"] is None
    assert overlay.load_geometry(path).position == "free"
    assert overlay.attach_timer_based_overlay(None, settings_path=path, position="dynamic-island")["position"] == "dynamic-island"


@pytest.mark.parametrize("fail_stream", [False, True])
def test_live_wires_audio_meter_and_always_stops_worker(monkeypatch, tmp_path, fail_stream):
    events = []
    queue = SimpleNamespace(close=lambda: events.append("queue closed"))
    class Transcriber:
        def __init__(self, sr, model, language, directory, **kwargs):
            assert sr == 48000
            assert language is None
            assert kwargs["display_queue"] is queue
        def start(self):
            events.append("started")
        def feed(self, audio):
            np.testing.assert_array_equal(audio, [.25, .5])
            events.append("fed")
        def get_audio_level(self):
            return (123, .5)
        def stop(self):
            events.append("stopped")
            return {"jsonl_path": "session.jsonl"}
    class Stream:
        def __init__(self, **kwargs):
            assert kwargs["device"] == 3
            self.callback = kwargs["callback"]
        def __enter__(self):
            if fail_stream:
                raise RuntimeError("microphone unavailable")
            self.callback(np.array([[.25], [.5]]), 2, None, None)
        def __exit__(self, *args):
            events.append("stream closed")
    def run_overlay(display_queue, **kwargs):
        assert display_queue is queue
        assert kwargs["level_source"]() == (123, .5)
        assert kwargs["position"] == "free"
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(cli, "platform", SimpleNamespace(machine=lambda: "arm64"))
    monkeypatch.setattr(cli, "importlib", SimpleNamespace(util=SimpleNamespace(find_spec=lambda _: object())))
    monkeypatch.setattr(cli, "session_lock", nullcontext)
    monkeypatch.setattr(cli.multiprocessing, "get_context", lambda _: SimpleNamespace(Queue=lambda **kwargs: queue))
    monkeypatch.setattr(overlay, "_require_appkit", lambda: None)
    monkeypatch.setattr(overlay, "run_console_overlay", run_overlay)
    monkeypatch.setattr(live_transcriber, "LiveTranscriber", Transcriber)
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(
        query_devices=lambda *args: {"name": "Test microphone", "default_samplerate": 48000}, InputStream=Stream))
    args = build_parser().parse_args(["live", "--device", "3", "--language", "auto", "--captions-position", "free", "--transcript-dir", str(tmp_path)])
    if fail_stream:
        with pytest.raises(RuntimeError, match="microphone unavailable"):
            cli.run(args)
    else:
        cli.run(args)
        assert "fed" in events and "stream closed" in events
    assert events[-2:] == ["stopped", "queue closed"]
