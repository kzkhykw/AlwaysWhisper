"""System capture protocol, shared timeline, and failure handling. No live audio/API."""
import io
import sys
import threading

import numpy as np
import pytest

from alwayswhisper.live import system_audio as sa


def encoded(samples, timestamp=10.0):
    pcm = np.asarray(samples, dtype="<f4")
    return sa.HEADER.pack(b"AUD0", len(pcm), timestamp) + pcm.tobytes()


def test_protocol_handles_short_pipe_reads_and_clean_eof():
    class ShortReads(io.BytesIO):
        def read(self, size):
            return super().read(min(size, 3))
    stream = ShortReads(encoded([0.25, -0.5]) + encoded([1.0], 12))
    timestamp, samples = sa.read_packet(stream)
    assert timestamp == 10
    np.testing.assert_array_equal(samples, [0.25, -0.5])
    assert sa.read_packet(stream)[0] == 12
    assert sa.read_packet(stream) is None


@pytest.mark.parametrize("data", [
    b"AUD", sa.HEADER.pack(b"nope", 1, 1),
    sa.HEADER.pack(b"AUD0", 0, 1), sa.HEADER.pack(b"AUD0", 48001, 1),
    sa.HEADER.pack(b"AUD0", 1, float("nan")),
    sa.HEADER.pack(b"AUD0", 10, 1) + b"short",
    encoded([float("inf")]), encoded([float("nan")]),
])
def test_protocol_rejects_invalid_and_truncated_audio(data):
    with pytest.raises(ValueError):
        sa.read_packet(io.BytesIO(data))


def test_two_sources_share_one_timeline_with_silence_and_clip_protection():
    mixer = sa.TimedAudioMixer(("system", "mic"), sr=10, origin=100)
    # Out-of-order delivery must not concatenate or duplicate the duration.
    mixer.push("mic", np.full(5, 0.75), 100.3)
    mixer.push("system", np.full(5, 0.5), 100.1)
    np.testing.assert_array_equal(
        mixer.pull(10), [0, .5, .5, 1, 1, 1, .75, .75, 0, 0])
    assert mixer.cursor == 10
    np.testing.assert_array_equal(mixer.pull(10), np.zeros(10))


def test_ring_wrap_does_not_replay_old_audio_or_overwrite_unread_audio():
    mixer = sa.TimedAudioMixer(("system",), sr=10, origin=0, seconds=1)
    mixer.push("system", np.ones(20), 0)
    assert mixer.dropped_frames == 10  # bounded to one second, even for bad clocks
    np.testing.assert_array_equal(mixer.pull(8), np.ones(8))
    mixer.push("system", np.full(4, .25), 1.0)
    np.testing.assert_array_equal(mixer.pull(8), [1, 1, .25, .25, .25, .25, 0, 0])
    mixer.push("system", np.ones(10), 0)  # late packets cannot reappear next wrap
    np.testing.assert_array_equal(mixer.pull(10), np.zeros(10))
    assert mixer.dropped_frames == 20


def test_stall_skips_stale_audio_and_keeps_future_audio():
    mixer = sa.TimedAudioMixer(("system",), sr=10, origin=0, seconds=1)
    mixer.push("system", np.ones(10), 0)
    mixer.skip_to(8)
    np.testing.assert_array_equal(mixer.pull(4), [1, 1, 0, 0])
    mixer.skip_to(1000)
    np.testing.assert_array_equal(mixer.pull(10), np.zeros(10))


def _fake_helper(tmp_path, mode):
    # Real pipes and short-lived processes exercise READY/EOF/termination. The
    # fixture never reads audio hardware or imports the ASR/API implementation.
    script = tmp_path / "capture-fixture"
    script.write_text(f"#!{sys.executable}\n" + """
import struct, sys, time
mode = sys.argv[1]
if mode == 'denied':
    print('ERROR: recording permission denied', file=sys.stderr, flush=True)
    sys.stdin.buffer.read()
elif mode == 'eof':
    print('READY', file=sys.stderr, flush=True)
else:
    print('READY', file=sys.stderr, flush=True)
    if mode == 'stream':
        for _ in range(150):
            pcm = struct.pack('<960f', *([.25] * 960))
            sys.stdout.buffer.write(struct.pack('<4sId', b'AUD0', 960, time.monotonic() - .02) + pcm)
            sys.stdout.buffer.flush()
            time.sleep(.02)
    else:
        sys.stdin.buffer.read()
""")
    script.chmod(0o700)
    return sa.SystemAudioTap(helper=script, app_ids=[mode], latency=.04)


def test_system_stream_reaches_consumer_and_process_is_closed(tmp_path):
    tap = _fake_helper(tmp_path, "stream")
    chunks = []
    received = threading.Event()
    def feed(chunk):
        chunks.append(chunk)
        if np.any(chunk):
            received.set()
    tap.on_chunk = feed
    with tap:
        assert received.wait(2), "system audio never reached ASR input"
        tap.raise_if_failed()
    assert tap._process.poll() is not None
    assert not any(t.is_alive() for t in tap._threads)
    assert tap._mic is None
    assert np.max(np.concatenate(chunks)) == .25


def test_permission_failure_does_not_fall_back_to_microphone(tmp_path):
    tap = _fake_helper(tmp_path, "denied")
    called = []
    tap.microphone_factory = lambda: called.append(True)
    with pytest.raises(RuntimeError, match="permission denied"):
        with tap:
            pytest.fail("capture must fail closed")
    assert called == []
    assert tap._process.poll() is not None


def test_unexpected_helper_exit_is_reported_even_after_ready(tmp_path):
    tap = _fake_helper(tmp_path, "eof")
    with pytest.raises(RuntimeError, match="キャプチャプロセスが終了"):
        with tap:
            tap._threads[0].join(timeout=2)
            tap.raise_if_failed()
    assert tap._process.poll() is not None


def test_both_inputs_share_48k_and_microphone_is_closed_on_failure(tmp_path):
    tap = _fake_helper(tmp_path, "idle")
    events = []
    class Mic:
        on_chunk = None
        def __enter__(self):
            events.append("open")
        def __exit__(self, *args):
            events.append("close")
    mic = Mic()
    tap.microphone_factory = lambda: mic
    tap.mixer = sa.TimedAudioMixer(("system", "mic"))
    with pytest.raises(RuntimeError, match="test failure"):
        with tap:
            assert tap.sr == 48000
            assert mic.on_chunk == tap._microphone_chunk
            tap._fail("test failure")
            tap.raise_if_failed()
    assert events == ["open", "close"]
    assert tap._process.poll() is not None


@pytest.mark.parametrize("exception", [None, KeyboardInterrupt])
def test_shutdown_flushes_pending_timeline_before_consumer_finalizes(monkeypatch, exception):
    tap = sa.SystemAudioTap(helper="unused")
    tap.mixer = sa.TimedAudioMixer(("system",), origin=100)
    tap.mixer.push("system", np.full(4800, .25), 100)
    chunks = []
    tap.on_chunk = chunks.append
    monkeypatch.setattr(sa.time, "monotonic", lambda: 100.1)
    tap.__exit__(exception, None, None)
    np.testing.assert_array_equal(np.concatenate(chunks), np.full(4800, .25))
