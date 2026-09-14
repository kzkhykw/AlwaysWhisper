"""Unit tests for the pure logic of alwayswhisper.live.live_transcriber.py.

These exercise resampling, speech segmentation, and transcript file-writing
with no audio hardware, no mlx/whisper model, and no multiprocessing -- the
parts with real logic. The mlx_whisper / sounddevice / multiprocessing glue
(_worker, LiveTranscriber) is exercised by the manual end-to-end smoke test.
"""
import json
import os
import queue
import sys
import time
import types

import numpy as np

from alwayswhisper.live import live_transcriber as lt  # noqa: E402


# ------------------------------------------------------------- test signals ---
def _noise(n, amp=0.001, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n).astype(np.float32) * amp).astype(np.float32)


def _tone(n, sr, freq=300.0, amp=0.2):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _push_chunks(buf, signal, chunk_size):
    segs = []
    for i in range(0, len(signal), chunk_size):
        segs.extend(buf.push(signal[i:i + chunk_size]))
    return segs


# ------------------------------------------------------------------ resample ---
def test_resample_48k_to_16k_length():
    x = np.zeros(48000, dtype=np.float32)  # 1.0s
    y = lt.resample_to_16k(x, 48000)
    assert abs(len(y) - 16000) <= 1


def test_resample_preserves_dominant_frequency():
    sr = 48000
    n = int(sr * 0.5)  # exactly 220 cycles of 440Hz -> bin-aligned, no leakage
    t = np.arange(n) / sr
    x = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    y = lt.resample_to_16k(x, sr)
    mag = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / 16000)
    peak = freqs[int(np.argmax(mag))]
    assert abs(peak - 440.0) < 15.0


def test_resample_16k_passthrough_identity():
    x = (np.random.default_rng(0).random(2000).astype(np.float32) * 2 - 1)
    y = lt.resample_to_16k(x, 16000)
    assert np.array_equal(x, y)
    assert y is not x  # copy, not the same object


def test_resample_output_dtype_is_always_float32():
    x64 = np.zeros(1000, dtype=np.float64)
    assert lt.resample_to_16k(x64, 44100).dtype == np.float32
    assert lt.resample_to_16k(x64, 16000).dtype == np.float32


# ------------------------------------------------------------- SegmentBuffer ---
def test_segment_buffer_silence_emits_nothing():
    sr = 16000
    buf = lt.SegmentBuffer(sr)
    silence = _noise(int(2.0 * sr), amp=0.001, seed=1)
    segs = _push_chunks(buf, silence, 1024)
    assert segs == []
    assert buf.flush() is None


def test_segment_buffer_burst_between_silence():
    sr = 16000
    pre_roll = 0.3
    silence_sec = 0.8
    chunk_size = 1024
    buf = lt.SegmentBuffer(sr, silence_sec=silence_sec, pre_roll_sec=pre_roll)
    lead = _noise(int(1.0 * sr), seed=2)
    burst = _tone(int(1.0 * sr), sr, freq=300.0, amp=0.2)
    tail = _noise(int(2.0 * sr), seed=3)
    signal = np.concatenate([lead, burst, tail])

    segs = _push_chunks(buf, signal, chunk_size)
    assert len(segs) == 1
    seg = segs[0]

    burst_start_s = len(lead) / sr
    burst_end_s = (len(lead) + len(burst)) / sr
    epsilon = chunk_size / sr  # chunk-granular decisions can only resolve to ~1 chunk

    assert seg.start_s <= burst_start_s + epsilon
    assert seg.start_s >= burst_start_s - pre_roll - epsilon
    assert seg.end_s >= burst_end_s  # includes the trailing silence, not cut early
    assert seg.end_s <= burst_end_s + silence_sec + 2 * epsilon
    assert buf.flush() is None  # nothing left pending after the cut


def test_segment_buffer_short_click_discarded():
    sr = 16000
    buf = lt.SegmentBuffer(sr, silence_sec=0.3, min_speech_sec=0.3, pre_roll_sec=0.1)
    lead = _noise(int(0.5 * sr), seed=4)
    click = _tone(int(0.06 * sr), sr, freq=300.0, amp=0.2)  # 60ms, well above enter_thr
    tail = _noise(int(1.0 * sr), seed=5)
    signal = np.concatenate([lead, click, tail])

    segs = _push_chunks(buf, signal, 512)
    assert segs == []
    assert buf.flush() is None


def test_segment_buffer_force_cuts_long_speech_contiguously():
    sr = 16000
    max_sec = 2.0
    total_sec = 6.0
    chunk_size = 1000  # evenly divides max_samples (32000) -> exact, no quantization slop
    buf = lt.SegmentBuffer(sr, silence_sec=0.8, min_speech_sec=0.3, max_sec=max_sec, pre_roll_sec=0.2)
    n = int(total_sec * sr)
    speech = _tone(n, sr, freq=300.0, amp=0.2)

    segs = _push_chunks(buf, speech, chunk_size)
    assert len(segs) >= 3

    for a, b in zip(segs, segs[1:]):
        assert abs(b.start_s - a.end_s) < 1e-9  # contiguous: no gap, no overlap

    final = buf.flush()
    total_samples = sum(len(s.samples) for s in segs) + (len(final.samples) if final else 0)
    assert total_samples == n  # every pushed sample accounted for somewhere


def test_segment_buffer_flush_returns_pending_speech_still_active():
    sr = 16000
    buf = lt.SegmentBuffer(sr, silence_sec=0.8, min_speech_sec=0.3, pre_roll_sec=0.2)
    lead = _noise(int(0.3 * sr), seed=6)
    speech = _tone(int(1.0 * sr), sr, freq=300.0, amp=0.2)
    signal = np.concatenate([lead, speech])  # ends mid-speech: no cut should fire yet

    segs = _push_chunks(buf, signal, 512)
    assert segs == []

    final = buf.flush()
    assert final is not None
    assert final.end_s - final.start_s >= 1.0 - (512 / sr)
    assert buf.flush() is None  # idempotent


def test_segment_buffer_variable_chunk_sizes_agree():
    sr = 16000
    lead = _noise(int(0.5 * sr), seed=7)
    burst1 = _tone(int(1.0 * sr), sr, freq=300.0, amp=0.2)
    mid = _noise(int(1.2 * sr), seed=8)
    burst2 = _tone(int(0.6 * sr), sr, freq=500.0, amp=0.2)
    tail = _noise(int(1.2 * sr), seed=9)
    signal = np.concatenate([lead, burst1, mid, burst2, tail])

    results = {}
    for chunk_size in (256, 1024, 4096):
        buf = lt.SegmentBuffer(sr, silence_sec=0.8, min_speech_sec=0.3, pre_roll_sec=0.2)
        results[chunk_size] = _push_chunks(buf, signal, chunk_size)

    counts = {cs: len(segs) for cs, segs in results.items()}
    assert len(set(counts.values())) == 1, counts
    assert counts[256] == 2

    tol = 4096 / sr + 1e-9  # coarsest chunk size used, per-boundary tolerance (+ float slop)
    base = results[256]
    for cs in (1024, 4096):
        other = results[cs]
        for a, b in zip(base, other):
            assert abs(a.start_s - b.start_s) <= tol
            assert abs(a.end_s - b.end_s) <= tol



# -------------------------------------------------- SegmentBuffer speech gate ---
# push() classifies a chunk with max(adaptive_floor * K, absolute_floor). In a
# quiet room the adaptive floor collapses toward the room tone, so it is the
# ABSOLUTE floor -- SegmentBuffer's `gate` -- that a quiet voice actually has to
# clear. These pin the default and prove the knob is wired end to end.
def _rms_tone(n, sr, rms, freq=300.0):
    """A tone with exactly the requested RMS (a sine's RMS is amp / sqrt(2))."""
    return _tone(n, sr, freq=freq, amp=rms * float(np.sqrt(2.0)))


def test_segment_buffer_default_gate_admits_quiet_speech():
    # RMS 0.0085 (~-41 dBFS) is softly-spoken-but-clearly-audible. It clears
    # the 0.006 default gate; under the 0.015 that was hardcoded here until
    # 2026-09-03 this utterance never opened a segment at all, in any room.
    sr = 16000
    signal = np.concatenate([_noise(int(1.0 * sr), seed=40),
                             _rms_tone(int(1.0 * sr), sr, rms=0.0085),
                             _noise(int(1.2 * sr), seed=41)])
    segs = _push_chunks(lt.SegmentBuffer(sr), signal, 512)
    assert len(segs) == 1
    assert segs[0].end_s - segs[0].start_s >= 1.0


def test_segment_buffer_gate_is_configurable():
    sr = 16000
    signal = np.concatenate([_noise(int(1.0 * sr), seed=42),
                             _rms_tone(int(1.0 * sr), sr, rms=0.0085),
                             _noise(int(1.2 * sr), seed=43)])
    # the pre-2026-09-03 hardcoded floor: this exact utterance is inaudible to it
    loud_only = lt.SegmentBuffer(sr, gate=0.015)
    assert _push_chunks(loud_only, signal, 512) == []
    assert loud_only.flush() is None
    # and a lower gate still admits it
    assert len(_push_chunks(lt.SegmentBuffer(sr, gate=0.002), signal, 512)) == 1


def test_segment_buffer_stay_gate_holds_a_dip_that_could_not_start_speech():
    """Hysteresis: the stay floor is a fraction (STAY_GATE_RATIO) of the enter
    floor, so once speech is under way a quieter passage keeps it open. Without
    that, trailing-off sentence ends would cut a segment mid-thought."""
    sr = 16000
    lead = _noise(int(0.6 * sr), seed=44)
    onset = _rms_tone(int(0.3 * sr), sr, rms=0.035)   # clears the enter gate
    dip = _rms_tone(int(1.5 * sr), sr, rms=0.0045)    # between the stay and enter floors
    tail = _noise(int(1.2 * sr), seed=45)

    held = _push_chunks(lt.SegmentBuffer(sr), np.concatenate([lead, onset, dip, tail]), 512)
    assert len(held) == 1
    # the dip is carried inside the segment, not cut off where the onset ended
    assert held[0].end_s >= (len(lead) + len(onset) + len(dip)) / sr

    # ...and that same dip on its own never opens a segment: it is under the enter gate
    assert _push_chunks(lt.SegmentBuffer(sr), np.concatenate([lead, dip, tail]), 512) == []


def test_segment_buffer_default_min_speech_keeps_a_short_utterance():
    # 0.256s == exactly 8 chunks of 512 (chunk-granular classification, so an
    # aligned length keeps this exact): a one-word reply. Kept at the 0.2s
    # default; the 0.3s this used to be threw it away.
    sr = 16000
    signal = np.concatenate([_noise(10240, seed=46),          # 0.64s == 20 whole chunks
                             _tone(4096, sr, amp=0.2),        # 0.256s == 8 whole chunks
                             _noise(int(1.2 * sr), seed=47)])
    assert len(_push_chunks(lt.SegmentBuffer(sr), signal, 512)) == 1
    assert _push_chunks(lt.SegmentBuffer(sr, min_speech_sec=0.3), signal, 512) == []


# ------------------------------------------------------- SegmentBuffer.pos_s ---
def test_segment_buffer_pos_s_starts_at_zero():
    buf = lt.SegmentBuffer(16000)
    assert buf.pos_s == 0.0


def test_segment_buffer_pos_s_tracks_all_pushed_audio_regardless_of_classification():
    # pos_s is the running total of *every* sample ever pushed -- silence and
    # speech alike (see SegmentBuffer.push: self._total accumulates
    # unconditionally, before any speech/silence classification happens) --
    # not just samples that ended up inside an emitted segment.
    sr = 16000
    buf = lt.SegmentBuffer(sr)
    silence = _noise(int(1.5 * sr), amp=0.001, seed=30)
    _push_chunks(buf, silence, 512)
    assert abs(buf.pos_s - 1.5) < 1e-9


def test_segment_buffer_pos_s_matches_emitted_segment_end_s_same_basis():
    # pos_s and seg.start_s/end_s must share one coordinate system (absolute
    # samples pushed so far, divided by the native sr) so a caller can compare
    # a live pos_s directly against a previously emitted segment's end_s with
    # no unit conversion or offset -- see SegmentBuffer's docstring and
    # _worker, which does exactly that via writer.maybe_flush(buf.pos_s).
    sr = 16000
    buf = lt.SegmentBuffer(sr, silence_sec=0.8, min_speech_sec=0.3, pre_roll_sec=0.2)
    lead = _noise(int(0.5 * sr), seed=31)
    burst = _tone(int(1.0 * sr), sr, freq=300.0, amp=0.2)
    tail = _noise(int(0.8 * sr), seed=32)  # == silence_sec -> forces the cut on this exact push

    segs = []
    for chunk in (lead, burst, tail):
        segs = buf.push(chunk)
    assert len(segs) == 1  # the tail push is the one that emits

    # Right after the push() call that emitted it -- before any further audio
    # is pushed -- pos_s must equal exactly the emitted segment's own end_s.
    assert buf.pos_s == segs[0].end_s


# ------------------------------------------------- SegmentBuffer wall clock ---
# 2026-09 redesign: push(chunk, wall=...) additionally records the real
# wall-clock instant (time.time() basis) a chunk was captured at, so
# wall_of()/wall_now/AudioSegment.wall_start/wall_end can recover true wall
# time even across a gap where the *stream* clock (pos_s/_total, samples
# pushed / sr) stalls but wall time doesn't -- e.g. the Mac sleeping, the
# bug this whole feature exists to fix (see live_transcriber's module
# docstring and TranscriptWriter's below). Every test above this point never
# passes wall=, so it already proves the no-wall path is untouched by this
# feature's mere existence; these tests cover the feature itself.
def test_segment_buffer_wall_of_and_wall_now_none_before_any_wall_stamp():
    sr = 16000
    buf = lt.SegmentBuffer(sr)
    assert buf.wall_now is None
    assert buf.wall_of(0) is None
    # Pushing without wall= (the legacy call shape) must not manufacture a
    # stamp from nothing.
    buf.push(_noise(400, seed=60))
    assert buf.wall_now is None
    assert buf.wall_of(0) is None


def test_segment_buffer_wall_now_tracks_the_most_recent_stamp():
    sr = 16000
    buf = lt.SegmentBuffer(sr)
    t0 = 1_700_000_000.0
    buf.push(_noise(1600, amp=0.001, seed=61), wall=t0)          # 0.1s of audio
    assert buf.wall_now == t0
    buf.push(_noise(1600, amp=0.001, seed=62), wall=t0 + 0.1)    # +0.1s of audio, +0.1s of wall
    assert buf.wall_now == t0 + 0.1


def test_segment_buffer_wall_of_extrapolates_earlier_positions_from_latest_stamp():
    # wall_of(pos) for a pos *behind* the most recent stamp is derived by
    # walking back along the sample clock from that one reference point --
    # see wall_of's docstring -- not by remembering each chunk's own stamp
    # individually (there is only ever one reference point: the latest).
    sr = 16000
    buf = lt.SegmentBuffer(sr)
    t0 = 1_700_000_000.0
    buf.push(_noise(1600, amp=0.001, seed=63), wall=t0)  # samples [0, 1600), stamped "ends at t0"
    assert buf.wall_of(1600) == t0
    assert abs(buf.wall_of(0) - (t0 - 0.1)) < 1e-9      # start of that same chunk, 0.1s earlier
    assert abs(buf.wall_of(800) - (t0 - 0.05)) < 1e-9   # midpoint


def test_segment_buffer_push_without_wall_does_not_erase_a_prior_stamp():
    # A push() call that omits wall= must leave the existing reference point
    # alone (not reset it to None) -- see push()'s docstring. In production
    # every chunk LiveTranscriber.feed() enqueues always carries a wall
    # stamp, so this never actually happens there; it matters for a caller
    # that mixes stamped and unstamped pushes (as the mixed-shape _worker
    # test below does).
    sr = 16000
    buf = lt.SegmentBuffer(sr)
    t0 = 1_700_000_000.0
    buf.push(_noise(1600, amp=0.001, seed=64), wall=t0)
    buf.push(_noise(1600, amp=0.001, seed=65))  # no wall= this time
    # wall_now must still extrapolate from the first (only) stamp, now 0.1s
    # further along the sample clock than when it was recorded.
    assert abs(buf.wall_now - (t0 + 0.1)) < 1e-9


def test_segment_buffer_emitted_segment_wall_start_end_match_contiguous_stamps():
    # "Contiguous" = wall time advances in lockstep with the sample clock,
    # exactly as a real, uninterrupted capture stream does (each chunk's
    # wall stamp is capture time, which advances by that chunk's own
    # duration each time). Under that condition wall_start/wall_end must
    # equal the very first stamp minus however far start_s/end_s sit from
    # the origin -- i.e. wall_start == t0 + seg.start_s, wall_end == t0 +
    # seg.end_s, for whatever t0 is the wall time at absolute sample 0.
    sr = 16000
    buf = lt.SegmentBuffer(sr, silence_sec=0.8, min_speech_sec=0.3, pre_roll_sec=0.0)
    t0 = 1_700_000_000.0
    lead = _noise(int(0.5 * sr), seed=70)
    burst = _tone(int(1.0 * sr), sr, freq=300.0, amp=0.2)
    tail = _noise(int(0.8 * sr), seed=71)  # >= silence_sec -> forces the cut on this exact push

    segs = []
    pos = 0
    for chunk in (lead, burst, tail):
        pos += len(chunk)
        segs = buf.push(chunk, wall=t0 + pos / sr)  # wall advances exactly with the sample clock
    assert len(segs) == 1
    seg = segs[0]

    assert abs(seg.wall_start - (t0 + seg.start_s)) < 1e-6
    assert abs(seg.wall_end - (t0 + seg.end_s)) < 1e-6


def test_segment_buffer_wall_start_jumps_with_a_sleep_gap_stamp():
    # The actual failure mode this redesign fixes: a +19h wall-clock jump
    # between two chunks (the Mac sleeping) while the *sample* clock only
    # advances by one ordinary chunk's worth -- pos_s barely moves, but the
    # next emitted segment's wall_start must jump by essentially the same
    # ~19h, because wall_of() always extrapolates from the single most
    # recent stamp (see its docstring), which the post-sleep chunk replaces
    # outright.
    sr = 16000
    buf = lt.SegmentBuffer(sr, silence_sec=0.8, min_speech_sec=0.3, pre_roll_sec=0.0)
    t0 = 1_700_000_000.0
    SLEEP_S = 19 * 3600 + 24 * 60  # 19h24m, matching the measured real-world gap

    lead = _noise(int(0.5 * sr), seed=72)
    buf.push(lead, wall=t0)  # audio right before the Mac sleeps

    # Wake up SLEEP_S later; the very next chunk pushed is stamped that far
    # ahead even though it's chronologically adjacent on the sample clock.
    post_sleep_wall = t0 + SLEEP_S
    burst = _tone(int(1.0 * sr), sr, freq=300.0, amp=0.2)
    tail = _noise(int(0.8 * sr), seed=73)  # forces the cut

    pos = len(lead)
    segs = []
    for chunk in (burst, tail):
        pos += len(chunk)
        # Every post-sleep chunk is stamped as if captured post_sleep_wall
        # plus its own small offset from the wake moment (a real, contiguous
        # stream again after the gap).
        wall = post_sleep_wall + (pos - len(lead)) / sr
        segs = buf.push(chunk, wall=wall)
    assert len(segs) == 1
    seg = segs[0]

    # seg.start_s (session-relative, stream-clock basis) stayed tiny -- this
    # is the very first/only segment, so it starts near sample 0 regardless
    # -- while seg.wall_start - t0 (wall-clock basis) reflects essentially
    # the whole ~19h24m sleep gap: the same absolute-sample-0 origin, two
    # wildly different elapsed times depending which clock measures it.
    assert seg.start_s < 3.0
    assert abs(seg.wall_start - post_sleep_wall) < 1.0
    assert (seg.wall_start - t0) > SLEEP_S - 3.0


# ----------------------------------------------------------- TranscriptWriter ---
def _meta(**overrides):
    """Base meta dict for TranscriptWriter tests. date_str/session_hms are the
    explicit, deterministic (no wall-clock) inputs the writer needs for the
    daily-markdown header -- normally supplied by LiveTranscriber."""
    m = {"started_at": "2026-07-16T16:42:03+09:00", "samplerate": 48000,
         "model": "mlx-community/whisper-large-v3-turbo", "language": "ja",
         "date_str": "2026-07-16", "session_hms": "16:42:03"}
    m.update(overrides)
    return m


def test_transcript_writer_meta_first_line(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-16.md"
    meta = _meta()
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["type"] == "meta"
    assert rec["samplerate"] == 48000
    assert rec["model"] == meta["model"]
    assert rec["language"] == "ja"
    w.close()


def test_transcript_writer_fresh_md_gets_exactly_one_h1(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-16.md"
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), _meta())
    w.write(3.0, 5.0, "hello")
    w.close()
    lines = md_path.read_text(encoding="utf-8").splitlines()
    h1_lines = [l for l in lines if l.startswith("# ")]
    assert h1_lines == ["# 2026-07-16 文字起こし"]


def test_transcript_writer_session_header_format(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-16.md"
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), _meta())
    w.close()
    content = md_path.read_text(encoding="utf-8")
    assert "## 16:42:03 セッション (mlx-community/whisper-large-v3-turbo, lang=ja)" in content


def test_transcript_writer_session_header_lang_auto_when_language_none(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-16.md"
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), _meta(language=None))
    w.close()
    assert "lang=auto" in md_path.read_text(encoding="utf-8")


def test_transcript_writer_second_session_appends_header_not_h1(tmp_path):
    jsonl_path1 = tmp_path / "t1.jsonl"
    jsonl_path2 = tmp_path / "t2.jsonl"
    md_path = tmp_path / "2026-07-16.md"  # shared by both sessions, same as real usage

    w1 = lt.TranscriptWriter(str(jsonl_path1), str(md_path), _meta(session_hms="09:00:00"))
    w1.write(0.0, 1.0, "first session")
    w1.close()

    w2 = lt.TranscriptWriter(str(jsonl_path2), str(md_path), _meta(session_hms="14:30:00"))
    w2.write(0.0, 1.0, "second session")
    w2.close()

    lines = md_path.read_text(encoding="utf-8").splitlines()
    h1_lines = [l for l in lines if l.startswith("# ")]
    h2_lines = [l for l in lines if l.startswith("## ")]
    assert len(h1_lines) == 1
    assert len(h2_lines) == 2
    assert "09:00:00" in h2_lines[0]
    assert "14:30:00" in h2_lines[1]


def test_transcript_writer_segment_lines_rounding_and_md_bullet_format(tmp_path):
    # jsonl start/end stay session-relative seconds (source of truth for
    # audio alignment); the md bullet stamp is wall-clock (started_at +
    # start_s) -- _meta()'s started_at is 2026-07-16T16:42:03+09:00, so
    # start_s=1.234/3.567 land at 16:42:04 / 16:42:06, not 00:00:01/00:00:03.
    # The gap between them (3.567-3.567=0s) is far under paragraph_gap_sec,
    # so they join onto ONE md line stamped with the first segment's wall
    # time -- jsonl still records both as separate segment records regardless.
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-16.md"
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), _meta())
    w.write(1.234, 3.567, "こんにちは")
    w.write(3.567, 4.5, "テストです")
    w.close()

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # meta + 2 segments
    assert json.loads(lines[0])["type"] == "meta"

    seg1 = json.loads(lines[1])
    assert seg1["type"] == "segment"
    assert seg1["start"] == 1.23
    assert seg1["end"] == 3.57
    assert seg1["text"] == "こんにちは"

    seg2 = json.loads(lines[2])
    assert seg2["start"] == 3.57
    assert seg2["end"] == 4.5

    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert md_lines == ["- [16:42:04] こんにちは テストです"]


def test_transcript_writer_segment_line_uses_wall_clock_not_elapsed(tmp_path):
    # started_at 16:42:03 + start_s=3.0 -> wall-clock 16:42:06 (not the old
    # elapsed-time "00:00:03" -- see the date-rollover redesign).
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-16.md"
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), _meta())
    w.write(3.0, 5.0, "text")
    w.close()
    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert md_lines == ["- [16:42:06] text"]


def test_transcript_writer_flushes_immediately_while_open(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-16.md"
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), _meta())
    w.write(0.0, 1.0, "hello")
    # read back via an independent handle while the writer is still open
    assert '"hello"' in jsonl_path.read_text(encoding="utf-8")
    assert "hello" in md_path.read_text(encoding="utf-8")
    w.close()


# ----------------------------------------------------- paragraph gap-splitting ---
# 2026-07-27 redesign (superseding an earlier same-day wall-clock-bucket
# design that shipped, untracked, an hour before this one -- nothing else
# depended on it): the md side does not write one bulleted line per
# recognized segment (that read as choppy one-sentence-per-line fragments --
# see the class docstring). A paragraph is one burst of speech: a segment
# joins the currently-open md line (single half-width space, no new stamp)
# as long as the silence gap since the previous segment's end is <
# paragraph_gap_sec; a new line (with its own "- [HH:MM:SS]" stamp) starts
# only when that gap is >= paragraph_gap_sec, a day rolls over, or the
# writer closes. This is gap-based, not wall-clock-bucketed, so segments can
# straddle any wall-clock boundary (a 15-minute mark, an hour, whatever) and
# still join as long as the actual silence between them is short -- see
# test_transcript_writer_short_gap_joins_across_old_bucket_boundary below.
def test_transcript_writer_short_gap_segments_join_one_line(tmp_path):
    # A 2s silence gap is far under the default paragraph_gap_sec (60s) --
    # the two segments join onto ONE line, stamped with the first's wall time.
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-27.md"
    meta = _meta(started_at="2026-07-27T13:07:00+09:00", date_str="2026-07-27", session_hms="13:07:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(0.0, 1.0, "textA")  # wall 13:07:00, ends at 1.0s
    w.write(3.0, 4.0, "textB")  # starts at 3.0s -- 2s gap since textA ended
    w.close()

    content = md_path.read_text(encoding="utf-8")
    assert content.endswith("\n")  # close() terminates the open line
    md_lines = [l for l in content.splitlines() if l.startswith("- ")]
    assert md_lines == ["- [13:07:00] textA textB"]

    # jsonl keeps recording one segment record per write() regardless of how
    # the md side joins them onto a line.
    jsonl_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(jsonl_lines) == 3  # meta + 2 segments
    seg1, seg2 = json.loads(jsonl_lines[1]), json.loads(jsonl_lines[2])
    assert (seg1["text"], seg2["text"]) == ("textA", "textB")


def test_transcript_writer_short_gap_joins_across_old_bucket_boundary(tmp_path):
    # Key behavioral difference vs. the earlier wall-clock-bucket design:
    # 13:14:59 and 13:15:02 cross a 15-minute wall-clock mark (the old
    # paragraph_minutes bucket design would have split these onto two
    # lines), but the gap between them is only 2s -- far under
    # paragraph_gap_sec -- so gap-based splitting joins them onto ONE line.
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-27.md"
    meta = _meta(started_at="2026-07-27T13:14:59+09:00", date_str="2026-07-27", session_hms="13:14:59")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(0.0, 1.0, "textA")  # wall 13:14:59, ends at 1.0s
    w.write(3.0, 4.0, "textB")  # wall 13:15:02 -- crosses the :15 mark, gap only 2s
    w.close()

    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert md_lines == ["- [13:14:59] textA textB"]


def test_transcript_writer_gap_exactly_at_threshold_starts_new_line(tmp_path):
    # >= paragraph_gap_sec starts a new paragraph -- a gap of exactly 60.0s
    # counts as "at the threshold", not "still joined".
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-27.md"
    meta = _meta(started_at="2026-07-27T13:00:00+09:00", date_str="2026-07-27", session_hms="13:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(0.0, 10.0, "textA")    # ends at 10.0s
    w.write(70.0, 71.0, "textB")   # starts at 70.0s -- gap is exactly 60.0s -> new line
    w.close()

    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert md_lines == ["- [13:00:00] textA", "- [13:01:10] textB"]


def test_transcript_writer_gap_just_under_threshold_joins(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-27.md"
    meta = _meta(started_at="2026-07-27T13:00:00+09:00", date_str="2026-07-27", session_hms="13:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(0.0, 10.0, "textA")    # ends at 10.0s
    w.write(69.0, 70.0, "textB")   # starts at 69.0s -- gap is 59.0s -> joins
    w.close()

    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert md_lines == ["- [13:00:00] textA textB"]


def test_transcript_writer_custom_paragraph_gap_sec_honored(tmp_path):
    # paragraph_gap_sec is honored: a custom threshold distinct from the (now
    # 60s) default changes where lines split. 20.0s here is shorter than
    # both gaps below, so -- unlike under the default, where the first 55s
    # gap would still join (55s < 60s) -- every gap here splits onto its own
    # new line.
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-27.md"
    meta = _meta(started_at="2026-07-27T13:00:00+09:00", date_str="2026-07-27", session_hms="13:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta, paragraph_gap_sec=20.0)
    w.write(0.0, 10.0, "a")     # ends at 10.0s
    w.write(65.0, 66.0, "b")    # gap 55s since "a" ended -- over the custom 20s threshold -> new line
    w.write(200.0, 201.0, "c")  # gap 134s since "b" ended -- also over 20s -> new line
    w.close()

    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert md_lines == ["- [13:00:00] a", "- [13:01:05] b", "- [13:03:20] c"]


def test_transcript_writer_close_terminates_open_line(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-27.md"
    meta = _meta(started_at="2026-07-27T09:00:00+09:00", date_str="2026-07-27", session_hms="09:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(0.0, 1.0, "hello")
    # before close(): the paragraph line is open, not yet newline-terminated
    assert not md_path.read_text(encoding="utf-8").endswith("\n")
    w.close()
    assert md_path.read_text(encoding="utf-8").endswith("\n")


def test_transcript_writer_append_only_invariant_across_writes(tmp_path):
    # Once md content is newline-terminated it must never be rewritten --
    # only ever appended to (sync_transcripts_notion.py's line cursor
    # depends on this). After every write(), the file's content from just
    # before that write is a strict (byte-for-byte) prefix of its new
    # content -- true whether the write joins an open line, terminates one
    # to start a new paragraph, or is the first write of the session.
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-27.md"
    meta = _meta(started_at="2026-07-27T13:07:00+09:00", date_str="2026-07-27", session_hms="13:07:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)

    before = md_path.read_text(encoding="utf-8")
    # 0.0/30.0: gap 29s since "a" ends at 1.0 -> joins. 900.0: gap 869s since
    # "b" ends at 31.0 -- >= 60s threshold -> new paragraph. 930.0: gap 29s
    # since "c" ends at 901.0 -> joins again. Exercises both transitions.
    for start_s, text in [(0.0, "a"), (30.0, "b"), (900.0, "c"), (930.0, "d")]:
        w.write(start_s, start_s + 1.0, text)
        after = md_path.read_text(encoding="utf-8")
        assert after.startswith(before)
        before = after
    w.close()
    assert md_path.read_text(encoding="utf-8").startswith(before)


# ---------------------------------------------------- date-rollover / wall clock ---
# TranscriptWriter derives every wall-clock value (bullet stamp AND the
# rollover decision itself) from self._session_start + start_s, parsed once
# from meta["started_at"] in __init__ -- never datetime.now() -- so these
# tests stay exact and deterministic regardless of when they actually run.
def test_transcript_writer_same_day_wall_clock_stamp(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-18.md"
    meta = _meta(started_at="2026-07-18T10:00:00+09:00", date_str="2026-07-18", session_hms="10:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(65.0, 66.0, "hello")  # 10:00:00 + 65s = 10:01:05, still the 18th
    w.close()

    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert md_lines == ["- [10:01:05] hello"]
    # no extra file materialized for a same-day write
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2026-07-18.md", "t.jsonl"]


def test_transcript_writer_midnight_rollover_creates_next_day_file(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-18.md"
    meta = _meta(started_at="2026-07-18T23:59:00+09:00", date_str="2026-07-18", session_hms="23:59:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)

    w.write(30.0, 31.0, "late night")     # 23:59:30 -- still the 18th
    w.write(90.0, 91.0, "past midnight")  # 00:00:30 the 19th -> rolls over
    w.close()  # must not raise even though the writer moved on to a new file

    day18 = md_path.read_text(encoding="utf-8")
    assert "- [23:59:30] late night" in day18
    assert "past midnight" not in day18  # the rolled-over line does not stay behind

    day19_path = tmp_path / "2026-07-19.md"
    assert day19_path.exists()
    day19 = day19_path.read_text(encoding="utf-8")
    assert day19.startswith("# 2026-07-19 文字起こし\n")
    assert "セッション (mlx-community/whisper-large-v3-turbo, lang=ja) — 2026-07-18から継続" in day19
    assert "- [00:00:30] past midnight" in day19


def test_transcript_writer_rollover_terminates_multi_segment_paragraph(tmp_path):
    # A paragraph that has already joined two segments (still open, no
    # trailing "\n") must be correctly terminated by _rollover() before the
    # old day's file is closed -- not just a freshly-opened single-segment
    # line (see test_transcript_writer_midnight_rollover_creates_next_day_file
    # above for that simpler case). Day rollover forces a new paragraph
    # unconditionally: the gap to the segment that triggers it (58s) is
    # itself under paragraph_gap_sec (60s) and would otherwise have joined,
    # but crossing midnight always wins over the gap check.
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-18.md"
    meta = _meta(started_at="2026-07-18T23:59:00+09:00", date_str="2026-07-18", session_hms="23:59:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(0.0, 1.0, "first")     # wall 23:59:00, ends at 1.0s
    w.write(3.0, 4.0, "second")    # wall 23:59:03 -- gap 2s -> joins "first"
    w.write(62.0, 63.0, "third")   # wall 00:00:02 the 19th -- gap 58s (<60s) but rolls over anyway
    w.close()

    day18 = md_path.read_text(encoding="utf-8")
    assert day18.endswith("- [23:59:00] first second\n")  # terminated before rollover, joined intact
    assert "third" not in day18

    day19_path = tmp_path / "2026-07-19.md"
    day19 = day19_path.read_text(encoding="utf-8")
    assert day19.startswith("# 2026-07-19 文字起こし\n")
    assert "から継続" in day19
    assert day19.endswith("- [00:00:02] third\n")  # fresh paragraph line in the new file


def test_transcript_writer_multi_day_jump_goes_straight_to_correct_file(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-18.md"
    meta = _meta(started_at="2026-07-18T08:44:50+09:00", date_str="2026-07-18", session_hms="08:44:50")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(50 * 3600, 50 * 3600 + 1, "two days later")  # -> 2026-07-20 10:44:50
    w.close()

    names = sorted(p.name for p in tmp_path.iterdir())
    assert "2026-07-19.md" not in names  # no file for the skipped day in between
    assert "2026-07-20.md" in names

    day18 = md_path.read_text(encoding="utf-8")
    assert "two days later" not in day18

    day20 = (tmp_path / "2026-07-20.md").read_text(encoding="utf-8")
    assert day20.startswith("# 2026-07-20 文字起こし\n")
    assert "2026-07-18から継続" in day20
    assert "- [10:44:50] two days later" in day20


def test_transcript_writer_rollover_appends_to_existing_next_day_file(tmp_path):
    day19_path = tmp_path / "2026-07-19.md"
    day19_path.write_text(
        "# 2026-07-19 文字起こし\n\n## 07:00:00 セッション (other-model, lang=ja)\n\n"
        "- [00:00:05] earlier unrelated entry\n",
        encoding="utf-8",
    )

    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-18.md"
    meta = _meta(started_at="2026-07-18T23:59:00+09:00", date_str="2026-07-18", session_hms="23:59:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(90.0, 91.0, "rolled over")  # 00:00:30 the 19th
    w.close()

    content = day19_path.read_text(encoding="utf-8")
    assert content.count("# 2026-07-19 文字起こし") == 1  # H1 not duplicated
    assert "earlier unrelated entry" in content
    assert content.index("earlier unrelated entry") < content.index("rolled over")  # original content kept first
    assert "- [00:00:30] rolled over" in content


def test_transcript_writer_jsonl_stays_session_relative_across_rollover(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-18.md"
    meta = _meta(started_at="2026-07-18T23:59:00+09:00", date_str="2026-07-18", session_hms="23:59:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(30.0, 31.0, "before")
    w.write(90.0, 91.5, "after")  # rolls over, but the jsonl record must not know that
    w.close()

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    seg1, seg2 = json.loads(lines[1]), json.loads(lines[2])
    assert (seg1["start"], seg1["end"]) == (30.0, 31.0)
    assert (seg2["start"], seg2["end"]) == (90.0, 91.5)


def test_transcript_writer_second_rollover_still_credits_original_session_start(tmp_path):
    # A session spanning three calendar days: self._session_start never
    # changes after __init__, so BOTH rollovers' continuation headers must
    # cite the true original start (2026-07-18), not "continued from
    # yesterday's file".
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-18.md"
    meta = _meta(started_at="2026-07-18T23:00:00+09:00", date_str="2026-07-18", session_hms="23:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(1800, 1801, "day18")                       # 23:30:00 -- still the 18th
    w.write(5400, 5401, "day19")                       # +1h30m -> 2026-07-19 00:30:00
    w.write(5400 + 86400, 5400 + 86400 + 1, "day20")    # +24h more -> 2026-07-20 00:30:00
    w.close()

    day19 = (tmp_path / "2026-07-19.md").read_text(encoding="utf-8")
    day20 = (tmp_path / "2026-07-20.md").read_text(encoding="utf-8")
    expected_header = "23:00:00 セッション (mlx-community/whisper-large-v3-turbo, lang=ja) — 2026-07-18から継続"
    assert expected_header in day19
    assert expected_header in day20
    assert "- [00:30:00] day19" in day19
    assert "- [00:30:00] day20" in day20


# -------------------------------------------------------------- maybe_flush ---
# Closes a still-open paragraph line during *idle* silence -- i.e. without
# waiting for a next segment to arrive and discover the gap in write() (see
# TranscriptWriter's class docstring). _worker calls this once per queue
# item, with now_s = buf.pos_s (see SegmentBuffer.pos_s above); these tests
# call it directly so they stay independent of the audio/segmentation layer
# (the worker-level wiring itself is covered separately below, under
# "_worker (glue)").
def test_maybe_flush_closes_open_line_when_idle_gap_reaches_threshold(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-28.md"
    meta = _meta(started_at="2026-07-28T10:00:00+09:00", date_str="2026-07-28", session_hms="10:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)  # default paragraph_gap_sec (60.0)
    w.write(0.0, 1.0, "hello")
    before_md = md_path.read_text(encoding="utf-8")
    before_jsonl = jsonl_path.read_text(encoding="utf-8")
    assert not before_md.endswith("\n")  # line still open

    assert w.maybe_flush(61.0) is True  # 61.0 - 1.0 = 60.0s idle -- exactly the default threshold

    after_md = md_path.read_text(encoding="utf-8")
    assert after_md == before_md + "\n"  # only a bare terminator was appended
    assert jsonl_path.read_text(encoding="utf-8") == before_jsonl  # jsonl untouched
    w.close()


def test_maybe_flush_noop_when_idle_gap_under_threshold(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-28.md"
    meta = _meta(started_at="2026-07-28T10:00:00+09:00", date_str="2026-07-28", session_hms="10:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(0.0, 1.0, "hello")
    before = md_path.read_text(encoding="utf-8")

    assert w.maybe_flush(60.9) is False  # 59.9s idle -- just under the 60.0s default threshold

    assert md_path.read_text(encoding="utf-8") == before  # untouched, byte for byte
    w.close()


def test_maybe_flush_noop_when_no_line_open(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-28.md"
    meta = _meta(started_at="2026-07-28T10:00:00+09:00", date_str="2026-07-28", session_hms="10:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)  # nothing written yet -- no line open
    before = md_path.read_text(encoding="utf-8")

    assert w.maybe_flush(10_000.0) is False

    assert md_path.read_text(encoding="utf-8") == before
    w.close()


def test_rollover_after_maybe_flush_still_terminates_and_headers_correctly(tmp_path):
    # Interaction with day-rollover: maybe_flush already closed the line
    # (reset _last_end_s to None) during a long idle silence; the *next*
    # write() also happens to cross midnight. _rollover()'s own "was a line
    # open" check (self._last_end_s is not None) must not be confused by
    # maybe_flush having already put it in that same state -- no extra
    # blank line on the old (day18) file, and the new (day19) file still
    # gets its continuation header and the crossing segment's own line.
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-18.md"
    meta = _meta(started_at="2026-07-18T23:00:00+09:00", date_str="2026-07-18", session_hms="23:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(0.0, 1.0, "before the long silence")  # wall 23:00:00
    assert w.maybe_flush(3601.0) is True  # 3601 - 1 = 3600s idle -- closes the line
    before = md_path.read_text(encoding="utf-8")
    assert before.endswith("\n")

    w.write(3800.0, 3801.0, "past midnight")  # 23:00:00 + 3800s -> 2026-07-19 00:03:20
    w.close()

    day18 = md_path.read_text(encoding="utf-8")
    assert day18 == before  # maybe_flush's terminator untouched by the later rollover -- no extra "\n"

    day19 = (tmp_path / "2026-07-19.md").read_text(encoding="utf-8")
    assert day19.startswith("# 2026-07-19 文字起こし\n")
    assert "から継続" in day19
    assert day19.endswith("- [00:03:20] past midnight\n")


def test_write_after_maybe_flush_starts_fresh_paragraph_with_no_blank_line(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-28.md"
    meta = _meta(started_at="2026-07-28T10:00:00+09:00", date_str="2026-07-28", session_hms="10:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(0.0, 1.0, "first")
    assert w.maybe_flush(61.0) is True
    w.write(61.0, 62.0, "second")  # a later utterance, after the idle-flushed gap
    w.close()

    content = md_path.read_text(encoding="utf-8")
    assert "- [10:00:00] first\n- [10:01:01] second" in content  # exactly one "\n" between them
    assert "- [10:00:00] first\n\n" not in content  # no blank paragraph left by the flush/write handoff
    bullet_lines = [l for l in content.splitlines() if l.startswith("- ")]
    assert bullet_lines == ["- [10:00:00] first", "- [10:01:01] second"]


def test_close_after_maybe_flush_does_not_append_extra_newline(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-28.md"
    meta = _meta(started_at="2026-07-28T10:00:00+09:00", date_str="2026-07-28", session_hms="10:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(0.0, 1.0, "hello")
    assert w.maybe_flush(61.0) is True
    after_flush = md_path.read_text(encoding="utf-8")

    w.close()

    assert md_path.read_text(encoding="utf-8") == after_flush  # close() is a no-op: no line is open


# --------------------------------------------- wall-clock-primary redesign ---
# 2026-09 fix: write(..., wall_start=, wall_end=) and maybe_flush(...,
# wall_now=) let a caller supply the real wall-clock instant a segment was
# captured at (see SegmentBuffer.wall_of/AudioSegment.wall_start/wall_end
# above and _worker below). When given, it is the PRIMARY source for the
# bullet stamp, the rollover-day decision, and the paragraph-gap decision;
# omitting it (every test above this point) reproduces the original
# session_start + start_s behavior byte for byte -- already proven by every
# preceding test in this file continuing to pass unmodified. This section
# tests the wall_start/wall_end/wall_now path itself, including the actual
# real-world bug it fixes: a sleep gap barely moves start_s but wall_start
# jumps by however long the Mac was actually asleep, which must still (a)
# roll the .md over to the correct calendar day and (b) force a new
# paragraph, even when the stream-clock gap alone would not have.
#
# NOTE: all local-time assertions below assume the test machine's local
# timezone is JST (+09:00) -- the same assumption already baked into every
# "+09:00" started_at fixture and literal "HH:MM:SS" expectation elsewhere
# in this file (see _meta()). wall_start/wall_end are plain time.time()-
# basis Unix timestamps (no timezone of their own); TranscriptWriter renders
# them via datetime.fromtimestamp(...).astimezone() -- i.e. the *system's*
# local timezone -- so these tests build their wall_start/wall_end fixtures
# from "...+09:00" ISO strings' .timestamp() and assert against the same
# HH:MM:SS a JST system produces, exactly like every neighboring test.
def test_transcript_writer_write_without_wall_kwargs_matches_legacy_call_shape(tmp_path):
    # The explicit regression guard for the redesign's fallback path:
    # calling write() exactly as every pre-existing caller did -- positional
    # start_s/end_s/text only -- must reproduce the original session_start +
    # start_s stamp and jsonl shape byte for byte, with no "wall" key at all.
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-16.md"
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), _meta())
    w.write(3.0, 5.0, "legacy call")
    w.close()

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    seg = json.loads(lines[1])
    assert seg == {"type": "segment", "start": 3.0, "end": 5.0, "text": "legacy call"}
    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert md_lines == ["- [16:42:06] legacy call"]  # started_at 16:42:03 + 3.0s, exactly as before


def test_transcript_writer_write_wall_start_used_for_bullet_stamp(tmp_path):
    # started_at is 16:42:03 -- if the legacy session_start + start_s formula
    # were still driving the stamp, a start_s of 3.0 would land at 16:42:06
    # (see the legacy test above). Supplying wall_start pointing at a
    # completely different wall-clock moment must override that.
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-16.md"
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), _meta())
    wall_start = lt.datetime.datetime.fromisoformat("2026-07-16T20:15:30+09:00").timestamp()
    w.write(3.0, 5.0, "wall-clock stamped", wall_start=wall_start, wall_end=wall_start + 2.0)
    w.close()

    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert md_lines == ["- [20:15:30] wall-clock stamped"]


def test_transcript_writer_jsonl_wall_field_present_when_wall_start_given(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-16.md"
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), _meta())
    wall_start = lt.datetime.datetime.fromisoformat("2026-07-16T20:15:30+09:00").timestamp()
    w.write(3.0, 5.0, "hi", wall_start=wall_start, wall_end=wall_start + 2.0)
    w.close()

    seg = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[1])
    assert seg["start"] == 3.0 and seg["end"] == 5.0  # untouched, still session-relative
    assert seg["wall"] == "2026-07-16T20:15:30+09:00"  # ISO-8601, local time, seconds precision


def test_transcript_writer_wall_start_rollover_crosses_midnight_despite_small_start_s(tmp_path):
    # The actual bug this redesign fixes: session-relative start_s barely
    # moves across a sleep gap (a few seconds of real audio either side of
    # it), but the real wall clock jumped many hours and crossed midnight.
    # Without wall_start, write() would derive the day purely from
    # session_start + start_s and never roll over -- exactly the measured
    # 2026-08-27 bug (a ~19h24m gap stayed on the stale day's file, and
    # 2026-08-27.md was never created). With wall_start supplied, the
    # rollover must use it instead of start_s.
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-08-26.md"
    meta = _meta(started_at="2026-08-26T19:00:00+09:00", date_str="2026-08-26", session_hms="19:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)

    wall_start = lt.datetime.datetime.fromisoformat("2026-08-27T14:28:40+09:00").timestamp()
    w.write(2.0, 3.0, "after the sleep", wall_start=wall_start, wall_end=wall_start + 1.0)
    w.close()  # must not raise even though the writer moved on to a new file

    day26 = md_path.read_text(encoding="utf-8")
    assert "after the sleep" not in day26  # must not land on the stale-day file

    day27_path = tmp_path / "2026-08-27.md"
    assert day27_path.exists()
    day27 = day27_path.read_text(encoding="utf-8")
    assert day27.startswith("# 2026-08-27 文字起こし\n")
    assert "セッション (mlx-community/whisper-large-v3-turbo, lang=ja) — 2026-08-26から継続" in day27
    assert "- [14:28:40] after the sleep" in day27


def test_transcript_writer_paragraph_breaks_on_wall_gap_even_when_stream_gap_is_small(tmp_path):
    # The other half of the same bug: even without crossing midnight (a
    # 2h gap, same calendar day -- so this test isolates the gap-decision
    # from the rollover-decision tested separately above), a real sleep gap
    # must still force a new paragraph line, because the silence that
    # actually mattered to a listener was hours long, not the handful of
    # stream-clock seconds either side of the sleep.
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-08-26.md"
    meta = _meta(started_at="2026-08-26T19:00:00+09:00", date_str="2026-08-26", session_hms="19:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta, paragraph_gap_sec=60.0)

    wall_before = lt.datetime.datetime.fromisoformat("2026-08-26T19:05:00+09:00").timestamp()
    w.write(10.0, 11.0, "before the sleep", wall_start=wall_before, wall_end=wall_before + 1.0)

    # Stream gap is only 5s (10.0 -> 16.0), far under the 60s threshold, but
    # the wall gap is 2h -- must still start a new paragraph.
    wall_after = wall_before + 2 * 3600
    w.write(16.0, 17.0, "after the sleep", wall_start=wall_after, wall_end=wall_after + 1.0)
    w.close()

    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert len(md_lines) == 2  # two separate paragraph lines, not joined onto one
    assert md_lines[0] == "- [19:05:00] before the sleep"
    assert md_lines[1].endswith("after the sleep")
    assert "before the sleep" not in md_lines[1]


def test_transcript_writer_paragraph_joins_on_small_wall_gap_same_as_before(tmp_path):
    # Sanity complement to the sleep-gap test above: when the wall gap is
    # ALSO small (a normal, uninterrupted pause), two segments must still
    # join onto one paragraph line exactly as the no-wall path always did --
    # the wall-gap preference doesn't change ordinary same-session behavior.
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-08-26.md"
    meta = _meta(started_at="2026-08-26T19:00:00+09:00", date_str="2026-08-26", session_hms="19:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta, paragraph_gap_sec=60.0)

    wall1 = lt.datetime.datetime.fromisoformat("2026-08-26T19:05:00+09:00").timestamp()
    w.write(10.0, 11.0, "first", wall_start=wall1, wall_end=wall1 + 1.0)
    wall2 = wall1 + 3.0  # 3s gap, same on both clocks -- well under threshold
    w.write(14.0, 15.0, "second", wall_start=wall2, wall_end=wall2 + 1.0)
    w.close()

    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert md_lines == ["- [19:05:00] first second"]


def test_maybe_flush_wall_now_closes_on_wall_gap_when_stream_gap_is_small(tmp_path):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-08-26.md"
    meta = _meta(started_at="2026-08-26T19:00:00+09:00", date_str="2026-08-26", session_hms="19:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)  # default paragraph_gap_sec (60.0)
    wall_start = lt.datetime.datetime.fromisoformat("2026-08-26T19:05:00+09:00").timestamp()
    wall_end = wall_start + 1.0
    w.write(10.0, 11.0, "hello", wall_start=wall_start, wall_end=wall_end)

    # now_s (stream clock) has only advanced 5s -- far under the 60s
    # threshold -- but wall_now is ~2h after wall_end, well over it.
    assert w.maybe_flush(16.0, wall_now=wall_end + 2 * 3600) is True

    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert md_lines == ["- [19:05:00] hello"]  # already terminated
    w.close()


def test_maybe_flush_wall_now_none_falls_back_to_stream_gap(tmp_path):
    # wall_now defaults to None -- calling maybe_flush exactly as every
    # pre-existing caller/test did must behave byte for byte as before.
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-08-26.md"
    meta = _meta(started_at="2026-08-26T19:00:00+09:00", date_str="2026-08-26", session_hms="19:00:00")
    w = lt.TranscriptWriter(str(jsonl_path), str(md_path), meta)
    w.write(0.0, 1.0, "hello")  # no wall kwargs -> self._last_wall_end stays None too
    assert w.maybe_flush(60.9) is False   # 59.9s idle -- under the 60.0s default threshold
    assert w.maybe_flush(61.0) is True    # 60.0s idle -- reaches it
    w.close()


# ----------------------------------------------------------- _worker (glue) ---
# _worker itself is otherwise exercised only by the manual end-to-end smoke
# test (see this module's docstring) -- these two are deliberately narrow
# exceptions: they drive the *real* _worker function end-to-end (real
# SegmentBuffer, real TranscriptWriter, a real queue.Queue -- _worker only
# ever calls .get() on it, so a plain synchronous Queue works fine in-process,
# no multiprocessing needed) to prove the maybe_flush wiring specifically,
# since that's the one new behavior that lives in _worker's loop itself
# rather than in a directly-testable pure-logic function. mlx_whisper is
# faked via sys.modules (it isn't installed in the test env, and
# clean_transcription() -- which is what actually matters about its result --
# already has its own direct unit tests above).
class _FakeMlxWhisper:
    """Stand-in for the real mlx_whisper module: transcribe() ignores every
    argument and returns a fixed, always-kept result (see clean_transcription
    -- no_speech_prob/avg_logprob/compression_ratio all comfortably pass)."""

    def __init__(self, text="hello"):
        self._result = {"text": text, "segments": [
            {"text": text, "no_speech_prob": 0.05, "avg_logprob": -0.2, "compression_ratio": 1.1}]}

    def transcribe(self, *args, **kwargs):
        return self._result


# qwen3_asr_mlx is faked the same way, via sys.modules (it isn't installed in
# the test env either -- see requirements-transcribe.txt). The fake only needs
# to match the confirmed qwen3-asr-mlx==0.2.0 shape that _make_transcribe_fn
# actually uses: `from qwen3_asr_mlx import Qwen3ASR`, then
# `Qwen3ASR.from_pretrained(model_id_or_path)` -> an object whose
# `.transcribe(audio, language=None, ..., context=None)` returns a
# TranscriptionResult(text, language, duration) dataclass -- no segments/
# no_speech_prob/avg_logprob/compression_ratio, unlike mlx_whisper's result.
class _FakeQwenResult:
    def __init__(self, text, language="ja", duration=1.0):
        self.text = text
        self.language = language
        self.duration = duration


class _FakeQwenModel:
    """Stand-in for the object Qwen3ASR.from_pretrained() returns. Records
    every transcribe() call (audio/language/context) so tests can assert on
    exactly what _make_transcribe_fn passed through."""

    def __init__(self, model_id, text):
        self.model_id = model_id
        self._text = text
        self.calls = []

    def transcribe(self, audio, language=None, context=None, **kwargs):
        self.calls.append({"audio": audio, "language": language, "context": context})
        return _FakeQwenResult(self._text)


def _fake_qwen_module(text="qwen hello", models=None):
    """Stand-in `qwen3_asr_mlx` module: exposes Qwen3ASR.from_pretrained()
    exactly the way _make_transcribe_fn imports it. `models`, if a list,
    collects every fake model instance from_pretrained() creates (usually
    just one) so a test can inspect its .calls afterward."""
    class Qwen3ASR:
        @staticmethod
        def from_pretrained(model_id_or_path):
            m = _FakeQwenModel(model_id_or_path, text)
            if models is not None:
                models.append(m)
            return m

    return types.SimpleNamespace(Qwen3ASR=Qwen3ASR)


def _worker_cfg(tmp_path, sr=16000, paragraph_gap_sec=60.0):
    jsonl_path = tmp_path / "t.jsonl"
    md_path = tmp_path / "2026-07-28.md"
    meta = _meta(started_at="2026-07-28T09:00:00+09:00", date_str="2026-07-28", session_hms="09:00:00",
                 model="fake-model")
    cfg = {
        "sr": sr, "model": "fake-model", "language": "ja",
        "jsonl_path": str(jsonl_path), "md_path": str(md_path), "meta": meta,
        "silence_sec": 0.8, "min_speech_sec": 0.3, "max_sec": 25.0, "pre_roll_sec": 0.3,
        "paragraph_gap_sec": paragraph_gap_sec,
    }
    return cfg, jsonl_path, md_path


def test_worker_idle_silence_flushes_open_paragraph_mid_loop(tmp_path, monkeypatch):
    """Speech opens a paragraph line; then -- with no second utterance ever
    arriving -- enough trailing silence accumulates (measured via buf.pos_s,
    fed to writer.maybe_flush() after every queue item; see _worker) to cross
    paragraph_gap_sec. A small paragraph_gap_sec (2.0s) keeps the synthetic
    audio short; the default value itself is covered separately
    (test_maybe_flush_* above and test_worker_falls_back_to_60s_* below). A
    maybe_flush spy snapshots the md file the moment it returns True --
    *inside* the loop, before _worker ever reaches the None sentinel /
    close() -- which is what proves maybe_flush (not the final close())
    performed the termination.
    """
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper())
    cfg, jsonl_path, md_path = _worker_cfg(tmp_path, paragraph_gap_sec=2.0)
    sr = cfg["sr"]

    events = []  # ("flush", now_s, result) | ("snapshot", md_text)
    real_maybe_flush = lt.TranscriptWriter.maybe_flush

    def spy_maybe_flush(self, now_s):
        result = real_maybe_flush(self, now_s)
        events.append(("flush", now_s, result))
        if result:
            events.append(("snapshot", md_path.read_text(encoding="utf-8")))
        return result

    monkeypatch.setattr(lt.TranscriptWriter, "maybe_flush", spy_maybe_flush)

    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))    # speech
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=40))        # >= silence_sec -> forces the emit
    for _ in range(4):                                      # idle silence, fed incrementally
        q.put(_noise(int(1.0 * sr), amp=0.001, seed=41))
    q.put(None)

    lt._worker(q, cfg)

    flush_results = [r for (_, _, r) in (e for e in events if e[0] == "flush")]
    assert flush_results == [False, False, False, True, False, False]

    snapshots = [text for (_, text) in (e for e in events if e[0] == "snapshot")]
    assert len(snapshots) == 1
    assert snapshots[0].endswith("- [09:00:00] hello\n")  # already terminated before shutdown/close()

    final_content = md_path.read_text(encoding="utf-8")
    assert final_content == snapshots[0]  # close() added nothing further


def test_worker_exits_quietly_on_ctrl_c_and_still_closes_the_transcript(tmp_path, monkeypatch):
    """Ctrl-C in a terminal is delivered to the whole foreground process group, so
    this child gets SIGINT too -- normally while blocked in q.get(). Letting the
    KeyboardInterrupt escape made multiprocessing dump a 20-line traceback over the
    parent's shutdown messages on every single Ctrl-C (observed end-to-end on
    live_avatar.py --transcribe-only, 2026-08-12). Shutdown is the parent's job
    (LiveTranscriber.stop()), so this process has nothing to add: it must return
    quietly -- while its finally block still terminates the open paragraph line,
    since that transcript is the thing the user keeps.
    """
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper())
    cfg, _jsonl_path, md_path = _worker_cfg(tmp_path)
    sr = cfg["sr"]

    class _CtrlCQueue:
        """Hands over real audio, then raises KeyboardInterrupt from get() the
        way a SIGINT'd multiprocessing queue read does."""

        def __init__(self, items):
            self._items = list(items)

        def get(self):
            if self._items:
                return self._items.pop(0)
            raise KeyboardInterrupt

    q = _CtrlCQueue([_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2),   # speech
                     _noise(int(1.0 * sr), amp=0.001, seed=43)])      # silence -> forces the emit

    lt._worker(q, cfg)   # returns; does not raise

    assert md_path.read_text(encoding="utf-8").endswith("- [09:00:00] hello\n")


def test_worker_word_display_streams_timestamp_units_on_one_line(tmp_path, monkeypatch, capsys):
    """Word display must affect the terminal only; persisted records stay
    segment-level for the daily-transcript/Notion contract."""
    result = {
        "text": "hello world",
        "segments": [{
            "text": "hello world", "no_speech_prob": 0.05,
            "avg_logprob": -0.2, "compression_ratio": 1.1,
            "words": [{"word": "hello", "start": 0.0, "end": 0.3},
                      {"word": " world", "start": 0.3, "end": 0.7}],
        }],
    }
    monkeypatch.setattr(lt, "_make_transcribe_fn", lambda cfg: (lambda audio: result, "whisper"))
    cfg, jsonl_path, _ = _worker_cfg(tmp_path)
    cfg["display_mode"] = "word"
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=42))
    q.put(None)

    lt._worker(q, cfg)

    out = capsys.readouterr().out
    assert "hello world\n" in out
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["text"] == "hello world"


def test_worker_falls_back_to_60s_paragraph_gap_when_cfg_key_missing(tmp_path, monkeypatch):
    """_worker builds its TranscriptWriter with
    cfg.get("paragraph_gap_sec", 60.0); this confirms that literal fallback
    constant agrees with the new default (60.0, not the old 180.0) by spying
    directly on the value _worker passes to TranscriptWriter.__init__ -- no
    audio/timing simulation needed since the fallback is a single constant,
    independent of any actual segmentation or flush timing.
    """
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper())
    captured = {}
    real_init = lt.TranscriptWriter.__init__

    def spy_init(self, jsonl_path, md_path, meta, paragraph_gap_sec=60.0):
        captured["paragraph_gap_sec"] = paragraph_gap_sec
        real_init(self, jsonl_path, md_path, meta, paragraph_gap_sec=paragraph_gap_sec)

    monkeypatch.setattr(lt.TranscriptWriter, "__init__", spy_init)

    cfg, _, _ = _worker_cfg(tmp_path)
    del cfg["paragraph_gap_sec"]  # omit the key -> exercises _worker's own fallback

    q = queue.Queue()
    q.put(None)  # immediate shutdown -- only the constructor call matters here
    lt._worker(q, cfg)

    assert captured["paragraph_gap_sec"] == 60.0


def test_worker_loading_log_includes_engine_name(tmp_path, monkeypatch, capsys):
    """The startup log line must name which engine is being loaded (whisper
    here) -- e.g. 'loading model <id> (engine: whisper) ...' -- so a user
    watching the console can tell at a glance which ASR backend is running,
    especially useful after a qwen->whisper fallback (see _make_transcribe_fn)."""
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper())
    cfg, _, _ = _worker_cfg(tmp_path)
    cfg["engine"] = "whisper"
    q = queue.Queue()
    q.put(None)
    lt._worker(q, cfg)
    out = capsys.readouterr().out
    assert "loading model fake-model (engine: whisper) ..." in out


# ---------------------------------------- feed()/_worker wall-stamped queue ---
# 2026-09: LiveTranscriber.feed() now puts (chunk, wall) tuples on the queue
# (wall = time.time() at capture) instead of a bare chunk array, so _worker
# can recover real wall time across a sleep gap (see the module docstring
# and the wall-clock-primary TranscriptWriter tests above). _worker accepts
# either shape -- see its own docstring -- so every existing test above this
# point, which pushes bare arrays directly, keeps working unmodified; these
# tests cover the new (chunk, wall) shape itself plus the None sentinel and
# mixed-shape tolerance.
def test_live_transcriber_feed_enqueues_a_chunk_wall_tuple(tmp_path):
    # feed() is only ever exercised here via a plain queue.Queue() stood in
    # for start()'s real multiprocessing.Queue -- start() itself is never
    # called (it would spawn a child process that loads a real Whisper
    # model, forbidden in this test environment).
    transcriber = lt.LiveTranscriber(16000, "fake-model", "ja", str(tmp_path))
    transcriber._queue = queue.Queue()
    chunk = _noise(400, seed=80)

    before = time.time()
    transcriber.feed(chunk)
    after = time.time()

    item = transcriber._queue.get_nowait()
    assert isinstance(item, tuple) and len(item) == 2
    fed_chunk, fed_wall = item
    assert np.array_equal(fed_chunk, np.asarray(chunk, dtype=np.float32))
    assert before <= fed_wall <= after  # time.time() taken during this feed() call


def test_live_transcriber_feed_copies_the_chunk_not_a_view(tmp_path):
    # Unchanged from before this feature (feed() has always .copy()'d), but
    # worth re-asserting now that the array travels inside a tuple: mutating
    # the caller's buffer after feed() must not affect the queued copy.
    transcriber = lt.LiveTranscriber(16000, "fake-model", "ja", str(tmp_path))
    transcriber._queue = queue.Queue()
    chunk = np.zeros(10, dtype=np.float32)
    transcriber.feed(chunk)
    chunk[:] = 1.0

    fed_chunk, _ = transcriber._queue.get_nowait()
    assert np.all(fed_chunk == 0.0)


def test_worker_unpacks_a_tuple_queue_item_and_stamps_the_bullet_with_wall_time(tmp_path, monkeypatch):
    # End-to-end through _worker's real loop (queue -> buf.push(chunk,
    # wall=) -> AudioSegment.wall_start -> writer.write(..., wall_start=)):
    # a (chunk, wall) item, exactly the shape LiveTranscriber.feed() now
    # produces, must land on a wall-clock bullet stamp -- not the legacy
    # session_start + start_s one (started_at is 09:00:00 -- see
    # _worker_cfg/_meta -- which would stamp a segment this short as
    # "09:00:0x", unmistakably different from the wall-clock stamp below).
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper())
    cfg, jsonl_path, md_path = _worker_cfg(tmp_path)
    cfg["pre_roll_sec"] = 0.0  # keeps buf_start exactly 0 -> simple, exact arithmetic below
    sr = cfg["sr"]

    # t0 is the wall time at absolute sample 0; each chunk's wall stamp
    # advances in lockstep with the sample clock (an ordinary, uninterrupted
    # capture stream), so the emitted segment's wall_start lands exactly on
    # t0 -- see test_segment_buffer_emitted_segment_wall_start_end_match_
    # contiguous_stamps above, which proves this same arithmetic directly on
    # SegmentBuffer. NOTE: assumes local tz JST, matching _meta()'s "+09:00"
    # fixtures elsewhere in this file.
    t0 = lt.datetime.datetime.fromisoformat("2026-07-28T23:41:00+09:00").timestamp()
    speech = _tone(int(1.0 * sr), sr, freq=300.0, amp=0.2)      # 1.0s
    silence = _noise(int(1.0 * sr), amp=0.001, seed=90)          # 1.0s, >= silence_sec -> forces the emit

    q = queue.Queue()
    q.put((speech, t0 + 1.0))
    q.put((silence, t0 + 2.0))
    q.put(None)

    lt._worker(q, cfg)

    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert md_lines == ["- [23:41:00] hello"]

    seg = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[1])
    assert seg["wall"] == "2026-07-28T23:41:00+09:00"


def test_worker_accepts_a_mix_of_bare_array_and_tuple_queue_items(tmp_path, monkeypatch):
    """_worker's per-item unpacking (see its own docstring) tolerates either
    shape on the very same queue -- a bare chunk (every existing test/legacy
    caller) or a (chunk, wall) tuple (LiveTranscriber.feed(), now) -- so it
    never assumes a queue is "all one shape or the other"; only the None
    shutdown sentinel is special-cased ahead of this."""
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper())
    cfg, jsonl_path, md_path = _worker_cfg(tmp_path)
    sr = cfg["sr"]

    speech = _tone(int(1.0 * sr), sr, freq=300.0, amp=0.2)
    silence = _noise(int(1.0 * sr), amp=0.001, seed=91)
    # Same calendar day as _worker_cfg/_meta's fixed started_at/date_str
    # (2026-07-28) -- using the real time.time() here would (correctly!)
    # roll the .md over to today's real date instead, since that's a
    # different calendar day than the fixture; this test is about the mixed
    # tuple/bare-array queue-item shape, not about rollover, so it keeps the
    # wall stamp deterministic and same-day like every other _worker test.
    wall_ts = lt.datetime.datetime.fromisoformat("2026-07-28T09:00:05+09:00").timestamp()

    q = queue.Queue()
    q.put(speech)                # bare array (legacy/no-wall shape)
    q.put((silence, wall_ts))    # (chunk, wall) tuple (new shape)
    q.put(None)

    lt._worker(q, cfg)  # must not raise

    md_lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    assert len(md_lines) == 1
    assert md_lines[0].endswith("hello")


# --------------------------------------------------- _make_transcribe_fn ---
def test_make_transcribe_fn_defaults_to_whisper_when_engine_key_missing(monkeypatch):
    """A cfg dict with no "engine" key at all (e.g. an older hand-built cfg,
    or the existing _worker_cfg() helper above) must behave exactly like
    engine="whisper" -- this is the "existing behavior stays unchanged"
    guarantee for --transcribe-engine's default."""
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper(text="plain whisper"))
    cfg = {"model": "fake-model", "language": "ja", "prompt": None}
    fn, engine = lt._make_transcribe_fn(cfg)
    assert engine == "whisper"
    assert fn(np.zeros(10, dtype=np.float32))["text"] == "plain whisper"


def test_make_transcribe_fn_whisper_engine_explicit_unchanged_call_shape(monkeypatch):
    """engine="whisper" reproduces the exact previous inline mlx_whisper.transcribe()
    call: path_or_hf_repo/language/initial_prompt/condition_on_previous_text/
    word_timestamps, and the raw (segments-bearing) result dict is returned
    as-is so clean_transcription's per-segment gates are unaffected."""
    captured = {}

    class _RecordingFakeMlxWhisper(_FakeMlxWhisper):
        def transcribe(self, audio, **kwargs):
            captured["audio"] = audio
            captured.update(kwargs)
            return super().transcribe(audio, **kwargs)

    monkeypatch.setitem(sys.modules, "mlx_whisper", _RecordingFakeMlxWhisper(text="hi"))
    cfg = {"engine": "whisper", "model": "fake-model", "language": "ja", "prompt": "glossary terms"}
    fn, engine = lt._make_transcribe_fn(cfg)
    assert engine == "whisper"
    audio = np.zeros(10, dtype=np.float32)
    result = fn(audio)
    assert result["text"] == "hi"
    assert "segments" in result  # unchanged: whisper's raw result, not normalized
    assert captured["path_or_hf_repo"] == "fake-model"
    assert captured["language"] == "ja"
    assert captured["initial_prompt"] == "glossary terms"
    assert captured["condition_on_previous_text"] is False
    assert captured["word_timestamps"] is False


def test_make_transcribe_fn_whisper_requests_word_timestamps_for_word_display(monkeypatch):
    captured = {}

    class _RecordingFakeMlxWhisper(_FakeMlxWhisper):
        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            return super().transcribe(audio, **kwargs)

    monkeypatch.setitem(sys.modules, "mlx_whisper", _RecordingFakeMlxWhisper(text="hi"))
    cfg = {"engine": "whisper", "model": "fake-model", "language": "ja", "prompt": None,
           "display_mode": "word"}
    fn, _ = lt._make_transcribe_fn(cfg)
    fn(np.zeros(10, dtype=np.float32))
    assert captured["word_timestamps"] is True


def test_extract_words_reads_whisper_segment_word_records():
    result = {"segments": [{"words": [{"word": " こんにちは"}, {"word": "世界 "}]}, {"words": []}]}
    assert lt.extract_words(result) == ["こんにちは", "世界"]


def test_extract_timed_words_and_sentence_end_detection():
    result = {"segments": [{"words": [
        {"word": "こんにちは", "start": 0.1, "end": 0.4},
        {"word": "。", "start": 0.4, "end": 0.5},
    ]}]}
    assert lt.extract_timed_words(result, offset_s=10.0) == [
        ("こんにちは", 10.1, 10.4), ("。", 10.4, 10.5),
    ]
    assert lt.ends_display_sentence("。")
    assert not lt.ends_display_sentence("こんにちは")


def test_make_transcribe_fn_qwen_success_passes_float32_audio_language_and_context(monkeypatch):
    models = []
    monkeypatch.setitem(sys.modules, "qwen3_asr_mlx",
                        _fake_qwen_module(text="qwen result", models=models))
    cfg = {"engine": "qwen", "model": "mlx-community/Qwen3-ASR-1.7B-bf16", "language": "ja",
           "prompt": "Notion Claude Fable"}
    fn, engine = lt._make_transcribe_fn(cfg)
    assert engine == "qwen"

    audio = np.linspace(-1.0, 1.0, 1600).astype(np.float32)
    result = fn(audio)
    assert result == {"text": "qwen result"}  # normalized -- no segments/no_speech_prob/etc.

    assert len(models) == 1
    assert models[0].model_id == "mlx-community/Qwen3-ASR-1.7B-bf16"  # from_pretrained(cfg["model"])
    assert len(models[0].calls) == 1
    call = models[0].calls[0]
    assert call["audio"].dtype == np.float32  # resample_to_16k's output stays float32 into transcribe()
    np.testing.assert_array_equal(call["audio"], audio)
    assert call["language"] == "ja"
    assert call["context"] == "Notion Claude Fable"  # cfg["prompt"] -> context=


def test_make_transcribe_fn_qwen_maps_auto_language_to_none(monkeypatch):
    models = []
    monkeypatch.setitem(sys.modules, "qwen3_asr_mlx", _fake_qwen_module(models=models))
    cfg = {"engine": "qwen", "model": "m", "language": "auto", "prompt": None}
    fn, _ = lt._make_transcribe_fn(cfg)
    fn(np.zeros(10, dtype=np.float32))
    assert models[0].calls[0]["language"] is None


def test_make_transcribe_fn_qwen_none_language_stays_none(monkeypatch):
    models = []
    monkeypatch.setitem(sys.modules, "qwen3_asr_mlx", _fake_qwen_module(models=models))
    cfg = {"engine": "qwen", "model": "m", "language": None, "prompt": None}
    fn, _ = lt._make_transcribe_fn(cfg)
    fn(np.zeros(10, dtype=np.float32))
    assert models[0].calls[0]["language"] is None


def test_make_transcribe_fn_qwen_explicit_language_code_passes_through(monkeypatch):
    models = []
    monkeypatch.setitem(sys.modules, "qwen3_asr_mlx", _fake_qwen_module(models=models))
    cfg = {"engine": "qwen", "model": "m", "language": "en", "prompt": None}
    fn, _ = lt._make_transcribe_fn(cfg)
    fn(np.zeros(10, dtype=np.float32))
    assert models[0].calls[0]["language"] == "en"


def test_make_transcribe_fn_qwen_import_error_falls_back_to_whisper(monkeypatch, capsys):
    """qwen3_asr_mlx genuinely not installed (sys.modules[name] = None forces
    the next `import qwen3_asr_mlx` to raise ImportError, the same idiom used
    for e.g. importlib.util.find_spec-style "not installed" simulation)."""
    monkeypatch.setitem(sys.modules, "qwen3_asr_mlx", None)
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper(text="fallback ok"))
    cfg = {"engine": "qwen", "model": "mlx-community/Qwen3-ASR-1.7B-bf16", "language": "ja", "prompt": None}
    fn, engine = lt._make_transcribe_fn(cfg)
    assert engine == "whisper"
    assert fn(np.zeros(10, dtype=np.float32))["text"] == "fallback ok"
    out = capsys.readouterr().out
    assert "qwen engine unavailable" in out
    assert "falling back to whisper" in out


def test_make_transcribe_fn_qwen_load_exception_falls_back_to_whisper(monkeypatch, capsys):
    """A qwen3_asr_mlx that IMPORTS fine but whose from_pretrained() raises
    (e.g. a corrupt/incompatible local HF cache, or a bad --transcribe-model
    override) must fall back exactly like an ImportError -- both are "qwen
    unusable at load time", per _make_transcribe_fn's docstring."""
    class _BoomQwen3ASR:
        @staticmethod
        def from_pretrained(model_id):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "qwen3_asr_mlx", types.SimpleNamespace(Qwen3ASR=_BoomQwen3ASR))
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper(text="fallback ok"))
    cfg = {"engine": "qwen", "model": "bad/model", "language": "ja", "prompt": None}
    fn, engine = lt._make_transcribe_fn(cfg)
    assert engine == "whisper"
    assert fn(np.zeros(10, dtype=np.float32))["text"] == "fallback ok"
    assert "qwen engine unavailable" in capsys.readouterr().out


def test_make_transcribe_fn_qwen_fallback_uses_whisper_default_not_qwen_model_id(monkeypatch):
    """cfg["model"] holds the QWEN model id (resolved for the engine the user
    actually asked for); falling back to whisper must NOT pass that id to
    mlx_whisper.transcribe (it is not a whisper checkpoint) -- it must use
    the whisper default (_WHISPER_FALLBACK_MODEL) instead."""
    monkeypatch.setitem(sys.modules, "qwen3_asr_mlx", None)
    captured = {}

    class _RecordingFakeMlxWhisper(_FakeMlxWhisper):
        def transcribe(self, *args, **kwargs):
            captured["path_or_hf_repo"] = kwargs.get("path_or_hf_repo")
            return super().transcribe(*args, **kwargs)

    monkeypatch.setitem(sys.modules, "mlx_whisper", _RecordingFakeMlxWhisper(text="ok"))
    cfg = {"engine": "qwen", "model": "mlx-community/Qwen3-ASR-1.7B-bf16", "language": "ja", "prompt": None}
    fn, _ = lt._make_transcribe_fn(cfg)
    fn(np.zeros(10, dtype=np.float32))
    assert captured["path_or_hf_repo"] == "mlx-community/whisper-large-v3-mlx"
    assert captured["path_or_hf_repo"] != cfg["model"]


def test_worker_loading_log_announces_qwen_before_attempting_it(tmp_path, monkeypatch, capsys):
    """cfg["engine"]="qwen" (even one that will succeed) must be named in the
    loading-model log line -- proving the log reflects the REQUESTED engine,
    not just a hardcoded "whisper"."""
    monkeypatch.setitem(sys.modules, "qwen3_asr_mlx", _fake_qwen_module(text="hello"))
    cfg, _, _ = _worker_cfg(tmp_path)
    cfg["engine"] = "qwen"
    cfg["model"] = "mlx-community/Qwen3-ASR-1.7B-bf16"
    q = queue.Queue()
    q.put(None)
    lt._worker(q, cfg)
    out = capsys.readouterr().out
    assert "loading model mlx-community/Qwen3-ASR-1.7B-bf16 (engine: qwen) ..." in out


# ------------------------------------------- _worker display_q (caption overlay) ---
# alwayswhisper.live.caption_overlay.py's CaptionLineModel consumes this same event
# stream via a real multiprocessing Queue in production (see
# LiveTranscriber.start()); here it's a duck-typed stub -- put_nowait() only,
# exactly the write-side surface _worker actually calls -- so these stay
# in-process with a plain queue.Queue() for the audio side, same idiom as
# every other _worker test above.
class _FakeDisplayQueue:
    """Records every put_nowait() call, in order. full=True makes every
    put_nowait() raise queue.Full, the way a saturated bounded mp.Queue
    would -- proving _worker drops display events silently instead of
    raising or blocking the transcription loop."""

    def __init__(self, full=False):
        self.events = []
        self.full = full

    def put_nowait(self, item):
        if self.full:
            raise queue.Full
        self.events.append(item)


def test_worker_display_queue_none_changes_nothing(tmp_path, monkeypatch):
    """display_q=None (the default) must be a complete no-op -- the pre-
    existing two-argument call shape _worker(q, cfg) keeps working exactly
    as before this feature existed."""
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper())
    cfg, _jsonl_path, md_path = _worker_cfg(tmp_path)
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=44))
    q.put(None)

    lt._worker(q, cfg)   # no display_q arg at all

    assert md_path.read_text(encoding="utf-8").endswith("- [09:00:00] hello\n")


def test_worker_sentence_mode_emits_one_chunk_then_eos_per_segment(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper(text="hello world"))
    cfg, _, _ = _worker_cfg(tmp_path)
    cfg["display_mode"] = "sentence"
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=45))
    q.put(None)
    display_q = _FakeDisplayQueue()

    lt._worker(q, cfg, display_q)

    assert display_q.events == [("chunk", "hello world"), ("eos",)]


def test_worker_word_mode_emits_chunk_per_word_matching_stdout_then_eos(tmp_path, monkeypatch, capsys):
    """Contiguous words, no pause/sentence-end mid-segment: a ("chunk", word)
    per printed word piece, verbatim (same leading-space spacing as stdout),
    then a single ("eos",) from the shutdown flush that also emits stdout's
    trailing newline (see test_worker_word_display_streams_timestamp_units_
    on_one_line above, same fixture shape)."""
    result = {
        "text": "hello world",
        "segments": [{
            "text": "hello world", "no_speech_prob": 0.05,
            "avg_logprob": -0.2, "compression_ratio": 1.1,
            "words": [{"word": "hello", "start": 0.0, "end": 0.3},
                      {"word": " world", "start": 0.3, "end": 0.7}],
        }],
    }
    monkeypatch.setattr(lt, "_make_transcribe_fn", lambda cfg: (lambda audio: result, "whisper"))
    cfg, _, _ = _worker_cfg(tmp_path)
    cfg["display_mode"] = "word"
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=46))
    q.put(None)
    display_q = _FakeDisplayQueue()

    lt._worker(q, cfg, display_q)

    out = capsys.readouterr().out
    assert "hello world\n" in out
    assert display_q.events == [("chunk", "hello"), ("chunk", " world"), ("eos",)]


def test_worker_word_mode_pause_break_emits_eos_mid_segment(tmp_path, monkeypatch):
    """A >=0.7s gap between two timestamp units inside ONE segment already
    makes _worker print a bare newline (closing that display line) before
    the next word -- see _worker's word-display branch. That closing print
    must carry a matching ("eos",) event, so the overlay starts a fresh line
    at exactly the point the terminal does, not just at segment boundaries."""
    result = {
        "text": "hello world",
        "segments": [{
            "text": "hello world", "no_speech_prob": 0.05,
            "avg_logprob": -0.2, "compression_ratio": 1.1,
            "words": [{"word": "hello", "start": 0.0, "end": 0.3},
                      {"word": " world", "start": 1.5, "end": 1.9}],  # 1.2s gap >= 0.7s
        }],
    }
    monkeypatch.setattr(lt, "_make_transcribe_fn", lambda cfg: (lambda audio: result, "whisper"))
    cfg, _, _ = _worker_cfg(tmp_path)
    cfg["display_mode"] = "word"
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=47))
    q.put(None)
    display_q = _FakeDisplayQueue()

    lt._worker(q, cfg, display_q)

    assert display_q.events == [
        ("chunk", "hello"), ("eos",),     # gap-triggered line close
        ("chunk", " world"), ("eos",),    # shutdown flush closes the 2nd line
    ]


def test_worker_word_mode_sentence_end_emits_eos_right_after_that_word(tmp_path, monkeypatch):
    result = {
        "text": "hello。 world",
        "segments": [{
            "text": "hello。 world", "no_speech_prob": 0.05,
            "avg_logprob": -0.2, "compression_ratio": 1.1,
            "words": [{"word": "hello。", "start": 0.0, "end": 0.3},
                      {"word": " world", "start": 0.3, "end": 0.7}],
        }],
    }
    monkeypatch.setattr(lt, "_make_transcribe_fn", lambda cfg: (lambda audio: result, "whisper"))
    cfg, _, _ = _worker_cfg(tmp_path)
    cfg["display_mode"] = "word"
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=48))
    q.put(None)
    display_q = _FakeDisplayQueue()

    lt._worker(q, cfg, display_q)

    assert display_q.events == [
        ("chunk", "hello。"), ("eos",),   # sentence-end punctuation closes the line
        ("chunk", " world"), ("eos",),    # shutdown flush closes the 2nd line
    ]


def test_worker_word_mode_fallback_without_word_timestamps_emits_chunk_and_eos(tmp_path, monkeypatch):
    """extract_timed_words() returns [] when the backend response carries no
    word timestamps -> the fallback branch prints the whole segment text in
    one shot. That single print() call both writes the text AND terminates
    the line (default end="\\n"), so it maps to a chunk immediately followed
    by an eos."""
    result = {"text": "hello world", "segments": [{
        "text": "hello world", "no_speech_prob": 0.05,
        "avg_logprob": -0.2, "compression_ratio": 1.1,
        # no "words" key -> extract_timed_words() returns []
    }]}
    monkeypatch.setattr(lt, "_make_transcribe_fn", lambda cfg: (lambda audio: result, "whisper"))
    cfg, _, _ = _worker_cfg(tmp_path)
    cfg["display_mode"] = "word"
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=49))
    q.put(None)
    display_q = _FakeDisplayQueue()

    lt._worker(q, cfg, display_q)

    assert display_q.events == [("chunk", "hello world"), ("eos",)]


def test_worker_display_queue_full_drops_silently_without_raising(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper(text="hello"))
    cfg, _jsonl_path, md_path = _worker_cfg(tmp_path)
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=50))
    q.put(None)
    display_q = _FakeDisplayQueue(full=True)

    lt._worker(q, cfg, display_q)   # must not raise despite every put_nowait() raising queue.Full

    assert display_q.events == []
    # the transcript itself is entirely unaffected by a saturated display queue
    assert md_path.read_text(encoding="utf-8").endswith("- [09:00:00] hello\n")


# ---------------------------------------------------------------- fmt_hms ---
def test_fmt_hms_zero():
    assert lt.fmt_hms(0) == "00:00:00"


def test_fmt_hms_hours_minutes_seconds():
    assert lt.fmt_hms(3661.5) == "01:01:01"


def test_fmt_hms_various():
    assert lt.fmt_hms(59) == "00:00:59"
    assert lt.fmt_hms(60) == "00:01:00"
    assert lt.fmt_hms(3600) == "01:00:00"


# ------------------------------------------------------------- extract_text ---
def test_extract_text_strips_whitespace():
    assert lt.extract_text({"text": "  hello world  "}) == "hello world"


def test_extract_text_handles_missing_or_none_or_empty():
    assert lt.extract_text({}) == ""
    assert lt.extract_text({"text": None}) == ""
    assert lt.extract_text({"text": ""}) == ""


def test_extract_text_japanese_passthrough():
    assert lt.extract_text({"text": "こんにちは。"}) == "こんにちは。"


# ------------------------------------ hallucination / repetition filtering ---
def test_looks_like_repetition_cjk_char_loops():
    assert lt.looks_like_repetition("剛" * 30)
    assert lt.looks_like_repetition("チェック" * 50)


def test_looks_like_repetition_phrase_loops():
    assert lt.looks_like_repetition("Let's do it. Let's do it. Let's do it.")           # 3x
    assert lt.looks_like_repetition("Let's do it. Let's do it. Let's do it. Let's do it.")  # 4x


def test_looks_like_repetition_leaves_normal_text():
    assert not lt.looks_like_repetition("今日はデータベースの設計について話します。")
    assert not lt.looks_like_repetition("Let's build the pipeline and ship it today.")
    assert not lt.looks_like_repetition("Thank you.")   # not a loop -> handled by the no_speech gate
    assert not lt.looks_like_repetition("no no no")     # 3x single word: keep (could be real emphasis)


def test_clean_transcription_drops_silent_hallucination():
    # "Thank you." on silence: high no_speech_prob, confident logprob -> Whisper's
    # own AND-logic would keep it; our hard no_speech gate drops it.
    r = {"text": "Thank you.", "segments": [
        {"text": " Thank you.", "no_speech_prob": 0.92, "avg_logprob": -0.2, "compression_ratio": 0.8}]}
    assert lt.clean_transcription(r) == ""


def test_clean_transcription_drops_char_loop_by_compression():
    r = {"text": "剛" * 30, "segments": [
        {"text": "剛" * 30, "no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 8.0}]}
    assert lt.clean_transcription(r) == ""


def test_clean_transcription_drops_phrase_loop_even_when_signals_ok():
    r = {"text": "x", "segments": [
        {"text": "Let's do it. Let's do it. Let's do it. Let's do it.",
         "no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.9}]}
    assert lt.clean_transcription(r) == ""


def test_clean_transcription_drops_low_confidence():
    r = {"text": "x", "segments": [
        {"text": " garbled noise", "no_speech_prob": 0.2, "avg_logprob": -2.0, "compression_ratio": 1.0}]}
    assert lt.clean_transcription(r) == ""


def test_clean_transcription_keeps_real_speech_and_joins_segments():
    r = {"text": "ignored top level", "segments": [
        {"text": " データベースの話をします。", "no_speech_prob": 0.05, "avg_logprob": -0.25, "compression_ratio": 1.1},
        {"text": " 次はAPIです。", "no_speech_prob": 0.10, "avg_logprob": -0.40, "compression_ratio": 1.0}]}
    assert lt.clean_transcription(r) == "データベースの話をします。 次はAPIです。"


def test_clean_transcription_fallback_without_segments():
    assert lt.clean_transcription({"text": "チェック" * 40}) == ""
    assert lt.clean_transcription({"text": "普通の文章です。"}) == "普通の文章です。"


def test_clean_transcription_handles_qwen_style_result_shape():
    """_make_transcribe_fn's qwen path normalizes TranscriptionResult to a
    plain {"text": ...} dict with no "segments"/no_speech_prob/avg_logprob/
    compression_ratio keys at all (unlike mlx_whisper's result) -- confirms
    clean_transcription's segments-less fallback branch (the `else` branch:
    top-level text, repetition- and hallucination-filtered) already handles
    that shape correctly with no code changes needed here."""
    assert lt.clean_transcription({"text": "普通の文章です。"}) == "普通の文章です。"
    assert lt.clean_transcription({"text": "Thank you for watching!"}) == ""       # outro hallucination
    assert lt.clean_transcription({"text": "剛" * 30}) == ""                        # repetition loop
    assert lt.clean_transcription({"text": "ご視聴ありがとうございました"}) == ""    # JA outro pair


# ------------------------------------------- text-level hallucination filter ---
def _seg(text, no_speech_prob=0.1, avg_logprob=-0.3, compression_ratio=1.5):
    """A single "passing"-confidence segment dict -- these signal values pass
    the existing no_speech/logprob/compression gates on their own, so any
    drop in the tests below can only be attributed to the new text filter."""
    return {"text": text, "no_speech_prob": no_speech_prob, "avg_logprob": avg_logprob,
            "compression_ratio": compression_ratio}


def test_is_hallucination_text_symbol_only():
    assert lt.is_hallucination_text("!")
    assert lt.is_hallucination_text(" ! ")
    assert lt.is_hallucination_text("！")   # full-width
    assert lt.is_hallucination_text("!!!")
    assert lt.is_hallucination_text("♪〜")
    assert lt.is_hallucination_text("。。。")


def test_is_hallucination_text_english_outro_phrases():
    assert lt.is_hallucination_text("Thank you for watching!")
    assert lt.is_hallucination_text("thank you for watching.")   # case/punct variant
    assert lt.is_hallucination_text("Thanks for watching!")      # short variant


def test_is_hallucination_text_japanese_outro_phrases():
    assert lt.is_hallucination_text("ご視聴ありがとうございました")
    assert lt.is_hallucination_text("最後までご視聴いただき本当にありがとうございました")
    assert lt.is_hallucination_text("チャンネル登録お願いします")
    assert lt.is_hallucination_text("高評価とチャンネル登録をお願いします")


def test_is_hallucination_text_keeps_real_speech():
    assert not lt.is_hallucination_text("先週ありがとうございました")        # embedded thanks: real speech
    assert not lt.is_hallucination_text("チャンネル登録の導線を見直します")  # keyword, no お願い/高評価 pair
    assert not lt.is_hallucination_text("次の動画で使う素材を選ぶ")


def test_is_hallucination_text_standalone_thanks_variants():
    # Standalone thanks -- meeting sign-offs and Whisper silence
    # hallucinations -- drop by exact match, punctuation-insensitive.
    assert lt.is_hallucination_text("ありがとうございました")
    assert lt.is_hallucination_text("ありがとうございました。")
    assert lt.is_hallucination_text("はい、ありがとうございました")
    assert lt.is_hallucination_text("はい、ありがとうございました。")


def test_clean_transcription_drops_symbol_only_segments():
    for txt in ("!", " ! ", "！", "!!!", "♪〜", "。。。"):
        r = {"text": txt, "segments": [_seg(txt)]}
        assert lt.clean_transcription(r) == "", f"expected drop for {txt!r}"


def test_clean_transcription_drops_thank_you_for_watching_variants():
    for txt in ("Thank you for watching!", "thank you for watching.", "Thanks for watching!"):
        r = {"text": txt, "segments": [_seg(txt)]}
        assert lt.clean_transcription(r) == "", f"expected drop for {txt!r}"


def test_clean_transcription_drops_japanese_outro_phrase():
    txt = "ご視聴ありがとうございました"
    r = {"text": txt, "segments": [_seg(txt)]}
    assert lt.clean_transcription(r) == ""


def test_clean_transcription_drops_japanese_outro_phrase_pair_variant():
    # "ご視聴" + "ありがとう" pair rule, not an exact-string match -- covers
    # the いただき/本当に insertions a real hallucination variant carries.
    txt = "最後までご視聴いただき本当にありがとうございました"
    r = {"text": txt, "segments": [_seg(txt)]}
    assert lt.clean_transcription(r) == ""


def test_clean_transcription_drops_subscribe_phrases():
    for txt in ("チャンネル登録お願いします", "高評価とチャンネル登録をお願いします"):
        r = {"text": txt, "segments": [_seg(txt)]}
        assert lt.clean_transcription(r) == "", f"expected drop for {txt!r}"


def test_clean_transcription_fallback_drops_hallucination_text():
    assert lt.clean_transcription({"text": "Thank you for watching!", "segments": []}) == ""


def test_clean_transcription_fallback_keeps_real_text():
    assert lt.clean_transcription({"text": "こんにちは", "segments": []}) == "こんにちは"


def test_clean_transcription_mixed_segments_drops_only_hallucinations():
    r = {"text": "ignored top level", "segments": [
        _seg("今日は編集作業をした"),
        _seg("!"),
        _seg("Thank you for watching!"),
    ]}
    assert lt.clean_transcription(r) == "今日は編集作業をした"


def test_clean_transcription_drops_standalone_thanks():
    # Standalone thanks -- meeting sign-offs and Whisper silence
    # hallucinations -- drop as exact (punctuation-insensitive) matches;
    # thanks embedded in a longer utterance still survives.
    for txt in ("ありがとうございました", "ありがとうございました。",
                "はい、ありがとうございました", "はい、ありがとうございました。"):
        r = {"text": txt, "segments": [_seg(txt)]}
        assert lt.clean_transcription(r) == "", f"expected drop for {txt!r}"
    kept = "先週ありがとうございました"
    assert lt.clean_transcription({"text": kept, "segments": [_seg(kept)]}) == kept


def test_clean_transcription_keeps_channel_registration_without_pair():
    # False-positive guard: "チャンネル登録" without お願い/高評価 is real
    # discussion about the channel, not the subscribe-request hallucination.
    txt = "チャンネル登録の導線を見直します"
    r = {"text": txt, "segments": [_seg(txt)]}
    assert lt.clean_transcription(r) == txt


def test_clean_transcription_keeps_unrelated_sentence():
    txt = "次の動画で使う素材を選ぶ"
    r = {"text": txt, "segments": [_seg(txt)]}
    assert lt.clean_transcription(r) == txt


# ------------------------------------------------ initial_prompt (bias) wiring ---
def test_live_transcriber_carries_prompt_into_cfg_and_meta(tmp_path):
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path), prompt="Notion Claude Fable")
    assert t.cfg["prompt"] == "Notion Claude Fable"          # -> _worker passes as initial_prompt
    assert t.cfg["meta"]["prompt"] == "Notion Claude Fable"  # -> recorded in the jsonl meta line


def test_live_transcriber_prompt_defaults_none(tmp_path):
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path))
    assert t.cfg["prompt"] is None


def test_live_transcriber_carries_word_display_mode_into_cfg(tmp_path):
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path), display_mode="word")
    assert t.cfg["display_mode"] == "word"


def test_live_transcriber_display_mode_defaults_to_word(tmp_path):
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path))
    assert t.cfg["display_mode"] == "word"


# -------------------------------------------- paragraph_gap_sec wiring (cfg) ---
def test_live_transcriber_carries_paragraph_gap_sec_default_into_cfg(tmp_path):
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path))
    assert t.cfg["paragraph_gap_sec"] == 60.0  # -> _worker passes it to TranscriptWriter


# ---------------------------------------------------------- gate wiring (cfg) ---
def test_live_transcriber_carries_gate_default_into_cfg(tmp_path):
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path))
    assert t.cfg["gate"] == lt.GATE_DEFAULT  # -> _worker passes it to SegmentBuffer


def test_live_transcriber_carries_explicit_gate_into_cfg(tmp_path):
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path), gate=0.004)
    assert t.cfg["gate"] == 0.004


def test_live_transcriber_carries_min_speech_default_into_cfg(tmp_path):
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path))
    assert t.cfg["min_speech_sec"] == 0.2


def test_worker_passes_cfg_gate_through_to_the_segmenter(tmp_path, monkeypatch):
    """cfg["gate"] must reach SegmentBuffer -- that is the entire point of the
    knob -- and a cfg dict written before the key existed must still work,
    since _worker reads it with .get() and falls back to GATE_DEFAULT (the
    same idiom as the no_speech/logprob gates)."""
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper())
    seen = []
    real_buffer = lt.SegmentBuffer

    class RecordingBuffer(real_buffer):
        def __init__(self, sr, **kw):
            seen.append(kw.get("gate"))
            super().__init__(sr, **kw)

    monkeypatch.setattr(lt, "SegmentBuffer", RecordingBuffer)

    cfg, _jsonl, _md = _worker_cfg(tmp_path)
    cfg["gate"] = 0.05
    q = queue.Queue()
    q.put(None)
    lt._worker(q, cfg)
    assert seen == [0.05]

    seen.clear()
    legacy_cfg, _jsonl2, _md2 = _worker_cfg(tmp_path)  # no "gate" key at all
    assert "gate" not in legacy_cfg
    q2 = queue.Queue()
    q2.put(None)
    lt._worker(q2, legacy_cfg)
    assert seen == [lt.GATE_DEFAULT]


# -------------------------------------------------------- engine wiring (cfg) ---
def test_live_transcriber_engine_defaults_to_whisper_in_cfg(tmp_path):
    """No engine= given (every pre-existing call site, e.g. live_avatar.py
    before --transcribe-engine existed) must keep cfg["engine"] == "whisper"
    -- _make_transcribe_fn's own default -- so _worker's behavior is
    unchanged for anyone/anything not passing the new kwarg."""
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path))
    assert t.cfg["engine"] == "whisper"


def test_live_transcriber_carries_explicit_engine_into_cfg(tmp_path):
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path), engine="qwen")
    assert t.cfg["engine"] == "qwen"


# --------------------------------------- display_queue wiring (caption overlay) ---
# display_queue rides as a separate Process arg alongside cfg (like the audio
# queue), not inside cfg itself (an mp.Queue isn't meaningful "config data") --
# see LiveTranscriber.start(), which passes self.display_queue as _worker's
# third positional argument. These pin the constructor-level contract only;
# _worker's actual event-emission behavior is covered by the
# "_worker display_q (caption overlay)" tests above.
def test_live_transcriber_display_queue_defaults_to_none(tmp_path):
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path))
    assert t.display_queue is None


def test_live_transcriber_carries_display_queue_through_unchanged(tmp_path):
    sentinel = object()
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path), display_queue=sentinel)
    assert t.display_queue is sentinel


# ------------------------------------------- immediate-repeat collapse (dedup) ---
# mlx-whisper's decoder sometimes emits one utterance twice back-to-back within
# a single result ("作ったら…できるよ。作ったら…できるよ。"). looks_like_repetition
# only catches >=3 repeats of a SHORT block, so a plain doubling of a long
# sentence used to reach the transcript verbatim (2026-09-03: 23 of 175
# segments in one session).
def test_collapse_immediate_repeat_two_copies():
    assert lt.collapse_immediate_repeat(
        "作ったらレビューすることができるよ。作ったらレビューすることができるよ。"
    ) == "作ったらレビューすることができるよ。"


def test_collapse_immediate_repeat_three_copies_whitespace_separated():
    """The copies may be joined with a space (two identical sub-segments) or
    run together (one segment decoded twice) -- both are the same artifact."""
    assert lt.collapse_immediate_repeat(
        "アップデータベース、 アップデータベース、 アップデータベース、"
    ) == "アップデータベース、"


def test_collapse_immediate_repeat_short_blocks():
    """Short doublings are collapsed too (owner request, 2026-09-03: "もっと機械
    的に置換して") -- three characters is enough for a doubling, two for three
    or more copies."""
    assert lt.collapse_immediate_repeat("前提として前提として") == "前提として"
    assert lt.collapse_immediate_repeat("これします? これします?") == "これします?"
    assert lt.collapse_immediate_repeat("はい、はい、") == "はい、"
    assert lt.collapse_immediate_repeat("だからこうしてだからこうして") == "だからこうして"


def test_collapse_immediate_repeat_keeps_reduplicated_words():
    """Japanese reduplications ARE the word -- collapsing them would rewrite
    real speech into nonsense -- and they are all two-character blocks, which
    is why a doubling needs three."""
    for word in ("いろいろ", "そろそろ", "ますます", "どんどん", "わくわく", "はいはい",
                 "ごちゃごちゃ", "もじもじ"):
        assert lt.collapse_immediate_repeat(word) == word


def test_collapse_immediate_repeat_inside_a_longer_utterance():
    """The repeat does not have to be the whole string: Whisper doubles a
    trailing or leading phrase just as often ("あ、いいです。いいです。")."""
    assert lt.collapse_immediate_repeat("あ、いいです。いいです。") == "あ、いいです。"
    assert lt.collapse_immediate_repeat(
        "A。B。A。B。終わりです。") == "A。B。終わりです。"   # a repeated GROUP of phrases


def test_collapse_immediate_repeat_runs_to_a_fixed_point():
    """Removing one repeat can expose another: the pair "コスプレ、写真…。" is only
    back-to-back once the doubled "コスプレ、" in front of it is gone."""
    assert lt.collapse_immediate_repeat(
        "コスプレ、コスプレ、写真撮るのも面白い。 コスプレ、写真撮るのも面白い。"
    ) == "コスプレ、写真撮るのも面白い。"
    assert lt.collapse_immediate_repeat("そうですねそうですね そうですね") == "そうですね"


def test_collapse_immediate_repeat_never_cuts_into_a_word():
    """A word whose own letters repeat is not a decoder repeat: only whole
    phrases (cut at punctuation/space) are ever compared and removed. These are
    real 32k-segment-replay regressions from a character-level scan."""
    assert lt.collapse_immediate_repeat("Oh, raha illallah.") == "Oh, raha illallah."
    assert lt.collapse_immediate_repeat("そこだけごちゃごちゃしてる") == "そこだけごちゃごちゃしてる"
    assert lt.collapse_immediate_repeat("日本の日本の最大の顧客") == "日本の日本の最大の顧客"


def test_collapse_immediate_repeat_keeps_the_final_punctuation_of_a_run():
    """Whisper re-punctuates freely between copies; the copy the speaker
    actually ended on owns the punctuation."""
    assert lt.collapse_immediate_repeat("Yeah, yeah, yeah!") == "Yeah!"
    assert lt.collapse_immediate_repeat("そうです、そうです。") == "そうです。"


def test_collapse_immediate_repeat_keeps_normal_spacing_of_the_kept_copy():
    """A kept copy keeps its own internal spacing; only whitespace orphaned by
    the removal is folded away."""
    assert lt.collapse_immediate_repeat(
        "カテゴライズをうまく使って、 キューアンドエージェントに。"
        "カテゴライズをうまく使って、 キューアンドエージェントに。"
    ) == "カテゴライズをうまく使って、 キューアンドエージェントに。"


def test_collapse_immediate_repeat_leaves_normal_text_untouched():
    text = "今日はデータベースの設計について話します。次はAPIです。"
    assert lt.collapse_immediate_repeat(text) == text


def test_clean_transcription_collapses_sentence_doubled_inside_one_segment():
    r = {"segments": [{"text": "いつでもすぐに自分のよく使うエージェントを呼び出せます。"
                               "いつでもすぐに自分のよく使うエージェントを呼び出せます。",
                       "no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.6}]}
    assert lt.clean_transcription(r) == "いつでもすぐに自分のよく使うエージェントを呼び出せます。"


def test_clean_transcription_collapses_two_identical_segments():
    """Same artifact, split across sub-segments: the join must not survive as
    a duplicate just because a space landed between the copies."""
    seg = {"text": "見ておきます。", "no_speech_prob": 0.1,
           "avg_logprob": -0.3, "compression_ratio": 1.2}
    assert lt.clean_transcription({"segments": [dict(seg), dict(seg)]}) == "見ておきます。"


def test_collapse_immediate_repeat_also_folds_a_genuine_double():
    """The accepted cost of the 2026-09-03 aggressiveness bump: a phrase the
    speaker really did say twice ("あ、そうですそうです。") is folded as well,
    because nothing in the text distinguishes it from the decoder saying it
    twice. The meaning survives; the duplicated log line was the bigger
    complaint."""
    assert lt.collapse_immediate_repeat("あ、そうですそうです。") == "あ、そうです。"


# -------------------------------------------------- bias-prompt echo (glossary) ---
# Whisper reads its own initial_prompt back as if it were speech on noisy /
# near-silent audio: the 2026-09-03 session emitted "データベース, Notion" and
# "AIエージェント, Notion" -- verbatim tails of the glossary bias prompt.
_PROMPT = "MCP, カスタムエージェント\nNotion AI, Notion AIエージェント, Notionデータベース, Notion"


def test_clean_transcription_drops_bias_prompt_echo():
    r = {"segments": [{"text": "データベース, Notion", "no_speech_prob": 0.1,
                       "avg_logprob": -0.3, "compression_ratio": 1.1}]}
    assert lt.clean_transcription(r, prompt=_PROMPT) == ""


def test_clean_transcription_drops_prompt_echo_after_collapsing_its_repeat():
    """The echo often arrives as several glossary entries in a row ("データベース,
    Notionデータベース, Notion..."), too long to be a single echo -- so it is
    judged phrase by phrase, after the repeat collapse."""
    r = {"segments": [{"text": "データベース, Notionデータベース, Notionデータベース, Notion",
                       "no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.4}]}
    assert lt.clean_transcription(r, prompt=_PROMPT) == ""


def test_clean_transcription_keeps_real_speech_mentioning_glossary_terms():
    """Only short text that IS (a chunk of) the prompt is an echo -- a real
    sentence about Notion must survive untouched."""
    r = {"segments": [{"text": "Notionデータベースの設計について話します。", "no_speech_prob": 0.1,
                       "avg_logprob": -0.3, "compression_ratio": 1.1}]}
    assert lt.clean_transcription(r, prompt=_PROMPT) == "Notionデータベースの設計について話します。"


def test_clean_transcription_keeps_a_real_sentence_split_into_short_phrases():
    """The per-phrase echo test must not fire on real speech that happens to be
    short and to mention glossary terms: its phrases carry particles the term
    list never contains."""
    r = {"segments": [{"text": "Notionは、便利です。", "no_speech_prob": 0.1,
                       "avg_logprob": -0.3, "compression_ratio": 1.1}]}
    assert lt.clean_transcription(r, prompt=_PROMPT) == "Notionは、便利です。"


def test_clean_transcription_without_prompt_keeps_glossary_shaped_text():
    """No bias prompt in play (prompt=None, the default) -> nothing to echo."""
    r = {"segments": [{"text": "データベース, Notion", "no_speech_prob": 0.1,
                       "avg_logprob": -0.3, "compression_ratio": 1.1}]}
    assert lt.clean_transcription(r) == "データベース, Notion"


def test_is_hallucination_text_thanks_in_both_tenses():
    """Whisper fabricates the present-tense sign-off too ("ありがとうございます"),
    and both tenses carry no content when they stand alone."""
    assert lt.is_hallucination_text("ありがとうございました")
    assert lt.is_hallucination_text("ありがとうございます。")
    assert lt.is_hallucination_text("はい、ありがとうございます")
    assert not lt.is_hallucination_text("先週はありがとうございます、助かりました")


# ---------------------------------------------------- clean_segments / display ---
def test_clean_segments_keeps_only_the_segments_that_pass_the_gates():
    r = {"segments": [
        {"text": "この機能は日本のXのユーザーから", "no_speech_prob": 0.1,
         "avg_logprob": -0.3, "compression_ratio": 1.1},
        {"text": "ご視聴ありがとうございました", "no_speech_prob": 0.1,
         "avg_logprob": -0.3, "compression_ratio": 1.1},
    ]}
    assert [s["text"] for s in lt.clean_segments(r)] == ["この機能は日本のXのユーザーから"]


def test_display_units_streams_words_when_they_reconstruct_the_text():
    segs = [{"text": "hello world",
             "words": [{"word": "hello", "start": 0.0, "end": 0.3},
                       {"word": " world", "start": 0.3, "end": 0.7}]}]
    assert lt.display_units(segs, "hello world", offset_s=10.0) == [
        ("hello", 10.0, 10.3), (" world", 10.3, 10.7)]


def test_display_units_falls_back_to_one_text_unit_when_words_disagree():
    """The words still carry the collapsed repeat, so streaming them would put
    on screen exactly what the transcript just removed: send the cleaned text
    as a single unit instead, spanning the segments' own time range."""
    segs = [{"text": "はい、テイラー。はい、テイラー。", "start": 1.0, "end": 3.0,
             "words": [{"word": "はい、テイラー。", "start": 1.0, "end": 2.0},
                       {"word": "はい、テイラー。", "start": 2.0, "end": 3.0}]}]
    assert lt.display_units(segs, "はい、テイラー。", offset_s=10.0) == [
        ("はい、テイラー。", 11.0, 13.0)]


def test_display_units_empty_without_text():
    assert lt.display_units([], "") == []


def test_worker_word_display_never_shows_a_filtered_hallucination(tmp_path, monkeypatch, capsys):
    """The bug this whole section exists for: the md/jsonl dropped Whisper's
    "ご視聴ありがとうございました" while the terminal (and the caption overlay,
    which mirrors it) printed it anyway, because the word-display path read
    the RAW result instead of the cleaned segments."""
    result = {
        "text": "ご視聴ありがとうございましたこの機能は",
        "segments": [
            {"text": "ご視聴ありがとうございました", "no_speech_prob": 0.1,
             "avg_logprob": -0.3, "compression_ratio": 1.1,
             "words": [{"word": "ご視聴ありがとうございました", "start": 0.0, "end": 0.5}]},
            {"text": "この機能は", "no_speech_prob": 0.1, "avg_logprob": -0.3,
             "compression_ratio": 1.1,
             "words": [{"word": "この機能は", "start": 0.5, "end": 1.0}]},
        ],
    }
    monkeypatch.setattr(lt, "_make_transcribe_fn", lambda cfg: (lambda audio: result, "whisper"))
    cfg, _, md_path = _worker_cfg(tmp_path)
    cfg["display_mode"] = "word"
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=47))
    q.put(None)
    display_q = _FakeDisplayQueue()

    lt._worker(q, cfg, display_q)

    out = capsys.readouterr().out
    assert "ありがとうございました" not in out
    assert "この機能は" in out
    assert display_q.events == [("chunk", "この機能は"), ("eos",)]
    assert "ありがとうございました" not in md_path.read_text(encoding="utf-8")


def test_worker_word_display_shows_the_collapsed_text_once(tmp_path, monkeypatch, capsys):
    """A doubled sentence reaches the screen once, matching what the
    transcript file records."""
    doubled = "作ったらレビューすることができるよ。作ったらレビューすることができるよ。"
    once = "作ったらレビューすることができるよ。"
    result = {
        "text": doubled,
        "segments": [{"text": doubled, "no_speech_prob": 0.1, "avg_logprob": -0.3,
                      "compression_ratio": 1.6, "start": 0.0, "end": 3.0,
                      "words": [{"word": once, "start": 0.0, "end": 1.5},
                                {"word": once, "start": 1.5, "end": 3.0}]}],
    }
    monkeypatch.setattr(lt, "_make_transcribe_fn", lambda cfg: (lambda audio: result, "whisper"))
    cfg, _, md_path = _worker_cfg(tmp_path)
    cfg["display_mode"] = "word"
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=48))
    q.put(None)
    display_q = _FakeDisplayQueue()

    lt._worker(q, cfg, display_q)

    assert capsys.readouterr().out.count(once) == 1
    assert display_q.events == [("chunk", once), ("eos",)]
    assert md_path.read_text(encoding="utf-8").count(once) == 1


def test_worker_passes_the_bias_prompt_into_the_cleaning_gates(tmp_path, monkeypatch):
    """cfg["prompt"] is what Whisper was biased with, so it is also what an
    echo has to be matched against -- _worker must hand it to the cleaner."""
    result = {"text": "データベース, Notion",
              "segments": [{"text": "データベース, Notion", "no_speech_prob": 0.1,
                            "avg_logprob": -0.3, "compression_ratio": 1.1}]}
    monkeypatch.setattr(lt, "_make_transcribe_fn", lambda cfg: (lambda audio: result, "whisper"))
    cfg, _, md_path = _worker_cfg(tmp_path)
    cfg["prompt"] = _PROMPT
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=49))
    q.put(None)

    lt._worker(q, cfg)

    assert "Notion" not in md_path.read_text(encoding="utf-8")


# ------------------------------------------------- per-segment timing (--transcribe-timing) ---
# Where the seconds between "you stopped talking" and "the line appears" go:
# the 0.8s silence gate, any audio backlog when inference can't keep up, and
# the inference itself. Diagnostic only -- off unless asked for, and on stderr
# so `2>timing.log` still leaves a clean transcript on stdout.
def test_format_timing_line_fields():
    line = lt.format_timing_line(audio_sec=12.5, infer_sec=3.2, lag_sec=4.6)
    assert line == "[timing] audio 12.5s  infer 3.20s (3.9x)  lag 4.6s"


def test_format_timing_line_marks_a_dropped_segment():
    assert lt.format_timing_line(1.0, 0.5, 1.6, dropped=True).endswith("  (dropped)")


def test_format_timing_line_survives_a_zero_inference_time():
    """A faked/instant transcribe must not divide by zero."""
    assert "(-x)" in lt.format_timing_line(1.0, 0.0, 1.0)


def test_worker_prints_no_timing_unless_asked(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper())
    cfg, _, _ = _worker_cfg(tmp_path)
    assert "timing" not in cfg          # legacy cfg dicts have no such key
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=51))
    q.put(None)

    lt._worker(q, cfg)

    assert "[timing]" not in capsys.readouterr().err


def test_worker_timing_reports_one_line_per_transcribed_segment(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper())
    cfg, _, _ = _worker_cfg(tmp_path)
    cfg["timing"] = True
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=52))
    q.put(None)

    lt._worker(q, cfg)

    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("[timing]")]
    assert len(lines) == 1
    assert " audio " in lines[0] and " infer " in lines[0] and " lag " in lines[0]


def test_worker_timing_also_reports_a_dropped_segment(tmp_path, monkeypatch, capsys):
    """A segment thrown away as a hallucination costs the same inference time,
    so it has to show up in the measurement too."""
    monkeypatch.setattr(lt, "_make_transcribe_fn",
                        lambda cfg: (lambda audio: {"text": "ご視聴ありがとうございました"}, "whisper"))
    cfg, _, _ = _worker_cfg(tmp_path)
    cfg["timing"] = True
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=53))
    q.put(None)

    lt._worker(q, cfg)

    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[timing]")]
    assert len(lines) == 1 and lines[0].endswith("  (dropped)")


def test_live_transcriber_timing_defaults_off_in_cfg(tmp_path):
    assert lt.LiveTranscriber(16000, "m", "ja", str(tmp_path)).cfg["timing"] is False


def test_live_transcriber_carries_timing_into_cfg(tmp_path):
    assert lt.LiveTranscriber(16000, "m", "ja", str(tmp_path), timing=True).cfg["timing"] is True


def test_live_transcriber_max_sec_defaults_to_the_module_constant(tmp_path):
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path))
    assert t.cfg["max_sec"] == lt.MAX_SEC_DEFAULT


def test_live_transcriber_carries_explicit_max_sec_into_cfg(tmp_path):
    """The forced cut that ends a segment when nobody pauses: shortening it is
    the one knob that moves the lag on continuous speech (inference is ~20x
    realtime, so it is never the wait)."""
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path), max_sec=15.0)
    assert t.cfg["max_sec"] == 15.0


# ------------------------------------------- runaway decode on near-silence ---
# Measured 2026-09-03 on a live session: near-silent 1.3-1.8s segments took
# 8-20s of inference each (real speech of the same length takes ~0.3s), because
# Whisper loops on non-speech and then retries the whole decode at every
# fallback temperature. The output was dropped by the filters, but the seconds
# were already spent -- and the audio queue behind it backed up, pushing the
# lag on real speech to 20-29s.
def test_sample_len_scales_with_audio_and_keeps_a_wide_margin():
    """Measured Japanese speech runs 2.9-5.8 tokens/s (6 real clips), so the
    cap sits about 3x above the fastest of them -- high enough never to cut a
    real utterance short, low enough to stop a loop in its tracks."""
    assert lt.decode_sample_len(1.5) == 56       # a noise blip: loop stops early
    assert lt.decode_sample_len(25.0) == 432     # 25s of speech needs ~132
    assert lt.decode_sample_len(0.0) == lt.DECODE_TOKENS_BASE


class _RecordingMlxWhisper:
    """Fake mlx_whisper that records the kwargs of every transcribe() call."""

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return {"text": "hello", "segments": [
            {"text": "hello", "no_speech_prob": 0.05, "avg_logprob": -0.2,
             "compression_ratio": 1.1, "tokens": [1, 2, 3]}]}


def test_whisper_call_disables_the_temperature_fallback(monkeypatch):
    """Whisper retries a bad decode at each of six fallback temperatures; on a
    looping non-speech segment that multiplies an already-slow decode. One pass
    is enough here -- a loop is dropped by clean_transcription either way."""
    fake = _RecordingMlxWhisper()
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    fn, engine = lt._make_transcribe_fn({"model": "m", "language": "ja", "display_mode": "word"})
    fn(np.zeros(16000 * 2, dtype=np.float32))
    assert engine == "whisper"
    assert fake.calls[0]["temperature"] == 0.0


def test_whisper_call_caps_sampled_tokens_by_audio_length(monkeypatch):
    fake = _RecordingMlxWhisper()
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    fn, _ = lt._make_transcribe_fn({"model": "m", "language": "ja", "display_mode": "word"})
    fn(np.zeros(16000 * 2, dtype=np.float32))          # 2.0s of audio
    assert fake.calls[0]["sample_len"] == lt.decode_sample_len(2.0)
    fn(np.zeros(16000 * 10, dtype=np.float32))         # 10.0s
    assert fake.calls[1]["sample_len"] == lt.decode_sample_len(10.0)


def test_worker_rejects_capped_text_from_files_and_displays(tmp_path, monkeypatch, capsys):
    """A capped result must not be published; the skip remains visible."""
    # More tokens than any cap this segment could have been given, so the test
    # doesn't have to predict the segment's exact length (pre-roll and trailing
    # silence are part of it).
    result = {"text": "本文", "segments": [
        {"text": "本文", "no_speech_prob": 0.05, "avg_logprob": -0.2,
         "compression_ratio": 1.1, "tokens": list(range(10_000))}]}
    monkeypatch.setattr(lt, "_make_transcribe_fn", lambda cfg: (lambda audio: result, "whisper"))
    cfg, jsonl_path, md_path = _worker_cfg(tmp_path)
    cfg["display_mode"] = "word"
    cfg["timing"] = True
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=61))
    q.put(None)

    display_q = _FakeDisplayQueue()
    lt._worker(q, cfg, display_q)

    captured = capsys.readouterr()
    assert "token cap" in captured.err
    assert "スキップ" in captured.err
    assert "(dropped)" in captured.err
    assert "本文" not in captured.out
    assert "本文" not in jsonl_path.read_text(encoding="utf-8")
    assert "本文" not in md_path.read_text(encoding="utf-8")
    assert not display_q.events


def test_token_cap_is_per_decode_window_not_entire_recording():
    result = {"segments": [
        {"seek": 0, "tokens": [1] * 30},
        {"seek": 100, "tokens": [1] * 30},
    ]}
    assert not lt.reached_token_cap(result, 48)
    result["segments"][1]["seek"] = 0
    assert lt.reached_token_cap(result, 48)
    assert not lt.reached_token_cap({"text": "qwen result"}, 48)


class _CappedThenRecoveredWhisper:
    def __init__(self, retry_caps=False, first_text=None):
        self.calls = []
        self.retry_caps = retry_caps
        self.first_text = first_text or (
            "アルファ版、 カスタムコマンド、 トークショップ、 カスタムアイテム、 カスタ"
        )

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        capped = len(self.calls) == 1 or self.retry_caps
        text = self.first_text if capped else "この機能を使います。"
        return {"text": text, "segments": [{
            "seek": 0, "text": text, "start": 0.0, "end": 1.0,
            "tokens": [1] * (kwargs["sample_len"] if capped else 5),
            "no_speech_prob": 0.05, "avg_logprob": -0.2, "compression_ratio": 1.1,
            "words": [{"word": text, "start": 0.0, "end": 1.0}],
        }]}


def test_whisper_recovers_capped_decode_without_glossary_once(monkeypatch):
    fake = _CappedThenRecoveredWhisper()
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    fn, _ = lt._make_transcribe_fn({
        "model": "m", "language": "ja", "prompt": "用語集", "display_mode": "word",
    })
    result = fn(np.zeros(16000, dtype=np.float32))
    assert result["text"] == "この機能を使います。"
    assert result["_token_cap_reached"] is False
    assert len(fake.calls) == 2
    assert fake.calls[0]["initial_prompt"] == "用語集"
    assert fake.calls[1] == dict(fake.calls[0], initial_prompt=None)


def test_whisper_retry_remains_bounded_when_it_also_hits_cap(monkeypatch):
    fake = _CappedThenRecoveredWhisper(retry_caps=True)
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    fn, _ = lt._make_transcribe_fn({"model": "m", "language": "ja", "prompt": "用語集"})
    result = fn(np.zeros(16000, dtype=np.float32))
    assert result["_token_cap_reached"] is True
    assert len(fake.calls) == 2
    assert fake.calls[0]["sample_len"] == fake.calls[1]["sample_len"] == 48


def test_whisper_does_not_retry_identical_promptless_decode(monkeypatch):
    fake = _CappedThenRecoveredWhisper()
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    fn, _ = lt._make_transcribe_fn({"model": "m", "language": "ja", "prompt": None})
    assert fn(np.zeros(16000, dtype=np.float32))["_token_cap_reached"] is True
    assert len(fake.calls) == 1


def test_whisper_does_not_retry_already_filtered_loop(monkeypatch):
    fake = _CappedThenRecoveredWhisper(first_text="サイト、" * 20)
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    fn, _ = lt._make_transcribe_fn({"model": "m", "language": "ja", "prompt": "用語集"})
    result = fn(np.zeros(16000, dtype=np.float32))
    assert not lt.clean_transcription(result)
    assert len(fake.calls) == 1


def test_worker_publishes_only_recovered_text(tmp_path, monkeypatch, capsys):
    fake = _CappedThenRecoveredWhisper()
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    cfg, jsonl_path, md_path = _worker_cfg(tmp_path)
    cfg.update(prompt="用語集", display_mode="word")
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(sr, sr, freq=300.0, amp=0.2))
    q.put(_noise(sr, amp=0.001, seed=61))
    q.put(None)
    display_q = _FakeDisplayQueue()
    lt._worker(q, cfg, display_q)

    captured = capsys.readouterr()
    assert "token cap" not in captured.err
    for output in (captured.out, jsonl_path.read_text(encoding="utf-8"),
                   md_path.read_text(encoding="utf-8")):
        assert "この機能を使います。" in output
        assert "カスタム" not in output
    assert display_q.events == [("chunk", "この機能を使います。"), ("eos",)]


def test_worker_stays_quiet_when_a_dropped_segment_hit_the_cap(tmp_path, monkeypatch, capsys):
    """A loop hitting the cap is the cap working as intended -- and it gets
    dropped -- so it must not cry wolf on every noise blip."""
    looped = "サイト、" * 20
    result = {"text": looped, "segments": [
        {"text": looped, "no_speech_prob": 0.05, "avg_logprob": -0.2,
         "compression_ratio": 1.1, "tokens": list(range(10_000))}]}
    monkeypatch.setattr(lt, "_make_transcribe_fn", lambda cfg: (lambda audio: result, "whisper"))
    cfg, _, _ = _worker_cfg(tmp_path)
    sr = cfg["sr"]
    q = queue.Queue()
    q.put(_tone(int(1.0 * sr), sr, freq=300.0, amp=0.2))
    q.put(_noise(int(1.0 * sr), amp=0.001, seed=62))
    q.put(None)

    lt._worker(q, cfg)

    assert "token cap" not in capsys.readouterr().err


def test_live_transcriber_silence_sec_defaults_to_the_module_constant(tmp_path):
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path))
    assert t.cfg["silence_sec"] == lt.SILENCE_SEC_DEFAULT


def test_live_transcriber_carries_explicit_silence_sec(tmp_path):
    """How long a speaker must stay quiet before their segment is cut and sent
    -- the fixed part of the lag between finishing a sentence and seeing it."""
    t = lt.LiveTranscriber(16000, "m", "ja", str(tmp_path), silence_sec=1.5)
    assert t.cfg["silence_sec"] == 1.5


def test_live_meter_reads_capture_even_when_transcription_queue_is_full(tmp_path):
    display = queue.Queue()
    t = lt.LiveTranscriber(16000, "fake", "ja", str(tmp_path), display_queue=display)
    t._queue = queue.Queue(maxsize=1)
    t._queue.put_nowait("backlog")
    before = time.monotonic()
    t.feed(np.array([.1, -.1, .1, -.1], dtype=np.float32))
    captured, rms = t.get_audio_level()
    assert before <= captured <= time.monotonic()
    assert np.isclose(rms, .1)
    assert display.empty()  # Meter updates cannot crowd out caption events.
    assert t.dropped == 1
    t.feed(np.zeros(256, dtype=np.float32))
    assert t.get_audio_level()[1] == 0


def test_no_meter_work_without_captions(tmp_path):
    t = lt.LiveTranscriber(16000, "fake", "ja", str(tmp_path))
    t.feed(np.ones(256, dtype=np.float32))
    assert t.get_audio_level() is None
