"""Tests for alwayswhisper.core.av_qa.run_av_qa.

extract_audio_clip is always monkeypatched on the av_qa module namespace so
no real ffmpeg call happens; transcriber is injected per-test (hermetic, no
network / Whisper call).
"""

import random
from pathlib import Path

import pytest

import alwayswhisper.core.av_qa as av_qa
from alwayswhisper.core.av_qa import _normalize, run_av_qa
from alwayswhisper.core.srt_parser import SrtEntry
from alwayswhisper.core.timestamp import Timestamp


def _entry(index, start_ms, end_ms, text):
    return SrtEntry(
        index=index,
        start=Timestamp.from_ms(start_ms),
        end=Timestamp.from_ms(end_ms),
        text=text,
    )


def _noop_extract(monkeypatch):
    """Stub extract_audio_clip: writes nothing meaningful (transcriber is
    also faked in these tests, so clip content never actually matters)."""

    def fake_extract(video_path, output_wav, start_sec, end_sec):
        Path(output_wav).write_bytes(b"")

    monkeypatch.setattr(av_qa, "extract_audio_clip", fake_extract)


def _expected_window(entry, pad_ms):
    start_sec = max(0.0, entry.start.to_ms() / 1000 - pad_ms / 1000)
    end_sec = max(0.0, entry.end.to_ms() / 1000 + pad_ms / 1000)
    return (round(start_sec, 3), round(end_sec, 3))


def _marker_extract(monkeypatch, window_to_text: dict):
    """extract_audio_clip stub that writes the expected entry's text into
    the wav file so an "echo" transcriber can read back which entry it's
    scoring -- correlates entry -> extraction -> transcription without
    changing the transcriber(wav_path) signature.
    """

    def fake_extract(video_path, output_wav, start_sec, end_sec):
        key = (round(start_sec, 3), round(end_sec, 3))
        text = window_to_text.get(key, "")
        Path(output_wav).write_text(text, encoding="utf-8")

    monkeypatch.setattr(av_qa, "extract_audio_clip", fake_extract)


def _echo_transcriber(wav_path) -> str:
    return Path(wav_path).read_text(encoding="utf-8")


class TestRunAvQaMatching:
    def test_matching_transcriber_passes_with_ratio_near_one(self, monkeypatch):
        entries = [
            _entry(1, 0, 2000, "今日はいい天気ですね"),
            _entry(2, 3000, 5000, "ありがとうございます"),
        ]
        pad_ms = 300
        _marker_extract(monkeypatch, {_expected_window(e, pad_ms): e.text for e in entries})

        report = run_av_qa(
            "video.mp4",
            entries,
            samples=2,
            min_ratio=0.5,
            pad_ms=pad_ms,
            rng=random.Random(0),
            transcriber=_echo_transcriber,
        )

        assert report["passed"] is True
        assert report["sampled"] == 2
        assert report["eligible"] == 2
        assert report["avg_ratio"] == pytest.approx(1.0)
        for s in report["samples"]:
            assert s["ratio"] == pytest.approx(1.0)
            assert s["ok"] is True


class TestRunAvQaMismatch:
    def test_wrong_text_transcriber_fails(self, monkeypatch):
        _noop_extract(monkeypatch)
        entries = [_entry(1, 0, 2000, "今日はいい天気ですね")]

        report = run_av_qa(
            "video.mp4",
            entries,
            samples=1,
            min_ratio=0.5,
            rng=random.Random(0),
            transcriber=lambda wav_path: "abcdefg12345",
        )

        assert report["passed"] is False
        assert report["samples"][0]["ok"] is False

    def test_empty_asr_gives_ratio_zero(self, monkeypatch):
        _noop_extract(monkeypatch)
        entries = [_entry(1, 0, 2000, "今日はいい天気ですね")]

        report = run_av_qa(
            "video.mp4",
            entries,
            samples=1,
            min_ratio=0.5,
            rng=random.Random(0),
            transcriber=lambda wav_path: "",
        )

        assert report["samples"][0]["ratio"] == 0.0
        assert report["samples"][0]["ok"] is False
        assert report["passed"] is False


class TestSampleCountClamping:
    def test_negative_samples_does_not_crash_and_yields_vacuous_pass(self, monkeypatch):
        # FIX 4: k = min(samples, len(eligible)) could go negative when
        # samples < 0 (e.g. a misconfigured --samples), and
        # random.sample(pop, negative_k) raises ValueError. k must clamp to
        # >= 0.
        _noop_extract(monkeypatch)
        entries = [_entry(1, 0, 2000, "今日はいい天気ですね")]

        report = run_av_qa(
            "video.mp4",
            entries,
            samples=-5,
            rng=random.Random(0),
            transcriber=lambda wav_path: "x",
        )

        assert report["sampled"] == 0
        assert report["samples"] == []
        assert report["passed"] is True
        assert report["eligible"] == 1  # eligibility itself is unaffected

    def test_zero_samples_is_a_no_op(self, monkeypatch):
        _noop_extract(monkeypatch)
        entries = [_entry(1, 0, 2000, "今日はいい天気ですね")]

        report = run_av_qa(
            "video.mp4", entries, samples=0, rng=random.Random(0),
            transcriber=lambda wav_path: "x",
        )

        assert report["sampled"] == 0
        assert report["passed"] is True


class TestEligibility:
    def test_short_entries_excluded_and_oversampling_handled(self, monkeypatch):
        _noop_extract(monkeypatch)
        entries = [
            _entry(1, 0, 400, "短すぎる"),  # 400ms < default min_entry_ms(500) -> excluded
            _entry(2, 1000, 1600, "ちょうどいい長さです"),  # 600ms -> eligible
            _entry(3, 2000, 3000, "   "),  # whitespace-only text -> excluded
        ]

        report = run_av_qa(
            "video.mp4",
            entries,
            samples=10,  # far more than eligible
            rng=random.Random(0),
            transcriber=lambda wav_path: "ちょうどいい長さです",
        )

        assert report["eligible"] == 1
        assert report["sampled"] == 1
        assert len(report["samples"]) == 1
        assert report["samples"][0]["index"] == 2

    def test_no_eligible_entries_returns_vacuous_pass(self, monkeypatch):
        _noop_extract(monkeypatch)
        entries = [_entry(1, 0, 100, "短い")]

        report = run_av_qa("video.mp4", entries, samples=5, rng=random.Random(0))

        assert report["eligible"] == 0
        assert report["sampled"] == 0
        assert report["samples"] == []
        assert report["passed"] is True


class TestSeededRng:
    def test_same_seed_gives_same_sample_selection(self, monkeypatch):
        _noop_extract(monkeypatch)
        entries = [_entry(i, i * 2000, i * 2000 + 1000, f"テキスト{i}") for i in range(1, 6)]

        def run_once():
            return run_av_qa(
                "video.mp4",
                entries,
                samples=2,
                rng=random.Random(42),
                transcriber=lambda wav_path: "x",
            )

        r1 = run_once()
        r2 = run_once()
        assert [s["index"] for s in r1["samples"]] == [s["index"] for s in r2["samples"]]


class TestNormalize:
    def test_fullwidth_and_halfwidth_ascii_converge(self):
        assert _normalize("ABC") == _normalize("ＡＢＣ")

    def test_japanese_punctuation_and_whitespace_stripped(self):
        assert _normalize("今日は、いい天気ですね。") == _normalize("今日はいい天気ですね")

    def test_ascii_punctuation_and_whitespace_stripped(self):
        assert _normalize("hello, world!") == _normalize("helloworld")

    def test_long_vowel_mark_is_preserved(self):
        assert "ー" in _normalize("スーパー")
        assert _normalize("スーパー") == "すーぱー" or _normalize("スーパー") == "スーパー".lower()

    def test_normalization_end_to_end_ignores_cosmetic_differences(self, monkeypatch):
        _noop_extract(monkeypatch)
        entries = [_entry(1, 0, 2000, "今日は、いい天気ですね。")]

        report = run_av_qa(
            "video.mp4",
            entries,
            samples=1,
            min_ratio=0.99,
            rng=random.Random(0),
            transcriber=lambda wav_path: "今日はいい天気ですね",
        )

        assert report["passed"] is True
        assert report["samples"][0]["ratio"] == pytest.approx(1.0)


class TestTranscriberRequired:
    """New in AlwaysWhisper: av_qa has no built-in transcription backend, so a
    transcriber callable is required-in-practice. It's still an optional
    keyword (default None) so a call that never needs to sample anything
    doesn't need to supply one -- see test_no_eligible_entries_returns_
    vacuous_pass and test_zero_samples_is_a_no_op above, which omit or
    never use it. But as soon as there's something to sample, omitting it
    is a caller error, not a silent no-op.
    """

    def test_none_transcriber_raises_when_sampling_is_needed(self, monkeypatch):
        _noop_extract(monkeypatch)
        entries = [_entry(1, 0, 2000, "今日はいい天気ですね")]

        with pytest.raises(ValueError):
            run_av_qa("video.mp4", entries, samples=1, rng=random.Random(0))
