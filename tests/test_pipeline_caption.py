"""Tests for alwayswhisper.pipeline.caption_video: AV QA gating, realignment,
gap-filling, and burn-mode dispatch.

realign_srt_by_words and SrtFile.fill_gaps are real (pure Python,
deterministic, no I/O) so these tests exercise the actual realign/gap-fill
code paths; the burn functions (add_captions_fast/add_captions_to_video),
run_av_qa, and create_backend are monkeypatched on alwayswhisper.pipeline's
namespace so no ffmpeg/MoviePy/Whisper call happens.
"""

import json
from pathlib import Path

import pytest

import alwayswhisper.pipeline as pipeline


def _write_video(tmp_path) -> Path:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake-video-bytes")
    return video


def _write_srt(tmp_path, name="synced.srt") -> Path:
    srt_path = tmp_path / name
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nテストです\n", encoding="utf-8"
    )
    return srt_path


def _patch_burn(monkeypatch):
    calls = {"fast": [], "standard": []}
    monkeypatch.setattr(
        pipeline,
        "add_captions_fast",
        lambda video, srt, out_path, style: calls["fast"].append(
            {"video": video, "srt": srt, "out_path": out_path, "style": style}
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "add_captions_to_video",
        lambda video, srt, out_path, style: calls["standard"].append(
            {"video": video, "srt": srt, "out_path": out_path, "style": style}
        ),
    )
    return calls


_PASS_REPORT = {
    "passed": True, "avg_ratio": 1.0, "samples": [],
    "sampled": 0, "eligible": 0, "min_ratio": 0.5,
}
_FAIL_REPORT = {
    "passed": False, "avg_ratio": 0.1,
    "samples": [{
        "index": 1, "start": "00:00:00,000", "end": "00:00:02,000",
        "srt_text": "a", "asr_text": "b", "ratio": 0.1, "ok": False,
    }],
    "sampled": 1, "eligible": 1, "min_ratio": 0.5,
}


class _FakeBackend:
    def transcribe_text(self, audio_path, language=None):
        return "unused"


def _patch_create_backend(monkeypatch, backend=None):
    backend = backend or _FakeBackend()
    monkeypatch.setattr(pipeline, "create_backend", lambda cfg: backend)
    return backend


class TestQaGating:
    def test_qa_pass_allows_burn_and_writes_report(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        out_path = tmp_path / "out.mp4"
        calls = _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        monkeypatch.setattr(pipeline, "run_av_qa", lambda *a, **k: _PASS_REPORT)

        report = pipeline.caption_video(
            video, srt_path, out_path, {"qa": {"enabled": True}}
        )

        assert len(calls["standard"]) == 1
        report_path = out_path.parent / "qa_report.json"
        assert report_path.exists()
        assert json.loads(report_path.read_text())["passed"] is True
        assert report["qa_report"]["passed"] is True

    def test_qa_fail_blocks_burn_but_still_writes_report(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        out_path = tmp_path / "out.mp4"
        calls = _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        monkeypatch.setattr(pipeline, "run_av_qa", lambda *a, **k: _FAIL_REPORT)

        with pytest.raises(RuntimeError, match="AV QA failed"):
            pipeline.caption_video(video, srt_path, out_path, {"qa": {"enabled": True}})

        assert calls["standard"] == []
        assert calls["fast"] == []
        report_path = out_path.parent / "qa_report.json"
        assert report_path.exists()
        assert json.loads(report_path.read_text())["passed"] is False

    def test_qa_disabled_skips_check_and_never_creates_backend(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        out_path = tmp_path / "out.mp4"
        calls = _patch_burn(monkeypatch)

        def fail_if_backend_created(cfg):
            raise AssertionError(
                "create_backend should not be called when qa.enabled is false"
            )

        monkeypatch.setattr(pipeline, "create_backend", fail_if_backend_created)

        def fail_if_qa_called(*a, **k):
            raise AssertionError("run_av_qa should not be called when qa.enabled is false")

        monkeypatch.setattr(pipeline, "run_av_qa", fail_if_qa_called)

        pipeline.caption_video(video, srt_path, out_path, {"qa": {"enabled": False}})

        assert len(calls["standard"]) == 1
        assert not (out_path.parent / "qa_report.json").exists()

    def test_qa_defaults_to_enabled_when_qa_key_missing(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        out_path = tmp_path / "out.mp4"
        calls = _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        called = {"n": 0}

        def fake_qa(*a, **k):
            called["n"] += 1
            return _PASS_REPORT

        monkeypatch.setattr(pipeline, "run_av_qa", fake_qa)

        pipeline.caption_video(video, srt_path, out_path, {})  # no "qa" key at all

        assert called["n"] == 1
        assert len(calls["standard"]) == 1

    def test_qa_fail_message_reports_per_sample_count_not_average(self, tmp_path, monkeypatch):
        # The pass/fail decision is per-sample (every sample must
        # individually meet min_ratio) -- avg_ratio can sit comfortably
        # above min_ratio while one sample still fails. The error message
        # must say so, not imply an average-vs-threshold comparison.
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        out_path = tmp_path / "out.mp4"
        _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        mixed_report = {
            "passed": False,
            "avg_ratio": 0.7167,  # above min_ratio(0.5) despite one failing sample
            "samples": [
                {"index": 1, "start": "00:00:00,000", "end": "00:00:02,000",
                 "srt_text": "a", "asr_text": "a", "ratio": 0.95, "ok": True},
                {"index": 2, "start": "00:00:03,000", "end": "00:00:05,000",
                 "srt_text": "b", "asr_text": "x", "ratio": 0.3, "ok": False},
                {"index": 3, "start": "00:00:06,000", "end": "00:00:08,000",
                 "srt_text": "c", "asr_text": "c", "ratio": 0.9, "ok": True},
            ],
            "sampled": 3, "eligible": 3, "min_ratio": 0.5,
        }
        monkeypatch.setattr(pipeline, "run_av_qa", lambda *a, **k: mixed_report)

        with pytest.raises(RuntimeError) as exc_info:
            pipeline.caption_video(video, srt_path, out_path, {"qa": {"enabled": True}})

        message = str(exc_info.value)
        assert "1/3 sample(s) below min_ratio=0.5" in message
        assert "avg_ratio=0.72" in message

    def test_qa_config_values_are_passed_through(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        out_path = tmp_path / "out.mp4"
        _patch_burn(monkeypatch)
        backend = _patch_create_backend(monkeypatch)
        captured = {}

        def fake_qa(
            video_path, entries, *,
            samples, min_ratio, pad_ms, min_entry_ms, language, transcriber, strip_phrases,
        ):
            captured.update(
                samples=samples, min_ratio=min_ratio, pad_ms=pad_ms,
                min_entry_ms=min_entry_ms, language=language,
                transcriber=transcriber, strip_phrases=strip_phrases,
            )
            return {**_PASS_REPORT, "min_ratio": min_ratio}

        monkeypatch.setattr(pipeline, "run_av_qa", fake_qa)

        pipeline.caption_video(
            video, srt_path, out_path,
            {
                "qa": {
                    "enabled": True, "samples": 3, "min_ratio": 0.7,
                    "pad_ms": 200, "min_entry_ms": 400,
                },
                "transcribe": {"language": "en"},
            },
        )

        assert captured["samples"] == 3
        assert captured["min_ratio"] == 0.7
        assert captured["pad_ms"] == 200
        assert captured["min_entry_ms"] == 400
        assert captured["language"] == "en"
        assert captured["transcriber"] == backend.transcribe_text
        assert captured["strip_phrases"] == ()  # language "en" -> no default strip phrases


class TestRealign:
    def test_realign_skipped_when_flag_false(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        words_path = tmp_path / "transcript_words.json"
        words_path.write_text(
            json.dumps([{"word": "テストです", "start": 0.0, "end": 2.0}], ensure_ascii=False),
            encoding="utf-8",
        )
        out_path = tmp_path / "out.mp4"
        _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        monkeypatch.setattr(pipeline, "run_av_qa", lambda *a, **k: _PASS_REPORT)

        report = pipeline.caption_video(
            video, srt_path, out_path,
            {"qa": {"enabled": True}, "caption": {"realign": False}},
            words_json=words_path,
        )

        assert "realign_stats" not in report
        assert not out_path.with_suffix(".realigned.srt").exists()

    def test_realign_skipped_when_words_json_not_given(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        out_path = tmp_path / "out.mp4"
        _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        monkeypatch.setattr(pipeline, "run_av_qa", lambda *a, **k: _PASS_REPORT)

        report = pipeline.caption_video(
            video, srt_path, out_path,
            {"qa": {"enabled": True}, "caption": {"realign": True}},
        )

        assert "realign_stats" not in report

    def test_realign_skipped_when_words_json_path_does_not_exist(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        out_path = tmp_path / "out.mp4"
        _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        monkeypatch.setattr(pipeline, "run_av_qa", lambda *a, **k: _PASS_REPORT)

        report = pipeline.caption_video(
            video, srt_path, out_path,
            {"qa": {"enabled": True}, "caption": {"realign": True}},
            words_json=tmp_path / "does_not_exist.json",
        )

        assert "realign_stats" not in report

    def test_realign_runs_when_flag_true_and_words_given(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        words_path = tmp_path / "transcript_words.json"
        words_path.write_text(
            json.dumps([{"word": "テストです", "start": 0.0, "end": 2.0}], ensure_ascii=False),
            encoding="utf-8",
        )
        out_path = tmp_path / "out.mp4"
        _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        monkeypatch.setattr(pipeline, "run_av_qa", lambda *a, **k: _PASS_REPORT)

        report = pipeline.caption_video(
            video, srt_path, out_path,
            {"qa": {"enabled": True}, "caption": {"realign": True}},
            words_json=words_path,
        )

        assert "realign_stats" in report
        realigned = out_path.with_suffix(".realigned.srt")
        assert realigned.exists()
        assert report["realigned_srt_path"] == realigned


class TestFillGaps:
    def test_fill_gaps_applied_before_burn(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = tmp_path / "synced.srt"
        srt_path.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nA\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\nB\n",
            encoding="utf-8",
        )
        out_path = tmp_path / "out.mp4"
        calls = _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        monkeypatch.setattr(pipeline, "run_av_qa", lambda *a, **k: _PASS_REPORT)

        pipeline.caption_video(video, srt_path, out_path, {"qa": {"enabled": True}})

        burned_srt = calls["standard"][0]["srt"]
        # Gap between entry 1 (ends 1000ms) and entry 2 (starts 2000ms) is
        # filled down to a 50ms minimum gap: entry 1 should now end at 1950ms.
        assert burned_srt.entries[0].end.to_ms() == 1950

    def test_qa_sees_unfilled_timing_not_gap_filled_timing(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = tmp_path / "synced.srt"
        srt_path.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nA\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\nB\n",
            encoding="utf-8",
        )
        out_path = tmp_path / "out.mp4"
        _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        seen = {}

        def fake_qa(video_path, entries, **kwargs):
            seen["end_ms"] = entries[0].end.to_ms()
            return _PASS_REPORT

        monkeypatch.setattr(pipeline, "run_av_qa", fake_qa)

        pipeline.caption_video(video, srt_path, out_path, {"qa": {"enabled": True}})

        assert seen["end_ms"] == 1000  # unfilled at QA time


class TestBurnModeDispatch:
    def test_fast_mode_calls_add_captions_fast(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        out_path = tmp_path / "out.mp4"
        calls = _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        monkeypatch.setattr(pipeline, "run_av_qa", lambda *a, **k: _PASS_REPORT)

        pipeline.caption_video(
            video, srt_path, out_path,
            {"qa": {"enabled": True}, "caption": {"fast_mode": True}},
        )

        assert len(calls["fast"]) == 1
        assert calls["standard"] == []

    def test_standard_mode_is_default(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        out_path = tmp_path / "out.mp4"
        calls = _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        monkeypatch.setattr(pipeline, "run_av_qa", lambda *a, **k: _PASS_REPORT)

        pipeline.caption_video(video, srt_path, out_path, {"qa": {"enabled": True}})

        assert len(calls["standard"]) == 1
        assert calls["fast"] == []


class TestStyleLoading:
    def test_default_style_is_loaded_and_passed_to_burn(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        out_path = tmp_path / "out.mp4"
        calls = _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        monkeypatch.setattr(pipeline, "run_av_qa", lambda *a, **k: _PASS_REPORT)

        pipeline.caption_video(video, srt_path, out_path, {"qa": {"enabled": True}})

        style = calls["standard"][0]["style"]
        assert style["text"]["font_size"] == 64

    def test_named_style_en_is_loaded(self, tmp_path, monkeypatch):
        video = _write_video(tmp_path)
        srt_path = _write_srt(tmp_path)
        out_path = tmp_path / "out.mp4"
        calls = _patch_burn(monkeypatch)
        _patch_create_backend(monkeypatch)
        monkeypatch.setattr(pipeline, "run_av_qa", lambda *a, **k: _PASS_REPORT)

        pipeline.caption_video(
            video, srt_path, out_path,
            {"qa": {"enabled": True}, "caption": {"style": "en"}},
        )

        style = calls["standard"][0]["style"]
        assert style["text"]["font_size"] == 44
