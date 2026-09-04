"""Tests for alwayswhisper.pipeline.transcribe_file: audio extraction, backend
dispatch, hallucination stripping, and language-aware segmentation.

extract_audio and create_backend are monkeypatched on alwayswhisper.pipeline's
namespace so no real ffmpeg or Whisper call happens; the word segmenters are
monkeypatched too for the dispatch/fallback tests, so language routing and
max_chars/min_chars fallback semantics can be asserted directly rather than
inferred from real segmentation output.
"""

import json
from pathlib import Path

import pytest

import alwayswhisper.pipeline as pipeline
from alwayswhisper.core.srt_parser import SrtFile


def _input_video(tmp_path) -> Path:
    path = tmp_path / "input.mp4"
    path.write_bytes(b"fake-video-bytes")
    return path


def _patch_extract_audio(monkeypatch):
    def fake_extract_audio(input_path, output_path):
        Path(output_path).write_bytes(b"fake-wav-bytes")

    monkeypatch.setattr(pipeline, "extract_audio", fake_extract_audio)


class _FakeBackend:
    def __init__(self, words):
        self._words = words
        self.calls: list[dict] = []

    def transcribe_words(self, audio_path, language=None, prompt=None):
        self.calls.append({"audio_path": audio_path, "language": language, "prompt": prompt})
        return list(self._words)


def _patch_create_backend(monkeypatch, words):
    backend = _FakeBackend(words)
    monkeypatch.setattr(pipeline, "create_backend", lambda cfg: backend)
    return backend


class TestWordsJsonOutput:
    def test_words_json_written_with_three_key_schema_and_unicode(self, tmp_path, monkeypatch):
        _patch_extract_audio(monkeypatch)
        words = [{"word": "こんにちは", "start": 0.0, "end": 0.5}]
        _patch_create_backend(monkeypatch, words)

        cfg = {"transcribe": {"language": "ja", "strip_phrases": []}, "srt": {}}
        result = pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        raw_text = result["words_path"].read_text(encoding="utf-8")
        assert "こんにちは" in raw_text  # ensure_ascii=False -- not \u-escaped
        assert "\\u" not in raw_text
        loaded = json.loads(raw_text)
        assert loaded == words
        for w in loaded:
            assert set(w.keys()) == {"word", "start", "end"}

    def test_return_dict_has_paths_and_counts(self, tmp_path, monkeypatch):
        _patch_extract_audio(monkeypatch)
        words = [
            {"word": "hello", "start": 0.0, "end": 0.4},
            {"word": "world.", "start": 0.4, "end": 0.9},
        ]
        _patch_create_backend(monkeypatch, words)

        cfg = {"transcribe": {"language": "en", "strip_phrases": []}, "srt": {}}
        result = pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        assert result["words_path"].exists()
        assert result["srt_path"].exists()
        assert result["word_count"] == 2
        assert result["entry_count"] >= 1
        assert result["hallucination_spans_removed"] == 0


class TestHallucinationStripping:
    def test_stripped_when_language_ja_and_strip_phrases_absent(self, tmp_path, monkeypatch):
        _patch_extract_audio(monkeypatch)
        words = [
            {"word": "お話", "start": 0.0, "end": 0.5},
            {"word": "ご視聴ありがとうございました", "start": 0.5, "end": 1.5},
        ]
        _patch_create_backend(monkeypatch, words)

        # strip_phrases key absent entirely -> None -> ja default applies.
        cfg = {"transcribe": {"language": "ja"}, "srt": {}}
        result = pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        loaded = json.loads(result["words_path"].read_text(encoding="utf-8"))
        assert loaded == [{"word": "お話", "start": 0.0, "end": 0.5}]
        assert result["hallucination_spans_removed"] == 1

    def test_not_stripped_when_language_en_and_strip_phrases_absent(self, tmp_path, monkeypatch):
        _patch_extract_audio(monkeypatch)
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "ご視聴ありがとうございました", "start": 0.5, "end": 1.5},
        ]
        _patch_create_backend(monkeypatch, words)

        # strip_phrases key absent -> None -> but language "en" -> no default
        # phrases at all (AlwaysWhisper's language-scoped i18n behavior).
        cfg = {"transcribe": {"language": "en"}, "srt": {}}
        result = pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        loaded = json.loads(result["words_path"].read_text(encoding="utf-8"))
        assert loaded == words  # untouched
        assert result["hallucination_spans_removed"] == 0


class TestSegmenterDispatch:
    def _patch_sentinel_segmenters(self, monkeypatch):
        sentinel_srt = SrtFile(entries=[])
        called = {"en": [], "ja": []}

        def fake_en(words, max_chars, min_chars):
            called["en"].append({"max_chars": max_chars, "min_chars": min_chars})
            return sentinel_srt

        def fake_ja(words, max_chars, min_chars):
            called["ja"].append({"max_chars": max_chars, "min_chars": min_chars})
            return sentinel_srt

        monkeypatch.setattr(pipeline, "words_to_srt_en", fake_en)
        monkeypatch.setattr(pipeline, "words_to_srt", fake_ja)
        return called

    def test_en_language_dispatches_to_words_to_srt_en(self, tmp_path, monkeypatch):
        _patch_extract_audio(monkeypatch)
        _patch_create_backend(monkeypatch, [{"word": "hello", "start": 0.0, "end": 0.1}])
        called = self._patch_sentinel_segmenters(monkeypatch)

        cfg = {"transcribe": {"language": "en", "strip_phrases": []}, "srt": {}}
        pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        assert len(called["en"]) == 1
        assert called["ja"] == []

    def test_ja_language_dispatches_to_words_to_srt(self, tmp_path, monkeypatch):
        _patch_extract_audio(monkeypatch)
        _patch_create_backend(monkeypatch, [{"word": "こんにちは", "start": 0.0, "end": 0.1}])
        called = self._patch_sentinel_segmenters(monkeypatch)

        cfg = {"transcribe": {"language": "ja", "strip_phrases": []}, "srt": {}}
        pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        assert len(called["ja"]) == 1
        assert called["en"] == []

    def test_zh_language_dispatches_to_words_to_srt(self, tmp_path, monkeypatch):
        # Chinese is in NO_SPACE_LANGUAGES -- same char-based segmenter as
        # Japanese (its grammar rules are JA-specific but degrade to plain
        # char-limit splitting for non-Japanese input).
        _patch_extract_audio(monkeypatch)
        _patch_create_backend(monkeypatch, [{"word": "你好", "start": 0.0, "end": 0.1}])
        called = self._patch_sentinel_segmenters(monkeypatch)

        cfg = {"transcribe": {"language": "zh", "strip_phrases": []}, "srt": {}}
        pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        assert len(called["ja"]) == 1
        assert called["en"] == []

    def test_de_language_dispatches_to_words_to_srt_en(self, tmp_path, monkeypatch):
        # German is space-delimited and NOT in NO_SPACE_LANGUAGES -- this is
        # the bug being fixed: the old dispatch was `language == "en"` vs.
        # "everything else", which wrongly ran German through the
        # Japanese-grammar segmenter.
        _patch_extract_audio(monkeypatch)
        _patch_create_backend(monkeypatch, [{"word": "Hallo", "start": 0.0, "end": 0.1}])
        called = self._patch_sentinel_segmenters(monkeypatch)

        cfg = {"transcribe": {"language": "de", "strip_phrases": []}, "srt": {}}
        pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        assert len(called["en"]) == 1
        assert called["ja"] == []

    def test_none_language_dispatches_to_words_to_srt_en(self, tmp_path, monkeypatch):
        # An unset language (None -- e.g. Whisper auto-detect with no
        # transcribe.language configured) must default to the generic
        # space-delimited segmenter, not the Japanese one.
        _patch_extract_audio(monkeypatch)
        _patch_create_backend(monkeypatch, [{"word": "hello", "start": 0.0, "end": 0.1}])
        called = self._patch_sentinel_segmenters(monkeypatch)

        cfg = {"transcribe": {"strip_phrases": []}, "srt": {}}
        pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        assert len(called["en"]) == 1
        assert called["ja"] == []

    def test_default_max_min_chars_fallback_en(self, tmp_path, monkeypatch):
        _patch_extract_audio(monkeypatch)
        _patch_create_backend(monkeypatch, [{"word": "hello", "start": 0.0, "end": 0.1}])
        called = self._patch_sentinel_segmenters(monkeypatch)

        cfg = {"transcribe": {"language": "en", "strip_phrases": []}, "srt": {}}
        pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        assert called["en"] == [{"max_chars": 42, "min_chars": 10}]

    def test_default_max_min_chars_fallback_ja(self, tmp_path, monkeypatch):
        _patch_extract_audio(monkeypatch)
        _patch_create_backend(monkeypatch, [{"word": "こんにちは", "start": 0.0, "end": 0.1}])
        called = self._patch_sentinel_segmenters(monkeypatch)

        cfg = {"transcribe": {"language": "ja", "strip_phrases": []}, "srt": {}}
        pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        assert called["ja"] == [{"max_chars": 35, "min_chars": 5}]

    def test_max_chars_zero_falls_back_or_semantics(self, tmp_path, monkeypatch):
        # max_chars uses `srt_cfg.get("max_chars") or <fallback>`, so an
        # explicit falsy value (0) still falls back -- matching s04's exact
        # semantics.
        _patch_extract_audio(monkeypatch)
        _patch_create_backend(monkeypatch, [{"word": "こんにちは", "start": 0.0, "end": 0.1}])
        called = self._patch_sentinel_segmenters(monkeypatch)

        cfg = {"transcribe": {"language": "ja", "strip_phrases": []}, "srt": {"max_chars": 0}}
        pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        assert called["ja"][0]["max_chars"] == 35

    def test_min_chars_zero_is_honored_not_fallback(self, tmp_path, monkeypatch):
        # min_chars uses plain `srt_cfg.get("min_chars", <fallback>)`, so an
        # explicit 0 is honored -- only an ABSENT key falls back.
        _patch_extract_audio(monkeypatch)
        _patch_create_backend(monkeypatch, [{"word": "こんにちは", "start": 0.0, "end": 0.1}])
        called = self._patch_sentinel_segmenters(monkeypatch)

        cfg = {"transcribe": {"language": "ja", "strip_phrases": []}, "srt": {"min_chars": 0}}
        pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        assert called["ja"][0]["min_chars"] == 0

    def test_explicit_max_min_chars_are_used(self, tmp_path, monkeypatch):
        _patch_extract_audio(monkeypatch)
        _patch_create_backend(monkeypatch, [{"word": "hello", "start": 0.0, "end": 0.1}])
        called = self._patch_sentinel_segmenters(monkeypatch)

        cfg = {
            "transcribe": {"language": "en", "strip_phrases": []},
            "srt": {"max_chars": 21, "min_chars": 3},
        }
        pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        assert called["en"] == [{"max_chars": 21, "min_chars": 3}]


class TestSegmentWordsDispatch:
    """segment_words() shares _dispatch_segmenter with transcribe_file --
    this exercises that second call site directly (no audio extraction or
    backend involved) so the language-routing fix is proven on both entry
    points, not just the one transcribe_file is tested through above.
    """

    def _patch_sentinel_segmenters(self, monkeypatch):
        sentinel_srt = SrtFile(entries=[])
        called = {"en": [], "ja": []}

        def fake_en(words, max_chars, min_chars):
            called["en"].append({"max_chars": max_chars, "min_chars": min_chars})
            return sentinel_srt

        def fake_ja(words, max_chars, min_chars):
            called["ja"].append({"max_chars": max_chars, "min_chars": min_chars})
            return sentinel_srt

        monkeypatch.setattr(pipeline, "words_to_srt_en", fake_en)
        monkeypatch.setattr(pipeline, "words_to_srt", fake_ja)
        return called

    def _words_json(self, tmp_path) -> Path:
        words_json = tmp_path / "transcript_words.json"
        words_json.write_text(
            json.dumps([{"word": "hello", "start": 0.0, "end": 0.1}]), encoding="utf-8"
        )
        return words_json

    @pytest.mark.parametrize(
        "language, expected_bucket",
        [
            ("ja", "ja"),
            ("zh", "ja"),
            ("en", "en"),
            ("de", "en"),
            (None, "en"),
        ],
    )
    def test_language_dispatch_matrix(self, tmp_path, monkeypatch, language, expected_bucket):
        called = self._patch_sentinel_segmenters(monkeypatch)
        cfg = {"transcribe": {"language": language}, "srt": {}}

        pipeline.segment_words(self._words_json(tmp_path), tmp_path / "out.srt", cfg)

        other_bucket = "en" if expected_bucket == "ja" else "ja"
        assert len(called[expected_bucket]) == 1
        assert called[other_bucket] == []


class TestEndToEndSegmentation:
    def test_transcript_raw_srt_written_with_real_segmenter(self, tmp_path, monkeypatch):
        _patch_extract_audio(monkeypatch)
        words = [
            {"word": "hello", "start": 0.0, "end": 0.4},
            {"word": "world.", "start": 0.4, "end": 0.9},
        ]
        _patch_create_backend(monkeypatch, words)

        cfg = {"transcribe": {"language": "en", "strip_phrases": []}, "srt": {}}
        result = pipeline.transcribe_file(_input_video(tmp_path), tmp_path / "work", cfg)

        srt = SrtFile.from_file(result["srt_path"])
        assert len(srt.entries) >= 1
        assert "hello" in srt.entries[0].text


class TestAudioCleanup:
    def test_wav_deleted_after_success(self, tmp_path, monkeypatch):
        _patch_extract_audio(monkeypatch)
        _patch_create_backend(monkeypatch, [{"word": "x", "start": 0.0, "end": 0.1}])

        workdir = tmp_path / "work"
        cfg = {"transcribe": {"language": "ja", "strip_phrases": []}, "srt": {}}
        pipeline.transcribe_file(_input_video(tmp_path), workdir, cfg)

        assert not (workdir / "audio_for_whisper.wav").exists()

    def test_wav_deleted_even_when_backend_raises(self, tmp_path, monkeypatch):
        _patch_extract_audio(monkeypatch)

        class _RaisingBackend:
            def transcribe_words(self, *a, **k):
                raise RuntimeError("boom")

        monkeypatch.setattr(pipeline, "create_backend", lambda cfg: _RaisingBackend())

        workdir = tmp_path / "work"
        cfg = {"transcribe": {"language": "ja"}, "srt": {}}

        with pytest.raises(RuntimeError, match="boom"):
            pipeline.transcribe_file(_input_video(tmp_path), workdir, cfg)

        assert not (workdir / "audio_for_whisper.wav").exists()
