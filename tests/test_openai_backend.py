"""Tests for alwayswhisper.backends.openai_backend.OpenAIBackend.

The real `openai` and `pydub` packages are not installed in this
environment (they live behind the `[api]` extra) -- which is itself part of
what's under test: constructing OpenAIBackend() and calling its methods
with a pre-set fake client must never require them. Only the "openai
package missing" and "API key missing" tests need a *minimal* fake `openai`
module in sys.modules, so `_get_client()` can get past the `import openai`
step and reach the code actually under test there.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import alwayswhisper.backends.openai_backend as openai_backend_module
from alwayswhisper.backends.openai_backend import OpenAIBackend


class _FakeTranscriptions:
    def __init__(self, response_factory):
        self.calls: list[dict] = []
        self._response_factory = response_factory

    def create(self, file, **kwargs):
        self.calls.append(kwargs)
        return self._response_factory(kwargs)


class _FakeClient:
    def __init__(self, response_factory):
        self.transcriptions = _FakeTranscriptions(response_factory)
        self.audio = SimpleNamespace(transcriptions=self.transcriptions)


def _words_response(kwargs):
    return SimpleNamespace(
        words=[
            SimpleNamespace(word="Hello", start=0.0, end=0.5),
            SimpleNamespace(word="world", start=0.5, end=1.0),
        ]
    )


def _text_response(kwargs):
    return SimpleNamespace(text="hello world")


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"fake-audio-bytes")
    return path


class TestTranscribeWordsSinglePath:
    def test_returns_three_key_dicts(self, audio_file):
        backend = OpenAIBackend()
        backend._client = _FakeClient(_words_response)

        words = backend.transcribe_words(audio_file, language="en", prompt="glossary")

        assert words == [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        for w in words:
            assert set(w.keys()) == {"word", "start", "end"}

    def test_sends_verbose_json_and_word_granularity(self, audio_file):
        backend = OpenAIBackend()
        client = _FakeClient(_words_response)
        backend._client = client

        backend.transcribe_words(audio_file, language="ja", prompt="term1, term2")

        call = client.transcriptions.calls[0]
        assert call["model"] == "whisper-1"
        assert call["response_format"] == "verbose_json"
        assert call["timestamp_granularities"] == ["word"]
        assert call["language"] == "ja"
        assert call["prompt"] == "term1, term2"

    def test_no_prompt_omits_prompt_kwarg(self, audio_file):
        backend = OpenAIBackend()
        client = _FakeClient(_words_response)
        backend._client = client

        backend.transcribe_words(audio_file, language="ja", prompt=None)

        assert "prompt" not in client.transcriptions.calls[0]


class TestTranscribeText:
    def test_returns_plain_text(self, audio_file):
        backend = OpenAIBackend()
        backend._client = _FakeClient(_text_response)

        text = backend.transcribe_text(audio_file, language="en")

        assert text == "hello world"

    def test_sends_no_prompt_and_no_word_timestamp_params(self, audio_file):
        backend = OpenAIBackend()
        client = _FakeClient(_text_response)
        backend._client = client

        backend.transcribe_text(audio_file, language="en")

        call = client.transcriptions.calls[0]
        assert call["model"] == "whisper-1"
        assert call["response_format"] == "json"
        assert "prompt" not in call
        assert "timestamp_granularities" not in call

    def test_language_none_omits_language_kwarg(self, audio_file):
        backend = OpenAIBackend()
        client = _FakeClient(_text_response)
        backend._client = client

        backend.transcribe_text(audio_file, language=None)

        assert "language" not in client.transcriptions.calls[0]


class TestChunkedWordsPath:
    def test_offsets_timestamps_across_chunks_and_cleans_up(self, audio_file, monkeypatch):
        # Force the chunked path without needing a real 24MB+ fixture: shrink
        # the module's chunk threshold below the tiny test file's size.
        monkeypatch.setattr(openai_backend_module, "MAX_FILE_SIZE", 4)

        class _FakeAudioSegment:
            def __init__(self, duration_ms=1000):
                self._duration_ms = duration_ms

            @classmethod
            def from_file(cls, path):
                return cls(duration_ms=1000)

            def __len__(self):
                return self._duration_ms

            def __getitem__(self, item):
                return self

            def export(self, path, format="mp3"):
                Path(path).write_bytes(b"fake-mp3")

        monkeypatch.setitem(
            sys.modules, "pydub", SimpleNamespace(AudioSegment=_FakeAudioSegment)
        )

        # Every chunk "transcribes" to the same relative word (0.0-0.5s);
        # the offset math must shift each chunk's copy onto the full
        # timeline, so later chunks must not still read as 0.0/0.5.
        backend = OpenAIBackend()
        backend._client = _FakeClient(_words_response)

        words = backend.transcribe_words(audio_file, language="en", prompt=None)

        assert len(words) > 2  # more than one chunk was produced
        assert max(w["start"] for w in words) > 0.5
        # No leftover chunk mp3 files (cleaned up in `finally`).
        assert list(audio_file.parent.glob("whisper_word_chunk_*.mp3")) == []


class TestMissingOpenAIPackage:
    def test_import_error_has_install_hint(self, audio_file, monkeypatch):
        # Setting sys.modules["openai"] = None makes `import openai` raise
        # ImportError, simulating "not installed" -- which it genuinely
        # isn't in this dev environment either way.
        monkeypatch.setitem(sys.modules, "openai", None)
        backend = OpenAIBackend()

        with pytest.raises(ImportError, match=r'pip install "alwayswhisper\[api\]"'):
            backend.transcribe_text(audio_file)


class TestMissingApiKey:
    def test_missing_api_key_raises_runtime_error(self, audio_file, monkeypatch):
        # Fake just enough of `openai` for `from openai import OpenAI` to
        # succeed, so this test reaches the API-key check itself rather than
        # failing earlier on the (separately tested) missing-package path.
        monkeypatch.setitem(
            sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: None)
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Prevent a real .env file (if one exists on disk) from supplying a
        # real key and masking the condition under test.
        import dotenv

        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)

        backend = OpenAIBackend()

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            backend.transcribe_text(audio_file)
