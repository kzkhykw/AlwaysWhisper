"""Tests for alwayswhisper.backends: factory dispatch and FasterWhisperBackend
word normalization.

FasterWhisperBackend imports `faster_whisper` lazily inside `_load_model`,
so these tests inject a fake module via monkeypatch.setitem(sys.modules,
...) before the backend ever touches it -- no real model load, no network,
no download, even though the real faster-whisper package happens to be
installed in this environment.
"""

import sys
from types import SimpleNamespace

import pytest

from alwayswhisper.backends import create_backend
from alwayswhisper.backends.faster_whisper_backend import FasterWhisperBackend


class _TrackingIterator:
    """Wraps an iterable and records whether it was driven to exhaustion."""

    def __init__(self, items):
        self._iter = iter(items)
        self.exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self.exhausted = True
            raise


def _fake_word(word, start, end):
    return SimpleNamespace(word=word, start=start, end=end, probability=0.99)


def _fake_segment(words=None, text=""):
    return SimpleNamespace(words=words, text=text)


class _FakeWhisperModel:
    """Records every constructor and .transcribe() call for assertions."""

    instances: list["_FakeWhisperModel"] = []

    def __init__(self, model_size_or_path, device="auto", compute_type="default", **kwargs):
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls: list[dict] = []
        self.last_segments: _TrackingIterator | None = None
        _FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.transcribe_calls.append({"audio": audio, **kwargs})
        if kwargs.get("word_timestamps"):
            items = [
                _fake_segment(words=[_fake_word(" Hello", 0.0, 0.5), _fake_word(" world", 0.5, 1.0)]),
                _fake_segment(words=[_fake_word("こんにちは", 1.0, 1.5), _fake_word("世界", 1.5, 2.0)]),
                _fake_segment(words=None),  # guard: segment.words can be None
                _fake_segment(words=[]),  # guard: segment.words can be empty
            ]
        else:
            items = [
                _fake_segment(text=" Hello world."),
                _fake_segment(text=" How are you?"),
            ]
        tracker = _TrackingIterator(items)
        self.last_segments = tracker
        info = SimpleNamespace(language=kwargs.get("language") or "en")
        return tracker, info


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    _FakeWhisperModel.instances = []
    fake_module = SimpleNamespace(WhisperModel=_FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    return fake_module


class TestCreateBackendDispatch:
    def test_faster_whisper_backend(self):
        backend = create_backend({"transcribe": {"backend": "faster-whisper"}})
        assert isinstance(backend, FasterWhisperBackend)

    def test_openai_backend(self):
        from alwayswhisper.backends.openai_backend import OpenAIBackend

        backend = create_backend({"transcribe": {"backend": "openai-api"}})
        assert isinstance(backend, OpenAIBackend)

    def test_defaults_to_faster_whisper_when_backend_key_missing(self):
        backend = create_backend({"transcribe": {}})
        assert isinstance(backend, FasterWhisperBackend)

    def test_faster_whisper_backend_reads_model_options_from_config(self):
        backend = create_backend(
            {
                "transcribe": {
                    "backend": "faster-whisper",
                    "model": "small",
                    "device": "cpu",
                    "compute_type": "int8",
                    "vad_filter": True,
                }
            }
        )
        assert backend.model_name == "small"
        assert backend.device == "cpu"
        assert backend.compute_type == "int8"
        assert backend.vad_filter is True

    def test_unknown_backend_raises_value_error_listing_valid_names(self):
        with pytest.raises(ValueError) as exc_info:
            create_backend({"transcribe": {"backend": "nonsense"}})
        message = str(exc_info.value)
        assert "nonsense" in message
        assert "faster-whisper" in message
        assert "openai-api" in message


class TestFasterWhisperBackendWordNormalization:
    def test_transcribe_words_strips_and_normalizes_shape(self, fake_faster_whisper):
        backend = FasterWhisperBackend(model="tiny")
        words = backend.transcribe_words("audio.wav", language="ja", prompt="glossary terms")

        assert [w["word"] for w in words] == ["Hello", "world", "こんにちは", "世界"]
        for w in words:
            assert set(w.keys()) == {"word", "start", "end"}
            assert isinstance(w["start"], float)
            assert isinstance(w["end"], float)

    def test_transcribe_words_fully_consumes_generator(self, fake_faster_whisper):
        backend = FasterWhisperBackend(model="tiny")
        backend.transcribe_words("audio.wav", language="ja", prompt=None)

        assert _FakeWhisperModel.instances[0].last_segments.exhausted is True

    def test_transcribe_words_passes_prompt_as_initial_prompt(self, fake_faster_whisper):
        backend = FasterWhisperBackend(model="tiny")
        backend.transcribe_words("audio.wav", language="ja", prompt="glossary terms")

        call = _FakeWhisperModel.instances[0].transcribe_calls[0]
        assert call["initial_prompt"] == "glossary terms"
        assert call["word_timestamps"] is True
        assert call["language"] == "ja"

    def test_empty_prompt_becomes_none(self, fake_faster_whisper):
        backend = FasterWhisperBackend(model="tiny")
        backend.transcribe_words("audio.wav", language="ja", prompt="")

        call = _FakeWhisperModel.instances[0].transcribe_calls[0]
        assert call["initial_prompt"] is None

    def test_no_prompt_becomes_none(self, fake_faster_whisper):
        backend = FasterWhisperBackend(model="tiny")
        backend.transcribe_words("audio.wav", language="ja", prompt=None)

        call = _FakeWhisperModel.instances[0].transcribe_calls[0]
        assert call["initial_prompt"] is None

    def test_vad_filter_forwarded(self, fake_faster_whisper):
        backend = FasterWhisperBackend(model="tiny", vad_filter=True)
        backend.transcribe_words("audio.wav", language="ja", prompt=None)

        call = _FakeWhisperModel.instances[0].transcribe_calls[0]
        assert call["vad_filter"] is True

    def test_model_constructed_once_across_two_calls(self, fake_faster_whisper):
        backend = FasterWhisperBackend(model="tiny")
        backend.transcribe_words("a.wav", language="ja", prompt=None)
        backend.transcribe_words("b.wav", language="ja", prompt=None)

        assert len(_FakeWhisperModel.instances) == 1
        assert len(_FakeWhisperModel.instances[0].transcribe_calls) == 2
        assert _FakeWhisperModel.instances[0].model_size_or_path == "tiny"


class TestFasterWhisperBackendTranscribeText:
    def test_joins_segment_texts_and_omits_bias_prompt(self, fake_faster_whisper):
        backend = FasterWhisperBackend(model="tiny")
        text = backend.transcribe_text("clip.wav", language="en")

        assert text == "Hello world. How are you?"
        call = _FakeWhisperModel.instances[0].transcribe_calls[0]
        assert "initial_prompt" not in call
        assert "word_timestamps" not in call

    def test_reuses_cached_model_from_transcribe_words(self, fake_faster_whisper):
        backend = FasterWhisperBackend(model="tiny")
        backend.transcribe_words("a.wav", language="ja", prompt=None)
        backend.transcribe_text("b.wav", language="ja")

        assert len(_FakeWhisperModel.instances) == 1


class TestFasterWhisperBackendMissingDependency:
    def test_import_error_has_install_hint(self, monkeypatch):
        # Setting sys.modules["faster_whisper"] = None makes `import
        # faster_whisper` raise ImportError, simulating "not installed"
        # without needing to actually uninstall the real package.
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        backend = FasterWhisperBackend(model="tiny")

        with pytest.raises(ImportError, match="pip install faster-whisper"):
            backend.transcribe_words("a.wav", language="ja", prompt=None)
