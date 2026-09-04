from __future__ import annotations

"""Local transcription backend backed by the faster-whisper package."""

from pathlib import Path


class FasterWhisperBackend:
    """Transcribes audio locally via faster-whisper (CTranslate2 + Whisper).

    The underlying WhisperModel is loaded lazily on first use and cached on
    the instance: QA spot-checks call transcribe_text repeatedly for a
    single caption pass, and reloading (and potentially re-downloading) the
    model on every call would be brutal.
    """

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
        vad_filter: bool = False,
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                import faster_whisper
            except ImportError as e:
                raise ImportError(
                    "The faster-whisper backend requires the 'faster-whisper' "
                    "package. Install it with: pip install faster-whisper"
                ) from e
            self._model = faster_whisper.WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe_words(
        self,
        audio_path: Path,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[dict]:
        """Transcribe audio to word-level timestamps.

        `segments` is a lazy generator -- transcription only happens as it's
        iterated, so the for-loop below both collects the words and fully
        drives the transcription to completion. `.word.strip()` normalizes
        faster-whisper's Latin leading-space convention (Japanese tokens
        have no surrounding spaces to begin with) so both this backend and
        OpenAIBackend emit the identical 3-key shape regardless of language.
        `segment.words` is guarded against None/empty (can happen for a
        segment faster-whisper decided has no words). Entries are kept even
        if the stripped text is empty -- callers that care (e.g.
        hallucination stripping) decide what to do with an empty token, not
        this backend.
        """
        model = self._load_model()
        segments, _info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            initial_prompt=prompt or None,
            vad_filter=self.vad_filter,
        )

        words: list[dict] = []
        for segment in segments:
            for w in segment.words or []:
                words.append(
                    {
                        "word": w.word.strip(),
                        "start": float(w.start),
                        "end": float(w.end),
                    }
                )
        return words

    def transcribe_text(self, audio_path: Path, language: str | None = None) -> str:
        """Transcribe a short audio clip to plain text -- no bias prompt.

        Used by the in-pipeline AV QA spot check, which must stay
        independent of the glossary/prompt bias so it can't "confirm"
        drifted captions instead of catching them.
        """
        model = self._load_model()
        segments, _info = model.transcribe(str(audio_path), language=language)
        return " ".join(segment.text.strip() for segment in segments).strip()
