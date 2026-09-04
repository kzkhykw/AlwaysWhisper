from __future__ import annotations

"""Common interface every transcription backend implements."""

from pathlib import Path
from typing import Protocol


class TranscriptionBackend(Protocol):
    """A source of Whisper-style transcription, local or remote."""

    def transcribe_words(
        self, audio_path: Path, language: str | None, prompt: str | None
    ) -> list[dict]:
        """Transcribe audio to word-level timestamps.

        Returns a list of ``{"word": str, "start": float, "end": float}``
        dicts (start/end in seconds), one per recognized word.
        """
        ...

    def transcribe_text(self, audio_path: Path, language: str | None) -> str:
        """Transcribe a short audio clip to plain text.

        Used for short clips (e.g. the in-pipeline AV QA spot check). MUST
        NOT apply any bias prompt -- independence from the glossary/prompt
        bias is the point of that check.
        """
        ...
