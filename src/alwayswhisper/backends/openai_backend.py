from __future__ import annotations

"""Hosted transcription backend backed by the OpenAI Whisper API.

The model is hardcoded to "whisper-1": as of this writing it is the only
OpenAI model that exposes word-level timestamps (via
response_format="verbose_json" + timestamp_granularities=["word"]).
"""

import os
from pathlib import Path

# OpenAI's upload limit is 25MB; chunk at 24MB for a safety margin.
CHUNK_SIZE_MB = 24
MAX_FILE_SIZE = CHUNK_SIZE_MB * 1024 * 1024

_API_EXTRA_HINT = 'pip install "alwayswhisper[api]"'


class OpenAIBackend:
    """Transcribes audio via OpenAI's hosted Whisper API (model: whisper-1).

    All third-party imports (openai, pydub, python-dotenv) are lazy, inside
    methods -- importing this module (or `alwayswhisper.backends`) never
    requires the `[api]` extra to be installed; only actually using this
    backend does. The client is created on first use and cached on the
    instance (mirrors FasterWhisperBackend's model caching: the in-pipeline
    AV QA spot check calls transcribe_text repeatedly).
    """

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                f"The openai-api backend requires the 'openai' package. "
                f"Install it with: {_API_EXTRA_HINT}"
            ) from e
        try:
            from dotenv import load_dotenv
        except ImportError as e:
            raise ImportError(
                f"The openai-api backend requires the 'python-dotenv' "
                f"package. Install it with: {_API_EXTRA_HINT}"
            ) from e

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set (checked the environment and .env). "
                "The openai-api backend requires it -- set it in your shell "
                "or in a .env file."
            )
        self._client = OpenAI(api_key=api_key)
        return self._client

    # -- word-level transcription -------------------------------------

    def transcribe_words(
        self,
        audio_path: Path,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[dict]:
        """Transcribe audio with word-level timestamps.

        Uses verbose_json + timestamp_granularities=["word"] to get
        per-word timing data. Returns a list of {"word", "start", "end"}
        dicts. Files over MAX_FILE_SIZE (OpenAI's 25MB upload limit, minus a
        safety margin) are split into mp3 chunks, transcribed independently,
        and the per-chunk timestamps are shifted back onto the full-file
        timeline.
        """
        audio_path = Path(audio_path)
        file_size = audio_path.stat().st_size

        if file_size <= MAX_FILE_SIZE:
            return self._transcribe_words_single(audio_path, language, prompt)
        return self._transcribe_words_chunked(audio_path, language, prompt)

    def _transcribe_words_single(
        self,
        audio_path: Path,
        language: str | None,
        prompt: str | None = None,
    ) -> list[dict]:
        """Transcribe a single file with word-level timestamps."""
        client = self._get_client()
        kwargs = {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        }
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(file=f, **kwargs)
        words = []
        for w in getattr(response, "words", []):
            words.append(
                {
                    "word": w.word if hasattr(w, "word") else w["word"],
                    "start": w.start if hasattr(w, "start") else w["start"],
                    "end": w.end if hasattr(w, "end") else w["end"],
                }
            )
        return words

    def _transcribe_words_chunked(
        self,
        audio_path: Path,
        language: str | None,
        prompt: str | None = None,
    ) -> list[dict]:
        """Transcribe audio in mp3 chunks, offsetting timestamps back onto
        the full-file timeline."""
        try:
            from pydub import AudioSegment
        except ImportError as e:
            raise ImportError(
                f"The openai-api backend requires the 'pydub' package for "
                f"files over {CHUNK_SIZE_MB}MB. Install it with: {_API_EXTRA_HINT}"
            ) from e

        audio = AudioSegment.from_file(str(audio_path))
        duration_ms = len(audio)
        file_size = audio_path.stat().st_size
        ms_per_byte = duration_ms / file_size
        chunk_duration_ms = int(MAX_FILE_SIZE * ms_per_byte * 0.9)  # 90% to be safe

        all_words: list[dict] = []
        offset_ms = 0
        chunk_idx = 0
        tmp_dir = audio_path.parent

        while offset_ms < duration_ms:
            chunk_end = min(offset_ms + chunk_duration_ms, duration_ms)
            chunk = audio[offset_ms:chunk_end]

            chunk_path = tmp_dir / f"whisper_word_chunk_{chunk_idx}.mp3"
            chunk.export(str(chunk_path), format="mp3")

            try:
                chunk_words = self._transcribe_words_single(chunk_path, language, prompt)
                offset_sec = offset_ms / 1000.0
                for w in chunk_words:
                    w["start"] += offset_sec
                    w["end"] += offset_sec
                    all_words.append(w)
            finally:
                chunk_path.unlink(missing_ok=True)

            offset_ms = chunk_end
            chunk_idx += 1

        return all_words

    # -- plain-text transcription (QA) ----------------------------------

    def transcribe_text(self, audio_path: Path, language: str | None = None) -> str:
        """Transcribe a short audio clip to plain text -- no bias prompt.

        Used by the in-pipeline AV QA spot check, which re-transcribes a
        few seconds of audio per sampled caption and fuzzy-compares it
        against the caption text. Deliberately omits the bias prompt used by
        transcribe_words: independence from that bias is the point of this
        check, since a biased re-transcription could "confirm" drifted
        captions instead of catching them. Clips are only a few seconds
        long, so no chunking is needed (unlike transcribe_words above).
        """
        client = self._get_client()
        kwargs = {
            "model": "whisper-1",
            "response_format": "json",
        }
        if language:
            kwargs["language"] = language
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(file=f, **kwargs)
        return response.text
