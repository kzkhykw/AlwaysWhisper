from __future__ import annotations

"""Backend factory: builds a TranscriptionBackend from config.

Backend implementations are imported lazily inside create_backend, not at
module import time, so importing `alwayswhisper.backends` never requires
faster-whisper *or* the `[api]` extra (openai/pydub/python-dotenv) to be
installed -- only the backend actually selected needs its dependency.
"""

from .base import TranscriptionBackend

VALID_BACKENDS = ("faster-whisper", "openai-api")


def create_backend(cfg: dict) -> TranscriptionBackend:
    """Construct the transcription backend named by cfg["transcribe"]["backend"].

    "faster-whisper" -> FasterWhisperBackend (local, needs the faster-whisper
    package). "openai-api" -> OpenAIBackend (hosted, needs the `[api]`
    extra). Any other value raises ValueError listing the valid names.
    """
    transcribe_cfg = cfg.get("transcribe", {}) if isinstance(cfg, dict) else {}
    backend_name = transcribe_cfg.get("backend", "faster-whisper")

    if backend_name == "faster-whisper":
        from .faster_whisper_backend import FasterWhisperBackend

        return FasterWhisperBackend(
            model=transcribe_cfg.get("model", "large-v3"),
            device=transcribe_cfg.get("device", "auto"),
            compute_type=transcribe_cfg.get("compute_type", "auto"),
            vad_filter=transcribe_cfg.get("vad_filter", False),
        )

    if backend_name == "openai-api":
        from .openai_backend import OpenAIBackend

        return OpenAIBackend()

    raise ValueError(
        f"Unknown transcribe.backend {backend_name!r}; valid values are: "
        + ", ".join(repr(b) for b in VALID_BACKENDS)
    )


__all__ = ["create_backend", "TranscriptionBackend"]
