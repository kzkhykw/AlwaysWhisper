"""AlwaysWhisper: local-Whisper transcription and stylish burned-in captions."""

from .config import load_config
from .pipeline import (
    auto_run,
    caption_video,
    qa_check,
    segment_words,
    transcribe_file,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "auto_run",
    "caption_video",
    "load_config",
    "qa_check",
    "segment_words",
    "transcribe_file",
]
