from __future__ import annotations

"""In-pipeline AV QA: spot-check subtitles against re-transcribed audio.

Before burning captions in, randomly sample a handful of SRT entries,
re-extract each entry's audio window, re-transcribe it independently (no
glossary bias prompt), and fuzzy-compare the result against the caption
text. A low match ratio usually means the caption text or its timestamp has
drifted from what's actually spoken in the video -- catching that here is
far cheaper than noticing it after an expensive caption burn-in.
"""

import difflib
import os
import random
import re
import string
import tempfile
import unicodedata
from pathlib import Path

from .ffmpeg import extract_audio_clip
from .hallucination_filter import DEFAULT_STRIP_PHRASES, strip_hallucination_text


# Characters stripped for comparison, on top of NFKC + lowercasing: all
# whitespace, ASCII punctuation, and common Japanese punctuation. The long
# vowel mark "ー" is deliberately NOT included -- it's phonetic content
# (e.g. コーヒー), not punctuation, and stripping it would corrupt words.
_STRIP_RE = re.compile(
    r"[\s、。・「」『』（）？！…〜" + re.escape(string.punctuation) + r"]"
)


def _normalize(text: str) -> str:
    """Normalize text for fuzzy comparison.

    NFKC folds full-width ASCII forms (Ａ-Ｚ, ！, ？, ...) onto their
    half-width equivalents; lowercasing then removes case differences;
    _STRIP_RE removes whitespace and punctuation so only the "content"
    characters are compared.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return _STRIP_RE.sub("", text)


def run_av_qa(
    video_path,
    entries,
    *,
    samples: int = 5,
    min_ratio: float = 0.5,
    pad_ms: int = 300,
    min_entry_ms: int = 500,
    language: str | None = None,
    transcriber=None,
    rng=None,
    strip_phrases=None,
) -> dict:
    """Randomly spot-check `entries` against re-transcribed audio clips.

    entries: list of SRT entry objects (.text, .start/.end Timestamps).
    transcriber is a required-in-practice callable (audio_path -> text),
    e.g. a backend's `transcribe_text`; this module has no built-in
    transcription backend of its own, unlike the rest of the pipeline this
    was extracted from. It's still declared as an optional keyword (default
    None) so callers that never end up sampling anything (no eligible
    entries, or `samples=0`) don't need to supply one -- but a ValueError is
    raised as soon as sampling actually needs to call it. rng is injectable
    too, so tests stay hermetic (no ffmpeg/Whisper calls, deterministic
    sampling). strip_phrases mechanically removes hallucinated stock
    phrases (e.g. "ご視聴"/"ありがとうございました") from each
    re-transcription before it's scored or stored in the report, so a
    hallucinated tail neither shows in QA logs nor drags the ratio down;
    None (the default) uses core.hallucination_filter's
    DEFAULT_STRIP_PHRASES.

    Returns a report dict; never raises for a failed check on its own -- the
    caller decides what to do with report["passed"].
    """
    rng = rng or random.Random()
    phrases = DEFAULT_STRIP_PHRASES if strip_phrases is None else strip_phrases

    eligible = [
        e for e in entries
        if e.text.strip() and (e.end.to_ms() - e.start.to_ms()) >= min_entry_ms
    ]
    k = max(0, min(samples, len(eligible)))
    sampled = rng.sample(eligible, k)

    if sampled and transcriber is None:
        raise ValueError(
            "run_av_qa requires a transcriber callable (audio_path -> text); "
            "pass backend.transcribe_text"
        )

    results = []
    for entry in sampled:
        start_sec = max(0.0, entry.start.to_ms() / 1000 - pad_ms / 1000)
        end_sec = max(0.0, entry.end.to_ms() / 1000 + pad_ms / 1000)

        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        wav_path = Path(tmp_name)
        try:
            extract_audio_clip(video_path, wav_path, start_sec, end_sec)
            asr_text = transcriber(wav_path)
        finally:
            wav_path.unlink(missing_ok=True)

        # Strip before scoring/storing -- a hallucinated tail must not drag
        # the ratio down or leak into the QA report/log.
        asr_text = strip_hallucination_text(asr_text, phrases)

        norm_srt = _normalize(entry.text)
        norm_asr = _normalize(asr_text)
        raw_ratio = (
            0.0 if not norm_asr
            else difflib.SequenceMatcher(None, norm_srt, norm_asr).ratio()
        )
        ok = raw_ratio >= min_ratio

        results.append({
            "index": entry.index,
            "start": entry.start.to_srt_string(),
            "end": entry.end.to_srt_string(),
            "srt_text": entry.text,
            "asr_text": asr_text,
            "ratio": round(raw_ratio, 4),
            "ok": ok,
        })

    avg_ratio = (
        sum(r["ratio"] for r in results) / len(results) if results else 1.0
    )
    passed = all(r["ok"] for r in results)

    return {
        "samples": results,
        "avg_ratio": round(avg_ratio, 4),
        "passed": passed,
        "min_ratio": min_ratio,
        "sampled": k,
        "eligible": len(eligible),
    }
