from __future__ import annotations

"""Mechanical removal of Whisper's stock outro hallucination.

Whisper hallucinates the phrase "ご視聴ありがとうございました" (stock YouTube
outro) -- and variants such as "はい、ありがとうございました" -- into
silent/trailing stretches of audio -- most often as a single short
word-timestamp token at the very end of a video, but sometimes split across
several consecutive tokens or fused with real trailing speech. Each
configured phrase (DEFAULT_STRIP_PHRASES below) is stripped independently
and mechanically (never via an LLM) wherever it occurs, so it never reaches
captions, transcript_words.json / transcript_raw.srt, or the AV QA
re-transcription log.
"""

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .srt_parser import SrtFile


# Inline default -- AlwaysWhisper's config deep-merges user YAML over the
# packaged defaults (see config.deep_merge), so a project config.yaml only
# needs to set the keys it wants to change. `transcribe.strip_phrases: null`
# means "use this constant's defaults when transcribe.language is 'ja'"
# (see resolve_strip_phrases in pipeline.py, which is also where the
# language-scoping lives); `[]` disables stripping entirely, and a
# non-empty list replaces this constant wholesale.
# Longest first: strip_hallucination_text/_words both match longest-phrase-
# first regardless of this order (see the sort in each), but keeping the
# constant itself longest-first documents intent and matches the config.
DEFAULT_STRIP_PHRASES: tuple[str, ...] = (
    "ご視聴ありがとうございました。",
    "はい、ありがとうございました。",
    "ご視聴ありがとうございました",
    "はい、ありがとうございました",
    "ありがとうございました。",
    "ありがとうございました",
)

# A "real content" character: any word character (Unicode letters/digits,
# which includes kanji/kana) except the underscore. Used to decide whether
# what's left after stripping is worth keeping, or is just punctuation
# debris (e.g. a lone "。") that should be dropped along with the phrase.
_HAS_ALNUM_RE = re.compile(r"[^\W_]")


def _resolve_phrases(phrases: Iterable[str] | None) -> Iterable[str]:
    """None -> DEFAULT_STRIP_PHRASES; anything else (including `()`/`[]`,
    which disables stripping) is used verbatim.

    A YAML `strip_phrases:` key with nothing after it parses to None, which
    is a *present* key -- `dict.get("strip_phrases", DEFAULT)` would not
    catch it, and iterating None directly raises TypeError. Every public
    function in this module runs its `phrases` argument through here first
    so no caller (config-driven or direct) can crash on a null value.
    """
    return DEFAULT_STRIP_PHRASES if phrases is None else phrases


def strip_hallucination_text(
    text: str, phrases: Iterable[str] | None = DEFAULT_STRIP_PHRASES
) -> str:
    """Remove every `phrases` substring from `text`, then collapse whitespace.

    Plain str.replace per phrase, longest phrase first, followed by
    collapsing any resulting whitespace runs into a single space and
    trimming the ends. Longest-first is not just cosmetic: phrases here
    routinely nest inside one another (e.g. bare "ありがとうございました"
    is a substring of both "はい、ありがとうございました" and
    "ご視聴ありがとうございました"), so removing a short phrase first would
    consume its occurrence inside a longer one and leave the longer
    phrase's own prefix ("はい、"/"ご視聴") behind as debris -- independent
    of what order the caller happens to list `phrases` in. `phrases=None`
    behaves like the default (see _resolve_phrases).
    """
    phrases = _resolve_phrases(phrases)
    for phrase in sorted((p for p in phrases if p), key=len, reverse=True):
        text = text.replace(phrase, "")
    return re.sub(r"\s+", " ", text).strip()


def strip_hallucination_words(
    words: list[dict], phrases: Iterable[str] | None = DEFAULT_STRIP_PHRASES
) -> tuple[list[dict], list[tuple[float, float]]]:
    """Strip hallucinated phrases from Whisper word-timestamp output.

    Handles any tokenization: a phrase may land in a single token
    ("ありがとうございました。"), be split across several consecutive
    tokens ("ご" "視聴" "ありがとう" "ございました"), or be fused with other
    text in one token ("と思いますありがとうございました").

    Algorithm: concatenate every word's text (in order), find all
    occurrences of every phrase in that concatenation (longer phrases
    first, so a phrase that is a substring of another can't eat into it),
    and mark those character positions. A word with none of its characters
    marked ("untouched" by any phrase match) always passes through
    byte-identical, regardless of its own content -- this matters because
    real Whisper output contains pre-existing empty ("") and
    punctuation-only ("・") tokens that must not be mistaken for
    hallucination debris just because they happen to have no letter/digit.
    Only a TOUCHED word is rebuilt from its unmarked characters: if the
    remainder has no letter/digit left (`_HAS_ALNUM_RE.search(...)` is None
    -- this drops punctuation-only leftovers like the "。" left behind by
    "。ありがとうございました") it is removed entirely; if real text is
    left, the word is kept with its ORIGINAL start/end (never shrunk).

    Returns a NEW list (the input list and its dicts are never mutated)
    plus the (start, end) spans in seconds of every contiguous run of
    fully-removed (touched-and-emptied) words -- an untouched word or a
    partially-trimmed-but-kept word both end any in-progress run without
    themselves becoming part of a span. Non-dict entries, and dicts missing
    "word" (as a str) / "start" / "end", pass through unchanged and are
    never part of a span. `phrases=None` behaves like the default (see
    _resolve_phrases).
    """

    def eligible(w: object) -> bool:
        return (
            isinstance(w, dict)
            and isinstance(w.get("word"), str)
            and "start" in w
            and "end" in w
        )

    phrases = _resolve_phrases(phrases)
    phrase_list = sorted((p for p in phrases if p), key=len, reverse=True)

    # Concatenate every eligible word's text, remembering each word's slice
    # (start offset + length) within the concatenation.
    concat_parts: list[str] = []
    slices: dict[int, tuple[int, int]] = {}
    pos = 0
    for i, w in enumerate(words):
        if not eligible(w):
            continue
        token = w["word"]
        concat_parts.append(token)
        slices[i] = (pos, len(token))
        pos += len(token)
    concat = "".join(concat_parts)

    removed_mask = bytearray(len(concat))
    for phrase in phrase_list:
        start = 0
        while True:
            idx = concat.find(phrase, start)
            if idx == -1:
                break
            for k in range(idx, idx + len(phrase)):
                removed_mask[k] = 1
            start = idx + len(phrase)

    out: list[dict] = []
    spans: list[tuple[float, float]] = []
    run_start: float | None = None
    run_end: float | None = None

    for i, w in enumerate(words):
        if not eligible(w):
            if run_start is not None:
                spans.append((float(run_start), float(run_end)))
                run_start = run_end = None
            out.append(w)
            continue

        offset, length = slices[i]
        mask_slice = removed_mask[offset : offset + length]
        touched = any(mask_slice)

        if not touched:
            # Not touched by any phrase match at all -- always passes
            # through byte-identical (same object), even if empty ("") or
            # punctuation-only ("・"). Ends any in-progress run exactly like
            # any other kept word; it must never itself start/end/count as
            # a span it had nothing to do with.
            if run_start is not None:
                spans.append((float(run_start), float(run_end)))
                run_start = run_end = None
            out.append(w)
            continue

        remainder = "".join(
            ch for ch, is_removed in zip(w["word"], mask_slice) if not is_removed
        )

        if _HAS_ALNUM_RE.search(remainder) is None:
            # Touched and nothing but punctuation left -- fully removed,
            # fold into the current contiguous span.
            if run_start is None:
                run_start = w["start"]
            run_end = w["end"]
            continue

        if run_start is not None:
            spans.append((float(run_start), float(run_end)))
            run_start = run_end = None

        new_word = dict(w)
        new_word["word"] = remainder
        out.append(new_word)

    if run_start is not None:
        spans.append((float(run_start), float(run_end)))

    return out, spans


def strip_hallucination_srt(
    srt: "SrtFile", phrases: Iterable[str] | None = DEFAULT_STRIP_PHRASES
) -> tuple[int, int]:
    """Strip hallucinated phrases from every entry's text, in place.

    An entry that contains NONE of the phrases as a substring is left
    completely untouched -- not even whitespace-normalised -- and never
    dropped (this matters because strip_hallucination_text's own whitespace
    collapse would otherwise strip a trailing space off e.g. "。 ", or an
    already-punctuation-only entry like "・" would otherwise look like
    hallucination debris once run through it, even though no phrase ever
    matched). Of the entries that DO contain a phrase: strip_hallucination_
    text() is applied, and one whose resulting text has no letter/digit
    left is dropped entirely (e.g. a lone "。" remnant of
    "。ありがとうございました"); one whose text changed but still has real
    content left is kept, trimmed. Reindexes afterward. `phrases=None`
    behaves like the default (see _resolve_phrases).

    Returns (dropped_count, trimmed_count).
    """
    phrases = list(_resolve_phrases(phrases))
    dropped = 0
    trimmed = 0
    kept = []
    for entry in srt.entries:
        if not any(p in entry.text for p in phrases if p):
            # Untouched by any phrase -- must stay byte-identical, never
            # dropped, regardless of its own content.
            kept.append(entry)
            continue

        new_text = strip_hallucination_text(entry.text, phrases)
        if new_text != entry.text:
            if _HAS_ALNUM_RE.search(new_text) is None:
                dropped += 1
                continue
            trimmed += 1
            entry.text = new_text
        kept.append(entry)

    srt.entries = kept
    srt.reindex()
    return dropped, trimmed
