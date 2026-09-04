from __future__ import annotations

"""Realign SRT entry start times against word-level timestamps.

The Whisper word-level timestamps (transcript_words.json) are treated as ground
truth. LLM-based correction passes can introduce small but accumulating drift
(~0.1s baseline + ~125 ppm per second observed with LLM-corrected SRTs). This
module anchors each SRT entry's start to the first matching word's start time
in the original transcript, then shifts the whole entry by the computed delta.

Only the start timestamp is realigned; the entry duration (end - start) is
preserved. Any resulting overlap with the next entry is clamped so prev.end
<= next.start.
"""

import json
from pathlib import Path

from .edit_plan import parse_removals
from .srt_parser import SrtFile
from .timestamp import Timestamp


def _build_edit_converters(edit_plan: dict):
    """Return (pre_to_post, post_to_pre) in milliseconds from edit_plan.json.

    The removal-interval list itself comes from the shared
    edit_plan.parse_removals() (a single source of truth shared by every
    consumer that needs to know what was cut, e.g. timestamp offsetting and
    word filtering) so they all agree on exactly which ranges were cut; this
    function only owns building the two converter closures.
    """
    removals = [
        (r["start_ms"], int(r["removed_ms"])) for r in parse_removals(edit_plan)
    ]

    def pre_to_post(t_ms: int) -> int:
        offset = 0
        for start_ms, rm in removals:
            if t_ms >= start_ms + rm:
                offset += rm
            elif t_ms > start_ms:
                return max(0, start_ms - offset)
        return max(0, t_ms - offset)

    def post_to_pre(t_ms: int) -> int:
        cum = 0
        for start_ms, rm in removals:
            if t_ms <= start_ms - cum:
                return t_ms + cum
            cum += rm
        return t_ms + cum

    return pre_to_post, post_to_pre


def _build_char_index(words: list[dict]) -> tuple[str, list[int]]:
    """Build a concatenated char string and per-char pre-edit start time (ms)."""
    concat_parts: list[str] = []
    char_pre_ms: list[int] = []
    for w in words:
        wt = w["word"]
        s_ms = int(round(w["start"] * 1000))
        e_ms = int(round(w["end"] * 1000))
        dur = max(1, e_ms - s_ms)
        n = len(wt)
        if n == 0:
            continue
        concat_parts.append(wt)
        for j in range(n):
            char_pre_ms.append(s_ms + (dur * j) // max(1, n))
    return "".join(concat_parts), char_pre_ms


def _norm(text: str) -> str:
    return "".join(text.split())


def _find_anchor(
    head: str, concat: str, char_pre_ms: list[int], lo_ms: int, hi_ms: int, hint_ms: int
) -> int:
    """Return the char position in concat where head best matches near hint.

    Only positions with pre-edit time in [lo_ms, hi_ms] are considered.
    Among candidates, the one closest to hint_ms wins. Returns -1 if none.
    """
    if not head:
        return -1
    best_pos = -1
    best_dist: int | None = None
    pos = 0
    while True:
        pos = concat.find(head, pos)
        if pos < 0:
            break
        if pos < len(char_pre_ms):
            t = char_pre_ms[pos]
            if lo_ms <= t <= hi_ms:
                d = abs(t - hint_ms)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_pos = pos
        pos += 1
    return best_pos


def _find_best_match(
    text_clean: str,
    concat: str,
    char_pre_ms: list[int],
    lo_ms: int,
    hi_ms: int,
    hint_ms: int,
) -> int:
    """Find the best anchor by trying longest-prefix match first.

    Prefers a longer prefix match; only falls back to shorter prefixes if
    no occurrence of the longer one lies within the time window. This avoids
    confusing short but highly repetitive phrases like 'ありがとう...'.
    """
    max_head = min(len(text_clean), 40)
    # Try descending lengths; stop at the first length with a valid match.
    for head_len in range(max_head, 1, -1):
        pos = _find_anchor(
            text_clean[:head_len], concat, char_pre_ms, lo_ms, hi_ms, hint_ms
        )
        if pos >= 0:
            return pos
    return -1


def realign_srt_by_words(
    srt: SrtFile,
    words: list[dict],
    edit_plan: dict,
    search_window_ms: int = 20_000,
) -> dict:
    """Realign SRT start times using word-level timestamps.

    For each entry: finds the entry's leading characters in the concatenated
    word transcript near the expected pre-edit time, converts the matched
    word's start to post-edit time, and shifts the entry so its start equals
    that time. Duration is preserved. Overlaps are clamped afterwards.

    Args:
        srt: SrtFile, modified in place.
        words: list of {word, start, end} from transcript_words.json.
        edit_plan: parsed edit_plan.json.
        search_window_ms: +/- window (pre-edit ms) around the hint to search.

    Returns:
        Stats: {adjusted, unchanged, failed, max_shift_ms, mean_shift_ms}.
    """
    if not srt.entries or not words:
        return {
            "adjusted": 0,
            "unchanged": 0,
            "failed": len(srt.entries),
            "max_shift_ms": 0,
            "mean_shift_ms": 0.0,
        }

    pre_to_post, post_to_pre = _build_edit_converters(edit_plan)
    concat, char_pre_ms = _build_char_index(words)

    adjusted = 0
    unchanged = 0
    failed = 0
    shifts: list[int] = []

    for entry in srt.entries:
        text_clean = _norm(entry.text)
        if not text_clean:
            failed += 1
            continue

        hint_pre_ms = post_to_pre(entry.start.to_ms())
        lo_ms = hint_pre_ms - search_window_ms
        hi_ms = hint_pre_ms + search_window_ms

        best_pos = _find_best_match(
            text_clean, concat, char_pre_ms, lo_ms, hi_ms, hint_pre_ms
        )

        if best_pos < 0:
            failed += 1
            continue

        new_pre_ms = char_pre_ms[best_pos]
        new_post_ms = pre_to_post(new_pre_ms)

        old_start_ms = entry.start.to_ms()
        duration_ms = entry.end.to_ms() - old_start_ms
        delta_ms = new_post_ms - old_start_ms

        if delta_ms == 0:
            unchanged += 1
            continue

        entry.start = Timestamp.from_ms(max(0, new_post_ms))
        entry.end = Timestamp.from_ms(max(0, new_post_ms + duration_ms))
        adjusted += 1
        shifts.append(delta_ms)

    # Clamp overlaps: prev.end <= next.start
    for i in range(len(srt.entries) - 1):
        cur = srt.entries[i]
        nxt = srt.entries[i + 1]
        if cur.end > nxt.start:
            cur.end = nxt.start
        if cur.start >= cur.end:
            # Ensure non-zero duration by nudging end up to next.start
            cur.end = nxt.start

    max_shift = max((abs(s) for s in shifts), default=0)
    mean_shift = sum(shifts) / len(shifts) if shifts else 0.0
    return {
        "adjusted": adjusted,
        "unchanged": unchanged,
        "failed": failed,
        "max_shift_ms": max_shift,
        "mean_shift_ms": mean_shift,
    }


def realign_srt_file(
    srt_path: Path,
    words_path: Path,
    edit_plan_path: Path,
    output_path: Path,
    search_window_ms: int = 20_000,
) -> dict:
    """Load an SRT, realign against word timestamps, save result."""
    srt = SrtFile.from_file(srt_path)
    words = json.loads(Path(words_path).read_text(encoding="utf-8"))
    edit_plan = (
        json.loads(Path(edit_plan_path).read_text(encoding="utf-8"))
        if Path(edit_plan_path).exists()
        else {}
    )
    stats = realign_srt_by_words(srt, words, edit_plan, search_window_ms)
    srt.save(output_path)
    return stats
