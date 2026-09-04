from __future__ import annotations

"""Shared parsing of edit_plan.json into audio-removal intervals.

Timestamp shifting and word filtering both need the same answer to "what
time ranges were physically cut from the audio" -- this module is the
single source of truth for that derivation so callers never drift apart.
"""

from .timestamp import Timestamp


def parse_removals(edit_plan: dict) -> list[dict]:
    """Parse edit_plan.json into [{"start_ms": int, "removed_ms": int}, ...].

    Mirrors the removal-interval derivation used elsewhere for timestamp
    synchronization:
    - filler entries derive removed_ms from (end - start) of their SRT
      timestamp strings (a detected filler is always cut in full).
    - pause entries use their own `removed_ms` field (only the excess above
      the kept minimum pause is actually cut from the audio).
    Entries missing required fields are skipped defensively -- this is a
    deliberate divergence from earlier inline parsing logic, which had no
    such guard and would raise KeyError on a malformed entry. The result is
    sorted by start_ms.
    """
    removals: list[dict] = []

    for filler in edit_plan.get("filler_removals", []):
        try:
            start = Timestamp.from_srt_string(filler["start"])
            removed_ms = filler.get("removed_ms", start.to_ms())
            if removed_ms == start.to_ms():
                end = Timestamp.from_srt_string(filler["end"])
                removed_ms = end.to_ms() - start.to_ms()
        except (KeyError, ValueError):
            continue
        removals.append({"start_ms": start.to_ms(), "removed_ms": removed_ms})

    for pause in edit_plan.get("pause_removals", []):
        if "start" not in pause:
            continue
        try:
            start = Timestamp.from_srt_string(pause["start"])
            removed_ms = pause.get("removed_ms")
            if removed_ms is None and "end" in pause:
                end = Timestamp.from_srt_string(pause["end"])
                removed_ms = end.to_ms() - start.to_ms()
        except (KeyError, ValueError):
            continue
        if removed_ms is not None:
            removals.append({"start_ms": start.to_ms(), "removed_ms": removed_ms})

    removals.sort(key=lambda r: r["start_ms"])
    return removals
