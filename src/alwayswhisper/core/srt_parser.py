from __future__ import annotations

"""SRT file parser, writer, and manipulation utilities."""

from dataclasses import dataclass, field
from pathlib import Path

from .timestamp import Timestamp, proportional_split


@dataclass
class SrtEntry:
    """A single SRT subtitle entry."""
    index: int
    start: Timestamp
    end: Timestamp
    text: str

    @property
    def duration_ms(self) -> int:
        return self.end.to_ms() - self.start.to_ms()

    def to_srt_block(self) -> str:
        """Format as SRT block."""
        return (
            f"{self.index}\n"
            f"{self.start.to_srt_string()} --> {self.end.to_srt_string()}\n"
            f"{self.text}\n"
        )


@dataclass
class SrtFile:
    """Collection of SRT entries with manipulation methods."""
    entries: list[SrtEntry] = field(default_factory=list)

    @classmethod
    def parse(cls, content: str) -> "SrtFile":
        """Parse SRT content string."""
        entries = []
        blocks = content.strip().split("\n\n")
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue
            try:
                index = int(lines[0].strip())
            except ValueError:
                continue
            timestamp_line = lines[1].strip()
            if " --> " not in timestamp_line:
                continue
            start_str, end_str = timestamp_line.split(" --> ")
            start = Timestamp.from_srt_string(start_str)
            end = Timestamp.from_srt_string(end_str)
            text = "\n".join(lines[2:]).strip()
            entries.append(SrtEntry(index=index, start=start, end=end, text=text))
        return cls(entries=entries)

    @classmethod
    def from_file(cls, path: str | Path) -> "SrtFile":
        """Load SRT from file."""
        content = Path(path).read_text(encoding="utf-8")
        return cls.parse(content)

    def to_string(self) -> str:
        """Convert to SRT format string."""
        return "\n".join(entry.to_srt_block() for entry in self.entries) + "\n"

    def save(self, path: str | Path) -> None:
        """Save to file."""
        Path(path).write_text(self.to_string(), encoding="utf-8")

    def reindex(self) -> None:
        """Reindex all entries starting from 1."""
        for i, entry in enumerate(self.entries, 1):
            entry.index = i

    def fill_gaps(self, min_gap_ms: int = 0) -> int:
        """Extend each entry's end toward the next entry's start.

        When min_gap_ms > 0, a small gap of that size is preserved so the
        previous caption ends before the next one starts, preventing
        single-frame visual overlap on the burn-in renderer.
        Returns the number of gaps filled.
        """
        filled = 0
        for i in range(len(self.entries) - 1):
            cur = self.entries[i]
            nxt = self.entries[i + 1]
            gap_ms = nxt.start.to_ms() - cur.end.to_ms()
            if gap_ms > min_gap_ms:
                target_ms = max(cur.start.to_ms(), nxt.start.to_ms() - min_gap_ms)
                cur.end = Timestamp.from_ms(target_ms)
                filled += 1
        return filled

    def merge_entries(self, idx1: int, idx2: int) -> None:
        """Merge entry at idx2 into idx1 (by list position)."""
        if idx1 < 0 or idx2 < 0 or idx1 >= len(self.entries) or idx2 >= len(self.entries):
            return
        e1 = self.entries[idx1]
        e2 = self.entries[idx2]
        e1.text = e1.text + e2.text
        e1.end = e2.end
        self.entries.pop(idx2)
        self.reindex()

    def split_entry(self, idx: int, split_pos: int) -> None:
        """Split entry at character position, adjusting timestamps proportionally."""
        if idx < 0 or idx >= len(self.entries):
            return
        entry = self.entries[idx]
        text = entry.text
        if split_pos <= 0 or split_pos >= len(text):
            return
        text1 = text[:split_pos]
        text2 = text[split_pos:]
        mid = proportional_split(entry.start, entry.end, len(text1), len(text2))
        new_entry = SrtEntry(
            index=entry.index + 1,
            start=mid,
            end=entry.end,
            text=text2,
        )
        entry.end = mid
        entry.text = text1
        self.entries.insert(idx + 1, new_entry)
        self.reindex()

    def offset_all(self, offset_ms: int) -> None:
        """Apply time offset to all entries."""
        for entry in self.entries:
            entry.start = Timestamp.from_ms(max(0, entry.start.to_ms() + offset_ms))
            entry.end = Timestamp.from_ms(max(0, entry.end.to_ms() + offset_ms))

    def remove_entry(self, idx: int) -> None:
        """Remove entry by list position."""
        if 0 <= idx < len(self.entries):
            self.entries.pop(idx)
            self.reindex()

    def entries_longer_than(self, max_chars: int) -> list[int]:
        """Return indices of entries with text longer than max_chars."""
        return [i for i, e in enumerate(self.entries) if len(e.text) > max_chars]

    def entries_shorter_than(self, min_chars: int) -> list[int]:
        """Return indices of entries with text shorter than min_chars."""
        return [i for i, e in enumerate(self.entries) if len(e.text) < min_chars]
