"""Timestamp utilities for SRT processing."""

from dataclasses import dataclass


@dataclass
class Timestamp:
    """SRT timestamp representation (hours, minutes, seconds, milliseconds)."""
    hours: int
    minutes: int
    seconds: int
    milliseconds: int

    @classmethod
    def from_ms(cls, total_ms: int) -> "Timestamp":
        """Create Timestamp from total milliseconds."""
        if total_ms < 0:
            total_ms = 0
        ms = total_ms % 1000
        total_seconds = total_ms // 1000
        s = total_seconds % 60
        total_minutes = total_seconds // 60
        m = total_minutes % 60
        h = total_minutes // 60
        return cls(hours=h, minutes=m, seconds=s, milliseconds=ms)

    @classmethod
    def from_srt_string(cls, s: str) -> "Timestamp":
        """Parse SRT timestamp string like '00:01:23,456'."""
        s = s.strip()
        time_part, ms_part = s.split(",")
        parts = time_part.split(":")
        return cls(
            hours=int(parts[0]),
            minutes=int(parts[1]),
            seconds=int(parts[2]),
            milliseconds=int(ms_part),
        )

    def to_ms(self) -> int:
        """Convert to total milliseconds."""
        return (
            self.hours * 3600000
            + self.minutes * 60000
            + self.seconds * 1000
            + self.milliseconds
        )

    def to_srt_string(self) -> str:
        """Convert to SRT timestamp string like '00:01:23,456'."""
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d},{self.milliseconds:03d}"

    def __sub__(self, other: "Timestamp") -> int:
        """Return difference in milliseconds."""
        return self.to_ms() - other.to_ms()

    def __add__(self, ms: int) -> "Timestamp":
        """Add milliseconds, return new Timestamp."""
        return Timestamp.from_ms(self.to_ms() + ms)

    def __lt__(self, other: "Timestamp") -> bool:
        return self.to_ms() < other.to_ms()

    def __le__(self, other: "Timestamp") -> bool:
        return self.to_ms() <= other.to_ms()

    def __gt__(self, other: "Timestamp") -> bool:
        return self.to_ms() > other.to_ms()

    def __ge__(self, other: "Timestamp") -> bool:
        return self.to_ms() >= other.to_ms()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Timestamp):
            return NotImplemented
        return self.to_ms() == other.to_ms()


def offset_timestamp(ts: Timestamp, offset_ms: int) -> Timestamp:
    """Apply offset to timestamp."""
    return Timestamp.from_ms(ts.to_ms() + offset_ms)


def proportional_split(start: Timestamp, end: Timestamp, text1_len: int, text2_len: int) -> Timestamp:
    """Split timestamp range proportionally by text length.

    Returns the midpoint timestamp where text1 ends and text2 begins.
    """
    total_len = text1_len + text2_len
    if total_len == 0:
        return start
    duration_ms = end.to_ms() - start.to_ms()
    split_ms = start.to_ms() + int(duration_ms * text1_len / total_len)
    return Timestamp.from_ms(split_ms)
