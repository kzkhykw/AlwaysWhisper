"""Typewriter animation calculation.

Determines how many characters to show at any given time
during a subtitle's display period.
"""


def calculate_visible_chars(
    elapsed_ms: int,
    total_duration_ms: int,
    total_chars: int,
    completion_ms: int = 400,
) -> int:
    """Calculate number of visible characters at a given time.

    Args:
        elapsed_ms: Time elapsed since subtitle start
        total_duration_ms: Total subtitle display duration
        total_chars: Total number of characters in text
        completion_ms: Fixed time in ms to complete all chars (default 400ms)

    Returns:
        Number of characters to display
    """
    if total_chars <= 0 or total_duration_ms <= 0:
        return total_chars

    if elapsed_ms <= 0:
        return 0

    if elapsed_ms >= total_duration_ms:
        return total_chars

    if elapsed_ms >= completion_ms:
        return total_chars

    # Linear interpolation over fixed completion time
    progress = elapsed_ms / completion_ms
    visible = int(progress * total_chars)

    # Always show at least 1 char after start
    return max(1, min(visible, total_chars))


def generate_char_timestamps(
    total_duration_ms: int,
    total_chars: int,
    completion_ms: int = 400,
) -> list[int]:
    """Generate timestamps (ms) for when each character appears.

    Returns list of length total_chars, where each element is the
    millisecond offset when that character becomes visible.
    """
    if total_chars <= 0:
        return []

    if total_duration_ms <= 0:
        return [0] * total_chars

    timestamps = []
    for i in range(total_chars):
        t = int((i / total_chars) * completion_ms)
        timestamps.append(t)

    return timestamps


def apply_cursor(
    visible_text: str,
    visible_chars: int,
    total_chars: int,
    cursor_char: str = "|",
) -> str:
    """Append a typing cursor if text is still being revealed.

    Args:
        visible_text: The currently visible portion of text
        visible_chars: Number of characters currently visible
        total_chars: Total number of characters in the full text
        cursor_char: Character to use as cursor

    Returns:
        Text with cursor appended if still typing, otherwise unchanged
    """
    if not visible_text:
        return visible_text
    if visible_chars < total_chars:
        return visible_text + cursor_char
    return visible_text
