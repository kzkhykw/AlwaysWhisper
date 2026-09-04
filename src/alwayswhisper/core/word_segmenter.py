from __future__ import annotations

"""Reconstruct SRT entries from word-level timestamps.

Strategy: Use natural pauses (gaps between words) as primary split points,
then apply post-processing to merge short segments and fix bad splits.
"""

import re

from .srt_parser import SrtEntry, SrtFile
from .timestamp import Timestamp

DEFAULT_MAX_CHARS = 35
DEFAULT_MIN_CHARS = 5
DEFAULT_PAUSE_THRESHOLD_SEC = 0.3

# --- Sentence-ending detection for 1-caption-per-sentence splits ---

# Sorted longest-first for greedy matching
SENTENCE_ENDINGS = [
    'と思っております', 'と思っています', 'していただければ',
    'いただきました', 'いただきます', 'と思います',
    'になります', 'ございます', 'してください', 'でしょうか', 'でしょう',
    'んですけれども', 'ですけれども', 'ますけれども',
    'んですけども', 'ですけども', 'ますけども',
    'んですけど', 'ですけど', 'ますけど',
    'なんですよ', 'なんですね',
    'んですが', 'ですが', 'ますが',
    'ですよね', 'ますよね', 'ですかね', 'ますかね',
    'ですね', 'ますね', 'ですよ', 'ますよ', 'ですし', 'ますし',
    'ですか', 'ますか', 'でした', 'ました', 'ません',
    'です', 'ます',
]

# Next chars that should delay/cancel a split at a detected ending.
# Includes chars that form longer endings (wait) and non-ending continuations (cancel).
ENDING_EXTENSIONS: dict[str, set[str]] = {
    'です': {'か', 'け', 'が', 'ね', 'よ', 'し', 'の'},  # の→ですので
    'ます': {'か', 'け', 'が', 'ね', 'よ', 'し', 'の'},  # の→ますので
    'でした': {'の'},
    'ました': {'の'},
    'ですか': {'ら', 'ね'},   # ら→ですから, ね→ですかね
    'ますか': {'ら', 'ね'},
    'ですけど': {'も'},       # も→ですけども
    'ますけど': {'も'},
    'ですよ': {'ね'},         # ね→ですよね
    'ますよ': {'ね'},
    'でしょう': {'か'},       # か→でしょうか
}


def _detect_sentence_ending(text: str, next_char: str | None) -> str | None:
    """Detect if text ends with a sentence-ending expression.

    Args:
        text: Accumulated segment text.
        next_char: The next character (or None if end of stream).

    Returns:
        The matched ending string if confirmed, None if no ending or should wait.
    """
    for ending in SENTENCE_ENDINGS:
        if text.endswith(ending):
            if next_char is not None:
                wait_chars = ENDING_EXTENSIONS.get(ending, set())
                if next_char in wait_chars:
                    return None  # Wait for possible longer ending / continuation
            return ending
    return None


def words_to_srt(
    words: list[dict],
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    pause_threshold: float = DEFAULT_PAUSE_THRESHOLD_SEC,
) -> SrtFile:
    """Convert word-level timestamps to SRT entries."""
    if not words:
        return SrtFile(entries=[])

    # Use char-limit-driven mode for tighter limits (<=25)
    if max_chars <= 25:
        segments = _group_by_char_limit(words, max_chars, min_chars, pause_threshold)
    else:
        segments = _group_by_pauses(words, pause_threshold)
        segments = _fix_word_splits(segments)
        segments = _split_long_segments(segments, words, max_chars)
        segments = _fix_word_splits(segments)

    merge_min = min_chars
    segments = _merge_short_segments(segments, merge_min, max_chars)
    # Skip _fix_boundaries in sentence-ending mode: splits are already at
    # grammatical boundaries and _fix_boundaries would merge them back.
    if max_chars > 25:
        segments = _fix_boundaries(segments, max_chars)

    return _segments_to_srt(segments)


def _group_by_pauses(words: list[dict], pause_threshold: float) -> list[dict]:
    """Group words into segments based on natural pauses."""
    segments = []
    current_words = []
    current_text = ""

    for i, word in enumerate(words):
        w_text = word["word"].strip()
        if not w_text:
            continue

        current_words.append(word)
        current_text += w_text

        # Check for pause after this word
        is_last = i == len(words) - 1
        has_pause = False
        if not is_last:
            gap = words[i + 1]["start"] - word["end"]
            if gap >= pause_threshold:
                has_pause = True

        if has_pause or is_last:
            segments.append({
                "text": current_text,
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
            })
            current_words = []
            current_text = ""

    return segments


def _group_by_char_limit(
    words: list[dict], max_chars: int, min_chars: int, pause_threshold: float
) -> list[dict]:
    """Group words into segments with sentence-ending detection.

    Priority order:
    1. Sentence ending (ます/です/etc.) → split immediately, even if < max_chars
    2. Natural pause → split if reasonable length accumulated
    3. Character limit → grammar-based or fallback split at ~max_chars
    """
    # Build flat list of (char, start, end, has_pause_after)
    chars: list[dict] = []
    for i, word in enumerate(words):
        w_text = word["word"].strip()
        if not w_text:
            continue
        is_last_word = i == len(words) - 1
        has_pause = False
        if not is_last_word:
            gap = words[i + 1]["start"] - word["end"]
            if gap >= pause_threshold:
                has_pause = True

        for ci, c in enumerate(w_text):
            is_last_char = ci == len(w_text) - 1
            chars.append({
                "char": c,
                "start": word["start"],
                "end": word["end"],
                "has_pause": has_pause and is_last_char,
            })

    if not chars:
        return []

    segments: list[dict] = []
    seg_start_idx = 0

    while seg_start_idx < len(chars):
        remaining = len(chars) - seg_start_idx

        # Scan char-by-char looking for sentence endings and pauses
        scan_end = min(seg_start_idx + max_chars + 5, len(chars))
        accumulated = ""
        split_pos = None  # chars count from seg_start_idx (1-based length)
        best_pause_pos = None

        for i in range(seg_start_idx, scan_end):
            accumulated += chars[i]["char"]
            pos_in_seg = i - seg_start_idx + 1  # 1-based length

            # --- Priority 1: Sentence ending detection ---
            next_char = chars[i + 1]["char"] if i + 1 < len(chars) else None
            ending = _detect_sentence_ending(accumulated, next_char)
            if ending is not None and pos_in_seg >= 2:
                split_pos = pos_in_seg
                break
            # If ending was detected but deferred (wait), a pause confirms it
            has_pause = chars[i]["has_pause"]
            if ending is None and has_pause:
                # Check if text ends with any base ending (pause confirms it)
                for e in SENTENCE_ENDINGS:
                    if accumulated.endswith(e) and pos_in_seg >= 2:
                        split_pos = pos_in_seg
                        break
                if split_pos is not None:
                    break

            # --- Priority 2: Pause-based split (interjections, natural breaks) ---
            if has_pause and pos_in_seg >= 1:
                split_pos = pos_in_seg
                break

            # --- Priority 3: At max_chars, scan ahead for nearby sentence ending ---
            if pos_in_seg >= max_chars:
                # Look ahead up to 5 chars for a sentence ending (allow small overflow)
                lookahead_end = min(i + 6, len(chars))
                lookahead_text = accumulated
                found_ahead = False
                for j in range(i + 1, lookahead_end):
                    lookahead_text += chars[j]["char"]
                    la_next = chars[j + 1]["char"] if j + 1 < len(chars) else None
                    la_ending = _detect_sentence_ending(lookahead_text, la_next)
                    if la_ending is not None:
                        split_pos = j - seg_start_idx + 1
                        found_ahead = True
                        break
                if found_ahead:
                    break

                # No nearby sentence ending — use pause or grammar fallback
                if best_pause_pos is not None:
                    split_pos = best_pause_pos
                else:
                    window_text = accumulated
                    grammar_split = _find_best_split(window_text, max_chars)
                    if grammar_split is not None and not _is_forbidden_split(window_text, grammar_split):
                        split_pos = grammar_split
                    else:
                        # Char-limit fallback
                        for offset in range(0, 6):
                            candidate = max_chars - offset
                            if candidate >= min_chars and candidate < len(window_text):
                                if not _is_forbidden_split(window_text, candidate):
                                    split_pos = candidate
                                    break
                        if split_pos is None:
                            for offset in range(1, 8):
                                candidate = max_chars + offset
                                if candidate < len(window_text):
                                    if not _is_forbidden_split(window_text, candidate):
                                        split_pos = candidate
                                        break
                break

        # Ultimate fallback
        if split_pos is None:
            split_pos = min(max_chars, remaining)

        # Emit segment
        split_idx = seg_start_idx + split_pos
        if split_idx > len(chars):
            split_idx = len(chars)

        text = "".join(c["char"] for c in chars[seg_start_idx:split_idx])
        seg_end = chars[split_idx - 1]["end"]

        segments.append({
            "text": text,
            "start": chars[seg_start_idx]["start"],
            "end": seg_end,
        })
        seg_start_idx = split_idx

    return segments



def _fix_word_splits(segments: list[dict]) -> list[dict]:
    """Fix segments where a pause-based split broke a word/compound verb.

    If the end of segment N + start of segment N+1 forms a known compound
    that shouldn't be split, merge the fragment into the appropriate segment.
    """
    # Fragments that should not start a new segment
    bad_start_fragments = [
        'いる', 'いた', 'いて', 'いない',
        'くれる', 'くれて', 'くれた',
        'もらう', 'もらって', 'もらった', 'もらえ',
        'あげる', 'あげて',
        'おく', 'おいた', 'おいて',
        'ほしい', 'ほしく',
        'ない', 'なく', 'なくて', 'なかった',
        'きる', 'きた', 'きて', 'きない',
        'しまう', 'しまって',
        'だける', 'だけた', 'だいて', 'だきた',  # いただ+ける
        'く',  # とにかく
        'ら',  # ながら
    ]

    for _ in range(3):
        changed = False
        new_segs = [segments[0]]
        for seg in segments[1:]:
            prev = new_segs[-1]
            prev_text = prev["text"]
            curr_text = seg["text"]

            merged = False
            # Check if curr starts with a fragment that should be merged
            for frag in bad_start_fragments:
                if curr_text.startswith(frag):
                    # Verify this looks like a broken compound
                    tail = prev_text[-3:] if len(prev_text) >= 3 else prev_text
                    compound = tail + frag
                    # Check against NO_SPLIT_PATTERNS
                    if NO_SPLIT_PATTERNS.search(compound) or frag in ['く', 'ら']:
                        move_len = len(frag)
                        prev["text"] = prev_text + curr_text[:move_len]
                        prev["end"] = seg["start"]
                        remaining = curr_text[move_len:]
                        if remaining:
                            seg["text"] = remaining
                            new_segs.append(seg)
                        merged = True
                        changed = True
                        break

            if not merged:
                new_segs.append(seg)

        segments = new_segs
        if not changed:
            break

    return segments



def _split_long_segments(segments: list[dict], words: list[dict], max_chars: int) -> list[dict]:
    """Split segments exceeding max_chars at the best grammatical point."""
    result = []

    for seg in segments:
        _recursive_split(seg, max_chars, result)

    return result


def _recursive_split(seg: dict, max_chars: int, result: list[dict]) -> None:
    """Recursively split a segment until all parts fit within max_chars."""
    text = seg["text"]
    if len(text) <= max_chars:
        result.append(seg)
        return

    # Try to find a good split point up to max_chars, then up to max_chars+10
    split_pos = _find_best_split(text, max_chars)
    if split_pos is None:
        split_pos = _find_best_split(text, min(max_chars + 10, len(text) - 3))
    if split_pos is None or split_pos <= 0 or split_pos >= len(text):
        # No good split - accept as-is rather than splitting mid-word
        result.append(seg)
        return

    text1 = text[:split_pos]
    text2 = text[split_pos:]

    # Proportionally split timing
    duration = seg["end"] - seg["start"]
    ratio = len(text1) / len(text)
    mid_time = seg["start"] + duration * ratio

    seg1 = {"text": text1, "start": seg["start"], "end": mid_time}
    seg2 = {"text": text2, "start": mid_time, "end": seg["end"]}

    _recursive_split(seg1, max_chars, result)
    _recursive_split(seg2, max_chars, result)


NO_SPLIT_PATTERNS = re.compile(
    # て形の複合動詞
    r"(している|していて|していた|してくる|してきた|してしまう|"
    r"ている|ていた|ていて|てくる|てきた|てしまう|"
    r"ておく|ておいて|てみた|てみて|てみる|"
    r"てくれる|てくれて|てくれた|てもらう|てもらって|てもらった|"
    r"てあげる|てあげて|てほしい|てほしく|"
    r"てない|てなく|"
    # できる系
    r"できる|できない|できた|できて|"
    # ない形
    r"なきゃ|なければ|ないん|なくて|なかった|なって|なっちゃ|"
    # っ系
    r"っている|っていう|っていた|っていて|ってない|ってくれ|"
    r"やってな|やってい|やってき|やってく|"
    # ます+助詞
    r"ますと|ですと|ましたと|でしたと|ますけど|ですけど|"
    r"ますが|ですが|ますね|ですね|ますよ|ですよ|"
    r"ますし|ですし|ますか|ですか|"
    # られる系
    r"なってい|なってき|なってく|"
    r"られる|られて|られた|られない|"
    r"させて|させる|させた|させない|"
    # 副詞等
    r"とにかく|やっぱり|やっぱ|"
    # その他複合
    r"かもしれ|かもです|"
    r"なくて|ながら|ければ|"
    r"ところ|ていただ)"
)
NO_SPLIT_KATAKANA = re.compile(r"[\u30A0-\u30FF\u30FC]+")
NO_SPLIT_ENGLISH = re.compile(r"[A-Za-z0-9]+")


def _is_forbidden_split(text: str, pos: int) -> bool:
    """Check if splitting at pos would break a word or compound verb."""
    for pat in [NO_SPLIT_PATTERNS, NO_SPLIT_KATAKANA, NO_SPLIT_ENGLISH]:
        for m in pat.finditer(text):
            if m.start() < pos < m.end():
                return True
    return False


def _find_best_split(text: str, max_chars: int) -> int | None:
    """Find the best position to split text for subtitles."""
    split_after = [
        (re.compile(r"(ます|です|ました|でした|ません)"), 100),
        (re.compile(r"(ですけど|ですが|ますが|ますね|ますよ)"), 95),
        (re.compile(r"(けど|けれども|ので|から|のに)"), 85),
        (re.compile(r"(ていて|していて|していた)"), 80),
        (re.compile(r"(って|たり|とか)"), 75),
        (re.compile(r"(して|できて)"), 70),
        (re.compile(r"(を|に|で)"), 60),
        (re.compile(r"(は|が|と|も)"), 55),
    ]

    best_pos = None
    best_score = -1

    for pattern, score in split_after:
        for m in pattern.finditer(text):
            pos = m.end()
            if pos > max_chars:
                continue
            if pos < 5 or len(text) - pos < 3:
                continue
            if _is_forbidden_split(text, pos):
                continue

            center = len(text) / 2
            balance = 1 - abs(pos - center) / len(text)
            effective = score + balance * 15

            if effective > best_score:
                best_score = effective
                best_pos = pos

    # No good split found: return None to keep the segment intact
    # rather than splitting mid-word
    return best_pos


def _merge_short_segments(
    segments: list[dict], min_chars: int, max_chars: int
) -> list[dict]:
    """Merge segments shorter than min_chars."""
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        combined_len = len(prev["text"]) + len(seg["text"])

        if len(seg["text"]) < min_chars and combined_len <= max_chars + 3:
            prev["text"] += seg["text"]
            prev["end"] = seg["end"]
        elif len(prev["text"]) < min_chars and combined_len <= max_chars + 3:
            prev["text"] += seg["text"]
            prev["end"] = seg["end"]
        else:
            merged.append(seg)

    return merged


def _fix_boundaries(segments: list[dict], max_chars: int) -> list[dict]:
    """Fix segments where text fragments are orphaned at boundaries.

    Moves short fragments (particles, verb endings) from the start of
    a segment to the end of the previous one.
    """
    # Patterns that should not start a segment (fragment to move to prev)
    bad_start_patterns = [
        # 1 char particles
        (re.compile(r"^([かをがもにではねよなわてたとけごくりれしきみ])"), 1),
        # 2-4 char fragments
        (re.compile(r"^(する|した|ない|たい|よう|さん|けど|ます|です|のを|ので|くて|ども|ろしく)"), None),
        # Verb continuations
        (re.compile(r"^(っている|っていう|っております|っちゃう|っていた|っていて|ってる|って)"), None),
        (re.compile(r"^(ている|ていた|ておい|ておいて|てくれ|てもら|ていただ)"), None),
        (re.compile(r"^(ながら|ことが|ものが|というか|という|れども|ればと)"), None),
    ]

    for _ in range(3):  # Multiple passes
        changed = False
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]

            for pat, _ in bad_start_patterns:
                m = pat.match(curr["text"])
                if m:
                    fragment = m.group(1) if m.lastindex else m.group(0)
                    if len(prev["text"]) + len(fragment) <= max_chars:
                        prev["text"] += fragment
                        curr["text"] = curr["text"][len(fragment):]
                        # Proportionally adjust timing
                        if curr["text"]:
                            total_dur = curr["end"] - curr["start"]
                            if total_dur > 0 and len(fragment) + len(curr["text"]) > 0:
                                ratio = len(fragment) / (len(fragment) + len(curr["text"]))
                                prev["end"] = curr["start"] + total_dur * ratio
                                curr["start"] = prev["end"]
                        changed = True
                        break

            # Remove empty segments
            if not curr["text"].strip():
                segments.pop(i)
                changed = True
                break

        # Pass 2: Move trailing fragments from curr to next
        for i in range(len(segments) - 1):
            curr = segments[i]
            nxt = segments[i + 1]
            curr_text = curr["text"]
            nxt_text = nxt["text"]

            # Check if tail of curr + head of nxt forms a word that got split
            for tail_len in [1, 2, 3]:
                if len(curr_text) <= tail_len + 3:
                    continue
                tail = curr_text[-tail_len:]
                combined = tail + nxt_text[:8]

                # Known words/patterns that should be restored
                restore_words = [
                    'なるほど', 'それで', 'あとは', 'そこで', 'ただ', 'また',
                    'なので', 'だから', 'でも', 'ちょっと', 'やっぱり',
                    'それが', 'これが', 'ここで', 'そして', 'つまり',
                    'ないし', 'なくて', 'ながら', 'できる', 'できない',
                    'ている', 'ていた', 'ていて', 'ておい',
                    'てくれ', 'てもら', 'ていただ',
                    'とにかく', 'かもしれ',
                ]
                for w in restore_words:
                    if combined.startswith(w) and not curr_text.endswith(w) and not nxt_text.startswith(w):
                        curr["text"] = curr_text[:-tail_len]
                        nxt["text"] = tail + nxt_text
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break

        if not changed:
            break

    return [s for s in segments if s["text"].strip()]


def _segments_to_srt(segments: list[dict]) -> SrtFile:
    """Convert segments to SrtFile."""
    entries = []
    for i, seg in enumerate(segments, 1):
        start = Timestamp.from_ms(int(seg["start"] * 1000))
        end = Timestamp.from_ms(int(seg["end"] * 1000))
        entries.append(SrtEntry(index=i, start=start, end=end, text=seg["text"]))
    return SrtFile(entries=entries)


# --- English-aware segmenter ---------------------------------------------

_EN_SENTENCE_END = re.compile(r"[.!?]['\")\]]?$")


def words_to_srt_en(
    words: list[dict],
    max_chars: int = 42,
    min_chars: int = 10,
    pause_threshold: float = 0.3,
) -> SrtFile:
    """Convert English word-level timestamps to SRT entries.

    English words are atomic — never split inside a word. Splits are taken
    (in priority order) at sentence-ending punctuation (`.`, `!`, `?`),
    natural pauses, or when adding the next word would exceed max_chars.
    """
    if not words:
        return SrtFile(entries=[])

    segments: list[dict] = []
    cur_text = ""
    cur_start: float | None = None
    cur_end: float | None = None

    def flush():
        nonlocal cur_text, cur_start, cur_end
        if cur_text and cur_start is not None and cur_end is not None:
            segments.append({"text": cur_text.strip(), "start": cur_start, "end": cur_end})
        cur_text = ""
        cur_start = None
        cur_end = None

    for i, word in enumerate(words):
        w_text = word["word"].strip()
        if not w_text:
            continue
        w_start = word["start"]
        w_end = word["end"]

        # Pause to the next word (used for pause-based splits).
        gap_after = 0.0
        if i + 1 < len(words):
            gap_after = words[i + 1]["start"] - w_end

        # Decide whether appending this word would overflow. If so, flush first.
        candidate = (cur_text + (" " if cur_text else "") + w_text).strip()
        if cur_text and len(candidate) > max_chars and len(cur_text) >= min_chars:
            flush()
            candidate = w_text

        # Append the current word.
        if cur_start is None:
            cur_start = w_start
        cur_text = candidate
        cur_end = w_end

        # Sentence-ending punctuation: split if we have at least min_chars.
        if _EN_SENTENCE_END.search(w_text) and len(cur_text) >= min_chars:
            flush()
            continue

        # Natural pause split, only if accumulated length is reasonable.
        if gap_after >= pause_threshold and len(cur_text) >= min_chars:
            flush()
            continue

    flush()

    # Merge runs that are still under min_chars with the next segment when
    # combining them stays within max_chars + small slack. Avoids dangling
    # 2-3 word fragments.
    merged: list[dict] = []
    for seg in segments:
        if merged and len(seg["text"]) < min_chars:
            prev = merged[-1]
            combined = (prev["text"] + " " + seg["text"]).strip()
            if len(combined) <= max_chars + 5:
                prev["text"] = combined
                prev["end"] = seg["end"]
                continue
        merged.append(seg)

    return _segments_to_srt(merged)
