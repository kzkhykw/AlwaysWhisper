from __future__ import annotations

"""Whisper bias-prompt token estimation, truncation, and glossary loading.

Whisper's decoder reserves a fixed token budget for the bias prompt
(``initial_prompt`` / ``prompt``); anything longer is silently dropped from
the *front*, so glossary terms placed early stop biasing the model without
any error. We can't run Whisper's exact tokenizer without pulling in the
heavy openai-whisper package, so we estimate conservatively (over-count so
the truncation warning fires before terms are actually dropped, never
after).
"""

import re
from pathlib import Path

# Whisper's decoder reserves 224 tokens for the prompt (both the
# faster-whisper `initial_prompt` and the OpenAI `whisper-1` `prompt` field
# share this limit).
WHISPER_PROMPT_TOKEN_LIMIT = 224


def _is_cjk(ch: str) -> bool:
    """True for CJK ideographs, kana, and full-width forms (heavier to tokenize)."""
    o = ord(ch)
    return (
        0x3000 <= o <= 0x30FF  # CJK punctuation, hiragana, katakana
        or 0x3400 <= o <= 0x4DBF  # CJK ext. A
        or 0x4E00 <= o <= 0x9FFF  # CJK unified ideographs
        or 0xF900 <= o <= 0xFAFF  # CJK compatibility ideographs
        or 0xFF00 <= o <= 0xFFEF  # full-width / half-width forms
    )


def estimate_whisper_tokens(text: str) -> int:
    """Conservative upper-bound estimate of Whisper multilingual token count.

    Whisper's gpt2-based multilingual tokenizer splits kana/kanji into roughly
    1-2 tokens each and Latin text into ~1 token per 4 chars. We count CJK at 2
    tokens (safe side) and other chars at 1 per 4, so the limit warning fires
    before terms are actually dropped rather than after.
    """
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return cjk * 2 + (other + 3) // 4


def truncate_prompt_to_tokens(
    prompt: str, max_tokens: int = WHISPER_PROMPT_TOKEN_LIMIT
) -> tuple[str, int, int]:
    """Trim a bias prompt to the final `max_tokens` (Whisper keeps the tail).

    Returns (possibly-truncated prompt, estimated tokens before, after). Drops
    from the front -- matching Whisper's own behaviour -- and snaps the cut to
    a whitespace boundary so no glossary term is left half-cut. Prints a
    warning when truncation actually happens, since a silently-dropped
    leading term is otherwise a hard-to-notice failure mode.
    """
    before = estimate_whisper_tokens(prompt)
    if before <= max_tokens:
        return prompt, before, before

    # Walk backwards, accumulating the token estimate until we'd exceed budget.
    running = 0.0
    keep_from = len(prompt)
    for i in range(len(prompt) - 1, -1, -1):
        running += 2 if _is_cjk(prompt[i]) else 0.25
        if running > max_tokens:
            break
        keep_from = i
    tail = prompt[keep_from:]

    # Snap to a term boundary: drop a leading partial word up to the first space.
    space = tail.find(" ")
    if space != -1 and space < len(tail) - 1:
        tail = tail[space + 1 :]

    tail = tail.strip()
    after = estimate_whisper_tokens(tail)
    print(
        f"WARNING: AlwaysWhisper: bias prompt ≈{before} tokens exceeds the "
        f"{max_tokens}-token limit -- Whisper keeps only the final "
        f"{max_tokens}, so leading terms were being ignored. Trimmed to the "
        f"tail (≈{after} tokens). Put must-fix terms LAST in your "
        f"glossary/prompt, or shorten it."
    )
    return tail, before, after


def load_glossary_text(path: str | Path) -> str:
    """Read a UTF-8 glossary/vocabulary text file for use as a bias prompt.

    Collapses every run of whitespace (including newlines) into a single
    space and strips the ends, so a glossary written as one term per line
    becomes a single flat line suitable for Whisper's prompt field.
    """
    text = Path(path).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", text).strip()
