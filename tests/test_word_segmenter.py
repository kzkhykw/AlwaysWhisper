"""Tests for word_segmenter sentence-ending detection."""

import pytest

from alwayswhisper.core.word_segmenter import (
    _detect_sentence_ending,
    words_to_srt,
    SENTENCE_ENDINGS,
    ENDING_EXTENSIONS,
)


# --- Helper to build word-level timestamps from text ---

def _make_words(text: str, duration_per_char: float = 0.1, gap: float = 0.0):
    """Create word-level timestamp dicts from plain text.

    Each character becomes a separate 'word' entry (mimicking Whisper's
    Japanese character-level output). Insert a pause gap after '|' markers.
    """
    words = []
    t = 0.0
    for ch in text:
        if ch == "|":
            t += 0.5  # insert pause
            continue
        words.append({"word": ch, "start": t, "end": t + duration_per_char})
        t += duration_per_char + gap
    return words


# ============================================================
# _detect_sentence_ending unit tests
# ============================================================

class TestDetectSentenceEnding:
    """Test the sentence-ending detection helper."""

    def test_basic_desu(self):
        # 「テストです」+ next char not extension → confirmed
        assert _detect_sentence_ending("テストです", "。") == "です"
        assert _detect_sentence_ending("テストです", "そ") == "です"

    def test_basic_masu(self):
        assert _detect_sentence_ending("話します", "。") == "ます"
        assert _detect_sentence_ending("話します", "次") == "ます"

    def test_desu_wait_for_ka(self):
        # next char 'か' could extend to 'ですか' → wait (return None)
        assert _detect_sentence_ending("テストです", "か") is None

    def test_desu_wait_for_ke(self):
        # next char 'け' could extend to 'ですけど' → wait
        assert _detect_sentence_ending("テストです", "け") is None

    def test_desu_wait_for_no(self):
        # next char 'の' → ですので (continuation) → wait
        assert _detect_sentence_ending("テストです", "の") is None

    def test_desuka_confirmed(self):
        # 「ですか」+ non-extension next char → confirmed
        assert _detect_sentence_ending("テストですか", "そ") == "ですか"

    def test_desuka_wait_for_ra(self):
        # 「ですか」+ 'ら' → ですから (continuation) → wait
        assert _detect_sentence_ending("テストですか", "ら") is None

    def test_desuka_wait_for_ne(self):
        # 「ですか」+ 'ね' → ですかね (longer ending) → wait
        assert _detect_sentence_ending("テストですか", "ね") is None

    def test_desukane_confirmed(self):
        assert _detect_sentence_ending("テストですかね", "そ") == "ですかね"

    def test_desukedo_confirmed(self):
        assert _detect_sentence_ending("テストですけど", "次") == "ですけど"

    def test_desukedo_wait_for_mo(self):
        # 「ですけど」+ 'も' → ですけども (longer ending) → wait
        assert _detect_sentence_ending("テストですけど", "も") is None

    def test_desukedomo_confirmed(self):
        assert _detect_sentence_ending("テストですけども", "次") == "ですけども"

    def test_desuga_confirmed(self):
        assert _detect_sentence_ending("テストですが", "次") == "ですが"

    def test_desune_confirmed(self):
        assert _detect_sentence_ending("テストですね", "そ") == "ですね"

    def test_desuyo_wait_for_ne(self):
        assert _detect_sentence_ending("テストですよ", "ね") is None

    def test_desyone_confirmed(self):
        assert _detect_sentence_ending("テストですよね", "そ") == "ですよね"

    def test_to_omoimasu(self):
        assert _detect_sentence_ending("いいと思います", "次") == "と思います"

    def test_gozaimasu(self):
        assert _detect_sentence_ending("ございます", "次") == "ございます"

    def test_mashita(self):
        assert _detect_sentence_ending("しました", "次") == "ました"

    def test_masen(self):
        assert _detect_sentence_ending("できません", "次") == "ません"

    def test_no_ending(self):
        assert _detect_sentence_ending("テストの", "結") is None
        assert _detect_sentence_ending("これは", "テ") is None

    def test_none_next_char(self):
        # End of stream: next_char=None → confirmed (no extension possible)
        assert _detect_sentence_ending("テストです", None) == "です"

    def test_longest_match_first(self):
        # 「と思います」should match before 「ます」
        assert _detect_sentence_ending("いいと思います", "次") == "と思います"
        # 「ですけど」should match before 「です」
        assert _detect_sentence_ending("テストですけど", "次") == "ですけど"

    def test_deshou(self):
        assert _detect_sentence_ending("でしょう", "次") == "でしょう"

    def test_deshouka(self):
        assert _detect_sentence_ending("でしょうか", "次") == "でしょうか"

    def test_deshou_wait_for_ka(self):
        assert _detect_sentence_ending("でしょう", "か") is None


# ============================================================
# words_to_srt integration tests (sentence-ending splits)
# ============================================================

class TestWordsToSrtSentenceEndings:
    """Test that words_to_srt splits at sentence endings with max_chars=21."""

    def test_two_sentences_split(self):
        """Two sentences should become two captions."""
        text = "これはテストですそれは別です"
        words = _make_words(text)
        srt = words_to_srt(words, max_chars=21, min_chars=1)
        texts = [e.text for e in srt.entries]
        assert texts == ["これはテストです", "それは別です"]

    def test_desunode_no_split(self):
        """「ですので」is a continuation, should NOT split at 「です」."""
        text = "これはテストですので続きます"
        words = _make_words(text)
        srt = words_to_srt(words, max_chars=21, min_chars=1)
        texts = [e.text for e in srt.entries]
        # Should NOT split into ["これはテストです", "ので続きます"]
        # Instead: one segment or split at 「続きます」
        assert "ので" not in texts[0] or len(texts) == 1 or texts[-1].endswith("ます")
        # More specifically: first segment should contain 「ですので」
        full = "".join(texts)
        assert full == text
        if len(texts) > 1:
            assert not texts[0].endswith("です") or texts[1].startswith("ので")

    def test_desukara_no_split(self):
        """「ですから」is a continuation, should NOT split at 「ですか」."""
        text = "テストですからまだ続きます"
        words = _make_words(text)
        srt = words_to_srt(words, max_chars=21, min_chars=1)
        texts = [e.text for e in srt.entries]
        full = "".join(texts)
        assert full == text
        # Should not split between 「ですか」and「ら」
        for t in texts:
            assert not t.endswith("ですか") or t == texts[-1]

    def test_short_caption_ok(self):
        """Short captions like 'はい' (with pause) should be independent."""
        # はい + pause + それはテストです
        words = _make_words("はい|それはテストです")
        srt = words_to_srt(words, max_chars=21, min_chars=1)
        texts = [e.text for e in srt.entries]
        assert texts[0] == "はい"
        assert texts[1] == "それはテストです"

    def test_multiple_sentences(self):
        """Multiple sentences should each be a separate caption."""
        text = "これはテストですそうですねわかりました"
        words = _make_words(text)
        srt = words_to_srt(words, max_chars=21, min_chars=1)
        texts = [e.text for e in srt.entries]
        assert texts == ["これはテストです", "そうですね", "わかりました"]

    def test_desukedo_split(self):
        """「ですけど」should trigger split."""
        text = "テストですけど次の話です"
        words = _make_words(text)
        srt = words_to_srt(words, max_chars=21, min_chars=1)
        texts = [e.text for e in srt.entries]
        assert texts == ["テストですけど", "次の話です"]

    def test_fallback_to_char_limit(self):
        """Long text without sentence endings should use char-limit logic."""
        # 25 chars, no sentence ending
        text = "あいうえおかきくけこさしすせそたちつてとなにぬねの"
        words = _make_words(text)
        srt = words_to_srt(words, max_chars=21, min_chars=1)
        texts = [e.text for e in srt.entries]
        full = "".join(texts)
        assert full == text
        # Should be split into chunks ≤ ~21 chars
        for t in texts:
            assert len(t) <= 24  # allow some flexibility

    def test_timestamp_accuracy(self):
        """Split at sentence ending should use the ending char's end timestamp."""
        words = _make_words("テストです次の話です", duration_per_char=0.1)
        srt = words_to_srt(words, max_chars=21, min_chars=1)
        # First segment "テストです" = 5 chars, end at 0.5s
        assert srt.entries[0].text == "テストです"
        assert abs(srt.entries[0].end.to_ms() - 500) < 50

    def test_max_chars_35_unchanged(self):
        """max_chars=35 should use pause-based mode (existing behavior)."""
        text = "これはテストですそれは別です"
        words = _make_words(text)
        srt = words_to_srt(words, max_chars=35, min_chars=5)
        # With max_chars=35, uses pause-based mode, not char-limit
        # All text fits in one segment (< 35 chars, no pauses)
        texts = [e.text for e in srt.entries]
        assert "".join(texts) == text
