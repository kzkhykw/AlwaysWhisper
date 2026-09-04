"""Tests for ASS subtitle generation (fast caption mode)."""

import pytest

from alwayswhisper.caption.ass_writer import (
    _build_typewriter_text,
    _color_to_ass,
    _escape_text,
    _format_ass_time,
    srt_to_ass,
)
from alwayswhisper.core.srt_parser import SrtEntry, SrtFile
from alwayswhisper.core.timestamp import Timestamp


class TestFormatAssTime:
    def test_zero(self):
        assert _format_ass_time(0) == "0:00:00.00"

    def test_centiseconds(self):
        assert _format_ass_time(1230) == "0:00:01.23"

    def test_minute(self):
        assert _format_ass_time(61_000) == "0:01:01.00"

    def test_hour(self):
        assert _format_ass_time(3_661_000) == "1:01:01.00"

    def test_negative_clamped(self):
        assert _format_ass_time(-50) == "0:00:00.00"

    def test_ms_truncates_to_cs(self):
        # 1.239s → 123cs (12.39 → floor)
        assert _format_ass_time(1239) == "0:00:01.23"


class TestColorToAss:
    def test_hex_white(self):
        assert _color_to_ass("#FFFFFF") == "&H00FFFFFF"

    def test_hex_dark_gray(self):
        assert _color_to_ass("#333333") == "&H00333333"

    def test_rgba_black_opaque(self):
        assert _color_to_ass([0, 0, 0, 255]) == "&H00000000"

    def test_rgba_alpha_inverted(self):
        # alpha=200 (78% opaque) → ASS alpha = 55 (0x37)
        assert _color_to_ass([0, 0, 0, 200]) == "&H37000000"

    def test_rgba_bgr_order(self):
        # RGB(255,0,0) → BBGGRR 0000FF
        assert _color_to_ass([255, 0, 0, 255]) == "&H000000FF"


class TestEscapeText:
    def test_plain(self):
        assert _escape_text("hello") == "hello"

    def test_braces(self):
        assert _escape_text("a{b}c") == "a\\{b\\}c"

    def test_newline(self):
        assert _escape_text("a\nb") == "a\\Nb"

    def test_backslash(self):
        assert _escape_text("a\\b") == "a\\\\b"


class TestBuildTypewriterText:
    def test_empty(self):
        assert _build_typewriter_text("", 1000, 400) == ""

    def test_single_char_no_cursor(self):
        # n=1: only one char, no cursor appended (cap not exceeded by next char)
        out = _build_typewriter_text("a", 1000, 400)
        assert out.startswith("{\\1a&H00&\\3a&H00&\\fscx100}a")
        assert "|" not in out

    def test_first_char_immediate(self):
        # First char is forced visible by the python max(1, ...) clamp at t=0+.
        out = _build_typewriter_text("abc", 1000, 400)
        assert out.startswith("{\\1a&H00&\\3a&H00&\\fscx100}a")

    def test_later_char_has_animation(self):
        out = _build_typewriter_text("abc", 1000, 400)
        # n=3, cap=400. char index 1 visible at ceil(2/3*400)=267ms
        assert (
            "{\\1a&HFF&\\3a&HFF&\\fscx0\\t(267,268,\\1a&H00&\\3a&H00&\\fscx100)}b"
        ) in out
        # char index 2 visible at ceil(3/3*400)=400ms
        assert (
            "{\\1a&HFF&\\3a&HFF&\\fscx0\\t(400,401,\\1a&H00&\\3a&H00&\\fscx100)}c"
        ) in out

    def test_cursor_appended_when_multichar(self):
        out = _build_typewriter_text("abc", 1000, 400)
        # cursor visible+full-width from start, hidden+zero-width at cap_ms
        assert out.endswith(
            "{\\1a&H00&\\3a&H00&\\fscx100\\t(400,401,\\1a&HFF&\\3a&HFF&\\fscx0)}|"
        )

    def test_cap_clamped_to_duration(self):
        # duration shorter than completion_ms → cap = duration
        out = _build_typewriter_text("ab", 100, 400)
        # n=2, cap=100. char 1 visible at ceil(2/2*100)=100ms
        assert (
            "{\\1a&HFF&\\3a&HFF&\\fscx0\\t(100,101,\\1a&H00&\\3a&H00&\\fscx100)}b"
        ) in out
        # cursor hidden at 100
        assert "\\t(100,101,\\1a&HFF&\\3a&HFF&\\fscx0)}|" in out


class TestSrtToAss:
    def _srt(self, entries):
        return SrtFile(entries=entries)

    def test_header_contains_play_res(self):
        srt = self._srt([])
        out = srt_to_ass(srt, {}, 1920, 1080)
        assert "PlayResX: 1920" in out
        assert "PlayResY: 1080" in out

    def test_style_uses_font_family_string(self):
        srt = self._srt([])
        out = srt_to_ass(
            srt, {"text": {"font_family": "Noto Sans JP", "font_size": 64}},
            1920, 1080,
        )
        assert "Style: Default,Noto Sans JP,64," in out

    def test_style_skips_path_font_family(self):
        srt = self._srt([])
        out = srt_to_ass(
            srt,
            {"text": {
                "font_family": ["/System/Library/Fonts/foo.ttc", "Hiragino Sans"]
            }},
            1920, 1080,
        )
        assert "Style: Default,Hiragino Sans," in out

    def test_dialogue_line_per_entry(self):
        srt = self._srt([
            SrtEntry(1, Timestamp(0, 0, 1, 0), Timestamp(0, 0, 2, 0), "ab"),
            SrtEntry(2, Timestamp(0, 0, 3, 0), Timestamp(0, 0, 4, 0), "cd"),
        ])
        out = srt_to_ass(srt, {}, 1920, 1080)
        assert out.count("Dialogue: 0,") == 2
        assert "0:00:01.00,0:00:02.00" in out
        assert "0:00:03.00,0:00:04.00" in out

    def test_skip_blank_entries(self):
        srt = self._srt([
            SrtEntry(1, Timestamp(0, 0, 1, 0), Timestamp(0, 0, 2, 0), "   "),
            SrtEntry(2, Timestamp(0, 0, 3, 0), Timestamp(0, 0, 4, 0), "ok"),
        ])
        out = srt_to_ass(srt, {}, 1920, 1080)
        assert out.count("Dialogue: 0,") == 1

    def test_japanese_text_preserved(self):
        srt = self._srt([
            SrtEntry(1, Timestamp(0, 0, 1, 0), Timestamp(0, 0, 2, 0), "あいうえお"),
        ])
        out = srt_to_ass(srt, {}, 1920, 1080)
        assert "あ" in out and "お" in out
