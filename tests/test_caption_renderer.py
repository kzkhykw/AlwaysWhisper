"""Tests for caption renderer."""

import pytest
from PIL import Image
from alwayswhisper.caption import renderer
from alwayswhisper.caption.renderer import render_caption_frame


class TestRenderCaptionFrame:
    def test_basic_render(self):
        frame = render_caption_frame("テスト", 1920, 1080)
        assert isinstance(frame, Image.Image)
        assert frame.size == (1920, 1080)
        assert frame.mode == "RGBA"

    def test_empty_text(self):
        frame = render_caption_frame("", 1920, 1080)
        assert frame.size == (1920, 1080)
        # Should be fully transparent
        data = frame.getdata()
        assert all(pixel[3] == 0 for pixel in data)

    def test_custom_style(self):
        style = {
            "background": {"color": [255, 0, 0, 128], "corner_radius": 8, "padding": 10},
            "text": {"color": "#00FF00", "font_size": 32, "font_family": []},
            "position": {"margin_bottom": 100},
        }
        frame = render_caption_frame("テスト", 1920, 1080, style)
        assert frame.size == (1920, 1080)

    def test_not_fully_transparent_with_text(self):
        frame = render_caption_frame("テスト", 1920, 1080)
        data = frame.getdata()
        # Should have some non-transparent pixels
        has_content = any(pixel[3] > 0 for pixel in data)
        assert has_content


class TestFontResolutionRobustness:
    """These deliberately avoid asserting which font gets found (that's
    machine-dependent -- CI boxes and dev laptops carry different fonts).
    They only pin the *fallback behavior*: bad candidates never crash
    rendering, and when truly nothing is found anywhere, PIL's built-in
    bitmap font kicks in with a loud warning rather than a silent one.
    """

    def test_nonsense_font_family_does_not_crash(self):
        # A style with only bogus font candidates should still fall through
        # to the platform defaults (or, worst case, the bitmap fallback)
        # instead of raising.
        style = {
            "text": {
                "font_family": ["/no/such/path.ttf", "NoSuchFontFamilyXYZ123"],
            }
        }
        frame = render_caption_frame("テスト", 640, 360, style)
        assert isinstance(frame, Image.Image)
        assert frame.size == (640, 360)

    def test_all_fonts_missing_falls_back_to_bitmap_font_with_warning(
        self, monkeypatch, capsys
    ):
        # Force every candidate (style-provided AND platform defaults) to
        # miss, so the loud-warning bitmap fallback path is deterministically
        # exercised regardless of what fonts happen to be installed.
        monkeypatch.setattr(renderer, "_platform_font_candidates", lambda: [])
        style = {"text": {"font_family": ["/no/such/path.ttf", "NoSuchFontFamilyXYZ123"]}}

        frame = render_caption_frame("テスト", 640, 360, style)

        assert isinstance(frame, Image.Image)
        assert frame.size == (640, 360)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "font_family" in captured.out
