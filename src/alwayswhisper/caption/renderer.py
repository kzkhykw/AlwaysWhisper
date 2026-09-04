from __future__ import annotations

"""Caption frame renderer using Pillow.

Renders caption text with:
- Black rounded rectangle background with alpha
- White text with shadow
- Cross-platform font resolution (style-configured candidates first, then
  platform defaults for macOS / Linux / Windows)
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Platform-default font candidates, tried in order, after anything the
# caller supplied via style["text"]["font_family"]. Entries may be absolute
# font file paths or installed family names -- both are attempted the same
# way in _find_font.
_MAC_FONT_CANDIDATES = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "Hiragino Sans",
]
_LINUX_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "NotoSansCJK-Bold",
    "Noto Sans CJK JP",
]
_WINDOWS_FONT_CANDIDATES = [
    "C:/Windows/Fonts/YuGothB.ttc",
    "C:/Windows/Fonts/meiryob.ttc",
    "Yu Gothic",
    "Meiryo",
]


DEFAULT_STYLE = {
    "position": {"align": "center_bottom", "margin_bottom": 60},
    "background": {"color": [0, 0, 0, 200], "corner_radius": 12, "padding": 16},
    "text": {
        "color": "#FFFFFF",
        # None = resolve purely from platform defaults (see _platform_font_candidates).
        "font_family": None,
        "font_size": 48,
        "font_weight": "bold",
    },
    "shadow": {"enabled": True, "offset": [2, 2], "color": "#333333"},
}


def _platform_font_candidates() -> list[str]:
    """Return this OS's default font candidates, in priority order."""
    if sys.platform == "darwin":
        return list(_MAC_FONT_CANDIDATES)
    if sys.platform.startswith("linux"):
        return list(_LINUX_FONT_CANDIDATES)
    if sys.platform.startswith("win"):
        return list(_WINDOWS_FONT_CANDIDATES)
    return []


def _find_font(font_family, size: int) -> ImageFont.FreeTypeFont:
    """Resolve a font, trying (in order): the style's configured
    font_family (a string or list; entries may be file paths or installed
    family names), then this platform's default candidates.

    If every candidate fails to load, falls back to PIL's built-in bitmap
    font (tiny, does not render Japanese) -- but prints a loud warning
    first, since a silent fallback to that font was a known trap (captions
    would render as near-invisible boxes with no clear cause).
    """
    if font_family is None:
        style_candidates: list[str] = []
    elif isinstance(font_family, str):
        style_candidates = [font_family]
    else:
        style_candidates = [str(f) for f in font_family]

    candidates = style_candidates + _platform_font_candidates()

    for font_name in candidates:
        try:
            path = Path(font_name)
            if path.exists():
                return ImageFont.truetype(str(path), size)
            return ImageFont.truetype(font_name, size)
        except (OSError, IOError):
            continue

    print(
        "WARNING: AlwaysWhisper could not load any of the configured or "
        f"platform-default fonts (tried: {candidates!r}); falling back to "
        "PIL's built-in bitmap font, which is tiny and cannot render "
        "Japanese text. Set the caption style's text.font_family to a "
        "valid font file path or an installed family name."
    )
    return ImageFont.load_default()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def render_caption_frame(
    text: str,
    frame_width: int,
    frame_height: int,
    style: dict | None = None,
    full_text: str | None = None,
) -> Image.Image:
    """Render a caption overlay frame (RGBA).

    Returns an RGBA image the same size as the video frame,
    with the caption text on a rounded rectangle background.

    Args:
        text: Text to render (may include cursor)
        frame_width: Video frame width
        frame_height: Video frame height
        style: Style configuration
        full_text: Full caption text for stable position calculation.
                   If provided, background size/position is based on this
                   instead of text, preventing layout shifts during typewriter.
    """
    if style is None:
        style = DEFAULT_STYLE

    # Merge with defaults
    bg_style = {**DEFAULT_STYLE["background"], **style.get("background", {})}
    text_style = {**DEFAULT_STYLE["text"], **style.get("text", {})}
    shadow_style = {**DEFAULT_STYLE["shadow"], **style.get("shadow", {})}
    pos_style = {**DEFAULT_STYLE["position"], **style.get("position", {})}

    # Create transparent overlay
    overlay = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if not text:
        return overlay

    # Load font
    font = _find_font(text_style.get("font_family"), text_style.get("font_size", 48))

    # Calculate text size based on full_text for stable positioning
    layout_text = full_text if full_text else text
    bbox = draw.textbbox((0, 0), layout_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    padding = bg_style.get("padding", 16)
    corner_radius = bg_style.get("corner_radius", 12)
    margin_bottom = pos_style.get("margin_bottom", 60)

    # Background rectangle dimensions
    bg_width = text_width + padding * 2
    bg_height = text_height + padding * 2

    # Position (center bottom)
    bg_x = (frame_width - bg_width) // 2
    bg_y = frame_height - margin_bottom - bg_height

    # Draw rounded rectangle background
    bg_color = tuple(bg_style.get("color", [0, 0, 0, 200]))
    draw.rounded_rectangle(
        [bg_x, bg_y, bg_x + bg_width, bg_y + bg_height],
        radius=corner_radius,
        fill=bg_color,
    )

    # Text position
    text_x = bg_x + padding
    text_y = bg_y + padding

    # Draw shadow
    if shadow_style.get("enabled", True):
        shadow_offset = shadow_style.get("offset", [2, 2])
        shadow_color = shadow_style.get("color", "#333333")
        if isinstance(shadow_color, str):
            shadow_rgb = _hex_to_rgb(shadow_color)
        else:
            shadow_rgb = tuple(shadow_color[:3])
        draw.text(
            (text_x + shadow_offset[0], text_y + shadow_offset[1]),
            text,
            font=font,
            fill=(*shadow_rgb, 255),
        )

    # Draw text
    text_color = text_style.get("color", "#FFFFFF")
    if isinstance(text_color, str):
        text_rgb = _hex_to_rgb(text_color)
    else:
        text_rgb = tuple(text_color[:3])
    draw.text((text_x, text_y), text, font=font, fill=(*text_rgb, 255))

    return overlay
