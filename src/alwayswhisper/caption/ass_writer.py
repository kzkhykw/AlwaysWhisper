from __future__ import annotations

"""ASS subtitle generator for libass-backed caption burn-in (fast mode).

Replicates the typewriter effect of overlay.py/renderer.py with ASS
\\alpha animations so libass can render captions at native speed,
avoiding the per-frame PIL/MoviePy bottleneck.

Visual differences from the standard path:
- Background is rectangular (libass BorderStyle=4 has no corner radius).
- Padding around the text matches the style's `padding` value loosely via
  font outline; visible bounds may be slightly tighter than the PIL render.
"""

from ..core.srt_parser import SrtFile


def _hex_to_bbggrr(hex_color: str) -> str:
    """Convert '#RRGGBB' to ASS 'BBGGRR' uppercase hex."""
    h = hex_color.lstrip("#")
    rr, gg, bb = h[0:2], h[2:4], h[4:6]
    return (bb + gg + rr).upper()


def _color_to_ass(value, default_rgba=(255, 255, 255, 255)) -> str:
    """Convert hex string or [R,G,B,A] list to ASS '&HAABBGGRR'.

    ASS alpha is inverted (00=opaque, FF=transparent).
    """
    if isinstance(value, str):
        return f"&H00{_hex_to_bbggrr(value)}"
    if isinstance(value, (list, tuple)):
        r, g, b = value[0], value[1], value[2]
        alpha = value[3] if len(value) >= 4 else 255
    else:
        r, g, b, alpha = default_rgba
    ass_alpha = 255 - int(alpha)
    return f"&H{ass_alpha:02X}{int(b):02X}{int(g):02X}{int(r):02X}"


def _format_ass_time(ms: int) -> str:
    """Format milliseconds as H:MM:SS.cc (centiseconds, libass format)."""
    if ms < 0:
        ms = 0
    cs_total = ms // 10
    h = cs_total // 360000
    cs_total -= h * 360000
    m = cs_total // 6000
    cs_total -= m * 6000
    s = cs_total // 100
    cs = cs_total - s * 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_text(text: str) -> str:
    """Escape ASS-meaningful characters in subtitle body text."""
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{").replace("}", "\\}")
    text = text.replace("\n", "\\N")
    return text


def _build_typewriter_text(text: str, duration_ms: int, completion_ms: int) -> str:
    """Per-character ASS overrides that mirror calculate_visible_chars + cursor.

    Strategy:
    - Each character starts hidden (alpha=FF) and animates to visible (alpha=00)
      at the millisecond it would first appear under the Python typewriter.
    - The first character is treated as immediately visible to match the
      `max(1, ...)` clamp in calculate_visible_chars.
    - A trailing '|' cursor stays visible until the reveal is complete, then
      animates out at `cap_ms`.
    """
    n = len(text)
    if n == 0:
        return ""

    cap_ms = min(completion_ms, duration_ms) if duration_ms > 0 else completion_ms
    if cap_ms <= 0:
        return _escape_text(text)

    reveal_ms: list[int] = []
    last = -1
    for i in range(n):
        # Find earliest elapsed_ms (1..cap_ms) at which char index i is visible.
        # Probe at integer ms; this matches the Python implementation's int().
        t_visible = cap_ms  # default to end if never falls earlier
        # Char i becomes visible when calculate_visible_chars > i.
        # The transition happens at ceil((i+1)/n * cap_ms) for i>=1; char 0
        # is forced visible at any t>0.
        if i == 0:
            t_visible = 0
        else:
            # Find smallest t in [1, cap_ms] with visible >= i+1
            # Use the closed-form: t such that int(t/cap_ms * n) >= i+1
            # → t/cap_ms >= (i+1)/n → t >= (i+1)/n * cap_ms (ceil to int ms)
            raw = (i + 1) * cap_ms / n
            t_visible = int(raw)
            if t_visible * n < (i + 1) * cap_ms:
                t_visible += 1  # ceil
            if t_visible > cap_ms:
                t_visible = cap_ms
        # Enforce monotonic non-decreasing (defensive)
        if t_visible <= last:
            t_visible = last + 1
        reveal_ms.append(t_visible)
        last = t_visible

    # We animate \1a (text fill) + \3a (outline) only, leaving \4a (box back)
    # at its style default so the background stays continuous across hidden
    # glyphs. We also collapse hidden glyphs to zero advance via \fscx0 so the
    # box width tracks the visible text (and the trailing cursor sits right
    # after the latest revealed character, matching the PIL renderer).
    parts: list[str] = []
    for i, ch in enumerate(text):
        ch_esc = _escape_text(ch)
        t = reveal_ms[i]
        if t <= 0:
            parts.append(f"{{\\1a&H00&\\3a&H00&\\fscx100}}{ch_esc}")
        else:
            parts.append(
                f"{{\\1a&HFF&\\3a&HFF&\\fscx0"
                f"\\t({t},{t + 1},\\1a&H00&\\3a&H00&\\fscx100)}}{ch_esc}"
            )

    # Cursor: shown next to the latest revealed char; hidden + zero-width
    # once the line has fully revealed at cap_ms.
    if n > 1:
        parts.append(
            f"{{\\1a&H00&\\3a&H00&\\fscx100"
            f"\\t({cap_ms},{cap_ms + 1},\\1a&HFF&\\3a&HFF&\\fscx0)}}|"
        )

    return "".join(parts)


def _pick_font_name(font_family) -> str:
    """Pick a family name from the style's font_family list (skip paths)."""
    if isinstance(font_family, str):
        return font_family
    if isinstance(font_family, (list, tuple)):
        for cand in font_family:
            if cand and not str(cand).startswith("/"):
                return str(cand)
        # Fall back to first entry even if path-like
        if font_family:
            name = str(font_family[0])
            # Strip directory for libass/fontconfig
            return name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return "Hiragino Sans"


def srt_to_ass(
    srt: SrtFile,
    style: dict,
    video_width: int,
    video_height: int,
) -> str:
    """Build a full .ass file content string from SRT + style config."""
    text_style = style.get("text", {})
    bg_style = style.get("background", {})
    shadow_style = style.get("shadow", {})
    pos_style = style.get("position", {})
    anim_style = style.get("animation", {})

    font_name = _pick_font_name(text_style.get("font_family", ["Hiragino Sans"]))
    font_size = int(text_style.get("font_size", 48))
    bold_flag = -1 if str(text_style.get("font_weight", "bold")).lower() == "bold" else 0

    primary = _color_to_ass(text_style.get("color", "#FFFFFF"))
    back = _color_to_ass(bg_style.get("color", [0, 0, 0, 200]))
    outline_col = _color_to_ass(shadow_style.get("color", "#333333"))

    shadow_offset = shadow_style.get("offset", [2, 2])
    shadow_dist = (
        int(shadow_offset[0]) if shadow_style.get("enabled", True) else 0
    )

    margin_v = int(pos_style.get("margin_bottom", 60))
    completion_ms = int(anim_style.get("completion_ms", 400))

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_width}\n"
        f"PlayResY: {video_height}\n"
        # WrapStyle 0 = automatic smart wrapping (lines balanced). Long captions
        # that exceed the usable width wrap onto an extra line instead of being
        # clipped off both screen edges (WrapStyle 2 = no wrap → clipping).
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # BorderStyle=3 is the ASS spec value for "opaque box" (libass).
        # Outline becomes the box padding; we approximate the style.background
        # padding by setting Outline to its half-width so the box hugs text.
        f"Style: Default,{font_name},{font_size},{primary},&H000000FF,"
        f"{outline_col},{back},{bold_flag},0,0,0,100,100,0,0,3,"
        f"{int(style.get('background', {}).get('padding', 16) / 2)},"
        f"{shadow_dist},2,40,40,{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    body_lines: list[str] = []
    for entry in srt.entries:
        if not entry.text.strip():
            continue
        start = _format_ass_time(entry.start.to_ms())
        end = _format_ass_time(entry.end.to_ms())
        text = _build_typewriter_text(
            entry.text, entry.duration_ms, completion_ms
        )
        body_lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

    return header + "".join(body_lines)


__all__ = ["srt_to_ass"]
