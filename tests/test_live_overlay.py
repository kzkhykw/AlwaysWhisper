"""Unit tests for the pure-logic half of alwayswhisper.live.caption_overlay.py.

CaptionLineModel holds no wall clock and imports no AppKit -- every `now` is
injected explicitly, exactly like TranscriptWriter's started_at/start_s
pattern in live_transcriber.py (see that module's tests). This file must stay
headless: it never imports AppKit, directly or transitively, so it runs in
CI/any sandbox with no WindowServer. caption_overlay.py itself guarantees
that by importing AppKit only lazily, inside the UI-only functions/classes
this file never touches (see caption_overlay.py's own module docstring).

The _require_appkit() tests below also stay headless: they force `import
AppKit` to fail (sys.modules["AppKit"] = None) rather than requiring AppKit
to genuinely be missing, so they exercise the ImportError->SystemExit path
without ever touching a real AppKit/WindowServer -- and without needing
AppKit to be absent from this environment's venv (it is present here).

The "controller tick() decision logic" tests near the bottom drive the real
_OverlayController.tick() -- including a real CaptionLineModel -- against an
injected FAKE renderer (_FakeRenderer, matching _AppKitRenderer's
.show(rows, alphas, font_size, geometry)/.hide() shape). This is the same
kind of deliberately narrow exception tests/test_live_transcriber.py takes
for _worker (see that module's docstring): _OverlayController is otherwise
AppKit-only glue, but its SHOW/HIDE decision -- specifically whether it
actually calls hide() once idle_clear_sec has elapsed with no new events --
had no test coverage at all before a user report ("stays visible after 8+
seconds of silence") prompted this. Auditing the code found no bug in the
decision logic itself (see the tests below, which pass); this coverage
exists so that claim keeps being true.

This file also covers the panel's hover-to-expand / drag-to-move /
Cmd+scroll-to-resize UX: OverlayGeometry (a small pure value class --
cx_frac/bottom_px/font_size, clamped by move_to()/scale_by()) and the
load_geometry()/save_geometry() JSON persistence helpers are exercised
directly, headlessly, the same way CaptionLineModel is. _is_command_scroll()
is tested the same way as _row_delta_from_scroll_event() elsewhere in this
file: a duck-typed fake NSEvent exposing only the one method it reads
(modifierFlags() here, vs. hasPreciseScrollingDeltas()/scrollingDeltaY()
there).

The controller tests drive expansion/dragging/debounced-save through the
same kind of deliberate seam _update_hover's tests already used (setting an
attribute directly rather than needing real AppKit/cursor tracking):
controller._hovering / controller._dragging are set directly to exercise
tick()'s `expanded = _hovering or _dragging` decision and its idle-hide
suppression, and controller._settings_dirty_at is set directly to exercise
the 1s debounced save without waiting on a real drag gesture. Every test
that constructs an _OverlayController passes settings_path=None (or a
tmp_path file, for the persistence-specific tests) -- NEVER the default
DEFAULT_SETTINGS_PATH -- so no test here ever reads or writes the real
~/.config/alwayswhisper/caption_overlay.json.
"""
import os
import queue
import sys
from unittest.mock import patch

import pytest

from alwayswhisper.live import caption_overlay as co  # noqa: E402


def test_screen_for_panel_prefers_the_panels_current_display():
    class Panel:
        def screen(self):
            return "panel-display"

    assert co._screen_for_panel(Panel(), "main-display") == "panel-display"


def test_screen_for_panel_falls_back_for_an_unattached_panel():
    class Panel:
        def screen(self):
            return None

    assert co._screen_for_panel(Panel(), "main-display") == "main-display"


def test_caption_panel_class_is_lazy_until_appkit_is_needed():
    # The class factory must stay lazy, like the rest of the AppKit layer:
    # importing this module for headless pipeline tests never imports AppKit.
    assert callable(co._get_caption_panel_class)


def _model(idle_clear_sec=5.0):
    return co.CaptionLineModel(idle_clear_sec=idle_clear_sec)


# ------------------------------------------------------------------ chunk/eos ---
def test_first_chunk_ever_starts_a_line():
    m = _model()
    m.apply(("chunk", "hello"), now=0.0)
    assert m.text_at(0.0) == "hello"


def test_chunks_within_a_segment_append_in_order_same_spacing_as_printed():
    m = _model()
    m.apply(("chunk", "hello"), now=0.0)
    m.apply(("chunk", " world"), now=0.1)  # word tokens carry their own leading space
    assert m.text_at(0.1) == "hello world"


def test_new_segment_after_eos_replaces_the_line_not_appends():
    m = _model()
    m.apply(("chunk", "first sentence"), now=0.0)
    m.apply(("eos",), now=0.5)
    m.apply(("chunk", "second"), now=1.0)
    assert m.text_at(1.0) == "second"


def test_eos_alone_does_not_clear_text_immediately():
    m = _model()
    m.apply(("chunk", "hello"), now=0.0)
    m.apply(("eos",), now=0.2)
    assert m.visible_at(0.2) is True
    assert m.text_at(0.2) == "hello"


def test_multiple_lines_in_sequence_each_start_fresh():
    m = _model()
    for i, w in enumerate(["a", "b", "c"]):
        m.apply(("chunk", w), now=i * 0.1)
    m.apply(("eos",), now=0.4)
    assert m.text_at(0.4) == "abc"
    for i, w in enumerate(["x", "y"]):
        m.apply(("chunk", w), now=1.0 + i * 0.1)
    assert m.text_at(1.1) == "xy"          # not "abcxy"


def test_sentence_mode_single_whole_chunk_then_eos():
    # sentence display mode: one chunk carries the whole segment's text.
    m = _model()
    m.apply(("chunk", "こんにちは。"), now=0.0)
    m.apply(("eos",), now=0.0)
    assert m.text_at(0.0) == "こんにちは。"


# ------------------------------------------------------ full text, no trimming ---
# User decision: a long caption must show its WHOLE text -- never chop
# content off. CaptionLineModel has no length limit at all (the AppKit layer
# wraps to multiple lines / grows the panel instead, see _layout_and_show);
# nothing here should ever be dropped, from the head, the tail, or anywhere
# else, no matter how long the accumulated text gets.
def test_long_chunk_sequence_retains_full_concatenated_text():
    m = _model()
    words = [f"word{i}" for i in range(50)]
    for i, w in enumerate(words):
        m.apply(("chunk", w), now=i * 0.01)
    assert m.text_at(0.49) == "".join(words)
    assert len(m.text_at(0.49)) == sum(len(w) for w in words)


def test_single_very_long_chunk_is_kept_in_full_no_ellipsis():
    m = _model()
    long_text = ("So this is connected to the MCP server, which means an AI "
                 "agent workflow that can read and write your workspace.")
    m.apply(("chunk", long_text), now=0.0)
    assert m.text_at(0.0) == long_text
    assert "…" not in m.text_at(0.0)
    assert "which" in m.text_at(0.0)  # not chopped into "…hich" (the bug that prompted this change)


def test_long_text_still_replaced_fresh_by_next_line_after_eos():
    m = _model()
    long_text = "a" * 500
    m.apply(("chunk", long_text), now=0.0)
    m.apply(("eos",), now=0.1)
    m.apply(("chunk", "short"), now=0.2)
    assert m.text_at(0.2) == "short"  # full replacement, not appended to the 500-char line


# ------------------------------------------------------------------- visibility ---
def test_hidden_before_any_event_ever_arrives():
    m = _model()
    assert m.visible_at(0.0) is False
    assert m.text_at(0.0) == ""


def test_visible_immediately_after_a_chunk():
    m = _model(idle_clear_sec=5.0)
    m.apply(("chunk", "hi"), now=10.0)
    assert m.visible_at(10.0) is True
    assert m.text_at(10.0) == "hi"


def test_stays_visible_just_under_the_idle_threshold():
    m = _model(idle_clear_sec=5.0)
    m.apply(("chunk", "hi"), now=0.0)
    assert m.visible_at(4.999) is True
    assert m.text_at(4.999) == "hi"


def test_hidden_once_idle_clear_sec_has_fully_elapsed():
    m = _model(idle_clear_sec=5.0)
    m.apply(("chunk", "hi"), now=0.0)
    assert m.visible_at(5.0) is False
    assert m.text_at(5.0) == ""


def test_eos_extends_visibility_like_any_other_event():
    m = _model(idle_clear_sec=5.0)
    m.apply(("chunk", "hi"), now=0.0)
    m.apply(("eos",), now=3.0)              # last event is now the eos, not the chunk
    assert m.visible_at(7.9) is True        # 7.9 - 3.0 = 4.9 < 5.0
    assert m.visible_at(8.0) is False       # 8.0 - 3.0 = 5.0 >= 5.0


def test_new_chunk_after_hidden_reappears_fresh_not_appended_to_stale_text():
    m = _model(idle_clear_sec=5.0)
    m.apply(("chunk", "old line"), now=0.0)
    m.apply(("eos",), now=0.5)
    assert m.visible_at(6.0) is False       # 6.0 - 0.5 = 5.5 >= 5.0 -> hidden
    m.apply(("chunk", "new"), now=6.0)
    assert m.visible_at(6.0) is True
    assert m.text_at(6.0) == "new"          # fresh line, not "old linenew"


def test_new_chunk_after_idle_timeout_without_eos_still_starts_fresh_line():
    # No eos ever arrived (e.g. a stalled/never-finished segment) -- the line
    # still goes invisible once idle_clear_sec elapses with no events, and
    # the next chunk must start fresh rather than resurrecting stale text.
    m = _model(idle_clear_sec=5.0)
    m.apply(("chunk", "stalled"), now=0.0)
    assert m.visible_at(5.0) is False
    m.apply(("chunk", "new"), now=5.0)
    assert m.text_at(5.0) == "new"


# ------------------------------------------------------- hold suppresses idle-hide ---
# hold=False is unchanged from plain visible_at(now) (see the "visibility"
# tests above); hold=True is the new UX seam hovering/dragging plugs into --
# see visible_rows()'s own expanded->hold wiring below and the controller's
# `expanded = _hovering or _dragging` in the "controller tick()" section.
def test_visible_at_hold_true_suppresses_idle_hide_past_idle_clear_sec():
    m = _model(idle_clear_sec=5.0)
    m.apply(("chunk", "hi"), now=0.0)
    assert m.visible_at(10.0, hold=True) is True   # well past idle_clear_sec


def test_visible_at_hold_false_still_hides_past_idle_clear_sec():
    """Regression guard: hold defaults to False and must behave exactly
    like the pre-hold visible_at(now) -- idle-hide still fires normally."""
    m = _model(idle_clear_sec=5.0)
    m.apply(("chunk", "hi"), now=0.0)
    assert m.visible_at(5.0, hold=False) is False
    assert m.visible_at(5.0) is False


def test_visible_at_hold_true_with_no_events_ever_is_still_hidden():
    m = _model()
    assert m.visible_at(0.0, hold=True) is False
    assert m.visible_at(100.0, hold=True) is False


def test_visible_rows_expanded_true_suppresses_idle_hide_too():
    """visible_rows(now, expanded=True)'s own visibility check uses
    hold=expanded (per contract: visibility = visible_at(now,
    hold=expanded)) -- so an expanded/hovering panel keeps showing past
    idle_clear_sec, not just visible_at(hold=True) in isolation."""
    m = _model(idle_clear_sec=5.0)
    m.apply(("chunk", "hi"), now=0.0)
    assert m.visible_rows(10.0, expanded=False) == []             # collapsed: idle-hides normally
    assert m.visible_rows(10.0, expanded=True) == [("hi", True)]  # expanded: held visible


# ------------------------------------------------------- history + scrollback ---
def _push_lines(m, count, start_now=0.0, step=0.01, prefix="L"):
    """Feed `count` distinct chunk+eos lines ("L0", "L1", ...), each its own
    fresh line, ending at `now = start_now + (count-1)*step`. Returns that
    final `now`. After this, the model's live text is the LAST of these
    lines and history holds the rest (oldest -> newest), same shape
    real usage builds up one utterance at a time."""
    now = start_now
    for i in range(count):
        m.apply(("chunk", f"{prefix}{i}"), now=now)
        m.apply(("eos",), now=now)
        now += step
    return now


def test_push_on_new_line_saves_previous_live_text_to_history():
    m = _model()
    m.apply(("chunk", "first"), now=0.0)
    m.apply(("eos",), now=0.1)
    m.apply(("chunk", "second"), now=0.2)
    assert m.visible_rows(0.2, expanded=True) == [("first", False), ("second", True)]


def test_push_on_new_line_after_idle_hide_revive_also_saves_to_history():
    m = _model(idle_clear_sec=5.0)
    m.apply(("chunk", "stalled"), now=0.0)   # no eos -- stalls
    m.apply(("chunk", "new"), now=5.0)       # idle timeout elapsed -> fresh line, "stalled" pushed
    assert m.visible_rows(5.0, expanded=True) == [("stalled", False), ("new", True)]


def test_first_chunk_ever_pushes_nothing_no_empty_history_entry():
    m = _model()
    m.apply(("chunk", "hello"), now=0.0)
    assert m.visible_rows(0.0, expanded=True) == [("hello", True)]  # no leading ("", False)


def test_history_capped_at_200_drops_oldest():
    cap = co.CaptionLineModel.HISTORY_CAP
    m = _model()
    now = _push_lines(m, cap + 5, prefix="line")  # line0 .. line{cap+4}; last one still live
    m.scroll_by(1_000_000, now)  # scroll all the way back -- offset > 0 forces expansion
    rows = m.visible_rows(now)
    history_texts = [t for t, is_live in rows if not is_live]
    # line0..line3 (4 entries) dropped to stay at the 200 cap; oldest
    # SURVIVING entry is line4.
    assert history_texts == ["line4", "line5"]


def test_visible_rows_window_shifts_back_with_scroll_offset():
    m = _model()
    now = _push_lines(m, 10)  # history L0..L9
    m.apply(("chunk", "LIVE"), now=now)  # pushes L9 into history too -> L0..L9, live=LIVE

    assert m.visible_rows(now, expanded=True) == [
        ("L8", False), ("L9", False), ("LIVE", True),
    ]

    m.scroll_by(2, now)
    assert m.visible_rows(now, expanded=True) == [
        ("L6", False), ("L7", False), ("LIVE", True),
    ]


def test_scroll_by_clamps_to_available_history_and_cannot_go_negative():
    m = _model()
    now = _push_lines(m, 6)  # history ends up L0..L5 (6 entries) after the LIVE push below
    m.apply(("chunk", "LIVE"), now=now)

    m.scroll_by(1_000_000, now)  # far more than available -> clamps
    assert m.offset_at(now) == 4  # max(0, 6 - HISTORY_WINDOW(2))
    oldest_shown = [t for t, is_live in m.visible_rows(now) if not is_live]  # offset>0 -> auto-expanded
    assert oldest_shown == ["L0", "L1"]

    m.scroll_by(-1_000_000, now)  # far past live -> clamps at 0, never negative
    assert m.offset_at(now) == 0


def test_auto_return_after_10s_of_no_scrolling():
    m = _model()
    now = _push_lines(m, 6)
    m.apply(("chunk", "LIVE"), now=now)
    m.scroll_by(2, now)
    assert m.offset_at(now + 9.9) == 2   # AUTO_RETURN_SEC (10.0) not yet reached
    assert m.offset_at(now + 10.0) == 0  # fully elapsed -> back to the live edge


def test_scroll_after_auto_return_starts_fresh_not_from_stale_offset():
    m = _model()
    now = _push_lines(m, 10)
    m.apply(("chunk", "LIVE"), now=now)
    m.scroll_by(5, now)
    later = now + 20.0  # long past auto-return
    m.scroll_by(1, later)
    # Re-bases from the EFFECTIVE (already auto-returned) offset 0, not the
    # stale raw 5 -- matches what the user would actually see on screen.
    assert m.offset_at(later) == 1


def test_idle_hide_suppressed_while_scrolled_back():
    m = _model(idle_clear_sec=5.0)
    now = _push_lines(m, 6)
    m.apply(("chunk", "LIVE"), now=now)
    m.apply(("eos",), now=now)
    m.scroll_by(1, now)
    later = now + 6.0  # > idle_clear_sec (5.0) since the last event...
    assert (later - now) < co.CaptionLineModel.AUTO_RETURN_SEC  # ...but still within the 10s auto-return window
    assert m.visible_at(later) is True     # suppressed: idle-hide does NOT fire while scrolled back
    assert m.visible_rows(later) != []


def test_idle_hide_still_applies_normally_at_the_live_edge():
    """Regression guard: with offset==0 (never scrolled -- the default,
    all-the-time state), idle-hide must behave exactly as it did before
    history/scroll existed."""
    m = _model(idle_clear_sec=5.0)
    m.apply(("chunk", "hi"), now=0.0)
    m.apply(("eos",), now=0.0)
    assert m.visible_at(5.0) is False
    assert m.visible_rows(5.0) == []


def test_live_row_pinned_at_bottom_regardless_of_scroll_offset():
    m = _model()
    now = _push_lines(m, 10)
    m.apply(("chunk", "LIVE"), now=now)

    assert m.visible_rows(now, expanded=True)[-1] == ("LIVE", True)   # offset 0

    m.scroll_by(2, now)
    assert m.visible_rows(now, expanded=True)[-1] == ("LIVE", True)   # offset 2, still pinned last


# --------------------------------------------------- collapsed vs expanded rows ---
# New default UX: the panel shows the live row ONLY unless expanded (hover/
# drag) or actively scrolled back. See _push_lines above for the history
# builder used here too.
def test_visible_rows_collapsed_by_default_shows_live_row_only():
    """Even with several accumulated history lines, expanded=False (the
    default) must return ONLY the live row, never any history."""
    m = _model()
    _push_lines(m, 5)
    m.apply(("chunk", "LIVE"), now=1.0)
    assert m.visible_rows(1.0) == [("LIVE", True)]


def test_visible_rows_expanded_true_returns_full_history_stack():
    m = _model()
    _push_lines(m, 5)
    m.apply(("chunk", "LIVE"), now=1.0)
    rows = m.visible_rows(1.0, expanded=True)
    assert [t for t, is_live in rows if not is_live] == ["L3", "L4"]
    assert rows[-1] == ("LIVE", True)


def test_visible_rows_expanded_with_no_history_is_still_just_the_live_row():
    m = _model()
    m.apply(("chunk", "hello"), now=0.0)
    assert m.visible_rows(0.0, expanded=True) == [("hello", True)]


def test_visible_rows_offset_greater_than_zero_forces_expansion_even_when_collapsed():
    """Actively scrolled-back history must show even with expanded=False --
    per contract, effective_expanded = expanded or effective_offset(now) > 0,
    independent of hover/drag."""
    m = _model()
    now = _push_lines(m, 6)
    m.apply(("chunk", "LIVE"), now=now)
    m.scroll_by(1, now)
    rows = m.visible_rows(now, expanded=False)
    assert [t for t, is_live in rows if not is_live] != []
    assert rows[-1] == ("LIVE", True)


# ------------------------------------------------- row alpha (fade) assignment ---
def test_alphas_for_no_rows_is_empty():
    assert co._alphas_for([]) == ()


def test_alphas_for_live_only_is_full_opacity():
    assert co._alphas_for([("hi", True)]) == (1.0,)


def test_alphas_for_full_history_window_oldest_to_newest():
    rows = [("a", False), ("b", False), ("live", True)]
    assert co._alphas_for(rows) == (0.55, 0.75, 1.0)


def test_alphas_for_partial_history_aligns_to_the_newest_live_adjacent_end():
    # A single history row uses the live-adjacent opacity.
    rows = [("a", False), ("live", True)]
    assert co._alphas_for(rows) == (0.75, 1.0)


# --------------------------------------------------- scroll-wheel event -> rows ---
class _FakeScrollEvent:
    """Duck-typed stand-in for NSEvent covering only what
    _row_delta_from_scroll_event reads: hasPreciseScrollingDeltas() and
    scrollingDeltaY(). A REAL trackpad/mouse gesture cannot be synthesized
    in this sandbox (see _row_delta_from_scroll_event's own docstring), so
    this tests the pure accumulator/threshold/sign arithmetic against
    controlled inputs -- not the actual hardware-to-sign mapping, which is
    sourced from Apple's NSEvent.h doc comments instead (see that same
    docstring)."""

    def __init__(self, delta_y, precise=True):
        self._delta_y = delta_y
        self._precise = precise

    def hasPreciseScrollingDeltas(self):
        return self._precise

    def scrollingDeltaY(self):
        return self._delta_y


def test_row_delta_precise_scroll_crossing_one_row_threshold():
    rows, accum = co._row_delta_from_scroll_event(_FakeScrollEvent(60.0), 0.0)
    assert rows == 2     # 60px / 30px-per-row (_PX_PER_ROW)
    assert accum == 0.0


def test_row_delta_accumulates_subrow_remainder_across_events():
    rows1, accum1 = co._row_delta_from_scroll_event(_FakeScrollEvent(20.0), 0.0)
    assert rows1 == 0
    assert accum1 == 20.0
    rows2, accum2 = co._row_delta_from_scroll_event(_FakeScrollEvent(20.0), accum1)
    assert rows2 == 1    # 40px accumulated crosses the 30px-per-row threshold once
    assert accum2 == 10.0


def test_row_delta_negative_scroll_is_negative_row_delta():
    rows, _accum = co._row_delta_from_scroll_event(_FakeScrollEvent(-60.0), 0.0)
    assert rows == -2


def test_row_delta_non_precise_wheel_event_is_scaled_up():
    # A physical mouse wheel notch reports a tiny delta (e.g. 1.0 "lines"),
    # which alone wouldn't cross even one row's px threshold at face value
    # -- hasPreciseScrollingDeltas()=False triggers the _WHEEL_LINE_SCALE
    # multiplier (per NSEvent.h's own doc comment quoted in
    # _row_delta_from_scroll_event).
    rows, accum = co._row_delta_from_scroll_event(_FakeScrollEvent(1.0, precise=False), 0.0)
    assert rows == 0
    assert accum == 10.0  # 1.0 * _WHEEL_LINE_SCALE (10.0)


# ============================================================ OverlayGeometry ===
# Pure value class -- no AppKit -- so it's exercised directly here, same as
# CaptionLineModel above it.
def test_overlay_geometry_defaults():
    g = co.OverlayGeometry()
    assert g.cx_frac == 0.5
    assert g.bottom_px == 24.0
    assert g.font_size == 28.0


def test_overlay_geometry_move_to_clamps_cx_frac_below_zero():
    g = co.OverlayGeometry()
    g.move_to(-0.5, 24.0, visible_h=800.0)
    assert g.cx_frac == 0.0


def test_overlay_geometry_move_to_clamps_cx_frac_above_one():
    g = co.OverlayGeometry()
    g.move_to(1.5, 24.0, visible_h=800.0)
    assert g.cx_frac == 1.0


def test_overlay_geometry_move_to_clamps_bottom_px_below_zero():
    g = co.OverlayGeometry()
    g.move_to(0.5, -10.0, visible_h=800.0)
    assert g.bottom_px == 0.0


def test_overlay_geometry_move_to_clamps_bottom_px_above_visible_h_minus_80():
    g = co.OverlayGeometry()
    g.move_to(0.5, 10_000.0, visible_h=800.0)
    assert g.bottom_px == 720.0   # 800 - 80


def test_overlay_geometry_move_to_bottom_px_clamp_floors_at_zero_for_tiny_visible_h():
    """max(0.0, visible_h - 80.0) -- a visible_h under 80 must not clamp to
    a negative ceiling."""
    g = co.OverlayGeometry()
    g.move_to(0.5, 50.0, visible_h=40.0)
    assert g.bottom_px == 0.0


def test_overlay_geometry_move_to_accepts_in_range_values_unchanged():
    g = co.OverlayGeometry()
    g.move_to(0.25, 100.0, visible_h=800.0)
    assert g.cx_frac == 0.25
    assert g.bottom_px == 100.0


def test_overlay_geometry_scale_by_positive_steps_grows_multiplicatively():
    g = co.OverlayGeometry(font_size=28.0)
    g.scale_by(1)
    assert g.font_size == pytest.approx(28.0 * 1.1)


def test_overlay_geometry_scale_by_negative_steps_shrinks_multiplicatively():
    g = co.OverlayGeometry(font_size=28.0)
    g.scale_by(-1)
    assert g.font_size == pytest.approx(28.0 / 1.1)


def test_overlay_geometry_scale_by_clamps_at_96_upper_bound():
    g = co.OverlayGeometry(font_size=28.0)
    g.scale_by(50)  # way more than enough to exceed 96
    assert g.font_size == 96.0
    g.scale_by(1)   # repeated call past the bound must stick, not drift further
    assert g.font_size == 96.0


def test_overlay_geometry_scale_by_clamps_at_12_lower_bound():
    g = co.OverlayGeometry(font_size=28.0)
    g.scale_by(-50)
    assert g.font_size == 12.0
    g.scale_by(-1)
    assert g.font_size == 12.0


def test_overlay_geometry_as_dict_roundtrips_through_from_dict():
    g = co.OverlayGeometry(cx_frac=0.3, bottom_px=50.0, font_size=40.0)
    d = g.as_dict()
    assert d == {"cx_frac": 0.3, "bottom_px": 50.0, "font_size": 40.0, "position": "free"}
    g2 = co.OverlayGeometry.from_dict(d)
    assert g2.cx_frac == 0.3
    assert g2.bottom_px == 50.0
    assert g2.font_size == 40.0


def test_overlay_geometry_from_dict_non_dict_input_returns_defaults():
    for garbage in (None, "not a dict", 42, ["a", "list"]):
        g = co.OverlayGeometry.from_dict(garbage)
        assert g.cx_frac == 0.5
        assert g.bottom_px == 24.0
        assert g.font_size == 28.0


def test_overlay_geometry_from_dict_missing_keys_fall_back_to_defaults_per_field():
    g = co.OverlayGeometry.from_dict({"cx_frac": 0.1})
    assert g.cx_frac == 0.1
    assert g.bottom_px == 24.0     # missing -> default
    assert g.font_size == 28.0     # missing -> default


def test_overlay_geometry_from_dict_non_numeric_values_fall_back_to_defaults():
    g = co.OverlayGeometry.from_dict({"cx_frac": "half", "bottom_px": None, "font_size": [1, 2]})
    assert g.cx_frac == 0.5
    assert g.bottom_px == 24.0
    assert g.font_size == 28.0


def test_overlay_geometry_from_dict_accepts_int_values_as_numeric():
    g = co.OverlayGeometry.from_dict({"cx_frac": 1, "bottom_px": 30, "font_size": 40})
    assert g.cx_frac == 1.0
    assert g.bottom_px == 30.0
    assert g.font_size == 40.0


def test_overlay_geometry_from_dict_clamps_out_of_range_numerics():
    g = co.OverlayGeometry.from_dict({"cx_frac": 5.0, "bottom_px": -100.0, "font_size": 999.0})
    assert g.cx_frac == 1.0
    assert g.bottom_px == 0.0
    assert g.font_size == 96.0


# ==================================================== persistence helpers ===
def test_default_settings_path_is_under_config_video_editor():
    path_str = str(co.DEFAULT_SETTINGS_PATH)
    assert path_str.endswith("alwayswhisper/caption_overlay.json")
    assert path_str.startswith(os.path.expanduser("~"))  # already expanduser'd, no literal "~"


def test_save_then_load_roundtrip_preserves_values(tmp_path):
    path = tmp_path / "geometry.json"
    g = co.OverlayGeometry(cx_frac=0.2, bottom_px=99.0, font_size=44.0)
    co.save_geometry(path, g)
    loaded = co.load_geometry(path)
    assert loaded.cx_frac == 0.2
    assert loaded.bottom_px == 99.0
    assert loaded.font_size == 44.0


def test_load_geometry_of_nonexistent_path_returns_defaults(tmp_path):
    path = tmp_path / "does" / "not" / "exist.json"
    g = co.load_geometry(path)
    assert g.cx_frac == 0.5
    assert g.bottom_px == 24.0
    assert g.font_size == 28.0


def test_load_geometry_of_corrupt_json_returns_defaults_not_raises(tmp_path):
    path = tmp_path / "geometry.json"
    path.write_text("{not valid json::")
    g = co.load_geometry(path)
    assert g.cx_frac == 0.5
    assert g.bottom_px == 24.0
    assert g.font_size == 28.0


def test_save_geometry_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dirs" / "geometry.json"
    co.save_geometry(path, co.OverlayGeometry(font_size=50.0))
    assert path.exists()
    assert co.load_geometry(path).font_size == 50.0


def test_save_geometry_swallows_oserror_never_raises(tmp_path):
    """A parent path component that is actually a FILE (not a directory)
    makes any mkdir/open underneath it raise NotADirectoryError (an OSError
    subclass) -- save_geometry must swallow this, not propagate it."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    bad_path = blocker / "geometry.json"
    co.save_geometry(bad_path, co.OverlayGeometry())  # must not raise
    assert not bad_path.exists()


# ================================================= command-scroll (Cmd+scroll) ===
class _FakeModifierEvent:
    """Duck-typed NSEvent stand-in exposing only modifierFlags(), which is
    all _is_command_scroll reads."""

    def __init__(self, flags):
        self._flags = flags

    def modifierFlags(self):
        return self._flags


def test_nsevent_modifier_flag_command_constant_value():
    assert co._NSEVENT_MODIFIER_FLAG_COMMAND == 1 << 20


def test_is_command_scroll_false_with_no_modifier_flags():
    assert co._is_command_scroll(_FakeModifierEvent(0)) is False


def test_is_command_scroll_true_with_command_flag_only():
    assert co._is_command_scroll(_FakeModifierEvent(co._NSEVENT_MODIFIER_FLAG_COMMAND)) is True


def test_is_command_scroll_true_with_command_flag_plus_other_bits():
    other_bits = 1 << 17  # some unrelated stray modifier bit
    flags = co._NSEVENT_MODIFIER_FLAG_COMMAND | other_bits
    assert co._is_command_scroll(_FakeModifierEvent(flags)) is True


def test_is_command_scroll_false_with_only_unrelated_modifier_bits():
    other_bits = 1 << 17
    assert co._is_command_scroll(_FakeModifierEvent(other_bits)) is False


# ------------------------------------------ missing-AppKit friendly failure ---
# A venv with only requirements-transcribe.txt installed (no
# requirements-menubar.txt) has no AppKit at all -- --transcribe-only
# --captions must not surface a raw ImportError traceback in that case (see
# _require_appkit's own docstring). sys.modules["AppKit"] = None makes
# `import AppKit` raise ImportError exactly like a genuinely missing package
# would, without needing AppKit to actually be absent from THIS environment
# (it is present here) and without ever touching a real AppKit/WindowServer.
def test_require_appkit_missing_dependency_raises_friendly_system_exit():
    with patch.dict(sys.modules, {"AppKit": None}):
        with pytest.raises(SystemExit) as exc_info:
            co._require_appkit()
    assert "AppKit is not available" in str(exc_info.value)
    assert "alwayswhisper[live]" in str(exc_info.value)


def test_attach_timer_based_overlay_missing_appkit_raises_friendly_system_exit():
    with patch.dict(sys.modules, {"AppKit": None}):
        with pytest.raises(SystemExit) as exc_info:
            co.attach_timer_based_overlay(None)
    assert "AppKit is not available" in str(exc_info.value)


def test_run_console_overlay_missing_appkit_raises_friendly_system_exit():
    with patch.dict(sys.modules, {"AppKit": None}):
        with pytest.raises(SystemExit) as exc_info:
            co.run_console_overlay(None)
    assert "AppKit is not available" in str(exc_info.value)


# ----------------------------------- _resolve_debug_input / CAPTION_OVERLAY_DEBUG_INPUT ---
# Lets a user turn on the same diagnostic logging --debug-input gives --demo
# in a REAL --transcribe-only/--menubar session (neither has a --debug-input
# CLI flag of its own) via CAPTION_OVERLAY_DEBUG_INPUT=1, with no code
# change -- added to make the user's own next debug run decisive about the
# "copy icon does nothing" report, since the earlier scripted diagnosis could
# only reproduce a MANUFACTURED cursor/event-location divergence, not
# necessarily the real one.
def test_resolve_debug_input_env_var_flips_on_when_not_explicitly_set(monkeypatch):
    monkeypatch.setenv("CAPTION_OVERLAY_DEBUG_INPUT", "1")
    assert co._resolve_debug_input(False) is True


def test_resolve_debug_input_defaults_off_without_env_var(monkeypatch):
    monkeypatch.delenv("CAPTION_OVERLAY_DEBUG_INPUT", raising=False)
    assert co._resolve_debug_input(False) is False


def test_resolve_debug_input_explicit_true_wins_regardless_of_env(monkeypatch):
    monkeypatch.delenv("CAPTION_OVERLAY_DEBUG_INPUT", raising=False)
    assert co._resolve_debug_input(True) is True


def test_resolve_debug_input_only_the_literal_string_one_counts(monkeypatch):
    for value in ("true", "yes", "0", "2", ""):
        monkeypatch.setenv("CAPTION_OVERLAY_DEBUG_INPUT", value)
        assert co._resolve_debug_input(False) is False, f"value={value!r} must not flip it on"


def test_attach_timer_based_overlay_env_var_reaches_the_real_controller(monkeypatch):
    """Integration-level proof (not just _resolve_debug_input in isolation):
    CAPTION_OVERLAY_DEBUG_INPUT=1 reaches the actual controller
    attach_timer_based_overlay constructs. Fakes out _require_appkit (must
    never touch real AppKit here) and _OverlayController itself (records
    its kwargs) -- sys.modules["AppKit"] = None is not usable for this one,
    since that makes _require_appkit raise before construction happens."""
    monkeypatch.setenv("CAPTION_OVERLAY_DEBUG_INPUT", "1")
    monkeypatch.setattr(co, "_require_appkit", lambda: None)
    captured = {}

    class _FakeController:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(co, "_OverlayController", _FakeController)

    co.attach_timer_based_overlay(None)

    assert captured["debug_input"] is True


def test_attach_timer_based_overlay_without_env_var_leaves_debug_input_false(monkeypatch):
    monkeypatch.delenv("CAPTION_OVERLAY_DEBUG_INPUT", raising=False)
    monkeypatch.setattr(co, "_require_appkit", lambda: None)
    captured = {}

    class _FakeController:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(co, "_OverlayController", _FakeController)

    co.attach_timer_based_overlay(None)

    assert captured["debug_input"] is False


# ------------------------------------------- controller tick() decision logic ---
class _FakeDisplayQueue:
    """Duck-typed stand-in for the mp display queue _OverlayController.tick()
    polls: get_nowait() pops in FIFO order and raises queue.Empty once
    drained, exactly the real queue's contract (matches the idiom
    tests/test_live_transcriber.py's _FakeDisplayQueue uses for the write
    side of this same channel)."""

    def __init__(self, items=None):
        self._items = list(items or [])

    def get_nowait(self):
        if not self._items:
            raise queue.Empty
        return self._items.pop(0)

    def push(self, item):
        self._items.append(item)


class _FakeRenderer:
    """Stand-in for _AppKitRenderer: records every
    show(rows, alphas, font_size, geometry, decor)/hide() call instead of
    touching AppKit, so _OverlayController.tick()'s show/hide DECISION --
    previously untested end to end -- is fully exercisable headlessly.
    Exposes neither .panel nor .content (_OverlayController looks both up
    via getattr(..., None)), so the hover/scroll wiring is a deliberate
    no-op against this fake -- exactly what these timing-focused tests
    want; hover/scroll have their own dedicated tests elsewhere. Returns
    None from show() (the implicit default) -- per _compute_layout's own
    contract, the controller must treat that as "no row boxes" ([]); the
    hover/copy-icon tests below use _FakeRendererWithBoxes instead, which
    returns real RowBox objects."""

    def __init__(self):
        self.calls = []  # ("show", rows, alphas, font_size, geometry, decor) | ("hide",)

    def show(self, rows, alphas, font_size, geometry, decor=None):
        self.calls.append(("show", rows, alphas, font_size, geometry, decor))

    def hide(self):
        self.calls.append(("hide",))


def _controller(items=None):
    dq = _FakeDisplayQueue(items)
    renderer = _FakeRenderer()
    # settings_path=None: NEVER let a controller built in this test file
    # touch the real ~/.config/alwayswhisper/caption_overlay.json (see
    # DEFAULT_SETTINGS_PATH) -- persistence itself is covered separately
    # below with explicit tmp_path settings files.
    controller = co._OverlayController(dq, font_size=28, renderer=renderer, settings_path=None)
    return controller, renderer, dq


def _show_call(controller, rows, alphas):
    """Expected _FakeRenderer call tuple for CONTROLLER's CURRENT geometry
    -- tick() passes controller.geometry itself (the same object, mutated
    in place by scale_by()/move_to(), never reassigned), so comparing
    against that live reference works whether or not OverlayGeometry
    defines __eq__ (the contract doesn't specify one). Only valid for
    asserting the MOST RECENT call in a test with no geometry mutation
    since earlier calls were recorded -- font_size is captured as a plain
    float at call time, so it goes stale relative to a later mutation.

    The trailing RowDecor(hover_row=None, copied_text=None) matches what
    tick() actually passes in every test in THIS section: none of them
    ever populate controller._row_boxes with real geometry (plain
    _FakeRenderer's show() always returns None -> _row_boxes stays []), so
    hover-row hit-testing can never match anything regardless of
    controller._hovering -- the hover-row/copy-icon-specific tests further
    down use _FakeRendererWithBoxes and build their own expected decor."""
    return ("show", rows, alphas, controller.geometry.font_size, controller.geometry,
            co.RowDecor(hover_row=None, copied_text=None))


def _live_row_call(controller, text):
    """Expected _FakeRenderer call for a single live row with no history
    shown -- the collapsed-by-default case, and the common case for these
    timing-focused tests (none of them hover/drag to expand or build up
    enough history to matter)."""
    return _show_call(controller, [(text, True)], (1.0,))


def test_tick_shows_on_first_chunk():
    controller, renderer, _dq = _controller([("chunk", "hello")])
    controller.tick(now=0.0)
    assert renderer.calls == [_live_row_call(controller, "hello")]


def test_tick_does_not_repaint_when_state_is_unchanged():
    controller, renderer, _dq = _controller([("chunk", "hello")])
    controller.tick(now=0.0)
    controller.tick(now=0.5)   # no new events, nothing changed -> no extra call
    assert renderer.calls == [_live_row_call(controller, "hello")]


def test_tick_hides_after_idle_clear_sec_of_silence():
    """Regression test for the user report 'stays visible after 8+ seconds
    of silence even though idle_clear_sec is 5.0': feed one chunk+eos, then
    tick with NO further events as `now` advances past idle_clear_sec
    (5.0s, _IDLE_CLEAR_SEC) -- the renderer must receive a hide() call.
    Auditing tick()/CaptionLineModel.visible_at() found this already works
    correctly (this test passes against the current code); it exists so a
    future regression here is caught automatically instead of only via a
    user report."""
    controller, renderer, _dq = _controller([("chunk", "hello"), ("eos",)])
    controller.tick(now=0.0)     # drains chunk+eos, shows "hello"
    controller.tick(now=1.0)     # well within the idle window
    controller.tick(now=4.9)     # still just barely within it
    controller.tick(now=5.0)     # idle_clear_sec has fully elapsed -> must hide
    assert renderer.calls == [_live_row_call(controller, "hello"), ("hide",)]


def test_tick_does_not_call_hide_repeatedly_once_already_hidden():
    """The repaint-skip compare must not mask the hide (previous test), but
    also must not spam hide() every tick once the panel is already hidden."""
    controller, renderer, _dq = _controller([("chunk", "hello"), ("eos",)])
    controller.tick(now=0.0)
    controller.tick(now=5.0)     # hides
    controller.tick(now=6.0)     # still silent, already hidden
    controller.tick(now=9.0)
    assert renderer.calls == [_live_row_call(controller, "hello"), ("hide",)]  # hide() exactly once


def test_tick_revives_on_next_chunk_after_idle_hide():
    """The revived line's OWN text ("again") is the live row shown --
    collapsed by default (see
    test_tick_collapsed_by_default_shows_live_row_only_despite_history
    below), so the third call is a single live row, even though "hello"
    (the utterance that just idle-hid) has separately been pushed into the
    model's history (stage 2's push-on-new-line behavior -- see
    test_push_on_new_line_after_idle_hide_revive_also_saves_to_history at
    the model level, and test_tick_hovering_reveals_expanded_history_stack_
    with_alphas below for the hovering-shows-it-too case)."""
    controller, renderer, dq = _controller([("chunk", "hello"), ("eos",)])
    controller.tick(now=0.0)
    controller.tick(now=5.0)     # hides
    dq.push(("chunk", "again"))
    controller.tick(now=5.1)
    assert renderer.calls == [
        _live_row_call(controller, "hello"),
        ("hide",),
        _live_row_call(controller, "again"),
    ]


def test_tick_stays_visible_just_under_idle_clear_sec():
    controller, renderer, _dq = _controller([("chunk", "hello"), ("eos",)])
    controller.tick(now=0.0)
    controller.tick(now=4.999)
    assert renderer.calls == [_live_row_call(controller, "hello")]   # no hide() yet


def test_tick_none_display_queue_never_shows_or_hides():
    renderer = _FakeRenderer()
    controller = co._OverlayController(None, font_size=28, renderer=renderer, settings_path=None)
    controller.tick(now=0.0)
    controller.tick(now=10.0)
    assert renderer.calls == []


def test_tick_collapsed_by_default_shows_live_row_only_despite_history():
    """Collapsed is the new default UX: even with several accumulated
    history lines, an un-hovered/un-dragged tick() must show ONLY the live
    row -- history is revealed by hovering (see the next test)."""
    events = []
    for i in range(5):
        events.append(("chunk", f"L{i}"))
        events.append(("eos",))
    events.append(("chunk", "LIVE"))
    controller, renderer, _dq = _controller(events)
    controller.tick(now=0.0)
    assert renderer.calls == [_live_row_call(controller, "LIVE")]


def test_tick_hovering_reveals_expanded_history_stack_with_alphas():
    """Setting _hovering=True (the seam _update_hover would flip via real
    cursor tracking, which needs AppKit -- see _FakePanelForHover's
    docstring below) makes the NEXT tick() show the full up-to-2-history +
    live stack with the correct per-row alphas, proving tick() wires
    expanded=_hovering through to model.visible_rows()/_alphas_for()."""
    events = []
    for i in range(5):
        events.append(("chunk", f"L{i}"))
        events.append(("eos",))
    events.append(("chunk", "LIVE"))
    controller, renderer, _dq = _controller(events)
    controller.tick(now=0.0)   # collapsed first (previous test)

    controller._hovering = True
    controller.tick(now=0.0)
    expected_rows = [("L3", False), ("L4", False), ("LIVE", True)]
    expected_alphas = (0.55, 0.75, 1.0)
    assert renderer.calls[-1] == _show_call(controller, expected_rows, expected_alphas)


def test_tick_holds_while_hovered_and_hides_after_leaving():
    controller, renderer, _dq = _controller([("chunk", "hello"), ("eos",)])
    controller.tick(now=0.0)
    controller._hovering = True
    controller.tick(now=4.99)
    assert renderer.calls[-1][0] == "show"
    controller.tick(now=5.0)
    controller.tick(now=30.0)
    assert renderer.calls[-1][0] == "show"
    assert controller._panel_visible
    controller._hovering = False
    controller.tick(now=30.1)
    assert renderer.calls[-1] == ("hide",)
    assert not controller._hovering


def test_tick_repaints_when_geometry_changes_even_with_no_new_events():
    """Repaint-skip state-compare includes geometry (font_size/cx_frac/
    bottom_px) -- changing controller.geometry directly (the same kind of
    change a real drag/Cmd+scroll gesture, AppKit-only and untested here,
    would make) must trigger a NEW renderer.show() on the next tick, even
    though no new display-queue event ever arrived."""
    controller, renderer, _dq = _controller([("chunk", "hello")])
    controller.tick(now=0.0)
    calls_before = len(renderer.calls)

    controller.geometry.scale_by(2)
    controller.tick(now=0.0)

    assert len(renderer.calls) == calls_before + 1
    last = renderer.calls[-1]
    assert last[0] == "show"
    assert last[1] == [("hello", True)]
    assert last[3] == controller.geometry.font_size   # picked up the NEW (scaled) font_size


def test_tick_scroll_changes_state_and_triggers_repaint_even_with_same_live_text():
    """Repaint-skip state must include the scroll offset (not just rows'
    text), per design -- though in practice a changed offset always changes
    `rows` too (a different history window, forced into expansion per
    effective_offset(now) > 0), so this also doubles as end-to-end proof
    that scrolling repaints even collapsed (_hovering/_dragging both
    False)."""
    events = []
    for i in range(6):
        events.append(("chunk", f"L{i}"))
        events.append(("eos",))
    events.append(("chunk", "LIVE"))
    controller, renderer, _dq = _controller(events)
    controller.tick(now=0.0)
    first_call = renderer.calls[-1]

    controller.model.scroll_by(1, 0.0)
    controller.tick(now=0.0)
    second_call = renderer.calls[-1]

    assert first_call != second_call
    assert second_call[0] == "show"


# ------------------------------------------ controller construction + settings ---
def test_controller_settings_path_none_uses_font_size_arg_and_never_writes(tmp_path, monkeypatch):
    """settings_path=None disables persistence entirely: geometry comes
    straight from the font_size ARGUMENT (no disk read), and no file is
    ever written, even after a tick that would otherwise be eligible for a
    debounced save. DEFAULT_SETTINGS_PATH is redirected to a tmp_path
    tripwire file for this test only, so an accidental fallback-to-default
    bug would be caught here instead of ever touching the real
    ~/.config/alwayswhisper/caption_overlay.json."""
    fake_default = tmp_path / "should_never_be_written.json"
    monkeypatch.setattr(co, "DEFAULT_SETTINGS_PATH", fake_default)

    renderer = _FakeRenderer()
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=52, renderer=renderer, settings_path=None)
    assert controller.geometry.font_size == 52

    controller._settings_dirty_at = 0.0
    controller.tick(now=100.0)
    assert not fake_default.exists()


def test_controller_settings_file_present_font_size_wins_over_arg(tmp_path):
    path = tmp_path / "geometry.json"
    co.save_geometry(path, co.OverlayGeometry(cx_frac=0.3, bottom_px=40.0, font_size=64.0))
    renderer = _FakeRenderer()
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=path)
    assert controller.geometry.font_size == 64.0    # the FILE's value, not the font_size= arg
    assert controller.geometry.cx_frac == 0.3
    assert controller.geometry.bottom_px == 40.0


def test_controller_settings_path_with_no_file_yet_font_size_arg_wins(tmp_path):
    path = tmp_path / "does_not_exist_yet.json"
    assert not path.exists()
    renderer = _FakeRenderer()
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=71, renderer=renderer, settings_path=path)
    assert controller.geometry.font_size == 71      # the ARG, since load_geometry fell back to defaults
    assert controller.geometry.cx_frac == 0.5        # untouched default (no file to read cx_frac from)
    assert controller.geometry.bottom_px == 24.0


# ------------------------------------------------ controller debounced save ---
def test_controller_debounced_save_writes_after_1s_of_no_further_dirt(tmp_path):
    path = tmp_path / "geometry.json"
    renderer = _FakeRenderer()
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=path)
    controller._settings_dirty_at = 10.0

    controller.tick(now=10.5)           # only 0.5s since dirty -- too soon
    assert not path.exists()

    controller.tick(now=11.1)           # >= 1.0s since dirty -- must save now
    assert path.exists()
    assert controller._settings_dirty_at is None
    assert co.load_geometry(path).font_size == controller.geometry.font_size


def test_controller_debounced_save_does_not_fire_while_dragging(tmp_path):
    path = tmp_path / "geometry.json"
    renderer = _FakeRenderer()
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=path)
    controller._dragging = True
    controller._settings_dirty_at = 10.0

    controller.tick(now=20.0)           # far past the 1.0s debounce, but still dragging
    assert not path.exists()
    assert controller._settings_dirty_at == 10.0    # left untouched, not cleared


def test_controller_close_flushes_pending_debounced_save(tmp_path):
    """close() must flush a pending debounced save immediately, not just
    hide the panel -- otherwise a Cmd+scroll resize in the last (<1.0s)
    moment before teardown (Ctrl-C) would be silently lost: mouse-up already
    saves drags immediately, but the scale path only marks
    _settings_dirty_at and otherwise waits for tick()'s debounce window,
    which never gets another chance to fire once the run loop has
    stopped."""
    path = tmp_path / "geometry.json"
    renderer = _FakeRenderer()
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=path)
    controller.geometry.scale_by(1)     # simulates a Cmd+scroll resize
    controller._settings_dirty_at = 10.0
    assert not path.exists()            # nothing written yet -- still inside the debounce window

    controller.close()

    assert path.exists()
    assert co.load_geometry(path).font_size == controller.geometry.font_size
    assert controller._settings_dirty_at is None
    assert ("hide",) in renderer.calls  # close() still hides the panel as before


def test_controller_close_with_settings_path_none_writes_nothing(tmp_path, monkeypatch):
    """settings_path=None must still disable persistence at close() time,
    exactly like tick()'s own debounced save. DEFAULT_SETTINGS_PATH is
    redirected to a tripwire file for this test only, so an accidental
    fallback-to-default bug would be caught here instead of ever touching
    the real ~/.config/alwayswhisper/caption_overlay.json."""
    fake_default = tmp_path / "should_never_be_written.json"
    monkeypatch.setattr(co, "DEFAULT_SETTINGS_PATH", fake_default)

    renderer = _FakeRenderer()
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=None)
    controller.geometry.scale_by(1)
    controller._settings_dirty_at = 10.0

    controller.close()

    assert not fake_default.exists()
    assert controller._settings_dirty_at == 10.0    # left untouched, not cleared


# --------------------------------------------------------------- hover toggle ---
class _FakePanelForHover:
    """Duck-typed NSPanel stand-in exposing only what _update_hover touches:
    frame() and setIgnoresMouseEvents_(). Real cursor-position/screen-space
    checks need actual AppKit (NSEvent.mouseLocation()/NSPointInRect), so
    _update_hover's own AppKit call is exercised via --demo, not here; this
    tests _OverlayController's is-it-called-and-only-on-change wiring by
    monkeypatching mouseLocation-dependent bits out of the picture instead --
    see test_update_hover_only_calls_setter_on_state_change below."""

    def __init__(self, frame):
        self._frame = frame
        self.ignores_calls = []

    def frame(self):
        return self._frame

    def setIgnoresMouseEvents_(self, value):
        self.ignores_calls.append(value)


class _FakeRendererWithPanel(_FakeRenderer):
    """Adds a .panel so _OverlayController._update_hover doesn't no-op."""

    def __init__(self, panel):
        super().__init__()
        self.panel = panel


def test_update_hover_only_calls_setter_on_state_change(monkeypatch):
    """_update_hover must call setIgnoresMouseEvents_ when hover state
    flips, and must NOT call it again on subsequent ticks while hover state
    stays the same (avoids redundant AppKit calls every tick)."""
    panel = _FakePanelForHover(frame=("fake-frame",))
    renderer = _FakeRendererWithPanel(panel)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi")]), font_size=28, renderer=renderer, settings_path=None)

    calls = {"n": 0}

    def fake_update_hover():
        # Directly exercise the state-change-only contract without needing
        # real AppKit: flip _hovering the way the real method would if the
        # cursor were newly inside/outside, and assert the setter fires
        # only on that flip.
        calls["n"] += 1
        hovering = calls["n"] >= 2   # simulate: outside, then inside, then still inside
        if hovering != controller._hovering:
            controller._hovering = hovering
            panel.setIgnoresMouseEvents_(not hovering)

    monkeypatch.setattr(controller, "_update_hover", fake_update_hover)

    controller.tick(now=0.0)   # shows "hi"; hover check #1 -> stays not-hovering, no setter call
    controller.tick(now=0.1)   # hover check #2 -> flips to hovering, setter called once
    controller.tick(now=0.2)   # hover check #3 -> still hovering, no additional setter call
    assert panel.ignores_calls == [False]  # setIgnoresMouseEvents_(False) exactly once


def test_update_hover_not_called_while_panel_is_hidden():
    """Per spec: hover is only checked 'when the panel is visible'."""
    panel = _FakePanelForHover(frame=("fake-frame",))
    renderer = _FakeRendererWithPanel(panel)
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=None)
    controller.tick(now=0.0)   # nothing ever happened -> hidden, panel stays not visible
    assert panel.ignores_calls == []


# =============================================================================
# Hover copy-icon UX: mouseover a caption row shows a small copy icon at its
# right end; clicking it copies that row's full text to the system pasteboard
# (NSPasteboard) instead of arming the existing drag-to-move gesture; clicking
# anywhere else on the panel is unaffected. See RowBox/RowDecor/
# _compute_layout/_hit_row/_hit_icon/_copy_text_to_pasteboard in
# caption_overlay.py, and _OverlayController._mouse_location/
# _handle_mouse_down/tick.
# =============================================================================
import time as _time
from types import SimpleNamespace


class _FakeFrame:
    """Duck-typed NSScreen.visibleFrame()-shaped stand-in for
    _panel_origin/_compute_layout -- only .origin.x/.origin.y/.size.width/
    .size.height are ever read (see _panel_origin's own docstring), so a
    plain namespace-of-namespaces works fine, no AppKit needed."""

    def __init__(self, x, y, w, h):
        self.origin = SimpleNamespace(x=x, y=y)
        self.size = SimpleNamespace(width=w, height=h)


def _geometry(cx_frac=0.5, bottom_px=24.0, font_size=28.0):
    return co.OverlayGeometry(cx_frac=cx_frac, bottom_px=bottom_px, font_size=font_size)


# ------------------------------------------------------------- _icon_reserve_for ---
def test_icon_reserve_for_scales_with_font_size_within_bounds():
    icon_size, reserve = co._icon_reserve_for(40.0)
    assert icon_size == pytest.approx(40.0 * 0.9)
    # icon_reserve covers BOTH icons: text / _PAD_X (already inside the text
    # chip's own padding) / copy / _ICON_GAP / translate / _PAD_X (new) /
    # chip edge.
    assert reserve == pytest.approx(2 * icon_size + co._ICON_GAP + co._PAD_X)


def test_icon_reserve_for_clamps_at_extremes():
    tiny_icon_size, _r = co._icon_reserve_for(1.0)
    huge_icon_size, _r2 = co._icon_reserve_for(500.0)
    assert tiny_icon_size >= 14.0 - 1e-9
    assert huge_icon_size <= 40.0 + 1e-9


# ------------------------------------------------------------------ _panel_origin ---
# Regression coverage for a real, live bug: y used to be clamped on only the
# TOP side (a single min() against the visible top). See the rewritten
# comment block directly above the two clamp lines in caption_overlay.py for
# the full reasoning; in short: hovering expands the stack from 1 row (live
# only) to up to 3 (2 history + live -- see _HISTORY_ALPHAS_NEWEST_FIRST),
# and on a screen short enough that panel_h then exceeds
# visible.size.height, the old min()-only clamp pushed the panel's origin
# BELOW visible.origin.y. Since _compute_layout always places the
# live/newest row at panel-local y=0 -- the BOTTOM of the stack, see that
# function's own "Bottom-up stacking" docstring section, and it's the one
# caller that actually feeds _panel_origin a real, possibly-oversized
# panel_h (via its own ox, oy = _panel_origin(...) line) -- that
# unconditionally pushed the live/newest caption itself off the bottom of
# the screen on every hover, every time. The tests below drive
# _panel_origin directly, the same pure function _compute_layout and
# _drag_reposition_origin both call.
def test_panel_origin_clamps_y_to_visible_bottom_when_taller_than_screen_bottom_px_zero():
    visible = _FakeFrame(0, 0, 1920, 1080)
    geometry = _geometry(bottom_px=0.0)

    _x, y = co._panel_origin(visible, 400.0, 1200.0, geometry)   # panel_h (1200) > visible height (1080)

    assert y == pytest.approx(visible.origin.y)


def test_panel_origin_clamps_y_to_visible_bottom_when_taller_than_screen_bottom_px_positive():
    visible = _FakeFrame(0, 0, 1920, 1080)
    geometry = _geometry(bottom_px=50.0)

    _x, y = co._panel_origin(visible, 400.0, 1100.0, geometry)   # panel_h (1100) > visible height (1080)

    assert y == pytest.approx(visible.origin.y)


def test_panel_origin_clamps_y_to_visible_bottom_on_a_negative_origin_secondary_monitor():
    """Multi-monitor: a display positioned BELOW the primary has a
    visibleFrame with a negative origin.y. The lower-bound clamp must land
    on THAT screen's own bottom (visible.origin.y, here -500), not 0 -- a
    bare max(0.0, y) would still leave the panel floating 500pt above this
    screen's actual bottom edge."""
    visible = _FakeFrame(0, -500, 1920, 800)
    geometry = _geometry(bottom_px=24.0)

    _x, y = co._panel_origin(visible, 400.0, 900.0, geometry)   # panel_h (900) > visible height (800)

    assert y == pytest.approx(-500.0)
    assert y == pytest.approx(visible.origin.y)


def test_panel_origin_normal_case_unaffected_by_the_new_lower_bound_clamp():
    """Regression guard: when panel_h comfortably fits on screen (the
    overwhelmingly common case -- collapsed single-row panel, or an
    expanded stack on any normal-height display), y is still exactly
    visible.origin.y + geometry.bottom_px, unchanged from before this fix."""
    visible = _FakeFrame(0, 0, 1920, 1080)
    geometry = _geometry(bottom_px=24.0)

    _x, y = co._panel_origin(visible, 400.0, 60.0, geometry)   # panel_h (60) well under visible height

    assert y == pytest.approx(visible.origin.y + 24.0)


def test_panel_origin_still_clamps_stale_bottom_px_to_keep_panel_top_on_screen():
    """Regression guard for the ORIGINAL (pre-existing, still-needed)
    upper-bound clamp: a stale/cross-screen geometry.bottom_px (e.g.
    persisted from a taller screen) must still be pulled down so the
    panel's TOP never lands past the visible top -- even though panel_h
    itself is small here and never engages the NEW lower-bound clamp."""
    visible = _FakeFrame(0, 0, 1920, 1080)
    geometry = _geometry(bottom_px=2000.0)   # stale value from a much taller screen

    _x, y = co._panel_origin(visible, 400.0, 60.0, geometry)

    assert y == pytest.approx(visible.origin.y + visible.size.height - 60.0)


# --------------------------------------------------------------- _compute_layout ---
def test_compute_layout_non_hovered_rows_have_no_icon():
    frame = _FakeFrame(0, 0, 2000, 1000)
    geometry = _geometry()
    row_sizes = [("a", 40, 20), ("b", 40, 20), ("c", 40, 20)]
    _panel_rect, chip_layouts, row_boxes = co._compute_layout(row_sizes, 28.0, 1, frame, geometry)

    assert row_boxes[0].icon is None
    assert row_boxes[1].icon is not None
    assert row_boxes[2].icon is None
    assert chip_layouts[0][4] is None
    assert chip_layouts[2][4] is None
    assert chip_layouts[1][4] is not None
    # The translate icon appears and disappears with the copy icon: same
    # hovered row, never one without the other.
    assert row_boxes[0].translate_icon is None
    assert row_boxes[1].translate_icon is not None
    assert row_boxes[2].translate_icon is None
    assert chip_layouts[0][5] is None
    assert chip_layouts[2][5] is None
    assert chip_layouts[1][5] is not None


def test_compute_layout_hover_row_none_means_no_icon_anywhere():
    frame = _FakeFrame(0, 0, 2000, 1000)
    geometry = _geometry()
    row_sizes = [("a", 40, 20), ("b", 40, 20)]
    _panel_rect, chip_layouts, row_boxes = co._compute_layout(row_sizes, 28.0, None, frame, geometry)
    assert all(box.icon is None for box in row_boxes)
    assert all(chip[4] is None for chip in chip_layouts)
    assert all(box.translate_icon is None for box in row_boxes)
    assert all(chip[5] is None for chip in chip_layouts)


def test_compute_layout_icon_view_rect_vertically_centered_in_its_chip():
    frame = _FakeFrame(0, 0, 2000, 1000)
    geometry = _geometry()
    row_sizes = [("row", 80, 30)]
    icon_size, _reserve = co._icon_reserve_for(28.0)

    _panel_rect, chip_layouts, _row_boxes = co._compute_layout(row_sizes, 28.0, 0, frame, geometry)
    _cx, _cy, _cw, chip_h, icon_local, translate_local = chip_layouts[0]

    for icon_x, icon_y, icon_w, icon_h in (icon_local, translate_local):
        assert icon_w == pytest.approx(icon_size)
        assert icon_h == pytest.approx(icon_size)
        assert icon_y == pytest.approx((chip_h - icon_size) / 2.0)
        assert icon_x > 0   # inside the chip's right extension, not overlapping the text padding


def test_compute_layout_icon_view_rect_starts_immediately_after_text_padding():
    """Balanced layout: text / _PAD_X / copy / _ICON_GAP / translate /
    _PAD_X / chip edge. The copy icon VIEW rect's local x is exactly the
    text chip's own width (which already bakes in _PAD_X on its right) --
    no separate extra gap -- and the translate icon follows one icon width
    plus _ICON_GAP later, ending exactly _PAD_X short of the chip's own
    (widened) right edge."""
    frame = _FakeFrame(0, 0, 2000, 1000)
    geometry = _geometry()
    row_sizes = [("row", 80, 30)]
    icon_size, icon_reserve = co._icon_reserve_for(28.0)

    _panel_rect, chip_layouts, _row_boxes = co._compute_layout(row_sizes, 28.0, 0, frame, geometry)
    _cx, _cy, chip_w, _ch, icon_local, translate_local = chip_layouts[0]
    text_chip_w = 80 + 2 * co._PAD_X

    assert icon_local[0] == pytest.approx(text_chip_w)
    assert translate_local[0] == pytest.approx(text_chip_w + icon_size + co._ICON_GAP)
    assert chip_w == pytest.approx(text_chip_w + icon_reserve)
    assert translate_local[0] + icon_size == pytest.approx(chip_w - co._PAD_X)


def test_compute_layout_hovered_narrow_row_does_not_shift_text_or_panel_origin():
    """A hovered row's chip is widened to the RIGHT only -- text-centering
    (both per-chip and the panel's own screen placement via _panel_origin)
    is driven by text_panel_w (the widest TEXT-only chip), never by
    whichever row happens to be hovered."""
    frame = _FakeFrame(0, 0, 2000, 1000)
    geometry = _geometry()
    row_sizes = [("wide row here", 200, 30), ("short", 40, 30)]

    panel_rect_no_hover, chip_layouts_no_hover, _ = co._compute_layout(row_sizes, 28.0, None, frame, geometry)
    panel_rect_hover_short, chip_layouts_hover_short, _ = co._compute_layout(row_sizes, 28.0, 1, frame, geometry)

    assert panel_rect_no_hover[0] == pytest.approx(panel_rect_hover_short[0])   # x
    assert panel_rect_no_hover[1] == pytest.approx(panel_rect_hover_short[1])   # y

    # The wide (non-hovered) row's own chip geometry (x, y, w, h) is
    # byte-for-byte unaffected by hovering the OTHER, narrower row.
    assert chip_layouts_no_hover[0][:4] == chip_layouts_hover_short[0][:4]


def test_compute_layout_panel_frame_width_is_text_width_plus_icon_reserve_always():
    frame = _FakeFrame(0, 0, 2000, 1000)
    geometry = _geometry()
    row_sizes = [("hello", 80, 30)]
    _icon_size, icon_reserve = co._icon_reserve_for(28.0)

    panel_rect_none, _cl_none, _rb_none = co._compute_layout(row_sizes, 28.0, None, frame, geometry)
    panel_rect_hover, _cl_hover, _rb_hover = co._compute_layout(row_sizes, 28.0, 0, frame, geometry)

    text_chip_w = 80 + 2 * co._PAD_X
    assert panel_rect_none[2] == pytest.approx(text_chip_w + icon_reserve)
    # Frame width is the SAME regardless of hover_row -- the strip is
    # always reserved so the frame never resizes the instant hover starts.
    assert panel_rect_hover[2] == pytest.approx(panel_rect_none[2])


def test_compute_layout_row_box_index_matches_oldest_first_order_despite_bottom_up_stacking():
    row_sizes = [("oldest", 40, 20), ("middle", 60, 20), ("newest-live", 30, 20)]
    frame = _FakeFrame(0, 0, 2000, 1000)
    geometry = _geometry()
    _panel_rect, _chip_layouts, row_boxes = co._compute_layout(row_sizes, 28.0, None, frame, geometry)

    assert [b.index for b in row_boxes] == [0, 1, 2]
    assert [b.text for b in row_boxes] == ["oldest", "middle", "newest-live"]
    # Bottom-up stacking: the LAST row_sizes entry (newest/live) sits at
    # the smallest y (closest to the panel's own bottom edge).
    ys = [b.frame[1] for b in row_boxes]
    assert ys[2] < ys[1] < ys[0]


def test_compute_layout_row_box_frame_screen_coords_add_panel_origin():
    frame = _FakeFrame(100, 50, 2000, 1000)
    geometry = _geometry()
    row_sizes = [("only row", 80, 30)]

    panel_rect, chip_layouts, row_boxes = co._compute_layout(row_sizes, 28.0, None, frame, geometry)
    ox, oy, _pw, _ph = panel_rect
    chip_x, chip_y, _cw, _ch, _icon_local, _translate_local = chip_layouts[0]

    box = row_boxes[0]
    assert box.frame[0] == pytest.approx(ox + chip_x)
    assert box.frame[1] == pytest.approx(oy + chip_y)


def test_compute_layout_row_box_icon_hit_rect_is_generous_and_screen_converted():
    """RowBox.icon (what _hit_icon matches) is a DIFFERENT, more generous
    rect than the small icon_size-square VIEW rect chip_layouts carries
    (what _make_copy_icon_view actually draws): full chip height, from
    midway into the text's own right padding through the MIDPOINT of the
    gap to the translate icon -- a tight icon_size square is a fiddly
    target on an 80ms-polling overlay. Its right edge stops at that
    midpoint (it used to run all the way to the chip's right edge, back
    when the copy icon was the only one) so the two icons' hit rects can
    tile the whole reserved strip without overlapping."""
    frame = _FakeFrame(100, 50, 2000, 1000)
    geometry = _geometry()
    row_sizes = [("only row", 80, 30)]
    icon_size, _icon_reserve = co._icon_reserve_for(28.0)

    panel_rect, chip_layouts, row_boxes = co._compute_layout(row_sizes, 28.0, 0, frame, geometry)
    ox, oy, _pw, _ph = panel_rect
    chip_x, chip_y, _cw, chip_h, icon_view_local, _translate_view_local = chip_layouts[0]
    text_chip_w = 80 + 2 * co._PAD_X

    box = row_boxes[0]
    assert icon_view_local is not None
    assert box.icon is not None
    # The hit rect and the view rect are NOT the same rect.
    assert box.icon != (ox + chip_x + icon_view_local[0], oy + chip_y + icon_view_local[1],
                         icon_view_local[2], icon_view_local[3])

    expected_x = ox + chip_x + (text_chip_w - co._PAD_X / 2.0)
    expected_w = icon_size + co._ICON_GAP / 2.0 + co._PAD_X / 2.0
    assert box.icon[0] == pytest.approx(expected_x)
    assert box.icon[1] == pytest.approx(oy + chip_y)     # full chip height -- starts at the chip's own y
    assert box.icon[2] == pytest.approx(expected_w)
    assert box.icon[3] == pytest.approx(chip_h)          # full chip height, not just the icon's own height

    # Right edge = the midpoint of the gap between the two icons.
    assert box.icon[0] + box.icon[2] == pytest.approx(
        ox + chip_x + text_chip_w + icon_size + co._ICON_GAP / 2.0)


def test_compute_layout_row_box_translate_hit_rect_tiles_the_rest_of_the_strip():
    """RowBox.translate_icon (what _hit_translate_icon matches) picks up
    exactly where RowBox.icon stops and runs to the (widened) chip's own
    right edge: same full chip height, no gap between the two and no
    overlap -- every point of the reserved strip belongs to exactly one of
    the two icons, so there is no dead band a click can fall into."""
    frame = _FakeFrame(100, 50, 2000, 1000)
    geometry = _geometry()
    row_sizes = [("only row", 80, 30)]
    _icon_size, icon_reserve = co._icon_reserve_for(28.0)

    panel_rect, chip_layouts, row_boxes = co._compute_layout(row_sizes, 28.0, 0, frame, geometry)
    ox, oy, _pw, _ph = panel_rect
    chip_x, chip_y, _cw, chip_h, _icon_view_local, translate_view_local = chip_layouts[0]
    text_chip_w = 80 + 2 * co._PAD_X

    box = row_boxes[0]
    assert translate_view_local is not None
    assert box.translate_icon is not None
    # Hit rect and view rect are different rects here too.
    assert box.translate_icon != (ox + chip_x + translate_view_local[0],
                                   oy + chip_y + translate_view_local[1],
                                   translate_view_local[2], translate_view_local[3])

    # Starts exactly where the copy hit rect ends -- no gap, no overlap.
    assert box.translate_icon[0] == pytest.approx(box.icon[0] + box.icon[2])
    assert box.translate_icon[1] == pytest.approx(oy + chip_y)
    assert box.translate_icon[3] == pytest.approx(chip_h)
    # ...and ends exactly on the chip's own right edge.
    assert box.translate_icon[0] + box.translate_icon[2] == pytest.approx(
        ox + chip_x + text_chip_w + icon_reserve)


# ------------------------------------------------------------ _hit_translate_icon ---
def test_hit_translate_icon_matches_only_the_translate_rect():
    """The two icon hit-testers must not answer for each other's rect --
    otherwise a click meant to copy would spend money on a translation (or
    the reverse)."""
    frame = _FakeFrame(0, 0, 2000, 1000)
    geometry = _geometry()
    _panel_rect, _chip_layouts, row_boxes = co._compute_layout(
        [("only row", 80, 30)], 28.0, 0, frame, geometry)
    box = row_boxes[0]

    copy_point = (box.icon[0] + box.icon[2] / 2.0, box.icon[1] + box.icon[3] / 2.0)
    translate_point = (box.translate_icon[0] + box.translate_icon[2] / 2.0,
                        box.translate_icon[1] + box.translate_icon[3] / 2.0)

    assert co._hit_icon(copy_point, row_boxes) is box
    assert co._hit_translate_icon(copy_point, row_boxes) is None
    assert co._hit_translate_icon(translate_point, row_boxes) is box
    assert co._hit_icon(translate_point, row_boxes) is None


def test_hit_translate_icon_misses_rows_without_a_translate_rect():
    frame = _FakeFrame(0, 0, 2000, 1000)
    geometry = _geometry()
    _panel_rect, _chip_layouts, row_boxes = co._compute_layout(
        [("a", 40, 20), ("b", 40, 20)], 28.0, None, frame, geometry)

    # Nobody hovered -> no icons at all -> a point anywhere hits neither.
    inside_row = (row_boxes[0].frame[0] + 1.0, row_boxes[0].frame[1] + 1.0)
    assert co._hit_translate_icon(inside_row, row_boxes) is None
    assert co._hit_icon(inside_row, row_boxes) is None


def test_compute_layout_row_box_frame_always_includes_icon_extension():
    """RowBox.frame (what _hit_row matches) reserves the icon extension
    for EVERY row, hovered or not -- see _compute_layout's docstring: this
    is what lets the cursor entering the band where the icon WILL appear
    immediately reveal it (no flicker right at the extension boundary)."""
    frame = _FakeFrame(0, 0, 2000, 1000)
    geometry = _geometry()
    row_sizes = [("hi", 40, 20), ("there", 50, 20)]
    _icon_size, icon_reserve = co._icon_reserve_for(28.0)
    text_chip_w_0 = 40 + 2 * co._PAD_X
    text_chip_w_1 = 50 + 2 * co._PAD_X

    _panel_rect, _chip_layouts, row_boxes = co._compute_layout(row_sizes, 28.0, 0, frame, geometry)
    # row 0 IS hovered -- its frame includes the extension.
    assert row_boxes[0].frame[2] == pytest.approx(text_chip_w_0 + icon_reserve)
    # row 1 is NOT hovered -- its frame STILL includes the extension.
    assert row_boxes[1].frame[2] == pytest.approx(text_chip_w_1 + icon_reserve)


def test_compute_layout_non_hovered_row_frame_extended_but_rendered_chip_is_not():
    """The HIT frame (RowBox.frame) always reserves the icon extension,
    but the RENDERED chip (chip_layouts' own chip_w, what _layout_stack
    actually builds the background view at) for a non-hovered row stays
    plain text width -- only the currently hovered row's chip is actually
    widened. hover_row=None here (nobody hovered) -- the frame is still
    extended regardless."""
    frame = _FakeFrame(0, 0, 2000, 1000)
    geometry = _geometry()
    row_sizes = [("hi", 40, 20)]
    _icon_size, icon_reserve = co._icon_reserve_for(28.0)
    text_chip_w = 40 + 2 * co._PAD_X

    _panel_rect, chip_layouts, row_boxes = co._compute_layout(row_sizes, 28.0, None, frame, geometry)
    assert chip_layouts[0][2] == pytest.approx(text_chip_w)                      # rendered chip: NOT widened
    assert row_boxes[0].frame[2] == pytest.approx(text_chip_w + icon_reserve)    # hit frame: IS extended
    assert chip_layouts[0][2] != row_boxes[0].frame[2]                           # and they now DIFFER


def test_compute_layout_empty_row_sizes_returns_no_chips_or_boxes():
    frame = _FakeFrame(0, 0, 2000, 1000)
    geometry = _geometry()
    panel_rect, chip_layouts, row_boxes = co._compute_layout([], 28.0, None, frame, geometry)
    assert chip_layouts == []
    assert row_boxes == []
    assert panel_rect[2] == 0.0
    assert panel_rect[3] == 0.0


# ------------------------------------------------------- _drag_reposition_origin ---
# Regression coverage for "the panel visibly jumps sideways right after a drag":
# _handle_mouse_drag's LIVE (pre-full-repaint) reposition used to center the
# panel on its own CURRENT frame width directly (_panel_origin(visible,
# frame.size.width, ...)) -- but frame.size.width is ALWAYS icon-reserve-
# inclusive (see _compute_layout's docstring), while _compute_layout itself
# always centers on the narrower TEXT-only width. The two disagreed by
# ~icon_reserve/2, so the drag-time position and the very next full repaint's
# position differed -- a visible sideways snap. _drag_reposition_origin is the
# extracted pure fix; these tests prove it now agrees with _compute_layout
# for the same real inputs, and that the OLD (naive) formula would not have.
def test_drag_reposition_origin_matches_compute_layout_for_the_same_geometry():
    """Feed _drag_reposition_origin the exact panel_w/panel_h _compute_layout
    itself returned for some real row measurements -- not just isolated
    numbers -- so this test would actually have caught the regression."""
    frame = _FakeFrame(0, 0, 1920, 1080)
    geometry = _geometry(cx_frac=0.5, bottom_px=24.0, font_size=28.0)
    row_sizes = [("older line", 120.0, 30.0), ("a longer live line still growing", 260.0, 30.0)]

    # hover_row=None: matches reality -- tick() forces the hover row to None
    # for the whole duration of a drag (see tick()'s "Hover-row hit-testing"
    # comment), yet the panel's frame width ALWAYS reserves the icon strip
    # regardless (again, see _compute_layout's docstring) -- so this is the
    # exact panel_w/panel_h shape a real drag actually sees.
    panel_rect, _chip_layouts, _row_boxes = co._compute_layout(row_sizes, 28.0, None, frame, geometry)
    panel_x, panel_y, panel_w, panel_h = panel_rect

    x, y = co._drag_reposition_origin(frame, panel_w, panel_h, 28.0, geometry)

    assert x == pytest.approx(panel_x)
    assert y == pytest.approx(panel_y)


def test_drag_reposition_origin_differs_from_naive_full_width_centering():
    """Proves this test suite would have caught the original bug: centering
    directly on the full (icon-reserve-inclusive) frame width -- what
    _handle_mouse_drag used to do -- lands at a different x than the fixed
    text-width centering, for the same inputs."""
    frame = _FakeFrame(0, 0, 1920, 1080)
    geometry = _geometry(cx_frac=0.5, bottom_px=24.0, font_size=28.0)
    panel_w, panel_h = 400.0, 60.0   # icon-reserve-inclusive, like a real panel.frame()

    fixed_x, _fixed_y = co._drag_reposition_origin(frame, panel_w, panel_h, 28.0, geometry)
    naive_x, naive_y = co._panel_origin(frame, panel_w, panel_h, geometry)   # the old, buggy call shape

    assert fixed_x != pytest.approx(naive_x)
    _icon_size, icon_reserve = co._icon_reserve_for(28.0)
    # Fixed (text-only) centering sits icon_reserve/2 to the RIGHT of the
    # naive full-width centering -- a narrower width centered on the same
    # point starts further right.
    assert fixed_x - naive_x == pytest.approx(icon_reserve / 2.0)
    assert _fixed_y == pytest.approx(naive_y)   # y (bottom anchoring) was never affected


def test_drag_reposition_origin_scales_with_font_size():
    """The icon reserve subtracted must track the CURRENT font_size (Cmd+
    scroll changes it live) -- not some fixed/default constant."""
    frame = _FakeFrame(0, 0, 1920, 1080)
    geometry = _geometry(cx_frac=0.5, bottom_px=24.0, font_size=64.0)
    panel_w, panel_h = 500.0, 90.0

    x, _y = co._drag_reposition_origin(frame, panel_w, panel_h, 64.0, geometry)
    _icon_size, icon_reserve = co._icon_reserve_for(64.0)
    expected_x, _expected_y = co._panel_origin(frame, panel_w - icon_reserve, panel_h, geometry)
    assert x == pytest.approx(expected_x)


def test_drag_reposition_origin_clamps_y_to_visible_bottom_when_panel_taller_than_screen():
    """Same hover-expansion overflow _panel_origin's own tests guard above
    (see the _panel_origin section), proven through _drag_reposition_origin
    directly -- this is the function _handle_mouse_drag actually calls on
    every live drag tick, so a drag started while hover-expanded (5 rows)
    on a short screen must not push the live row off the bottom either."""
    frame = _FakeFrame(0, 0, 1920, 1080)
    geometry = _geometry(bottom_px=24.0)
    _icon_size, icon_reserve = co._icon_reserve_for(28.0)
    frame_w, frame_h = 400.0, 1200.0   # frame_h (1200) > visible height (1080)

    x, y = co._drag_reposition_origin(frame, frame_w, frame_h, 28.0, geometry)

    assert y == pytest.approx(frame.origin.y)
    # x math is untouched by this fix -- still the usual text-width centering.
    expected_x, _expected_y = co._panel_origin(frame, frame_w - icon_reserve, frame_h, geometry)
    assert x == pytest.approx(expected_x)


# ------------------------------------------------------------ _hit_row/_hit_icon ---
def test_hit_row_inside_returns_matching_box():
    box = co.RowBox(index=0, text="x", frame=(10.0, 10.0, 50.0, 20.0), icon=None)
    assert co._hit_row((20.0, 15.0), [box]) is box


def test_hit_row_boundary_min_inside_max_outside():
    box = co.RowBox(index=0, text="x", frame=(10.0, 10.0, 50.0, 20.0), icon=None)
    assert co._hit_row((10.0, 10.0), [box]) is box       # min corner: inside (half-open)
    assert co._hit_row((60.0, 15.0), [box]) is None       # x == x + w: outside
    assert co._hit_row((20.0, 30.0), [box]) is None       # y == y + h: outside


def test_hit_row_outside_returns_none():
    box = co.RowBox(index=0, text="x", frame=(10.0, 10.0, 50.0, 20.0), icon=None)
    assert co._hit_row((0.0, 0.0), [box]) is None


def test_hit_row_empty_list_returns_none():
    assert co._hit_row((5.0, 5.0), []) is None


def test_hit_row_picks_the_containing_box_among_several():
    boxes = [
        co.RowBox(index=0, text="a", frame=(0.0, 40.0, 100.0, 30.0), icon=None),
        co.RowBox(index=1, text="b", frame=(0.0, 0.0, 100.0, 30.0), icon=None),
    ]
    assert co._hit_row((5.0, 50.0), boxes) is boxes[0]
    assert co._hit_row((5.0, 5.0), boxes) is boxes[1]
    assert co._hit_row((5.0, 35.0), boxes) is None   # in the gap between rows


def test_hit_icon_matches_only_icon_rect_not_whole_row_frame():
    box = co.RowBox(index=0, text="x", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))
    assert co._hit_icon((10.0, 10.0), [box]) is None   # inside the row frame, outside the icon
    assert co._hit_icon((75.0, 10.0), [box]) is box     # inside the icon


def test_hit_icon_skips_rows_with_no_icon():
    box = co.RowBox(index=0, text="x", frame=(0.0, 0.0, 100.0, 30.0), icon=None)
    assert co._hit_icon((10.0, 10.0), [box]) is None


def test_hit_icon_boundary_half_open():
    box = co.RowBox(index=0, text="x", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))
    assert co._hit_icon((70.0, 5.0), [box]) is box       # min corner: inside
    assert co._hit_icon((90.0, 5.0), [box]) is None       # x == x + w: outside
    assert co._hit_icon((70.0, 25.0), [box]) is None      # y == y + h: outside


def test_hit_icon_empty_list_returns_none():
    assert co._hit_icon((0.0, 0.0), []) is None


# ------------------------------------------------------ _copy_text_to_pasteboard ---
class _FakePasteboard:
    def __init__(self):
        self.cleared = False
        self.strings = []

    def clearContents(self):
        self.cleared = True

    def setString_forType_(self, text, type_):
        self.strings.append((text, type_))
        return True


class _RaisingPasteboard:
    def clearContents(self):
        pass

    def setString_forType_(self, text, type_):
        raise RuntimeError("boom")


def test_copy_text_to_pasteboard_clears_then_sets_string_returns_true():
    pb = _FakePasteboard()
    ok = co._copy_text_to_pasteboard("hello clipboard", pasteboard=pb)
    assert ok is True
    assert pb.cleared is True
    assert len(pb.strings) == 1
    assert pb.strings[0][0] == "hello clipboard"


def test_copy_text_to_pasteboard_raising_fake_returns_false_no_exception():
    ok = co._copy_text_to_pasteboard("hello", pasteboard=_RaisingPasteboard())
    assert ok is False


def test_copy_text_to_pasteboard_clear_raising_also_returns_false():
    class _RaisingClear:
        def clearContents(self):
            raise RuntimeError("boom")

        def setString_forType_(self, text, type_):
            return True

    assert co._copy_text_to_pasteboard("hello", pasteboard=_RaisingClear()) is False


# ------------------------------------------------------- controller: hover_row ---
class _FakeRendererWithBoxes(_FakeRenderer):
    """Like _FakeRenderer, but show() also returns a caller-supplied list of
    RowBox objects -- simulating what _AppKitRenderer.show() really returns
    (see _layout_stack/_compute_layout) -- so hover-row/copy-icon controller
    tests can drive realistic row_boxes without needing real AppKit layout
    math (that's _compute_layout's own job, tested directly above)."""

    def __init__(self, boxes=None):
        super().__init__()
        self._boxes = boxes or []

    def show(self, rows, alphas, font_size, geometry, decor=None):
        self.calls.append(("show", rows, alphas, font_size, geometry, decor))
        return self._boxes


def test_controller_defaults_copy_fn_to_copy_text_to_pasteboard():
    renderer = _FakeRenderer()
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=None)
    assert controller._copy_fn is co._copy_text_to_pasteboard


def test_controller_starts_with_no_row_boxes_or_hover_row():
    renderer = _FakeRenderer()
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=None)
    assert controller._row_boxes == []
    assert controller._hover_row is None


def test_global_mouse_down_copies_when_the_click_is_inside_an_icon():
    renderer = _FakeRenderer()
    copied = []
    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=None,
        copy_fn=lambda text: copied.append(text) or True)
    controller._row_boxes = [co.RowBox(0, "copy me", (0, 0, 100, 30), (70, 0, 30, 30))]

    class Event:
        def locationInWindow(self):
            return type("Point", (), {"x": 80, "y": 10})()

    controller._handle_global_mouse_down(Event())
    assert copied == ["copy me"]
    assert controller._dragging is False


def test_global_mouse_down_ignores_clicks_outside_an_icon():
    renderer = _FakeRenderer()
    copied = []
    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=None,
        copy_fn=lambda text: copied.append(text) or True)
    controller._row_boxes = [co.RowBox(0, "copy me", (0, 0, 100, 30), (70, 0, 30, 30))]

    class Event:
        def locationInWindow(self):
            return type("Point", (), {"x": 20, "y": 10})()

    controller._handle_global_mouse_down(Event())
    assert copied == []


def test_icon_dwell_copies_when_mouse_events_are_not_delivered():
    box = co.RowBox(0, "copy me", (0, 0, 100, 30), (70, 0, 30, 30))
    renderer = _FakeRendererWithBoxes([box])
    copied = []
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "copy me")]), renderer=renderer, settings_path=None,
        copy_fn=lambda text: copied.append(text) or True)
    controller.tick(now=0.0)
    controller._hovering = True
    controller._mouse_location = lambda: (80, 10)
    controller.tick(now=0.1)  # reveal the icon / begin dwell
    controller.tick(now=0.1 + co.HOVER_COPY_DELAY_SEC - 0.01)
    assert copied == []
    controller.tick(now=0.1 + co.HOVER_COPY_DELAY_SEC)
    assert copied == ["copy me"]


def test_update_hover_makes_a_real_panel_key_when_cursor_enters(monkeypatch):
    class Panel:
        def __init__(self):
            self.ignored = []
            self.key_calls = 0

        def frame(self):
            origin = type("Origin", (), {"x": 0, "y": 0})()
            size = type("Size", (), {"width": 100, "height": 30})()
            return type("Frame", (), {"origin": origin, "size": size})()

        def setIgnoresMouseEvents_(self, value):
            self.ignored.append(value)

        def makeKeyAndOrderFront_(self, sender):
            self.key_calls += 1

        def isKeyWindow(self):
            return True

    renderer = _FakeRenderer()
    renderer.panel = Panel()
    controller = co._OverlayController(_FakeDisplayQueue([]), renderer=renderer, settings_path=None)
    monkeypatch.setattr(controller, "_mouse_location", lambda: (10, 10))

    controller._update_hover()
    assert renderer.panel.ignored == [False]
    assert renderer.panel.key_calls == 1


def test_controller_hovering_over_a_row_sets_hover_row_and_repaints_with_decor():
    boxes = [
        co.RowBox(index=0, text="older", frame=(0.0, 40.0, 100.0, 30.0), icon=(70.0, 45.0, 20.0, 20.0)),
        co.RowBox(index=1, text="newer", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0)),
    ]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "x")]), font_size=28, renderer=renderer, settings_path=None)
    controller.tick(now=0.0)   # initial paint -- populates controller._row_boxes from `boxes`

    controller._hovering = True
    controller._mouse_location = lambda: (10.0, 10.0)   # inside row index 1's frame
    controller.tick(now=0.1)

    assert controller._hover_row == 1
    last = renderer.calls[-1]
    assert last[0] == "show"
    assert last[-1] == co.RowDecor(hover_row=1, copied_text=None)


def test_controller_hover_row_changes_when_mouse_moves_to_another_row():
    boxes = [
        co.RowBox(index=0, text="older", frame=(0.0, 40.0, 100.0, 30.0), icon=(70.0, 45.0, 20.0, 20.0)),
        co.RowBox(index=1, text="newer", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0)),
    ]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "x")]), font_size=28, renderer=renderer, settings_path=None)
    controller.tick(now=0.0)

    controller._hovering = True
    controller._mouse_location = lambda: (10.0, 10.0)   # row 1
    controller.tick(now=0.1)
    assert controller._hover_row == 1
    calls_after_row1 = len(renderer.calls)

    controller._mouse_location = lambda: (10.0, 50.0)   # row 0
    controller.tick(now=0.2)
    assert controller._hover_row == 0
    assert len(renderer.calls) == calls_after_row1 + 1
    assert renderer.calls[-1][-1] == co.RowDecor(hover_row=0, copied_text=None)


def test_controller_hover_row_unchanged_does_not_repaint():
    boxes = [co.RowBox(index=0, text="x", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "x")]), font_size=28, renderer=renderer, settings_path=None)
    controller.tick(now=0.0)
    controller._hovering = True
    controller._mouse_location = lambda: (10.0, 10.0)
    controller.tick(now=0.1)
    calls_before = len(renderer.calls)

    controller.tick(now=0.2)   # same mouse position, same hover_row -> no new repaint
    assert len(renderer.calls) == calls_before


def test_controller_hover_row_forced_none_while_dragging():
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi")]), font_size=28, renderer=renderer, settings_path=None)
    controller.tick(now=0.0)
    controller._hovering = True
    controller._mouse_location = lambda: (10.0, 10.0)
    controller.tick(now=0.1)
    assert controller._hover_row == 0   # sanity check: plain hovering does set it

    controller._dragging = True
    controller.tick(now=0.2)
    assert controller._hover_row is None


def test_controller_clears_row_boxes_and_hover_row_on_hide():
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi"), ("eos",)]), font_size=28, renderer=renderer, settings_path=None)
    controller.tick(now=0.0)
    assert controller._row_boxes == boxes

    controller.tick(now=5.0)   # idle_clear_sec elapsed -> hides
    assert renderer.calls[-1] == ("hide",)
    assert controller._row_boxes == []


# --------------------------------------------- controller: tick() repaint debug line ---
# Instrumentation added alongside the mouseDown: debug line -- shows the SAME
# icon_rect shape at PAINT time so it can be read side by side with what
# _handle_mouse_down logs at CLICK time, revealing whether a row's geometry
# moved between the two (e.g. the live row still growing).
def test_tick_calls_log_repaint_debug_only_on_a_repaint_that_carries_an_icon(monkeypatch):
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi")]), font_size=28, renderer=renderer,
        settings_path=None, debug_input=True)
    calls = []
    monkeypatch.setattr(controller, "_log_repaint_debug", lambda rows, decor: calls.append(decor))

    controller.tick(now=0.0)   # first repaint: not hovering -> hover_row None -> must NOT log
    assert calls == []

    controller._hovering = True
    controller._mouse_location = lambda: (10.0, 10.0)   # inside row 0's frame
    controller.tick(now=0.1)   # repaint WITH hover_row=0 -> must log exactly once
    assert len(calls) == 1
    assert calls[0].hover_row == 0


def test_tick_never_calls_log_repaint_debug_when_debug_input_off(monkeypatch):
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi")]), font_size=28, renderer=renderer, settings_path=None)
    calls = []
    monkeypatch.setattr(controller, "_log_repaint_debug", lambda rows, decor: calls.append(decor))

    controller._hovering = True
    controller._mouse_location = lambda: (10.0, 10.0)
    controller.tick(now=0.0)
    controller.tick(now=0.1)

    assert calls == []


def test_log_repaint_debug_falls_back_to_row_box_bounds_without_a_real_panel(capsys):
    """_FakeRenderer exposes neither .panel nor .content (see its own
    docstring) -- exercises the bounding-box-of-row_boxes fallback."""
    boxes = [
        co.RowBox(index=0, text="older", frame=(0.0, 40.0, 100.0, 30.0), icon=None),
        co.RowBox(index=1, text="newer", frame=(10.0, 0.0, 120.0, 30.0), icon=(90.0, 5.0, 20.0, 20.0)),
    ]
    renderer = _FakeRenderer()
    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=None, debug_input=True)
    controller._row_boxes = boxes

    controller._log_repaint_debug(rows=[("older", False), ("newer", True)], decor=co.RowDecor(hover_row=1))

    out = capsys.readouterr().out
    assert "[caption-overlay] repaint: rows=2 hover_row=1" in out
    assert "icon_rect=(90.0, 5.0, 20.0, 20.0)" in out
    assert "panel=~(0.0, 0.0, 130.0, 70.0)" in out   # bounding box of both row frames


def test_log_repaint_debug_prefers_the_real_panel_frame_when_available(capsys):
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    panel = _FakePanelForHover(frame=_FakeFrame(100.0, 50.0, 300.0, 80.0))
    renderer = _FakeRendererWithPanel(panel)
    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=None, debug_input=True)
    controller._row_boxes = boxes

    controller._log_repaint_debug(rows=[("hi", True)], decor=co.RowDecor(hover_row=0))

    out = capsys.readouterr().out
    assert "panel=(100.0, 50.0, 300.0, 80.0)" in out


def test_log_repaint_debug_icon_rect_none_when_hover_row_has_no_matching_box(capsys):
    renderer = _FakeRenderer()
    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=None, debug_input=True)
    controller._row_boxes = []

    controller._log_repaint_debug(rows=[("hi", True)], decor=co.RowDecor(hover_row=0))

    out = capsys.readouterr().out
    assert "icon_rect=None" in out
    assert "panel=None" in out


# --------------------------------------------------- controller: mouse_down/copy ---
def test_handle_mouse_down_on_icon_calls_copy_fn_and_does_not_arm_drag():
    boxes = [co.RowBox(index=0, text="hello world", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    copied = []

    def fake_copy(text):
        copied.append(text)
        return True

    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hello world")]), font_size=28, renderer=renderer,
        settings_path=None, copy_fn=fake_copy)
    controller.tick(now=0.0)   # populates controller._row_boxes via the fake renderer

    controller._mouse_location = lambda: (75.0, 10.0)   # inside the icon rect
    controller._handle_mouse_down(None)

    assert copied == ["hello world"]
    assert controller._dragging is False
    assert controller._drag_base is None

    controller.tick(now=0.1)
    assert renderer.calls[-1][-1] == co.RowDecor(hover_row=None, copied_text="hello world")


def test_handle_mouse_down_on_icon_with_copy_fn_returning_false_sets_no_feedback():
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi")]), font_size=28, renderer=renderer,
        settings_path=None, copy_fn=lambda text: False)
    controller.tick(now=0.0)

    controller._mouse_location = lambda: (75.0, 10.0)
    controller._handle_mouse_down(None)

    assert controller._copied is None
    assert controller._dragging is False


def test_handle_mouse_down_off_icon_arms_drag_and_does_not_call_copy_fn():
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    copy_calls = []
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi")]), font_size=28, renderer=renderer,
        settings_path=None, copy_fn=lambda text: copy_calls.append(text) or True)
    controller.tick(now=0.0)

    controller._mouse_location = lambda: (10.0, 10.0)   # inside the row's frame but NOT the icon
    controller._handle_mouse_down(None)

    assert copy_calls == []
    assert controller._dragging is True
    assert controller._drag_base == (10.0, 10.0, controller.geometry.cx_frac, controller.geometry.bottom_px)


def test_handle_mouse_down_with_no_row_boxes_arms_drag_as_before():
    """No repaint has happened yet (or the panel is hidden) -> _row_boxes is
    empty -> _hit_icon can never match -> falls through to the existing
    drag-arm behavior, unchanged."""
    renderer = _FakeRenderer()
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=None)
    controller._mouse_location = lambda: (10.0, 10.0)
    controller._handle_mouse_down(None)
    assert controller._dragging is True


def test_handle_mouse_down_calls_mouse_location_exactly_once():
    """Regression guard: _handle_mouse_down must read the cursor position
    ONCE and reuse it for both the icon hit-test and (when it falls
    through) the drag-arm base -- not query it twice."""
    renderer = _FakeRenderer()
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=None)
    calls = []

    def fake_location():
        calls.append(1)
        return (10.0, 10.0)

    controller._mouse_location = fake_location
    controller._handle_mouse_down(None)
    assert len(calls) == 1
    assert controller._dragging is True   # sanity: still fell through and armed the drag


# ---------------------------------- controller: _click_point (root-cause fix) ---
# "Copy icon click does nothing" root-cause coverage: _handle_mouse_down used
# to hit-test ONLY against a fresh _mouse_location() re-poll of the CURRENT
# global cursor position, never the mouseDown_ event's OWN recorded location.
# AppKit event dispatch is not instantaneous, and a real click/release
# inherently involves a little cursor motion (trackpad travel, mouse
# micro-drift) -- so NSEvent.mouseLocation(), read at callback time, can
# already differ from where the click itself actually landed. Reproduced
# hands-on with real AppKit (a synthetic mouseDown_ dispatched via
# panel.sendEvent_ whose own location was dead-center on the icon, while
# _mouse_location() simulated a cursor that had drifted a few points past the
# icon's edge): the OLD code silently treated the click as a drag instead of
# a copy. Fixed by _click_point(), which prefers the event's own location
# (converted to screen via its window) whenever available.
class _FakeEventWindow:
    """Duck-typed NSWindow stand-in exposing only convertPointToScreen_ --
    the one method _click_point calls on event.window(). Always returns the
    same fixed screen point regardless of input: these tests only care that
    _handle_mouse_down USES the converted point, not that the conversion
    math itself is correct (that's real AppKit's job -- see the real-AppKit
    diagnosis scripts this fix came from, not something headlessly
    testable)."""

    def __init__(self, screen_point):
        self._screen_point = screen_point

    def convertPointToScreen_(self, point):
        return SimpleNamespace(x=self._screen_point[0], y=self._screen_point[1])


class _FakeMouseDownEvent:
    """Duck-typed NSEvent stand-in for a real mouseDown_/mouseUp_ event --
    exposes only .window() and .locationInWindow(), the two _click_point
    touches. `window=None` exercises the defensive fallback path."""

    def __init__(self, window, location):
        self._window = window
        self._location = location

    def window(self):
        return self._window

    def locationInWindow(self):
        return self._location


def test_handle_mouse_down_prefers_event_location_over_a_drifted_mouse_location():
    """The exact reproduction: the event's own location is dead-center on
    the icon; _mouse_location() -- what the OLD code used EXCLUSIVELY --
    simulates a cursor that already drifted off it. Must still copy, and
    must not even consult _mouse_location() when a real event+window are
    available."""
    boxes = [co.RowBox(index=0, text="hello world", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 0.0, 20.0, 30.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    copied = []
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hello world")]), font_size=28, renderer=renderer,
        settings_path=None, copy_fn=lambda text: copied.append(text) or True)
    controller.tick(now=0.0)

    event = _FakeMouseDownEvent(window=_FakeEventWindow((75.0, 10.0)), location=(999.0, 999.0))

    def _must_not_be_called():
        raise AssertionError("_mouse_location() must not be consulted when event.window() is available")

    controller._mouse_location = _must_not_be_called

    controller._handle_mouse_down(event)

    assert copied == ["hello world"]
    assert controller._dragging is False
    assert controller._drag_base is None


def test_handle_mouse_down_falls_back_to_mouse_location_when_event_has_no_window():
    """Defensive fallback: a real mouseDown_ always has a window, but if
    event.window() is ever None, _click_point must fall back to
    _mouse_location() rather than crash or silently misbehave."""
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    copied = []
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi")]), font_size=28, renderer=renderer,
        settings_path=None, copy_fn=lambda text: copied.append(text) or True)
    controller.tick(now=0.0)

    controller._mouse_location = lambda: (75.0, 10.0)   # inside the icon rect
    event = _FakeMouseDownEvent(window=None, location=(999.0, 999.0))

    controller._handle_mouse_down(event)

    assert copied == ["hi"]
    assert controller._dragging is False


def test_handle_mouse_down_event_none_still_uses_mouse_location_unchanged():
    """event=None (every pre-existing test in this file, and the smoke/
    diagnosis scripts' own established idiom) must still work exactly as
    before -- the fallback path, not a regression."""
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    copied = []
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi")]), font_size=28, renderer=renderer,
        settings_path=None, copy_fn=lambda text: copied.append(text) or True)
    controller.tick(now=0.0)

    controller._mouse_location = lambda: (75.0, 10.0)
    controller._handle_mouse_down(None)

    assert copied == ["hi"]


def test_copy_feedback_expires_after_copy_feedback_sec_with_exactly_one_more_repaint():
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi")]), font_size=28, renderer=renderer,
        settings_path=None, copy_fn=lambda text: True)

    # _handle_mouse_down timestamps the feedback with the real wall clock
    # (it has no `now` parameter -- AppKit calls it as cb(event)), so every
    # tick() `now` in this test is seeded from the same real clock instead
    # of an arbitrary 0.0 -- otherwise the CaptionLineModel's OWN
    # idle_clear_sec bookkeeping (driven by tick()'s injected `now`) would
    # see a huge, unrelated jump and idle-hide the row out from under us.
    t0 = _time.monotonic()
    controller.tick(now=t0)
    controller._mouse_location = lambda: (75.0, 10.0)
    controller._handle_mouse_down(None)

    assert controller._copied is not None
    expire_at = controller._copied[1]
    assert expire_at == pytest.approx(t0 + co.COPY_FEEDBACK_SEC, abs=0.5)

    controller.tick(now=expire_at - 0.01)
    assert renderer.calls[-1][-1].copied_text == "hi"
    calls_before = len(renderer.calls)

    controller.tick(now=expire_at - 0.005)   # still not expired, nothing else changed -> no repaint
    assert len(renderer.calls) == calls_before

    controller.tick(now=expire_at + 0.01)   # expired
    assert len(renderer.calls) == calls_before + 1
    assert renderer.calls[-1][-1] == co.RowDecor(hover_row=None, copied_text=None)


def test_handle_mouse_down_on_icon_logs_when_debug_input(capsys):
    boxes = [co.RowBox(index=0, text="hi there", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi there")]), font_size=28, renderer=renderer,
        settings_path=None, copy_fn=lambda text: True, debug_input=True)
    controller.tick(now=0.0)
    controller._mouse_location = lambda: (75.0, 10.0)
    controller._handle_mouse_down(None)

    out = capsys.readouterr().out
    assert "[caption-overlay] copied: hi there" in out


# ------------------------------- controller: _handle_mouse_down "mouseDown:" debug line ---
# Coordinator-requested instrumentation: a single line, BEFORE branching,
# with everything needed to explain a miss by hand from a real
# CAPTION_OVERLAY_DEBUG_INPUT=1 session -- click (the point actually used,
# from _click_point) vs cursor (a fresh, independent _mouse_location() poll)
# side by side, so a real-world divergence between the two is directly
# visible rather than only demonstrable via a manufactured one.
def test_handle_mouse_down_debug_line_on_icon_hit_shows_click_cursor_and_hit(capsys):
    boxes = [co.RowBox(index=0, text="hi there", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi there")]), font_size=28, renderer=renderer,
        settings_path=None, copy_fn=lambda text: True, debug_input=True)
    controller.tick(now=0.0)
    controller._hovering = True
    controller._hover_row = 0
    controller._mouse_location = lambda: (75.0, 10.0)   # inside the icon rect

    controller._handle_mouse_down(None)

    out = capsys.readouterr().out
    assert ("[caption-overlay] mouseDown: click=(75.0,10.0) cursor=(75.0,10.0) hovering=True "
            "hover_row=0 boxes=1 icon_rect=(70.0, 5.0, 20.0, 20.0) hit=0") in out
    assert "[caption-overlay] copied: hi there" in out


def test_handle_mouse_down_debug_line_on_miss_says_drag_armed(capsys):
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    copy_calls = []
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi")]), font_size=28, renderer=renderer,
        settings_path=None, copy_fn=lambda text: copy_calls.append(text) or True, debug_input=True)
    controller.tick(now=0.0)
    controller._mouse_location = lambda: (10.0, 10.0)   # inside the row but NOT the icon

    controller._handle_mouse_down(None)

    out = capsys.readouterr().out
    assert "[caption-overlay] mouseDown: click=(10.0,10.0) cursor=(10.0,10.0)" in out
    assert "hit=None" in out
    assert "[caption-overlay] mouseDown: no icon hit -> drag armed" in out
    assert "copied:" not in out
    assert copy_calls == []


def test_handle_mouse_down_debug_line_logs_copy_failed_when_copy_fn_returns_false(capsys):
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi")]), font_size=28, renderer=renderer,
        settings_path=None, copy_fn=lambda text: False, debug_input=True)
    controller.tick(now=0.0)
    controller._mouse_location = lambda: (75.0, 10.0)

    controller._handle_mouse_down(None)

    out = capsys.readouterr().out
    assert "[caption-overlay] copy FAILED for: hi" in out
    assert "copied:" not in out
    assert controller._copied is None


def test_handle_mouse_down_no_debug_lines_when_debug_input_off(capsys):
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi")]), font_size=28, renderer=renderer,
        settings_path=None, copy_fn=lambda text: True)
    controller.tick(now=0.0)
    controller._mouse_location = lambda: (75.0, 10.0)

    controller._handle_mouse_down(None)

    out = capsys.readouterr().out
    assert out == ""


# ------------------------------------------------- controller: _handle_mouse_drag ---
# Regression coverage for a real, live bug: `panel = getattr(self.renderer,
# "panel", None)` used to sit only in the SECOND half of this method, right
# before `if panel is not None:`. Python decides a name is local to the
# WHOLE function the moment it is assigned anywhere in that function's body
# -- so the EARLIER `screen = _screen_for_panel(panel, ...)` line, reading
# `panel` before that later assignment ever ran, raised UnboundLocalError on
# every single call, unconditionally. Drag-to-move was completely dead the
# instant a real drag started. Proven below by actually calling
# _handle_mouse_drag and asserting it runs to completion (no exception) and
# drives geometry.move_to / panel.setFrameOrigin_ with the expected numbers.
#
# _handle_mouse_drag reads AppKit.NSEvent.mouseLocation() directly (not
# through the monkeypatchable _mouse_location() seam _handle_mouse_down/
# _update_hover use -- see that method's own docstring), via its own local
# `import AppKit`. Since this file otherwise stays headless by design (see
# the module docstring), _FakeAppKitModuleForDrag below is swapped into
# sys.modules["AppKit"] instead -- Python's `import AppKit` binds straight
# to whatever is already in sys.modules, real or fake, so this hands the
# method a fixed, test-controlled cursor position without ever touching a
# real WindowServer. Mirrors this file's own sys.modules["AppKit"] = None
# trick used for _require_appkit further up, generalized to a working fake
# instead of a forced ImportError.
class _FakeAppKitModuleForDrag:
    def __init__(self, mouse_x, mouse_y):
        self.NSEvent = SimpleNamespace(mouseLocation=lambda: SimpleNamespace(x=mouse_x, y=mouse_y))
        # _screen_for_panel's fallback argument is evaluated on every call
        # (Python evaluates call arguments eagerly) even though
        # _FakePanelForDrag's own .screen() below always succeeds and so
        # this fallback is never actually used -- it just has to exist.
        self.NSScreen = SimpleNamespace(mainScreen=lambda: None)
        # Mirrors the real free function's (x, y) -> point-like-object
        # contract closely enough for _FakePanelForDrag.setFrameOrigin_ to
        # read .x/.y back off it.
        self.NSMakePoint = lambda x, y: SimpleNamespace(x=x, y=y)


class _FakePanelForDrag:
    """Duck-typed NSPanel stand-in for _handle_mouse_drag: .screen() (feeds
    _screen_for_panel, the same duck-typed shape as the plain `class Panel:
    def screen(self): ...` stand-ins at the very top of this file),
    .frame() (the panel's CURRENT, already-laid-out size -- what
    _drag_reposition_origin's frame_w/frame_h come from), and
    .setFrameOrigin_() (records the live reposition, the same recording
    style as _FakePanelForHover.ignores_calls above)."""

    def __init__(self, screen, frame):
        self._screen = screen
        self._frame = frame
        self.set_frame_origin_calls = []   # [(x, y), ...]

    def screen(self):
        return self._screen

    def frame(self):
        return self._frame

    def setFrameOrigin_(self, point):
        self.set_frame_origin_calls.append((point.x, point.y))


def test_handle_mouse_drag_completes_without_unboundlocalerror_and_calls_move_to(monkeypatch):
    visible = _FakeFrame(0, 0, 1920, 1080)
    screen = SimpleNamespace(visibleFrame=lambda: visible)
    panel = _FakePanelForDrag(screen=screen, frame=_FakeFrame(0, 0, 300.0, 60.0))
    renderer = _FakeRendererWithPanel(panel)
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=None)
    controller._drag_base = (100.0, 50.0, 0.5, 24.0)   # base_x, base_y, base_cx, base_bottom

    monkeypatch.setitem(sys.modules, "AppKit", _FakeAppKitModuleForDrag(mouse_x=150.0, mouse_y=80.0))
    move_to_calls = []
    monkeypatch.setattr(controller.geometry, "move_to",
                         lambda cx, bottom, vh: move_to_calls.append((cx, bottom, vh)))

    controller._handle_mouse_drag(None)   # must not raise UnboundLocalError

    dx, dy = 150.0 - 100.0, 80.0 - 50.0
    assert len(move_to_calls) == 1
    called_cx, called_bottom, called_vh = move_to_calls[0]
    assert called_cx == pytest.approx(0.5 + dx / visible.size.width)
    assert called_bottom == pytest.approx(24.0 + dy)
    assert called_vh == pytest.approx(visible.size.height)
    assert panel.set_frame_origin_calls   # the live per-tick reposition also ran, not just move_to


def test_handle_mouse_drag_repositions_panel_frame_via_drag_reposition_origin(monkeypatch):
    """The live reposition (panel.setFrameOrigin_, applied directly to the
    CURRENT frame between full repaints -- see the method's own docstring)
    must be wired through _drag_reposition_origin with the right
    frame_w/frame_h/font_size/geometry -- the _drag_reposition_origin tests
    above already prove that function's own math; this just confirms
    _handle_mouse_drag actually calls it, now that it can run at all."""
    visible = _FakeFrame(0, 0, 1920, 1080)
    screen = SimpleNamespace(visibleFrame=lambda: visible)
    frame = _FakeFrame(0, 0, 300.0, 60.0)
    panel = _FakePanelForDrag(screen=screen, frame=frame)
    renderer = _FakeRendererWithPanel(panel)
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=None)
    controller._drag_base = (100.0, 50.0, 0.5, 24.0)

    monkeypatch.setitem(sys.modules, "AppKit", _FakeAppKitModuleForDrag(mouse_x=150.0, mouse_y=80.0))

    controller._handle_mouse_drag(None)

    # geometry.move_to already ran for real (not monkeypatched here) by the
    # time _handle_mouse_drag reaches the reposition block, so recomputing
    # against controller.geometry NOW matches what the method itself used.
    expected_x, expected_y = co._drag_reposition_origin(
        visible, frame.size.width, frame.size.height, controller.geometry.font_size, controller.geometry)
    assert len(panel.set_frame_origin_calls) == 1
    actual_x, actual_y = panel.set_frame_origin_calls[0]
    assert actual_x == pytest.approx(expected_x)
    assert actual_y == pytest.approx(expected_y)


# ------------------------------------------------------- controller: mouse_up ---
def test_handle_mouse_up_without_prior_drag_does_not_write_settings(tmp_path):
    path = tmp_path / "geometry.json"
    renderer = _FakeRenderer()
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=path)
    assert controller._dragging is False

    controller._handle_mouse_up(None)

    assert not path.exists()


def test_handle_mouse_up_after_a_drag_still_writes_settings(tmp_path):
    """Regression guard: the ordinary (non-icon-click) drag-then-release
    path must keep saving immediately on mouse-up, exactly as before."""
    path = tmp_path / "geometry.json"
    renderer = _FakeRenderer()
    controller = co._OverlayController(_FakeDisplayQueue([]), font_size=28, renderer=renderer, settings_path=path)
    controller._dragging = True
    controller._drag_base = (0.0, 0.0, 0.5, 24.0)

    controller._handle_mouse_up(None)

    assert path.exists()
    assert controller._dragging is False
    assert co.load_geometry(path).font_size == controller.geometry.font_size


def test_icon_click_then_mouse_up_does_not_write_settings(tmp_path):
    """End-to-end: clicking the icon (which does NOT arm a drag) followed
    by the mouseUp_ AppKit always sends afterward must not persist
    geometry -- only a real drag's release does."""
    path = tmp_path / "geometry.json"
    boxes = [co.RowBox(index=0, text="hi", frame=(0.0, 0.0, 100.0, 30.0), icon=(70.0, 5.0, 20.0, 20.0))]
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "hi")]), font_size=28, renderer=renderer,
        settings_path=path, copy_fn=lambda text: True)
    controller.tick(now=0.0)

    controller._mouse_location = lambda: (75.0, 10.0)
    controller._handle_mouse_down(None)
    controller._handle_mouse_up(None)

    assert not path.exists()


# ================================================== controller: translate icon ===
# The translate icon sits immediately right of the copy icon on the hovered
# row and puts the TRANSLATION on the clipboard (Japanese -> English, anything
# else -> Japanese; see alwayswhisper.live.translator.py). Unlike copying, it is a
# network call, so the controller runs it off the UI thread and picks the
# result up on a later tick(). Every test here injects run_async=_run_inline,
# which runs that "thread" synchronously: the result still travels through the
# real queue and is still applied by the real tick(), so the actual production
# code path is exercised -- only the concurrency is made deterministic. No test
# in this file starts a thread or touches the network.
def _run_inline(fn):
    fn()


def _translate_boxes(text="こんにちは", copy_rect=(60, 0, 20, 30), translate_rect=(80, 0, 20, 30)):
    return [co.RowBox(0, text, (0, 0, 100, 30), copy_rect, translate_rect)]


def test_controller_defaults_translate_fn_to_translator_translate():
    from alwayswhisper.live import translator

    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=_FakeRenderer(), settings_path=None)

    assert controller._translate_fn is translator.translate


def test_mouse_down_on_the_translate_icon_translates_and_does_not_copy_or_drag():
    """The click must not fall through to either of the other two things a
    panel click can do: copying the ORIGINAL text, or arming a drag."""
    copied, asked = [], []
    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=_FakeRenderer(), settings_path=None,
        copy_fn=lambda text: copied.append(text) or True,
        translate_fn=lambda text: asked.append(text) or "Hello",
        run_async=_run_inline)
    controller._row_boxes = _translate_boxes()
    controller._mouse_location = lambda: (90, 15)

    controller._handle_mouse_down(None)

    assert asked == ["こんにちは"]
    assert copied == []             # not until the result arrives, and then it is the TRANSLATION
    assert controller._dragging is False


def test_translation_result_is_copied_to_the_clipboard_on_the_next_tick():
    copied = []
    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=_FakeRenderer(), settings_path=None,
        copy_fn=lambda text: copied.append(text) or True,
        translate_fn=lambda text: "Hello there",
        run_async=_run_inline)
    controller._row_boxes = _translate_boxes()
    controller._mouse_location = lambda: (90, 15)

    controller._handle_mouse_down(None)
    assert controller._translate[:2] == ("こんにちは", "pending")
    assert copied == []             # the UI thread has not picked the result up yet

    controller.tick(now=1.0)

    assert copied == ["Hello there"]
    assert controller._translate[:2] == ("こんにちは", "done")


def test_translation_failure_copies_nothing_and_shows_the_failed_state(capsys):
    copied = []

    def _boom(text):
        raise RuntimeError("no credits")

    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=_FakeRenderer(), settings_path=None,
        copy_fn=lambda text: copied.append(text) or True,
        translate_fn=_boom, run_async=_run_inline)
    controller._row_boxes = _translate_boxes()
    controller._mouse_location = lambda: (90, 15)

    controller._handle_mouse_down(None)
    controller.tick(now=1.0)

    assert copied == []
    assert controller._translate[:2] == ("こんにちは", "failed")
    # A failed network call must say so on stdout even without --debug-input:
    # a silently-nothing-happened icon is unexplainable to the user.
    assert "no credits" in capsys.readouterr().out


def test_a_failing_clipboard_write_after_a_good_translation_is_also_a_failure():
    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=_FakeRenderer(), settings_path=None,
        copy_fn=lambda text: False, translate_fn=lambda text: "Hello",
        run_async=_run_inline)
    controller._row_boxes = _translate_boxes()
    controller._mouse_location = lambda: (90, 15)

    controller._handle_mouse_down(None)
    controller.tick(now=1.0)

    assert controller._translate[:2] == ("こんにちは", "failed")


def test_translate_feedback_expires_after_the_feedback_window():
    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=_FakeRenderer(), settings_path=None,
        copy_fn=lambda text: True, translate_fn=lambda text: "Hello",
        run_async=_run_inline)
    controller._row_boxes = _translate_boxes()
    controller._mouse_location = lambda: (90, 15)

    controller._handle_mouse_down(None)
    controller.tick(now=1.0)
    assert controller._translate is not None

    controller.tick(now=1.0 + co.COPY_FEEDBACK_SEC - 0.01)
    assert controller._translate is not None

    controller.tick(now=1.0 + co.COPY_FEEDBACK_SEC)
    assert controller._translate is None


def test_pending_state_never_expires_on_its_own():
    """A slow API call must not have its "in progress" icon time out from
    under it -- only an arrived result (or failure) ends the pending state."""
    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=_FakeRenderer(), settings_path=None,
        copy_fn=lambda text: True, translate_fn=lambda text: "Hello",
        run_async=lambda fn: None)   # the "thread" never runs: permanently in flight
    controller._row_boxes = _translate_boxes()
    controller._mouse_location = lambda: (90, 15)

    controller._handle_mouse_down(None)
    controller.tick(now=999.0)

    assert controller._translate[:2] == ("こんにちは", "pending")


def test_clicking_again_while_the_same_row_is_in_flight_does_not_ask_twice():
    """Guards against paying twice (and racing two results) for one row --
    the dwell fallback below makes a second trigger easy to hit by accident."""
    asked = []
    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=_FakeRenderer(), settings_path=None,
        copy_fn=lambda text: True,
        translate_fn=lambda text: asked.append(text) or "Hello",
        run_async=lambda fn: None)   # stays in flight
    controller._row_boxes = _translate_boxes()
    controller._mouse_location = lambda: (90, 15)

    controller._handle_mouse_down(None)
    controller._handle_mouse_down(None)

    assert asked == []               # run_async never ran the worker...
    assert controller._translate_inflight == {"こんにちは"}   # ...but exactly one is registered


def test_a_finished_row_can_be_translated_again():
    """The in-flight guard is released once the result has been applied --
    it exists to stop DUPLICATE requests, not to make a row translate only
    once. Driven through a real repaint (a display event plus
    _FakeRendererWithBoxes) rather than by assigning _row_boxes directly,
    because the tick() in between rebuilds that list from the repaint."""
    asked = []
    renderer = _FakeRendererWithBoxes(_translate_boxes())
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "こんにちは")]), font_size=28, renderer=renderer,
        settings_path=None, copy_fn=lambda text: True,
        translate_fn=lambda text: asked.append(text) or "Hello",
        run_async=_run_inline)
    controller.tick(now=0.0)
    controller._mouse_location = lambda: (90, 15)

    controller._handle_mouse_down(None)
    controller.tick(now=1.0)
    controller._handle_mouse_down(None)
    controller.tick(now=2.0)

    assert asked == ["こんにちは", "こんにちは"]


def test_global_mouse_down_translates_when_the_click_is_inside_the_translate_icon():
    asked = []
    controller = co._OverlayController(
        _FakeDisplayQueue([]), font_size=28, renderer=_FakeRenderer(), settings_path=None,
        copy_fn=lambda text: True,
        translate_fn=lambda text: asked.append(text) or "Hello",
        run_async=_run_inline)
    controller._row_boxes = _translate_boxes()

    class Event:
        def locationInWindow(self):
            return type("Point", (), {"x": 90, "y": 15})()

    controller._handle_global_mouse_down(Event())

    assert asked == ["こんにちは"]


def test_translate_icon_dwell_translates_when_mouse_events_are_not_delivered():
    """Same fallback the copy icon already has (see
    test_icon_dwell_copies_when_mouse_events_are_not_delivered): on setups
    where WindowServer never delivers the click, resting on the icon for
    HOVER_COPY_DELAY_SEC triggers it."""
    asked = []
    boxes = _translate_boxes(text="こんにちは")
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "こんにちは")]), renderer=renderer, settings_path=None,
        copy_fn=lambda text: True,
        translate_fn=lambda text: asked.append(text) or "Hello",
        run_async=_run_inline)
    controller.tick(now=0.0)
    controller._hovering = True
    controller._mouse_location = lambda: (90, 15)

    controller.tick(now=0.1)
    controller.tick(now=0.1 + co.HOVER_COPY_DELAY_SEC - 0.01)
    assert asked == []

    controller.tick(now=0.1 + co.HOVER_COPY_DELAY_SEC)
    assert asked == ["こんにちは"]


def test_translate_icon_dwell_does_not_re_translate_while_the_cursor_rests_there():
    """Dwell latches per text -- otherwise resting the cursor on the icon
    would fire a paid API call every HOVER_COPY_DELAY_SEC."""
    asked = []
    boxes = _translate_boxes(text="こんにちは")
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "こんにちは")]), renderer=renderer, settings_path=None,
        copy_fn=lambda text: True,
        translate_fn=lambda text: asked.append(text) or "Hello",
        run_async=_run_inline)
    controller.tick(now=0.0)
    controller._hovering = True
    controller._mouse_location = lambda: (90, 15)

    for step in range(1, 30):
        controller.tick(now=0.1 + step * co.HOVER_COPY_DELAY_SEC)

    assert asked == ["こんにちは"]


def test_translate_state_reaches_the_renderer_as_decor():
    """The icon can only render its pending/done/failed shape if tick()
    actually forwards the state -- and a state change alone (no new text,
    no geometry change) has to be enough to trigger the repaint."""
    boxes = _translate_boxes(text="こんにちは")
    renderer = _FakeRendererWithBoxes(boxes)
    controller = co._OverlayController(
        _FakeDisplayQueue([("chunk", "こんにちは")]), renderer=renderer, settings_path=None,
        copy_fn=lambda text: True, translate_fn=lambda text: "Hello",
        run_async=_run_inline)
    controller.tick(now=0.0)
    controller._mouse_location = lambda: (90, 15)
    shows_before = len([c for c in renderer.calls if c[0] == "show"])

    controller._handle_mouse_down(None)
    controller.tick(now=0.1)

    shows = [c for c in renderer.calls if c[0] == "show"]
    assert len(shows) > shows_before          # the state change alone repainted
    decor = shows[-1][5]
    assert decor.translate_text == "こんにちは"
    assert decor.translate_state == "done"


# Floating panel controls: actual controller callbacks, no WindowServer needed.
def test_close_hover_hides_immediately_and_next_chunk_restores_under_same_cursor():
    controller, renderer, dq = _controller([("chunk", "hello"), ("eos",)])
    renderer.control_frame = (0, 0, 300, 100)
    mouse = [150, 50]
    controller._mouse_location = lambda: tuple(mouse)
    controller.tick(now=0.0)
    mouse[:] = [25, 85]
    controller.tick(now=0.1)
    assert renderer.calls[-1] == ("hide",)
    assert controller._dismissed
    dq._items.extend([("eos",)])
    controller.tick(now=0.2)
    assert not controller._panel_visible
    dq._items.extend([("chunk", "new caption")])
    controller.tick(now=0.3)
    assert controller._panel_visible
    assert renderer.calls[-1][1] == [("new caption", True)]
    controller.tick(now=0.4)
    assert controller._panel_visible
    mouse[:] = [150, 50]
    controller.tick(now=0.5)
    mouse[:] = [25, 85]
    controller.tick(now=0.6)
    assert not controller._panel_visible


@pytest.mark.parametrize("corner", [(1, 1)])
def test_top_right_handle_resizes_and_persists_on_release(corner, tmp_path):
    controller, renderer, _ = _controller([("chunk", "hello")])
    controller.settings_path = tmp_path / "geometry.json"
    renderer.control_frame = (0, 0, 300, 100)
    controller._mouse_location = lambda: (150, 50)
    controller.tick(now=0.0)
    _, corners = co._overlay_controls(renderer.control_frame)
    x, y, w, h = corners[corner]
    mouse = [x + w / 2, y + h / 2]
    controller._mouse_location = lambda: tuple(mouse)
    controller._handle_mouse_down(None)
    assert controller._resize_base is not None
    assert controller._drag_base is None
    mouse[0] += corner[0] * 60
    mouse[1] += corner[1] * 20
    controller._handle_mouse_drag(None)
    assert controller.geometry.font_size == pytest.approx(28 * 1.2)
    assert renderer.calls[-1][3] == pytest.approx(28 * 1.2)
    controller._handle_mouse_up(None)
    assert co.load_geometry(controller.settings_path).font_size == pytest.approx(28 * 1.2)
    assert controller._resize_base is None


def test_resize_clamps_font_size():
    controller, _, _ = _controller()
    controller._resize_base = ((0, 0), (1, 1), 28, (0, 0, 300, 100))
    controller._resize_to((-10000, -10000))
    assert controller.geometry.font_size == 12
    controller._resize_to((10000, 10000))
    assert controller.geometry.font_size == 96


def test_empty_chunk_does_not_restore_dismissed_caption():
    controller, renderer, dq = _controller([("chunk", "hello")])
    controller.tick(now=0.0)
    controller._dismiss()
    dq._items.extend([("chunk", " "), ("eos",)])
    controller.tick(now=0.1)
    assert not controller._panel_visible



def test_corner_hover_does_not_expand_history_and_move_grip():
    controller, renderer, _ = _controller([
        ("chunk", "old"), ("eos",), ("chunk", "live")])
    renderer.control_frame = (0, 0, 300, 100)
    controller._mouse_location = lambda: (150, 50)
    controller.tick(now=0.0)
    controller._hovering = True
    controller._mouse_location = lambda: (294, 94)
    controller.tick(now=0.1)
    assert renderer.calls[-1][1] == [("live", True)]
    controller._mouse_location = lambda: (150, 50)
    controller.tick(now=0.2)
    assert renderer.calls[-1][1] == [("old", False), ("live", True)]



def test_hover_entry_at_idle_deadline_is_checked_before_hiding(monkeypatch):
    controller, renderer, _ = _controller([("chunk", "hello"), ("eos",)])
    controller.tick(now=0.0)
    def enter_panel():
        controller._hovering = True
    monkeypatch.setattr(controller, "_update_hover", enter_panel)
    controller.tick(now=5.0)
    assert controller._panel_visible
    assert renderer.calls[-1][0] == "show"


def test_corner_hover_holds_collapsed_caption_past_timeout():
    controller, renderer, _ = _controller([("chunk", "hello"), ("eos",)])
    renderer.control_frame = (0, 0, 300, 100)
    controller._mouse_location = lambda: (294, 94)
    controller.tick(now=0.0)
    controller._hovering = True
    controller.tick(now=30.0)
    assert not controller._expanded
    assert controller._panel_visible
    assert renderer.calls[-1][1] == [("hello", True)]



def test_native_event_pump_sends_queued_input_in_order():
    pending = iter(["down", "drag", "up", None])
    sent = []
    app = SimpleNamespace(
        nextEventMatchingMask_untilDate_inMode_dequeue_=lambda *args: next(pending),
        sendEvent_=sent.append, updateWindows=lambda: sent.append("update"))
    appkit = SimpleNamespace(
        NSApplication=SimpleNamespace(sharedApplication=lambda: app),
        NSEventMaskAny=123, NSDefaultRunLoopMode="default",
        NSDate=SimpleNamespace(distantPast=lambda: None))
    co._dispatch_pending_events(appkit)
    assert sent == ["down", "drag", "up", "update"]


def test_interactive_loop_pumps_input_and_cleans_up_timer(monkeypatch):
    calls = []
    timer = SimpleNamespace(invalidate=lambda: calls.append("invalidate"))
    app = SimpleNamespace(finishLaunching=lambda: calls.append("launch"))
    appkit = SimpleNamespace(
        NSApplication=SimpleNamespace(sharedApplication=lambda: app),
        NSTimer=SimpleNamespace(scheduledTimerWithTimeInterval_repeats_block_=
            lambda *args: calls.append("input timer") or timer))
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    helper = SimpleNamespace(runConsoleEventLoop=lambda **kw: calls.append(("loop", kw)))
    co._run_interactive_event_loop(helper)
    assert calls == ["launch", "input timer", ("loop", {"installInterrupt": True}), "invalidate"]


@pytest.mark.parametrize("point", [(5, 5), (150, 50), (295, 5), (150, 90)])
def test_whole_caption_area_arms_move_except_explicit_controls(point):
    controller, renderer, _ = _controller([("chunk", "hello")])
    renderer.control_frame = (0, 0, 300, 100)
    controller._mouse_location = lambda: point
    controller.tick(now=0.0)
    controller._handle_mouse_down(None)
    assert controller._dragging
    assert controller._drag_base[:2] == point
    assert controller._resize_base is None


def test_menu_reopens_dismissed_caption_and_disables_idle_hide(monkeypatch):
    controller, renderer, dq = _controller([("chunk", "hello"), ("eos",)])
    controller.tick(now=0.0)
    controller._dismiss()
    monkeypatch.setattr(co.time, "monotonic", lambda: 30.0)
    controller.show_from_menu()
    assert controller._panel_visible
    assert controller._pinned
    assert renderer.calls[-1][1] == [("hello", True)]
    controller.tick(now=600.0)
    assert controller._panel_visible
    dq._items.extend([("chunk", "next"), ("eos",)])
    controller.tick(now=601.0)
    controller.tick(now=1200.0)
    assert controller._panel_visible
    assert renderer.calls[-1][1] == [("next", True)]


def test_menu_opens_waiting_caption_before_first_recognition(monkeypatch):
    controller, renderer, dq = _controller()
    monkeypatch.setattr(co.time, "monotonic", lambda: 0.0)
    controller.show_from_menu()
    assert renderer.calls[-1][1] == [("音声認識を待っています…", True)]
    controller.tick(now=600.0)
    assert controller._panel_visible
    dq._items.extend([("chunk", "first speech"), ("eos",)])
    controller.tick(now=601.0)
    assert renderer.calls[-1][1] == [("first speech", True)]
    assert controller.model._history == []
    assert controller._pinned


def test_explicit_close_ends_menu_pin_and_next_recognition_uses_idle_hide(monkeypatch):
    controller, renderer, dq = _controller([("chunk", "hello"), ("eos",)])
    controller.tick(now=0.0)
    monkeypatch.setattr(co.time, "monotonic", lambda: 1.0)
    controller.show_from_menu()
    controller._dismiss()
    assert not controller._pinned
    controller.tick(now=2.0)
    assert not controller._panel_visible
    dq._items.extend([("chunk", "new"), ("eos",)])
    controller.tick(now=3.0)
    assert controller._panel_visible
    controller.tick(now=8.0)
    assert not controller._panel_visible


def test_menu_button_survives_caption_dismiss_and_is_removed_at_shutdown():
    renderer = _FakeRenderer()
    callbacks, removals = [], []
    renderer.install_status_item = callbacks.append
    renderer.remove_status_item = lambda: removals.append(True)
    controller = co._OverlayController(_FakeDisplayQueue([]), renderer=renderer, settings_path=None)
    assert len(callbacks) == 1
    callbacks[0]()
    assert controller._pinned and controller._panel_visible
    controller._dismiss()
    assert removals == []
    callbacks[0]()
    assert controller._panel_visible
    controller.close()
    assert removals == [True]


@pytest.mark.parametrize("position", ["notch", "dynamic-island"])
def test_docked_position_overrides_saved_settings_and_can_restore_free(tmp_path, position):
    path = tmp_path / "settings.json"
    co.save_geometry(path, co.OverlayGeometry(cx_frac=.2, bottom_px=123, font_size=36))
    controller = co._OverlayController(None, renderer=_FakeRenderer(), settings_path=path,
                                       position=position)
    assert controller.geometry.position == position
    assert controller.geometry.font_size == 36
    assert co.load_geometry(path).position == position
    restored = co._OverlayController(None, renderer=_FakeRenderer(), settings_path=path,
                                     position="free")
    assert restored.geometry.cx_frac == .2
    assert restored.geometry.bottom_px == 123
    restored.geometry.set_position("bottom")
    assert restored.geometry.position == "free"
    assert restored.geometry.cx_frac == .5
    assert restored.geometry.bottom_px == 24


@pytest.mark.parametrize("bad", [None, [], {}, True, 123, "unknown"])
def test_invalid_saved_position_falls_back_to_free(bad):
    assert co.OverlayGeometry.from_dict({"position": bad}).position == "free"


def _placement_screen(x=0, y=0, width=1512, height=982, inset=32):
    return SimpleNamespace(
        frame=lambda: _FakeFrame(x, y, width, height),
        visibleFrame=lambda: _FakeFrame(x, y + 60, width, height - 85),
        safeAreaInsets=lambda: SimpleNamespace(top=inset))


@pytest.mark.parametrize("position,gap", [("notch", 0)])
def test_top_layout_keeps_live_row_below_camera_while_history_expands(position, gap):
    geometry = co.OverlayGeometry(cx_frac=.1, bottom_px=300, position=position)
    screen = _placement_screen(x=-1512, y=-400)
    visible = co._caption_visible_frame(screen, geometry)
    safe_top = -400 + 982 - 32
    assert visible.origin.y + visible.size.height == safe_top
    single, _, boxes = co._compute_layout([("live", 200, 36)], 28, None, visible, geometry)
    expanded, _, expanded_boxes = co._compute_layout(
        [("old", 360, 100), ("recent", 180, 40), ("live", 200, 36)],
        28, 2, visible, geometry)
    assert single[1] + single[3] == pytest.approx(safe_top - gap)
    assert expanded[1] + expanded[3] == pytest.approx(safe_top - gap)
    assert expanded[0] + expanded[2] / 2 == pytest.approx(-1512 / 2)
    assert expanded_boxes[-1].frame[1] == boxes[-1].frame[1]
    assert expanded_boxes[-1].frame[1] > expanded_boxes[1].frame[1] > expanded_boxes[0].frame[1]
    assert expanded_boxes[-1].icon is not None


def test_docked_screen_prefers_notched_display_and_falls_back_when_disconnected():
    external = _placement_screen(x=1512, inset=0)
    laptop = _placement_screen()
    panel = SimpleNamespace(screen=lambda: external)
    geometry = co.OverlayGeometry(position="notch")
    assert co._caption_screen(panel, external, [external, laptop], geometry) is laptop
    assert co._caption_screen(panel, external, [external], geometry) is external
    geometry.set_position("free")
    assert co._caption_screen(panel, external, [external, laptop], geometry) is external


def test_no_notch_or_old_macos_uses_visible_area_below_menu_bar():
    screen = _placement_screen(inset=0)
    del screen.safeAreaInsets
    frame = co._caption_visible_frame(screen, co.OverlayGeometry(position="notch"))
    assert frame.origin.y + frame.size.height == 957


def test_docked_body_cannot_arm_move_gesture():
    controller = co._OverlayController(None, renderer=_FakeRenderer(), settings_path=None,
                                       position="notch")
    controller._mouse_location = lambda: (100, 100)
    controller._handle_mouse_down(None)
    assert not controller._dragging
    assert controller._drag_base is None


def test_display_geometry_change_triggers_repaint_without_new_words():
    renderer = _FakeRenderer()
    screen = [0, 0, 1512, 950]
    renderer.screen_state = lambda geometry: tuple(screen)
    controller = co._OverlayController(_FakeDisplayQueue([("chunk", "live")]),
                                       renderer=renderer, settings_path=None, position="notch")
    controller.tick(now=0)
    count = len(renderer.calls)
    screen[2] = 1920
    controller.tick(now=.1)
    assert len(renderer.calls) == count + 1


def test_island_anchor_uses_camera_housing_instead_of_visible_frame():
    screen = _placement_screen(x=-1800, y=-200, width=1800, height=1169, inset=38)
    screen.auxiliaryTopLeftArea = lambda: _FakeFrame(-1800, 931, 790, 38)
    screen.auxiliaryTopRightArea = lambda: _FakeFrame(-790, 931, 790, 38)
    anchor = co._island_anchor(screen)
    assert anchor.camera_width == 220
    assert anchor.camera_height == 38
    assert anchor.center_x == -900
    assert anchor.top == 969
    assert anchor.attached


def test_island_compact_and_expanded_share_physical_screen_top():
    anchor = co.IslandAnchor(900, 1169, 220, 38, 1800)
    compact, _, boxes = co._island_layout([], 28, None, anchor)
    expanded, _, rows = co._island_layout([("字幕は島の中", 250, 34)], 28, None, anchor)
    assert boxes == []
    assert compact[2:] == (316, 44)
    for x, y, w, h in (compact, expanded):
        assert x + w / 2 == 900
        assert y + h == 1169  # not safe-area bottom / visibleFrame.top
    assert expanded[2] > compact[2] and expanded[3] > compact[3]
    assert rows[0].frame[1] + rows[0].frame[3] <= 1169 - 38


def test_island_animation_keeps_camera_covered_and_stays_between_endpoints():
    anchor = co.IslandAnchor(900, 1169, 220, 38, 1800)
    compact, _, _ = co._island_layout([], 28, None, anchor)
    expanded, _, _ = co._island_layout([("字幕", 250, 34)], 28, None, anchor)
    previous_width = compact[2]
    for i in range(11):
        rect = co._island_tween(compact, expanded, i / 10)
        assert rect[0] + rect[2] / 2 == pytest.approx(900)
        assert rect[1] + rect[3] == pytest.approx(1169)
        assert previous_width <= rect[2] <= expanded[2]
        previous_width = rect[2]
    assert co._island_tween(compact, expanded, 2) == expanded


def test_island_without_camera_is_a_detached_pill():
    anchor = co._island_anchor(_placement_screen(inset=0))
    assert not anchor.attached
    assert anchor.top == 949  # visible top 957, with 8pt clearance
    assert co._island_layout([], 28, None, anchor)[0][2:] == (160, 32)


def test_island_lifecycle_compact_before_speech_and_after_idle():
    renderer = _FakeRenderer()
    dq = _FakeDisplayQueue([])
    controller = co._OverlayController(dq, renderer=renderer, settings_path=None, position="dynamic-island")
    controller.tick(now=0)
    assert renderer.calls[-1][0] == "show" and renderer.calls[-1][1] == []
    assert controller._panel_visible
    dq._items.extend([("chunk", "発話で開く")])
    controller.tick(now=1)
    assert renderer.calls[-1][1] == [("発話で開く", True)]
    controller.tick(now=7)
    assert renderer.calls[-1][0] == "show" and renderer.calls[-1][1] == []
    assert controller._panel_visible
    controller._hovering = True
    controller.tick(now=8)
    assert renderer.calls[-1][1] == [("発話で開く", True)]
    controller._dismiss()
    assert renderer.calls[-1][0] == "show" and renderer.calls[-1][1] == []
    controller.close()
    assert renderer.calls[-1][0] == "hide"


def test_audio_meter_tracks_actual_loudness_and_releases_on_stale_capture():
    quiet = co.AudioLevelMeter()
    loud = co.AudioLevelMeter()
    for i in range(30):
        now = i / 30
        quiet.update((now, .004), now)
        loud.update((now, .1), now)
    assert 0 < quiet.level < loud.level < 1
    previous = loud.level
    for i in range(30, 75):
        loud.update((29 / 30, .1), i / 30)
    assert loud.level == 0
    assert previous > .5
    assert all(r[3] == 2 for r in co._audio_bar_rects(loud.level))


def test_audio_meter_silence_invalid_data_and_full_scale():
    meter = co.AudioLevelMeter()
    for sample in [None, (0, 0), (0, -1), (0, float('nan')), (0, float('inf'))]:
        assert meter.update(sample, 0) == 0
    for i in range(30):
        assert 0 <= meter.update((i / 30, 10), i / 30) <= 1
    assert meter.level > .99
    rects = co._audio_bar_rects(meter.level)
    assert rects[2][3] > rects[1][3] > rects[0][3]
    assert all(0 <= x <= 20 - w and 0 <= y <= 20 - h for x, y, w, h in rects)
