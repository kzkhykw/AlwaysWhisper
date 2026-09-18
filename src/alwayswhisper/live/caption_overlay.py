#!/usr/bin/env python3
"""macOS live captions: an animated notch island and a legacy draggable overlay.

AppKit loads only at runtime. Caption/history/layout logic is safe to import
without macOS. Start with `alwayswhisper live`; preview with `--demo`.
"""
import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

from . import translator   # stdlib-only sibling module; safe to import headlessly


# ============================================================= pure logic ===
class CaptionLineModel:
    """The live caption line, PLUS a scrollable history of previous lines --
    driven entirely by (event, now) pairs, no wall-clock reads inside this
    class, so it is fully deterministic and unit-testable
    (tests/test_caption_overlay.py injects `now` directly, the same idiom
    live_transcriber.TranscriptWriter uses for started_at/start_s).

    Events mirror exactly what live_transcriber.py's _worker prints to the
    terminal (see its display_q wiring / _emit_display):
      ("chunk", text) -- a piece of display text arrived. If the LIVE line
        had already ended (an "eos" was the most recent event, or the line
        had gone invisible from an idle timeout even with no "eos" ever
        seen -- see apply()), the previous live text (if any) is PUSHED
        into history (capped at HISTORY_CAP, oldest dropped) and this chunk
        REPLACES the live line (one caption line per utterance, the
        「1行ずつ」 behavior); otherwise it's appended, verbatim, to the
        current live line (Whisper word tokens already carry their own
        leading space, so no separator is added here).
      ("eos",) -- the current segment's display output is complete. Does
        NOT clear the text -- it stays on screen, unchanged, until
        idle_clear_sec has elapsed with no further event (see visible_at).

    No length limit and no trimming on the live line's text: it always
    accumulates in full, however long it gets (user decision -- long
    captions must show their WHOLE text, never chop off content). The
    AppKit layer wraps it to as many lines as needed and grows the panel
    instead (see _layout_stack), rather than this class ever dropping
    characters from a line's OWN text. History rows, once pushed, are
    likewise stored and shown verbatim.

    Collapsed vs. expanded (screen-space tradeoff): visible_rows(now,
    expanded) is COLLAPSED by default -- just the live row, no history --
    unless `expanded` is True (the caller passes its current hover/drag
    state -- see _OverlayController.tick) or the effective scroll offset is
    > 0 (the user just scrolled). Either way it EXPANDS to exactly the old
    always-on look: the newest HISTORY_WINDOW (2) history rows above the
    pinned live row.

    Scrollback (Apple Live Captions-style, only reachable while expanded):
    scroll_by(row_delta, now) moves a scroll OFFSET (positive = further
    back into history/older, negative = toward the live row/newer,
    clamped to [0, max(0, len(history) - HISTORY_WINDOW)]). The offset is a
    float so trackpad motion can slide continuously; the integer part
    selects which HISTORY_WINDOW-sized slice of history is shown and the
    fractional part is a clip-shift of the next-older row (see
    visible_rows / _compute_layout). The live row itself never moves,
    always shown, always last. AUTO_RETURN_SEC (10s) of no further
    scroll_by() lazily resets the effective offset back to 0 (no
    background timer -- computed fresh from `now` on every read, exactly
    like idle_clear_sec). While the effective offset is > 0, OR `hold` is
    passed to visible_at/visible_rows (hover/drag), idle-hide is suppressed
    (see visible_at) -- deliberately: a user actively reading back through
    history, or with the cursor resting on the panel, should not have the
    whole stack vanish out from under them just because the LIVE line's own
    idle timer expired (approved tradeoff -- see _OverlayController).
    """

    HISTORY_CAP = 200
    HISTORY_WINDOW = 2
    AUTO_RETURN_SEC = 10.0

    def __init__(self, idle_clear_sec=5.0):
        self.idle_clear_sec = idle_clear_sec
        self._text = ""
        self._ended = True          # nothing has started yet -> next chunk starts fresh
        self._last_event_at = None  # monotonic time of the last event, or None if none yet
        self._history = []          # oldest -> newest, capped at HISTORY_CAP
        self._offset = 0.0          # raw scroll offset: rows back from live (0 = at the live edge)
        self._last_scroll_at = None  # monotonic time of the last scroll_by(), or None if never scrolled

    def apply(self, event, now):
        """Advance the model by one event. `now` is the caller's monotonic
        clock reading at the moment this event was drained from the display
        queue (see _OverlayController.tick)."""
        kind = event[0]
        if kind == "chunk":
            text = event[1]
            # A fresh line starts not only right after an eos, but also if
            # the line had already gone invisible from an idle timeout with
            # no eos ever seen (a stalled/never-finished segment) -- either
            # way, resurrecting stale off-screen text into a live line would
            # be wrong, so both cases start clean -- and, new in this
            # version, push whatever the live line WAS onto history first
            # (only if it's non-empty: the very first chunk ever, or one
            # right after __init__, has nothing to push).
            fresh = self._ended or not self._live_visible_at(now)
            if fresh:
                if self._text:
                    self._history.append(self._text)
                    del self._history[:-self.HISTORY_CAP]  # no-op while under the cap
                self._text = text
            else:
                self._text = self._text + text
            self._ended = False
        elif kind == "eos":
            self._ended = True
        else:
            return  # unrecognized event kind: ignore rather than crash the UI poll loop
        self._last_event_at = now

    def _live_visible_at(self, now):
        """Whether the LIVE row itself is within idle_clear_sec of its last
        event, ignoring scroll -- the whole of what visible_at/text_at used
        to mean before history existed. Still used, unchanged, to decide
        `fresh` in apply() above and as visible_at's own o==0 case below."""
        if self._last_event_at is None or not self._text:
            return False
        return (now - self._last_event_at) < self.idle_clear_sec

    def _effective_offset(self, now):
        """The scroll offset as of `now`: the raw offset last set by
        scroll_by(), or 0 if AUTO_RETURN_SEC has elapsed since the last
        scroll_by() call with no further scrolling. Computed lazily on every
        read (this method and its callers never mutate self._offset) --
        the same injected-`now` idiom idle_clear_sec already uses, so this
        stays deterministic/testable with no background timer of its own."""
        if self._offset > 0 and self._last_scroll_at is not None:
            if now - self._last_scroll_at >= self.AUTO_RETURN_SEC:
                return 0
        return self._offset

    def offset_at(self, now):
        """Public accessor for the current effective scroll offset (0 = at
        the live edge, i.e. showing the newest history + live)."""
        return self._effective_offset(now)

    def visible_at(self, now, hold=False):
        """Whether the panel (collapsed live row, or expanded whole stack --
        that split is visible_rows()'s call, not this method's; this only
        decides SHOWN vs. HIDDEN) is currently shown. While actively
        scrolled back (effective offset > 0) OR `hold` is True (the caller
        is hovering/dragging -- see _OverlayController.tick), idle-hide is
        suppressed entirely -- stays shown regardless of idle_clear_sec, as
        long as anything has ever happened at all. Otherwise (offset == 0
        and not held), this is exactly the pre-history idle_clear_sec check
        (_live_visible_at)."""
        if hold or self._effective_offset(now) > 0:
            return self._last_event_at is not None
        return self._live_visible_at(now)

    def text_at(self, now):
        """The LIVE row's text specifically (not the history stack) -- ""
        when the whole stack is hidden. Kept for callers that only care
        about the live line; see visible_rows() for the full stack."""
        return self._text if self.visible_at(now) else ""

    def visible_rows(self, now, expanded=False, hold=False):
        """Ordered oldest -> newest list of (text, is_live) tuples. []
        when hidden -- visibility is visible_at(now, hold=expanded):
        passing expanded=True (hover/drag) suppresses idle-hide the same
        way an active scrollback does, since both mean "the user is
        looking at this right now".

        COLLAPSED (the default, and the common case whenever the cursor
        isn't over the panel and nothing is scrolled back): just
        [(live_text, True)] -- no history, however much has accumulated.

        EXPANDED -- `expanded` is True, or the effective scroll offset is
        > 0 (scroll_by() was just used, even if the cursor has since left
        the panel -- matches what's still on screen mid-auto-return) --
        exactly the pre-collapse behavior: up to HISTORY_WINDOW (2) history
        rows, the window scroll_by()'s offset selects (shifting toward
        older entries as the offset grows), THEN the live row, always
        last, always present (pinned at the bottom regardless of scroll
        offset). A fractional offset also includes the next-older history
        row so the layout can clip it in smoothly (see _compute_layout's
        scroll_frac)."""
        if not self.visible_at(now, hold=expanded or hold):
            return []
        effective_expanded = expanded or self._effective_offset(now) > 0
        if not effective_expanded:
            return [(self._text, True)]
        n = len(self._history)
        o = self._effective_offset(now)
        floor_o = int(math.floor(o))
        extra = 1 if (o - floor_o) > 1e-6 else 0
        start = max(0, n - self.HISTORY_WINDOW - extra - floor_o)
        end = n - floor_o
        window = self._history[start:end]
        return [(t, False) for t in window] + [(self._text, True)]

    def scroll_by(self, row_delta, now):
        """Move the scroll offset by `row_delta` rows: positive scrolls
        further back into history (older), negative scrolls forward toward
        the live row (newer). `row_delta` may be fractional so a trackpad
        can slide continuously. Clamped to [0, max(0, len(history) -
        HISTORY_WINDOW)] -- can't scroll past the oldest entry, or past the
        live edge. Re-bases from the EFFECTIVE offset (offset_at(now)), not
        the raw one: if AUTO_RETURN_SEC had already lazily reset the
        offset to 0 (see _effective_offset), a new scroll starts fresh from
        0 rather than jumping back to a stale pre-auto-return position --
        matching what the user would actually see (the stack visually
        already snapped back to live once auto-return's 10s elapsed).
        Always records `now` as the last-scroll time, restarting the
        auto-return countdown."""
        max_offset = max(0, len(self._history) - self.HISTORY_WINDOW)
        start = self._effective_offset(now)
        offset = max(0.0, min(float(max_offset), start + row_delta))
        self._offset = 0.0 if offset < 1e-6 else offset
        self._last_scroll_at = now


CAPTION_POSITIONS = ("free", "bottom", "notch", "dynamic-island")


class AudioLevelMeter:
    """Smoothed mic loudness; no ASR events, animation clock, or random signal."""

    def __init__(self):
        self.level = 0.0
        self.last_tick = None

    def update(self, sample, now):
        target = 0.0
        if sample is not None:
            captured_at, rms = sample
            if (math.isfinite(captured_at) and math.isfinite(rms)
                    and 0 <= now - captured_at <= .3 and rms > 0):
                # -60 dBFS silence floor; full display at -12 dBFS.
                target = max(0.0, min(1.0, (20 * math.log10(rms) + 60) / 48))
        dt = 1 / 30 if self.last_tick is None else max(0.0, now - self.last_tick)
        self.last_tick = now
        tau = .035 if target > self.level else .18
        self.level += (target - self.level) * (1 - math.exp(-dt / tau))
        if self.level < .005:
            self.level = 0.0
        return self.level


def _audio_bar_rects(level, width=20.0, height=20.0):
    """Five symmetric loudness bars (not separate frequency bands)."""
    bar_width = width / 10
    spacing = (width - 5 * bar_width) / 4
    return [(i * (bar_width + spacing), (height - h) / 2, bar_width, h)
            for i, weight in enumerate((.45, .75, 1.0, .75, .45))
            for h in [2 + (height - 2) * max(0.0, min(1.0, level)) * weight]]


class OverlayGeometry:
    """Panel placement + caption font size -- pure data/math, no AppKit.
    Draggable/scrollable at runtime (see _OverlayController's mouse-drag
    and Cmd+scroll handlers) and persisted across runs (see
    load_geometry/save_geometry below).

    (cx_frac, bottom_px) anchor the panel bottom-center: cx_frac is the
    panel's CENTER x as a FRACTION of the current screen's visibleFrame
    width (not raw points -- stays meaningful across differently sized/
    positioned screens), bottom_px is points from the visibleFrame's
    bottom edge up to the panel's bottom edge (what the old fixed
    _BOTTOM_MARGIN constant covered before dragging existed). font_size is
    the caption text size in points -- what the --captions-font-size CLI
    flag used to set unconditionally; now just that value's first-run
    default (see _OverlayController.__init__)."""

    def __init__(self, cx_frac=0.5, bottom_px=24.0, font_size=28.0, position="free"):
        self.cx_frac = cx_frac
        self.bottom_px = bottom_px
        self.font_size = font_size
        self.position = position

    @property
    def top_anchored(self):
        return self.position in ("notch", "dynamic-island")

    def set_position(self, position):
        if position not in CAPTION_POSITIONS:
            raise ValueError(f"Unknown caption position: {position}")
        # 'bottom' explicitly resets an old dragged placement; 'free' restores it.
        if position == "bottom":
            self.cx_frac, self.bottom_px = 0.5, 24.0
            position = "free"
        self.position = position

    def move_to(self, cx_frac, bottom_px, visible_h):
        """Reposition, clamping to stay on-screen: cx_frac to [0.0, 1.0]
        (panel center can't pass either screen edge); bottom_px to [0.0,
        max(0.0, visible_h - 80.0)] (bottom edge can't go below the screen
        bottom, or leave less than an 80pt headroom budget at the top -- a
        fixed budget rather than the actual current panel height, so this
        stays cheap/AppKit-free; _layout_stack applies the precise
        per-repaint top clamp against the real panel height)."""
        self.cx_frac = max(0.0, min(1.0, cx_frac))
        self.bottom_px = max(0.0, min(max(0.0, visible_h - 80.0), bottom_px))

    def scale_by(self, steps):
        """Multiply font_size by 1.1**steps (steps: int, may be negative --
        scroll-down shrinks), clamped to [12.0, 96.0]. Stays a float, never
        rounded, so repeated small steps compound smoothly."""
        self.font_size = max(12.0, min(96.0, self.font_size * (1.1 ** steps)))

    def as_dict(self):
        return {"cx_frac": self.cx_frac, "bottom_px": self.bottom_px,
                "font_size": self.font_size, "position": self.position}

    @classmethod
    def from_dict(cls, d):
        """Tolerant load: a non-dict `d`, a missing key, or a non-numeric
        value (bool included -- a stray JSON true/false is a typo, not a
        meaningful 0.0/1.0) falls back to that field's own default rather
        than raising; a numeric value is clamped to the field's valid
        range. Never raises -- see load_geometry, which must survive a
        hand-edited/corrupt settings file."""
        if not isinstance(d, dict):
            d = {}
        defaults = {"cx_frac": 0.5, "bottom_px": 24.0, "font_size": 28.0}
        ranges = {"cx_frac": (0.0, 1.0), "bottom_px": (0.0, math.inf), "font_size": (12.0, 96.0)}

        def _field(key):
            val = d.get(key, defaults[key])
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                val = defaults[key]
            lo, hi = ranges[key]
            return max(lo, min(hi, float(val)))

        geometry = cls(cx_frac=_field("cx_frac"), bottom_px=_field("bottom_px"), font_size=_field("font_size"))
        position = d.get("position", "free")
        geometry.set_position(position if position in CAPTION_POSITIONS else "free")
        return geometry


DEFAULT_SETTINGS_PATH = os.path.expanduser("~/.config/alwayswhisper/caption_overlay.json")


def load_geometry(path):
    """OverlayGeometry loaded from `path`'s JSON, or OverlayGeometry()
    defaults if the file is missing, unreadable, or not valid JSON --
    NEVER raises: a bad settings file must not block the overlay from
    starting (see _OverlayController.__init__)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return OverlayGeometry()
    return OverlayGeometry.from_dict(data)


def save_geometry(path, geometry):
    """Persist `geometry` as JSON to `path`, creating parent directories as
    needed. Persistence is a nice-to-have, never allowed to crash the
    overlay: any OSError (permission denied, disk full, missing parent
    that can't be created, ...) is swallowed after printing one warning
    line, never raised (see _OverlayController.tick's debounced save /
    _handle_mouse_up's immediate save, neither of which may propagate a
    filesystem error into the UI poll loop)."""
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(geometry.as_dict(), f)
    except OSError as exc:
        print(f"[caption-overlay] warning: could not save settings to {path}: {exc}")


# ============================================================ AppKit glue ===
# All of the below is UI plumbing: it imports AppKit lazily (inside each
# function/class, never at module scope -- see this module's docstring) and
# is exercised by `python caption_overlay.py --demo`, not by
# tests/test_caption_overlay.py.
_DEFAULT_FONT_SIZE = 28
_IDLE_CLEAR_SEC = 5.0
_POLL_INTERVAL_SEC = 0.03     # NSTimer tick driving the drain-and-repaint loop
_PANEL_BG_ALPHA = 0.85
_CORNER_RADIUS = 14.0
_PAD_X = 18.0
_PAD_Y = 10.0
_MAX_WIDTH_FRACTION = 0.92    # panel width clamp, as a fraction of the screen's visible width
# NOTE: the old fixed bottom-margin constant (24.0) now lives as
# OverlayGeometry's own bottom_px default -- see that class, and
# _panel_origin/_layout_stack, which read geometry.bottom_px instead.

# Hover icon UX (2026-08-24~): hovering a caption row shows two small icons
# at its right end -- copy, then translate (see RowBox/RowDecor/
# _compute_layout below). Clicking either one acts on that row instead of
# arming the drag gesture (see _OverlayController._handle_mouse_down) and
# shows a checkmark for COPY_FEEDBACK_SEC seconds as feedback (see tick()).
# Copy puts the row's own text on the clipboard; translate (2026-09-03~)
# puts its TRANSLATION there instead -- Japanese to English, anything else
# to Japanese, via translator.py/OpenRouter. Because that one is a network
# call it also has in-progress and failed states, which copy has no use for
# (see RowDecor.translate_state / _make_translate_icon_view).
COPY_FEEDBACK_SEC = 1.2
HOVER_COPY_DELAY_SEC = 0.5  # fallback where macOS will not deliver panel clicks
_ICON_SIZE_RATIO = 0.9     # icon_size ~= font_size * this ratio
_ICON_SIZE_MIN = 14.0
_ICON_SIZE_MAX = 40.0
# Gap BETWEEN the two icons. Deliberately half the padding that separates
# the pair from the text on one side and the chip edge on the other, so the
# two icons read as one control cluster rather than as two unrelated marks.
_ICON_GAP = _PAD_X / 2.0

_RoundedBackgroundView = None  # lazily-defined ObjC subclass, cached -- see _get_rounded_background_view_class


def _get_rounded_background_view_class():
    """Lazily define (once per process) the NSView subclass used as the
    caption panel's content view. Must be defined lazily -- subclassing an
    AppKit class needs AppKit imported first, and this module deliberately
    keeps AppKit out of its top-level scope (see the module docstring) -- and
    cached after the first call: PyObjC registers Objective-C classes
    globally by name, and re-declaring the same-named class a second time in
    one process raises an error."""
    global _RoundedBackgroundView
    if _RoundedBackgroundView is None:
        import AppKit

        class _CaptionBackgroundView(AppKit.NSView):
            """Fills itself with a rounded, translucent-black rect: plain
            NSBezierPath + NSColor in drawRect_, deliberately NOT a CALayer
            backgroundColor. CALayer.backgroundColor needs a CGColorRef, and
            this environment has only pyobjc-framework-Cocoa installed (no
            pyobjc-framework-Quartz -- and no new dependency may be added
            for this feature); without Quartz's bridge metadata, PyObjC can
            only hand back a CGColorRef as an untyped, unintrospectable
            opaque pointer (ObjCPointerWarning), not a real bridged object.
            A plain NSBezierPath fill needs no CoreGraphics types at all and
            renders the identical rounded translucent-black rect."""

            def drawRect_(self, rect):
                AppKit.NSColor.colorWithCalibratedWhite_alpha_(
                    0.0, getattr(self, "background_alpha", _PANEL_BG_ALPHA)).set()
                radius = getattr(self, "corner_radius", _CORNER_RADIUS)
                AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    self.bounds(), radius, radius
                ).fill()
                if getattr(self, "attached_top", False):
                    bounds = self.bounds()
                    AppKit.NSBezierPath.bezierPathWithRect_(AppKit.NSMakeRect(
                        0, bounds.size.height - radius, bounds.size.width, radius)).fill()

        _RoundedBackgroundView = _CaptionBackgroundView
    return _RoundedBackgroundView


_ScrollableContainerView = None  # lazily-defined ObjC subclass, cached -- see _get_scrollable_container_view_class
_CaptionPanelClass = None


def _get_caption_panel_class():
    """Return the interactive overlay panel class.

    A borderless non-activating NSPanel reports ``canBecomeKeyWindow ==
    False`` by default.  On this macOS configuration WindowServer then
    declines to dispatch an actual mouse-down to it, even after
    ``ignoresMouseEvents`` has been turned off.  Allowing key-window status
    is required for the copy control to be clickable; keeping it from
    becoming the app's main window limits this to the small transient panel.
    """
    global _CaptionPanelClass
    if _CaptionPanelClass is None:
        import AppKit

        class _CaptionPanel(AppKit.NSPanel):
            def canBecomeKeyWindow(self):
                return True

            def canBecomeMainWindow(self):
                return False

        _CaptionPanelClass = _CaptionPanel
    return _CaptionPanelClass


def _get_scrollable_container_view_class():
    """Lazily define (once per process; same rationale/caching contract as
    _get_rounded_background_view_class) the NSView subclass used as the
    panel's CONTENT view: a fully transparent container (draws nothing of
    its own -- the default NSView drawRect_ is a no-op -- only each ROW
    CHIP, a separate _CaptionBackgroundView instance per visible line,
    draws its own rounded rect; see _layout_stack) whose only other job is
    forwarding scroll-wheel and mouse-drag input to plain Python callables
    the controller assigns as the `on_scroll`/`on_mouse_down`/
    `on_mouse_drag`/`on_mouse_up` attributes after construction (ordinary
    Python attributes on a PyObjC ObjC-subclass instance -- verified
    against the installed pyobjc that this rides along fine). Mouse-down +
    drag on this view is how the panel is repositioned (see
    _OverlayController._handle_mouse_down/_drag/_up); AppKit keeps
    delivering mouseDragged_/mouseUp_ to whichever view got the
    mouseDown_, even once the cursor has left that view's bounds -- no
    extra tracking-area wiring needed.

    Two overrides below (acceptsFirstMouse_/hitTest_) exist specifically to
    fix a "drag/scroll do nothing" real-machine bug report -- root-caused
    and verified as follows (headless tests cannot touch AppKit at all, so
    this was verified with a throwaway on-machine pyobjc probe against the
    installed AppKit, the same "verified against the installed pyobjc"
    idiom this file already uses elsewhere, e.g. _make_row_label):

      - hitTest_: AppKit routes a mouse/scroll event to the DEEPEST view
        under the cursor (NSView.h declares `hitTest:` with no doc comment
        in this SDK snapshot, but its contract -- "the farthest-descendant
        view hit by the point" -- is exactly what the probe below
        reproduces). Each row chip's NSTextField label is sized to nearly
        fill its chip (see _layout_stack: label frame = chip minus
        _PAD_X/_PAD_Y), so almost every point a user would naturally click
        (i.e. on the caption text itself) hit-tests to the LABEL, not this
        container. Probe (content -> chip -> label, all real AppKit views,
        a point inside all three): `content.hitTest_(point)` returned the
        label, not the chip or content. That matters because NSControl
        (NSTextField's superclass) provides its OWN mouseDown_/scrollWheel_
        implementations -- confirmed via
        `NSControl.instanceMethodForSelector_("scrollWheel:")` /
        `("mouseDown:")` both being DIFFERENT pointers than NSView's own --
        i.e. a click/scroll landing on the label never reaches this class's
        overrides below via any responder-chain forwarding; NSControl's own
        cell-tracking logic runs instead and swallows it (this non-editable,
        non-selectable label has nothing to track/select, so it does
        nothing -- but "nothing" also means never telling the container).
        Fix: override hitTest_ so ANY point inside this window resolves to
        THIS view -- no subview of this overlay legitimately needs its own
        mouse events (every label is display-only: non-editable,
        non-selectable). Still returns None for points outside this view
        entirely (super's own out-of-bounds contract, preserved via the
        `is not None` check), so this never widens what the WINDOW itself
        considers a hit.
      - acceptsFirstMouse_: belt-and-suspenders for the same bug, covering
        the window-activation angle hitTest_ alone doesn't. Probe: a real
        NSPanel built with this file's exact style mask
        (NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel)
        reports canBecomeKeyWindow() == False, and stays isKeyWindow() ==
        False even after orderFrontRegardless() (what _layout_stack
        actually calls) AND after an explicit makeKeyWindow() probe call --
        i.e. structurally, permanently non-key (Apple's own official sample
        code comments -- developer.apple.com/library/archive/samplecode/
        RoundTransparentWindow's CustomWindow.m and GLEssentials'
        GLEssentialsFullscreenWindow.m -- both say plainly: "by default,
        borderless windows cannot be made key and input cannot go to
        them"; the generic "Panels can become the key window" claim in
        Apple's own Cocoa "How Panels Work" guide evidently assumes a
        title/utility/HUD style bit this panel doesn't have). A view's
        acceptsFirstMouse_ defaults to NO (probe: plain
        NSView().acceptsFirstMouse_(None) == False) -- the documented
        purpose of that method is precisely to let a view opt IN to
        receiving a real mouseDown_ even though the click's window isn't
        (and, here, structurally can't become) key, without that click
        being swallowed as pure window-activation. Returning True here
        does NOT make the panel key or steal keyboard focus -- keying and
        first-mouse acceptance are separate mechanisms (confirmed by the
        probe above: the panel stays non-key throughout) -- it only makes
        sure a click is delivered as a real mouseDown_ instead of being
        eaten. debug_input (see below) exists partly to let a real gesture
        prove this end to end, since none of the above can be exercised
        headlessly.

    debug_input (instance attribute, default False, set by
    _OverlayController.__init__ via the `debug_input=` constructor arg /
    main()'s --debug-input flag) makes every mouseDown_/mouseDragged_/
    mouseUp_/scrollWheel_ actually RECEIVED by this view print one stdout
    line (see _log_debug_input) -- logged here, at the point of receipt,
    rather than only inside the controller's on_* callbacks, so a
    --debug-input run can distinguish "AppKit never delivered the event to
    us at all" from "we got it but a callback/geometry bug ate it"."""
    global _ScrollableContainerView
    if _ScrollableContainerView is None:
        import AppKit
        import objc

        class _CaptionStackContainerView(AppKit.NSView):
            def drawRect_(self, rect):
                # Minimum nonzero 8-bit alpha: visually transparent, while
                # keeping padding and row gaps eligible for native mouse input.
                AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.0, 1.0 / 255.0).set()
                AppKit.NSBezierPath.bezierPathWithRect_(self.bounds()).fill()

            def acceptsFirstMouse_(self, event):
                return True

            def hitTest_(self, point):
                hit = objc.super(_CaptionStackContainerView, self).hitTest_(point)
                return self if hit is not None else None

            def scrollWheel_(self, event):
                if getattr(self, "debug_input", False):
                    _log_debug_input("scrollWheel_")
                cb = getattr(self, "on_scroll", None)
                if cb is not None:
                    cb(event)

            def mouseDown_(self, event):
                if getattr(self, "debug_input", False):
                    _log_debug_input("mouseDown_", event)
                cb = getattr(self, "on_mouse_down", None)
                if cb is not None:
                    cb(event)

            def mouseDragged_(self, event):
                if getattr(self, "debug_input", False):
                    _log_debug_input("mouseDragged_")
                cb = getattr(self, "on_mouse_drag", None)
                if cb is not None:
                    cb(event)

            def mouseUp_(self, event):
                if getattr(self, "debug_input", False):
                    _log_debug_input("mouseUp_")
                cb = getattr(self, "on_mouse_up", None)
                if cb is not None:
                    cb(event)

        _ScrollableContainerView = _CaptionStackContainerView
    return _ScrollableContainerView


def _log_debug_input(label, event=None):
    """--debug-input instrumentation: print one stdout line for `label`
    (an event/transition name -- e.g. "scrollWheel_", "ignoresMouseEvents ->
    False") plus the cursor's current SCREEN position, read fresh via
    NSEvent.mouseLocation() (screen coords, not the event's own
    window-relative locationInWindow -- matches what _update_hover/
    _handle_mouse_down/_handle_mouse_drag already read cursor position
    with, so a logged coordinate is directly comparable to a panel frame
    logged the same way). Only ever called when a debug_input flag is True
    (see _CaptionStackContainerView's scrollWheel_/mouseDown_/
    mouseDragged_/mouseUp_, and _OverlayController._update_hover/
    _handle_mouse_drag/_handle_scroll_wheel) -- the non-demo runtime path
    stays log-silent by default, per this feature's own design goal (see
    main()'s --debug-input help text).

    `event=None` (the default) logs only the cursor position, as before.
    When `event` is given (currently only the container's mouseDown_ --
    see _get_scrollable_container_view_class), also logs the event's OWN
    `locationInWindow()` (window-local, NOT converted to screen -- this is
    deliberately the raw AppKit value, so it can be cross-checked by eye
    against a screen-converted `click=` value logged elsewhere, e.g.
    _OverlayController._handle_mouse_down's own "mouseDown:" debug line) --
    confirms whether the event even reached this container at all, and
    with what location AppKit itself recorded for it, independent of
    anything this module's own callbacks compute afterward."""
    import AppKit

    mouse = AppKit.NSEvent.mouseLocation()
    suffix = ""
    if event is not None:
        loc = event.locationInWindow()
        suffix = f" locationInWindow=({loc.x:.1f}, {loc.y:.1f})"
    print(f"[caption-overlay] --debug-input: {label} at screen=({mouse.x:.1f}, {mouse.y:.1f}){suffix}")


def _make_panel(font_size):
    """Build (panel, content): a borderless, always-on-top NSPanel (click-
    through by DEFAULT -- _OverlayController toggles this live on hover,
    see _update_hover -- and only while NOT click-through does it receive
    the scroll/mouse-drag input that expands/moves it) whose content view
    is a transparent, scrollable, draggable container. Row chips (one per
    visible caption line -- history rows and the live row) are added to
    `content` as children on every repaint (see _layout_stack), not built
    here: how many there are, and their sizes, depend on the current text
    and scroll offset. Must be called on the main thread."""
    import AppKit

    style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
    panel = _get_caption_panel_class().alloc().initWithContentRect_styleMask_backing_defer_(
        AppKit.NSMakeRect(0, 0, 10, 10), style, AppKit.NSBackingStoreBuffered, False
    )
    # NSStatusWindowLevel (25) used to be used here, but that sits BELOW
    # another app's native fullscreen Space content -- a real-machine report
    # confirmed this overlay does not appear above a fullscreened Zoom/Meet/
    # browser/Keynote even with CanJoinAllSpaces|FullScreenAuxiliary below.
    # NSScreenSaverWindowLevel is the customary level for a truly
    # above-everything overlay -- CGWindowLevel.h's own doc comment defines
    # the ordering model ("Windows with a higher level are sorted in front
    # of windows with a lower level"; kCGStatusWindowLevel=25 vs.
    # kCGScreenSaverWindowLevel=1000, both confirmed on this machine via
    # AppKit.NSStatusWindowLevel/NSScreenSaverWindowLevel), and
    # rcaelers/workrave's MacOSOverlayWindow.mm (a mature, real-world
    # break-reminder app whose entire purpose is overlaying above other
    # apps' fullscreen Spaces) independently uses this exact same level for
    # this exact same problem. Apple's own Live Captions behaves the same
    # way. Tradeoff, deliberately accepted: at this level the panel also
    # floats above the menu bar/Dock during fullscreen -- intended, since a
    # caption that can't be seen during a fullscreen call is useless.
    panel.setLevel_(AppKit.NSScreenSaverWindowLevel)
    panel.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        | AppKit.NSWindowCollectionBehaviorStationary
    )
    panel.setIgnoresMouseEvents_(False)  # entire visible panel is draggable
    panel.setOpaque_(False)
    panel.setBackgroundColor_(AppKit.NSColor.clearColor())
    panel.setHasShadow_(False)
    # NSPanel (unlike plain NSWindow) defaults hidesOnDeactivate to YES, which
    # would hide this overlay the instant the user clicks into Zoom/Meet --
    # i.e. exactly when it needs to stay visible. Must be turned off.
    panel.setHidesOnDeactivate_(False)

    content = _get_scrollable_container_view_class().alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 10, 10))
    # The controller assigns real callbacks (and, for --debug-input, flips
    # debug_input to True) after construction.
    content.on_scroll = None
    content.on_mouse_down = None
    content.on_mouse_drag = None
    content.on_mouse_up = None
    content.debug_input = False
    content.setWantsLayer_(True)
    content.setClipsToBounds_(True)
    panel.setContentView_(content)

    return panel, content


def _make_row_label(font_size):
    """Build one row's NSTextField: non-editable, centered, white,
    word-wrapped (never truncated -- long captions must show their WHOLE
    text, see CaptionLineModel). NSTextField's cell actually already
    defaults to exactly this wrap configuration (verified against the
    installed pyobjc: a fresh cell's wraps()/usesSingleLineMode()/
    lineBreakMode() are True/False/NSLineBreakByWordWrapping already) --
    set explicitly anyway so this isn't relying on an unstated default."""
    import AppKit

    label = AppKit.NSTextField.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 10, 10))
    label.setEditable_(False)
    label.setSelectable_(False)
    label.setBordered_(False)
    label.setBezeled_(False)
    label.setDrawsBackground_(False)
    label.setTextColor_(AppKit.NSColor.whiteColor())
    label.setFont_(AppKit.NSFont.systemFontOfSize_weight_(font_size, AppKit.NSFontWeightMedium))
    label.setAlignment_(AppKit.NSTextAlignmentCenter)
    label.cell().setUsesSingleLineMode_(False)
    label.cell().setLineBreakMode_(AppKit.NSLineBreakByWordWrapping)
    label.cell().setWraps_(True)
    return label


_UNBOUNDED_HEIGHT = 100000.0  # "tall enough that wrapped text never runs out of room" for height measurement


def _measure_row_size(label, text, max_text_w):
    """(text_w, text_h), ceil()'d: hug `text` if it fits on one line within
    max_text_w, else use the full max_text_w and wrap. The single-panel
    design's exact sizing rule (see the pre-history _layout_and_show, now
    folded into _layout_stack per row), including the fix for the "which"
    -> "…hich" / "Hello." -> "…ello." truncation reports: BOTH branches
    measure through the label's OWN cell (cell.cellSize() /
    cellSizeForBounds_()), never a bare NSAttributedString, which
    empirically under-reports what NSTextFieldCell actually needs to draw
    by ~12-13px width / ~8px height at font size 28 (its own internal
    text-container insets that a bare attributed-string measurement
    doesn't include) -- verified against the installed pyobjc, consistent
    across English/Japanese/punctuation samples. `label` must already have
    its font set; this sets label's stringValue as part of measuring (the
    caller does not need to call setStringValue_ separately)."""
    label.setStringValue_(text)
    cell = label.cell()
    natural = cell.cellSize()  # unconstrained -- still single-line-sized even with wraps=True (verified)
    if natural.width <= max_text_w:
        return math.ceil(natural.width), math.ceil(natural.height)
    import AppKit
    wrapped = cell.cellSizeForBounds_(AppKit.NSMakeRect(0, 0, max_text_w, _UNBOUNDED_HEIGHT))
    return math.ceil(max_text_w), math.ceil(wrapped.height)


_ROW_GAP = 8.0        # vertical gap between stacked row chips
_LIVE_ALPHA = 1.0
# Per-row alphaValue by position within the CURRENTLY SHOWN window, newest
# history row first (i.e. index 0 = the history row immediately above the
# live row = "1 back", index 1 = "2 back", index 2 = the extra older row
# clipped in during a fractional scroll). Deliberately positional, not
# based on absolute distance back through all of history: even scrolled
# deep into history, the newest-shown row of the current window always
# gets the least-faded alpha.
_HISTORY_ALPHAS_NEWEST_FIRST = (0.75, 0.55, 0.35)


def _alphas_for(rows):
    """(alpha, ...) parallel to `rows` (oldest -> newest, live last -- see
    CaptionLineModel.visible_rows): _LIVE_ALPHA for the live row, and
    _HISTORY_ALPHAS_NEWEST_FIRST assigned to however many history rows are
    actually present (0-3), aligned to the NEWEST (live-adjacent) end --
    e.g. with only 2 history rows shown, they get (0.55, 0.75), not
    (0.75, 0.55): the OLDER of the two is "2 back" (0.55), not "1 back".
    Pure function, no AppKit -- unit-tested directly."""
    if not rows:
        return ()
    k = len(rows) - 1  # history rows shown = everything but the live row
    return tuple(reversed(_HISTORY_ALPHAS_NEWEST_FIRST[:k])) + (_LIVE_ALPHA,)


def _panel_origin(visible, panel_w, panel_h, geometry):
    """(x, y) panel origin (AppKit frame convention: bottom-left corner)
    for a panel_w x panel_h panel within `visible` (an
    NSScreen.visibleFrame()-shaped object -- only .origin.x/.origin.y/
    .size.width/.size.height are read, so a duck-typed stand-in works too),
    anchored per `geometry`: x from geometry.cx_frac (panel CENTER as a
    fraction of the visible width), y from geometry.bottom_px (points
    above the visible bottom edge) -- both clamped to keep the panel fully
    on-screen. Shared by _layout_stack (full repaint) and
    _OverlayController._handle_mouse_drag (cursor-rate reposition of the
    panel's CURRENT frame size, between _layout_stack's own <=0.08s-latency
    repaints -- see that method's docstring) so the two never drift apart."""
    center_x = visible.origin.x + (0.5 if geometry.top_anchored else geometry.cx_frac) * visible.size.width
    x = center_x - panel_w / 2.0
    x = max(visible.origin.x, min(visible.origin.x + visible.size.width - panel_w, x))

    # Bottom edge at geometry.bottom_px above the visible bottom; AppKit's
    # coordinate system has y=0 at the bottom, and a frame's y is its
    # BOTTOM edge with height extending upward from it. Clamped on BOTH
    # sides now, same as x above -- and in this specific order, upper bound
    # first, lower bound second, so the lower bound always wins when the
    # two disagree. The upper-bound clamp is the original one (a stale/
    # cross-screen bottom_px -- e.g. loaded from a taller screen's settings
    # -- must never push the panel's top past the visible top). The
    # lower-bound clamp exists because panel_h alone can already exceed
    # visible.size.height: hovering expands the stack from 1 row (live
    # only) up to 3 (2 history + live -- see _HISTORY_ALPHAS_NEWEST_FIRST),
    # and on a short enough screen that makes panel_h > visible height even
    # with geometry.bottom_px == 0, which used to push this origin BELOW
    # visible.origin.y with no floor at all. _compute_layout always places
    # the live/newest row at panel-local y=0 -- the BOTTOM of the stack
    # (see that function's own "Bottom-up stacking" docstring section) --
    # so once the lower-bound clamp wins, the live/newest caption (what the
    # viewer actually needs to read right now) stays fully on screen, and
    # any overflow is pushed off the TOP instead, clipping only the oldest
    # history rows. geometry.bottom_px is already >= 0 by construction (see
    # OverlayGeometry.move_to/from_dict), so this lower bound only ever
    # engages because of panel_h, never because of a negative bottom_px.
    if geometry.top_anchored:
        return x, visible.origin.y + visible.size.height - panel_h
    y = visible.origin.y + geometry.bottom_px
    y = min(y, visible.origin.y + visible.size.height - panel_h)
    y = max(y, visible.origin.y)
    return x, y


def _drag_reposition_origin(visible, frame_w, frame_h, font_size, geometry):
    """(x, y) to move the panel's frame origin to, LIVE, during a drag (see
    _OverlayController._handle_mouse_drag, the only caller -- it owns the
    real AppKit frame/screen reads and only hands this pure function
    already-extracted numbers). `frame_w`/`frame_h` are the panel's
    CURRENT, already-icon-reserve-inclusive frame size (panel.frame()'s
    own size -- see _compute_layout's docstring: panel_rect's width is
    ALWAYS text_panel_w + icon_reserve, regardless of hover state, so
    frame_w always includes that strip).

    Fixes a real regression: naively calling _panel_origin(visible,
    frame_w, frame_h, geometry) -- i.e. centering on frame_w directly, as
    this used to do -- disagrees with _compute_layout's OWN centering
    (which centers each repaint on text_panel_w, the widest TEXT-only
    chip, NEVER the icon-widened panel_w -- see _compute_layout's
    "Horizontal centering" docstring section) by ~icon_reserve/2. Since a
    drag is hovering==False -> _hover_row is always None during a drag
    (see tick()'s "Hover-row hit-testing" comment) but the panel's frame
    width already reserves the icon strip regardless either way, that
    mismatch shifted the LIVE drag reposition ~icon_reserve/2 to the
    RIGHT of where the very next full _layout_stack repaint (correctly
    centered on text_panel_w) puts it -- a visible sideways jump right
    after every drag. Subtracting the same icon reserve _compute_layout
    itself subtracts (see _icon_reserve_for) before centering keeps this
    function and _compute_layout in permanent agreement -- proven
    directly in tests/test_caption_overlay.py by feeding this function
    the exact panel_w _compute_layout returned for the same inputs and
    asserting the two x's match.

    Pure (no AppKit), unlike _handle_mouse_drag itself, which reads
    NSScreen/NSEvent/NSWindow directly and so cannot run headlessly."""
    text_w = frame_w - _icon_reserve_for(font_size)[1]
    return _panel_origin(visible, text_w, frame_h, geometry)


def _screen_for_panel(panel, fallback_screen):
    """Return the display that currently owns ``panel``.

    ``NSScreen.mainScreen()`` is not a stable choice for a floating panel on
    a multi-display desktop: macOS may change the main screen while another
    app is activated.  Repainting against that new screen would teleport the
    overlay (and its icon hit boxes) away from the cursor between hover and
    click.  NSWindow.screen() stays tied to the display containing the panel,
    so prefer it; ``fallback_screen`` covers a just-created/off-screen panel.

    Kept free of an AppKit import so this small selection rule can be tested
    headlessly.
    """
    if panel is not None:
        screen = panel.screen()
        if screen is not None:
            return screen
    return fallback_screen


def _safe_top_inset(screen):
    method = getattr(screen, "safeAreaInsets", None)
    return max(0.0, float(method().top)) if method is not None else 0.0


def _caption_screen(panel, fallback_screen, screens, geometry):
    """Dock to the notched display even when an external monitor is primary."""
    if geometry.top_anchored:
        for screen in screens:
            if _safe_top_inset(screen) > 0:
                return screen
    return _screen_for_panel(panel, fallback_screen)


def _caption_visible_frame(screen, geometry):
    visible = screen.visibleFrame()
    inset = _safe_top_inset(screen) if geometry.top_anchored else 0.0
    if not inset:
        return visible
    frame = screen.frame()
    # Use the camera housing's lower edge even with an auto-hidden menu bar.
    # https://developer.apple.com/documentation/appkit/nsscreen/safeareainsets
    top = frame.origin.y + frame.size.height - inset
    return SimpleNamespace(
        origin=SimpleNamespace(x=frame.origin.x, y=visible.origin.y),
        size=SimpleNamespace(width=frame.size.width, height=max(0.0, top - visible.origin.y)))


@dataclass(frozen=True)
class IslandAnchor:
    center_x: float
    top: float
    camera_width: float
    camera_height: float
    screen_width: float
    scale: float = 1.0

    @property
    def attached(self):
        return self.camera_height > 0

    @property
    def header_height(self):
        return max(32.0, self.camera_height)

    @property
    def caption_padding(self):
        # Layout uses points; keep the requested padding at two display pixels.
        return 2.0 / self.scale


def _island_anchor(screen):
    """Use the *housing*, including its menu-bar area, as the island's anchor.

    Apple's auxiliaryTopLeftArea/RightArea delimit the actual camera width;
    visibleFrame alone cannot describe this area (it excludes the whole strip).
    """
    frame = screen.frame()
    height = _safe_top_inset(screen)
    center = frame.origin.x + frame.size.width / 2
    width = 0.0
    if height:
        left = getattr(screen, "auxiliaryTopLeftArea", lambda: None)()
        right = getattr(screen, "auxiliaryTopRightArea", lambda: None)()
        if left is not None and right is not None and left.size.width and right.size.width:
            x0, x1 = left.origin.x + left.size.width, right.origin.x
            width = max(0.0, x1 - x0)
            center = (x0 + x1) / 2
        else:
            width = 220.0  # Older bindings: conservatively reserve the camera strip.
        top = frame.origin.y + frame.size.height
    else:
        visible = screen.visibleFrame()
        top = visible.origin.y + visible.size.height - 8.0
    scale = float(getattr(screen, "backingScaleFactor", lambda: 1.0)())
    return IslandAnchor(center, top, width, height, frame.size.width, max(1.0, scale))


def _island_layout(row_sizes, font_size, hover_row, anchor, scroll_frac=0.0):
    """One continuous surface from screen top, with text below the housing.

    No rows means the compact island: only safe left/right status wings.
    Caption text and history expand the SAME surface, not a second window.
    """
    if not row_sizes:
        width = max(160.0, anchor.camera_width + 96.0)
        height = anchor.header_height + (6.0 if anchor.attached else 0.0)
        return (anchor.center_x - width / 2, anchor.top - height, width, height), [], []
    reserve = _icon_reserve_for(font_size)[1]
    min_width = max(400.0, anchor.camera_width + 160.0)
    # Reserve action space on BOTH sides, even when icons are hidden. The
    # caption then stays centered in the island with the same wrap width.
    padded = [(text, max(w, min_width - 2 * reserve - 2 * _PAD_X), h)
              for text, w, h in row_sizes]
    visible = SimpleNamespace(
        origin=SimpleNamespace(x=anchor.center_x - anchor.screen_width / 2, y=0),
        size=SimpleNamespace(width=anchor.screen_width,
                             height=anchor.top - anchor.header_height))
    rect, chips, boxes = _compute_layout(padded, font_size, hover_row, visible,
                                        OverlayGeometry(position="notch"), pad_y=anchor.caption_padding,
                                        scroll_frac=scroll_frac)
    x, y, w, h = rect
    # Whole-point, even width prevents NSWindow's frame rounding from starting
    # another tiny resize (and nudging text) when only the hover state changes.
    full_width = 2 * math.ceil((w + reserve) / 2)
    added_width = full_width - w
    x, w = round(anchor.center_x - full_width / 2), full_width
    icon_screen_dx = x - rect[0] + added_width

    def shifted(rect, dx):
        return (rect[0] + dx, *rect[1:]) if rect else None

    bottom_extra = 2.0 / anchor.scale
    chips = [(0.0, cy + bottom_extra, w, ch, shifted(icon, added_width), shifted(translation, added_width))
             for cx, cy, cw, ch, icon, translation in chips]
    boxes = [RowBox(box.index, box.text, (x, box.frame[1], w, box.frame[3]),
                    shifted(box.icon, icon_screen_dx), shifted(box.translate_icon, icon_screen_dx))
             for box in boxes]
    return (x, y - bottom_extra, w, h + anchor.header_height + bottom_extra), chips, boxes


def _island_tween(start, target, progress):
    """Monotone ease-out; a shared top/center remains fixed during the morph."""
    p = max(0.0, min(1.0, progress))
    t = 1 - (1 - p) ** 3
    return tuple(a + (b - a) * t for a, b in zip(start, target))


def _island_should_morph(was_visible, was_compact, now_compact, reduce_motion,
                         same_anchor, start, target):
    """Whether _IslandSurface.show should run the 0.24s compact<->expanded
    size morph. Scroll/text updates while already expanded must NOT morph
    -- that resize animation is what made history scrolling feel jumpy.
    First show, reduce-motion, a display change, or an already-matching
    frame also skip the morph (draw the target immediately)."""
    if reduce_motion or not was_visible or not same_anchor or start == target:
        return False
    if not was_compact and not now_compact:
        return False
    return True


@dataclass(frozen=True)
class RowBox:
    """One row's hit-testable geometry, in SCREEN coordinates (the same
    bottom-left-origin space as AppKit.NSEvent.mouseLocation()) -- built
    fresh by _compute_layout on every repaint, oldest -> newest (matching
    CaptionLineModel.visible_rows()'s own order), regardless of the
    bottom-up visual stacking order chips are actually placed in (see
    _compute_layout). `frame` covers the WHOLE row chip, including the
    icons' right extension when present -- what _hit_row matches. `icon`
    is the copy-icon's own smaller hit rect and `translate_icon` the
    translate-icon's, or None (both of them, always together) when this row
    has no icons at all (only the currently hovered row does -- see
    RowDecor / _compute_layout's `hover_row` argument) -- what _hit_icon and
    _hit_translate_icon respectively match. The two never overlap: together
    they tile the whole reserved strip, so no click inside it is dead."""
    index: int
    text: str
    frame: tuple
    icon: tuple = None
    translate_icon: tuple = None


@dataclass(frozen=True)
class RowDecor:
    """Per-repaint hover/copy decoration, passed to renderer.show() as an
    extra `decor=` keyword (see _AppKitRenderer.show/_layout_stack) and
    folded into tick()'s repaint-skip state compare (see
    _OverlayController.tick), so a change in either field alone -- with no
    other change to the caption text/history/geometry -- still triggers a
    repaint. `hover_row` is an index into the current `rows` (oldest ->
    newest) for the row that should show a plain copy icon, or None for no
    icon anywhere. `copied_text` is the text of the row whose icon should
    render as a checkmark instead (~COPY_FEEDBACK_SEC seconds after a
    successful copy -- see _OverlayController._handle_mouse_down/tick), or
    None. `translate_text` + `translate_state` are the translate icon's
    equivalent, but with three states rather than one because translating
    is a network round trip: "pending" (in flight), "done" (the translation
    is on the clipboard) or "failed" -- see _make_translate_icon_view and
    _OverlayController._start_translate/_drain_translations. Both are None
    when the translate icon should render in its plain idle shape."""
    hover_row: int = None
    copied_text: str = None
    translate_text: str = None
    translate_state: str = None


def _point_in_rect(point, rect):
    """Half-open rect containment, matching AppKit.NSPointInRect's own
    contract (point.x in [rect.x, rect.x + rect.w), point.y in [rect.y,
    rect.y + rect.h)) -- pure, so _hit_row/_hit_icon stay AppKit-free and
    directly unit-testable against plain tuples. `point` is (x, y); `rect`
    is (x, y, w, h)."""
    px, py = point
    rx, ry, rw, rh = rect
    return rx <= px < rx + rw and ry <= py < ry + rh


def _hit_row(point, boxes):
    """The first RowBox in `boxes` whose .frame contains `point` (screen
    coords), or None if none does (including an empty `boxes`) -- used by
    _OverlayController.tick() to compute the currently-hovered row."""
    for box in boxes:
        if _point_in_rect(point, box.frame):
            return box
    return None


def _hit_icon(point, boxes):
    """Like _hit_row, but matches only a box's smaller .icon rect (rows
    with icon=None can never match) -- used by
    _OverlayController._handle_mouse_down to tell an icon click (copy)
    apart from a click anywhere else on the panel (drag)."""
    for box in boxes:
        if box.icon is not None and _point_in_rect(point, box.icon):
            return box
    return None


def _hit_translate_icon(point, boxes):
    """_hit_icon's counterpart for the translate icon (.translate_icon).
    The two rects never overlap (see _compute_layout), so exactly one of
    these two functions can ever match a given point -- which is what keeps
    a click meant to copy from spending money on a translation."""
    for box in boxes:
        if box.translate_icon is not None and _point_in_rect(point, box.translate_icon):
            return box
    return None


def _icon_reserve_for(font_size):
    """(icon_size, icon_reserve) for `font_size` -- icon_size scales with
    the caption font (bigger captions, bigger icons) but is clamped to
    [_ICON_SIZE_MIN, _ICON_SIZE_MAX] so it stays sane at either extreme of
    OverlayGeometry.scale_by's own [12, 96] font_size range. Both icons are
    the same size. icon_reserve (2 * icon_size + _ICON_GAP + _PAD_X) is how
    much wider than the plain TEXT chip width the hovered chip -- and the
    panel's own always-reserved strip, see _compute_layout -- becomes to fit
    BOTH, laid out as a BALANCED text / _PAD_X / copy / _ICON_GAP /
    translate / _PAD_X / chip-edge: the first _PAD_X is already the text
    chip's own existing right padding (nothing new), so only the second
    _PAD_X, the inter-icon gap, and the two icons are actually added here --
    reusing _PAD_X itself (rather than a separately-tuned outer gap
    constant) keeps the cluster's spacing visually consistent with the
    text's own padding on every other side of the chip."""
    icon_size = max(_ICON_SIZE_MIN, min(_ICON_SIZE_MAX, font_size * _ICON_SIZE_RATIO))
    return icon_size, 2 * icon_size + _ICON_GAP + _PAD_X


def _compute_layout(row_sizes, font_size, hover_row, visible_frame, geometry, *,
                    pad_y=_PAD_Y, scroll_frac=0.0, history_window=None):
    """Pure layout math for the caption stack (no AppKit) -- split out of
    _layout_stack so the hover copy-icon geometry has direct unit test
    coverage without a real WindowServer. `row_sizes` is [(text, text_w,
    text_h), ...] oldest -> newest (matching CaptionLineModel.
    visible_rows()), i.e. exactly what _layout_stack already measures per
    row via _measure_row_size before calling this. `hover_row` is an index
    into `row_sizes` (0 = oldest) for the row that should get a copy icon,
    or None for no icon at all. `visible_frame` is an
    NSScreen.visibleFrame()-shaped object, same duck-typed contract as
    _panel_origin's own `visible` argument.

    Returns (panel_rect, chip_layouts, row_boxes):
      panel_rect = (x, y, w, h), SCREEN coords, AppKit bottom-left-origin
        frame convention (see _panel_origin) -- w is ALWAYS
        text_panel_w + icon_reserve (see _icon_reserve_for), even with
        hover_row=None: the frame itself never resizes the instant
        hovering starts/stops (the extra strip is simply invisible against
        the panel's own transparent background until an icon is actually
        drawn in it -- see _layout_stack).
      chip_layouts = [(chip_x, chip_y, chip_w, chip_h, icon_view_rect,
        translate_view_rect), ...], oldest -> newest (matching row_sizes).
        chip_x/chip_y/chip_w/chip_h are PANEL-LOCAL (content-view)
        coordinates -- what _layout_stack uses to place each row's own
        background chip view. chip_w is the chip's actual RENDERED width:
        widened by icon_reserve for row `hover_row` ONLY, plain text width
        for every other row (unlike RowBox.frame below, which reserves the
        icon space for EVERY row, hovered or not). icon_view_rect and
        translate_view_rect, both non-None only for row `hover_row`, are
        the two tight icon_size-square VIEW rects, CHIP-LOCAL (relative to
        that chip's own origin) -- ready to hand straight to the icon
        subviews' setFrame_ (see _make_copy_icon_view /
        _make_translate_icon_view). These are DIFFERENT, smaller rects than
        RowBox.icon/RowBox.translate_icon below.
      row_boxes = [RowBox(index, text, frame, icon), ...], oldest ->
        newest, SCREEN coordinates -- what the controller hit-tests mouse
        clicks/hover against (see _hit_row/_hit_icon):
          frame is what _hit_row matches (hover-ROW detection): its width
            is ALWAYS text_w + icon_reserve, for EVERY row, hovered or
            not -- so the cursor entering the band where the icon WOULD
            appear (inside the panel's own always-reserved strip, see
            panel_rect above) immediately makes that row the hovered one
            and reveals the icon there, with no flicker right at the
            extension's boundary, even though the RENDERED chip
            (chip_layouts above) only actually widens once a row becomes
            hover_row. Row frames never overlap in y (the bottom-up
            stacking always leaves at least _ROW_GAP between rows), so
            _hit_row's first match is never ambiguous despite every row's
            frame reserving the same width.
          icon and translate_icon, both non-None only for row `hover_row`,
            are what _hit_icon and _hit_translate_icon match: deliberately
            GENEROUS click targets -- full chip height, together spanning
            from midway into the text's own right padding through the
            (widened) chip's right edge, split between them at the
            MIDPOINT of the gap separating the two icons. Each is larger
            than the small icon_size-square view rect chip_layouts carries,
            since a tight icon_size square is a fiddly target on an
            80ms-polling overlay; and because they tile that strip with no
            gap and no overlap, every click inside it does exactly one of
            the two things and none does nothing.

    Horizontal centering -- both each chip WITHIN the panel, and the panel
    itself on screen via _panel_origin -- is computed from text_panel_w
    (the widest TEXT-only chip), never the icon-widened panel_w: a chip
    growing an icon extends purely to its own right, its text never
    shifts, and cx_frac keeps meaning exactly what it always has (the TEXT
    stack's own center), regardless of which row (if any) is hovered.

    Bottom-up stacking (unchanged from the pre-icon _layout_stack): the
    LAST row_sizes entry (newest/live) sits at content-local y=0, each
    earlier (older) row stacked above it -- but RowBox.index still matches
    row_sizes' own oldest-first order, not the render/stacking order."""
    icon_size, icon_reserve = _icon_reserve_for(font_size)

    if not row_sizes:
        x, y = _panel_origin(visible_frame, 0.0, 0.0, geometry)
        return (x, y, 0.0, 0.0), [], []

    text_sizes = [(w + 2 * _PAD_X, h + 2 * pad_y) for _text, w, h in row_sizes]
    if geometry.top_anchored:
        # One centered black surface, including a stable reserve for actions.
        width = max(280.0, max(w for w, _h in text_sizes))
        text_sizes = [(width, h) for _w, h in text_sizes]
    text_panel_w = max(w for w, _h in text_sizes)
    n = len(row_sizes)
    if history_window is None:
        history_window = CaptionLineModel.HISTORY_WINDOW
    n_extra = max(0, n - 1 - history_window)
    frac = max(0.0, min(1.0, float(scroll_frac)))
    extra_block_h = 0.0
    shift = 0.0
    if n_extra > 0:
        extra_block_h = (sum(text_sizes[i][1] for i in range(n_extra))
                         + _ROW_GAP * n_extra)
        shift = frac * (text_sizes[n_extra - 1][1] + _ROW_GAP)
    panel_h = sum(h for _w, h in text_sizes) + _ROW_GAP * (len(text_sizes) - 1)
    panel_h -= extra_block_h
    panel_w = text_panel_w + icon_reserve   # ALWAYS reserved -- see docstring above

    ox, oy = _panel_origin(visible_frame,
                           panel_w if geometry.top_anchored else text_panel_w,
                           panel_h, geometry)

    y_positions = [0.0] * n
    y_cursor = 0.0
    for i in range(n - 1, -1, -1):
        y_positions[i] = y_cursor
        y_cursor += text_sizes[i][1] + _ROW_GAP
    if shift:
        for i in range(n - 1):  # live stays pinned; history slides toward it
            y_positions[i] -= shift
    if geometry.top_anchored:
        y_positions = [panel_h - y - text_sizes[i][1] for i, y in enumerate(y_positions)]

    chip_layouts = []
    row_boxes = []
    for i, (text, _tw, _th) in enumerate(row_sizes):
        text_w, chip_h = text_sizes[i]
        chip_x = (text_panel_w - text_w) / 2.0
        chip_y = y_positions[i]

        # RowBox.frame ALWAYS reserves the icon extension, independent of
        # hover_row -- see docstring above.
        hit_frame_w = text_w + icon_reserve

        if i == hover_row:
            # Icon VIEWS: two tight icon_size squares, the first starting
            # right where the text chip's own existing right padding ends
            # (text_w already bakes in 2*_PAD_X), the second _ICON_GAP
            # later -- balanced text / _PAD_X / copy / _ICON_GAP /
            # translate / _PAD_X / chip-edge (see _icon_reserve_for).
            icon_y = (chip_h - icon_size) / 2.0
            icon_view_local = (text_w, icon_y, icon_size, icon_size)
            translate_view_local = (text_w + icon_size + _ICON_GAP, icon_y, icon_size, icon_size)
            render_w = text_w + icon_reserve
            # Icon HIT rects: deliberately more generous than the views --
            # full chip height, together tiling the whole strip from half a
            # padding INTO the text's own right padding through the widened
            # chip's right edge, split at the MIDPOINT of the gap between
            # the two icons so neither can swallow the other's clicks and
            # no point in the strip belongs to neither.
            icon_hit_x = text_w - _PAD_X / 2.0
            split_x = text_w + icon_size + _ICON_GAP / 2.0
            icon_hit_screen = (ox + chip_x + icon_hit_x, oy + chip_y, split_x - icon_hit_x, chip_h)
            translate_hit_screen = (ox + chip_x + split_x, oy + chip_y,
                                     (text_w + icon_reserve) - split_x, chip_h)
        else:
            icon_view_local = None
            translate_view_local = None
            render_w = text_w
            icon_hit_screen = None
            translate_hit_screen = None

        chip_layouts.append((chip_x, chip_y, render_w, chip_h, icon_view_local, translate_view_local))
        row_boxes.append(RowBox(
            index=i,
            text=text,
            frame=(ox + chip_x, oy + chip_y, hit_frame_w, chip_h),
            icon=icon_hit_screen,
            translate_icon=translate_hit_screen,
        ))

    return (ox, oy, panel_w, panel_h), chip_layouts, row_boxes


_CONTROL_HEIGHT = 28.0
_CORNER_SIZE = 28.0


def _overlay_controls(frame):
    """Top-left close button and top-right scale handle, in frame coordinates."""
    x, y, w, h = frame
    c = _CORNER_SIZE
    close = (x, y + h - c, c, c)
    return close, {(1, 1): (x + w - c, y + h - c, c, c)}


_OverlayControlView = None


def _make_overlay_control(rect, kind):
    """Draw contrasting controls without relying on template-image tinting."""
    global _OverlayControlView
    import AppKit
    if _OverlayControlView is None:
        class _CaptionOverlayControl(AppKit.NSView):
            def drawRect_(self, dirty):
                bounds = self.bounds()
                size = min(bounds.size.width, bounds.size.height)
                circle = AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                    AppKit.NSMakeRect(1, 1, size - 2, size - 2))
                AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.04, 0.94).setFill()
                circle.fill()
                AppKit.NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.8).setStroke()
                circle.setLineWidth_(1.0)
                circle.stroke()
                path = AppKit.NSBezierPath.bezierPath()
                path.setLineWidth_(2.0)
                def line(x1, y1, x2, y2):
                    path.moveToPoint_(AppKit.NSMakePoint(size * x1, size * y1))
                    path.lineToPoint_(AppKit.NSMakePoint(size * x2, size * y2))
                if self.control_kind == "close":
                    line(.34, .34, .66, .66)
                    line(.34, .66, .66, .34)
                else:
                    line(.3, .3, .7, .7)
                    line(.3, .3, .3, .5)
                    line(.3, .3, .5, .3)
                    line(.7, .7, .5, .7)
                    line(.7, .7, .7, .5)
                AppKit.NSColor.whiteColor().setStroke()
                path.stroke()
        _OverlayControlView = _CaptionOverlayControl
    view = _OverlayControlView.alloc().initWithFrame_(AppKit.NSMakeRect(*rect))
    view.control_kind = kind
    view.setToolTip_("マウスを重ねると一時的に非表示" if kind == "close"
                     else "ドラッグして字幕サイズを変更")
    return view


def _layout_stack(panel, content, rows, alphas, font_size, geometry, decor=None,
                  scroll_frac=0.0):
    """Lay out `rows` (oldest -> newest, live last -- see
    CaptionLineModel.visible_rows) as a bottom-up stack of rounded row
    chips inside `content`, position the panel per `geometry` (bottom edge
    pinned at geometry.bottom_px above the screen, growing upward as the
    stack gets taller -- same anchoring the single-line design used;
    center x at geometry.cx_frac of the screen width -- see
    _panel_origin), and bring it on-screen without activating this process
    (so focus/keyboard input is never stolen from whatever meeting app is
    in front). Returns the [RowBox, ...] list _compute_layout worked out
    (oldest -> newest, screen coords), or [] when there is nothing to
    show/lay out -- so the caller (_AppKitRenderer.show ->
    _OverlayController.tick) can hit-test hover/clicks against it on
    subsequent ticks without redoing any AppKit measurement.

    Each row gets its OWN chip: a fresh _CaptionBackgroundView (rounded,
    translucent-black rect, drawRect_-based -- see
    _get_rounded_background_view_class) sized to hug THAT row's own text
    (clamped to the same _MAX_WIDTH_FRACTION budget and wrap rules as
    before, per row -- see _measure_row_size) and given its alphaValue from
    `alphas`. Chips are rebuilt from scratch on every call (old subviews of
    `content` are removed first) rather than pooled/reused: repaints only
    happen on an actual content/scroll-offset change (see
    _OverlayController.tick's state-compare), so this is infrequent -- at
    most a few times/sec during active speech, never during silence.

    TWO SEPARATE centering steps, not to be conflated: each chip centers
    WITHIN the panel's own width (the widest TEXT chip's width -- see
    _compute_layout); the PANEL ITSELF then centers on the screen. Getting
    these backwards would visibly misplace narrower history/live chips
    relative to a wider one, or the whole stack relative to the screen.

    `decor` (a RowDecor, or None -- treated the same as RowDecor(), i.e. no
    icons at all) says which row (if any) is currently hovered, which row's
    copy icon (if any) should render as a checkmark instead, and which
    row's translate icon (if any) is mid-request/done/failed -- see
    RowDecor / _compute_layout / _make_copy_icon_view /
    _make_translate_icon_view. All the actual per-row icon geometry is
    _compute_layout's job; this function only measures text (as before) and
    builds the AppKit views.
    """
    import AppKit

    # Keep an existing overlay on its current display.  mainScreen() may
    # change on a multi-monitor desktop while this process is inactive.
    screen = _caption_screen(panel, AppKit.NSScreen.mainScreen(), AppKit.NSScreen.screens(), geometry)
    if screen is None:
        return []
    visible = _caption_visible_frame(screen, geometry)
    max_text_w = max(10.0, visible.size.width * _MAX_WIDTH_FRACTION - 2 * _PAD_X)
    if geometry.top_anchored:
        max_text_w = max(10.0, min(640.0, visible.size.width * _MAX_WIDTH_FRACTION)
                         - 2 * _PAD_X - _icon_reserve_for(font_size)[1])

    for v in list(content.subviews()):
        v.removeFromSuperview()

    if not rows:
        return []

    if decor is None:
        decor = RowDecor()

    labels = []
    row_sizes = []  # (text, text_w, text_h), oldest -> newest, matching `rows`
    for (text, _is_live) in rows:
        label = _make_row_label(font_size)
        text_w, text_h = _measure_row_size(label, text, max_text_w)
        labels.append(label)
        row_sizes.append((text, text_w, text_h))

    panel_rect, chip_layouts, row_boxes = _compute_layout(
        row_sizes, font_size, decor.hover_row, visible, geometry,
        scroll_frac=scroll_frac)
    panel_x, panel_y, panel_w, panel_h = panel_rect

    panel_h += _CONTROL_HEIGHT
    # Keep the new control strip reachable at the top of the display.
    adjusted_y = max(visible.origin.y, min(panel_y,
                     visible.origin.y + visible.size.height - panel_h))
    if geometry.top_anchored:
        adjusted_y = panel_y - _CONTROL_HEIGHT
    delta_y = adjusted_y - panel_y
    if delta_y:
        def shifted(rect):
            return (rect[0], rect[1] + delta_y, rect[2], rect[3]) if rect else None
        row_boxes = [RowBox(b.index, b.text, shifted(b.frame), shifted(b.icon),
                            shifted(b.translate_icon)) for b in row_boxes]
        panel_y = adjusted_y

    if geometry.top_anchored:
        background = _get_rounded_background_view_class().alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, panel_w, panel_h))
        background.background_alpha = 1.0
        background.corner_radius = 22.0
        background.attached_top = geometry.position == "notch"
        content.addSubview_(background)

    for label, (text, text_w, text_h), chip_layout, alpha in zip(
            labels, row_sizes, chip_layouts, alphas):
        chip_x, chip_y, chip_w, chip_h, icon_rect, translate_rect = chip_layout
        chip = _get_rounded_background_view_class().alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, chip_w, chip_h))
        chip.setAlphaValue_(alpha)
        if geometry.top_anchored:
            chip.background_alpha = 0.0
        label.setFrame_(AppKit.NSMakeRect(_PAD_X, _PAD_Y, text_w, text_h))
        chip.addSubview_(label)
        if icon_rect is not None:
            icon_view = _make_copy_icon_view(icon_rect, checkmark=(decor.copied_text == text))
            chip.addSubview_(icon_view)
        if translate_rect is not None:
            # Only THIS row's own state is its icon's -- a translation
            # pending on a different row must not light this one up.
            state = decor.translate_state if decor.translate_text == text else None
            chip.addSubview_(_make_translate_icon_view(translate_rect, state))
        chip.setFrame_(AppKit.NSMakeRect(chip_x, chip_y, chip_w, chip_h))
        content.addSubview_(chip)

    # The right-side copy/translate reserve is transparent until used; keep
    # the scale handle attached to the black caption itself, not that reserve.
    content.controls_width = panel_w if geometry.top_anchored else panel_w - _icon_reserve_for(font_size)[1]
    close_rect, corners = _overlay_controls((0, 0, content.controls_width, panel_h))
    close_view = _make_overlay_control(close_rect, "close")
    content.addSubview_(close_view)
    content.control_views = [close_view]
    for (sx, sy), rect in corners.items():
        grip = _make_overlay_control(rect, "scale")
        content.addSubview_(grip)
        content.control_views.append(grip)
    for control in content.control_views:
        control.setHidden_(not getattr(content, "controls_visible", False))

    # y is independent of panel_h (never derived from it) -- what makes a
    # taller (more rows, or wrapped) panel grow upward while its bottom
    # edge stays put at geometry.bottom_px.
    panel.setFrame_display_(AppKit.NSMakeRect(panel_x, panel_y, panel_w, panel_h), True)
    content.setFrame_(AppKit.NSMakeRect(0, 0, panel_w, panel_h))
    panel.orderFrontRegardless()   # show without activating this process
    return row_boxes


_SYMBOL_POINT_SIZE_RATIO = 0.72  # SF Symbol point size ~= icon view size * this ratio


def _symbol_configuration_for(point_size):
    """An NSImageSymbolConfiguration for `point_size` at medium weight/
    scale, or None if unavailable. SF Symbol point-size configuration
    (macOS 11+, verified present against the installed pyobjc/AppKit --
    see this module's own AppKit-verification notes elsewhere) makes the
    glyph actually RENDER at its intended size/weight, rather than relying
    on NSImageScaleProportionallyUpOrDown alone to stretch whatever size
    the symbol's default representation happens to be -- the difference is
    visible as crisper, correctly-weighted glyphs at typical caption font
    sizes. Prefers configurationWithPointSize_weight_scale_ (adds an
    explicit NSImageSymbolScale); falls back to the older two-argument
    configurationWithPointSize_weight_ if only that exists; returns None
    if NEITHER does (a macOS old enough to have no SF Symbols
    configuration API at all) -- _make_copy_icon_view treats None as
    "skip, use the plain image as-is", never raising."""
    import AppKit

    cls = getattr(AppKit, "NSImageSymbolConfiguration", None)
    if cls is None:
        return None
    if hasattr(cls, "configurationWithPointSize_weight_scale_"):
        return cls.configurationWithPointSize_weight_scale_(
            point_size, AppKit.NSFontWeightMedium, AppKit.NSImageSymbolScaleMedium)
    if hasattr(cls, "configurationWithPointSize_weight_"):
        return cls.configurationWithPointSize_weight_(point_size, AppKit.NSFontWeightMedium)
    return None


def _make_icon_view(rect, symbol_name, description, fallback_glyph):
    """Build one small icon subview for a hovered chip's right extension --
    `rect` is the CHIP-LOCAL (x, y, w, h) _compute_layout already worked out
    (see _layout_stack), handed straight to setFrame_. Prefers the SF Symbol
    `symbol_name` rendered through an NSImageView, with an explicit
    NSImageSymbolConfiguration applied (see _symbol_configuration_for) so the
    glyph renders crisply at its intended size instead of just being
    stretched; falls back to a plain NSTextField showing `fallback_glyph` if
    imageWithSystemSymbolName_accessibilityDescription_ returns None (an
    older macOS with no SF Symbols support). Every icon uses the same white
    tint -- the SHAPE alone carries the state. The container's hitTest_
    already funnels every click to itself (see
    _get_scrollable_container_view_class), so these views need no mouse
    event handling of their own -- purely decorative.

    Shared by _make_copy_icon_view and _make_translate_icon_view; those two
    only decide WHICH symbol/glyph a state maps to."""
    import AppKit

    x, y, w, h = rect
    frame = AppKit.NSMakeRect(x, y, w, h)
    tint = AppKit.NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.75)

    image = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol_name, description)
    if image is not None:
        view = AppKit.NSImageView.imageViewWithImage_(image)
        view.setFrame_(frame)
        view.setContentTintColor_(tint)
        view.setImageScaling_(AppKit.NSImageScaleProportionallyUpOrDown)
        cfg = _symbol_configuration_for(min(w, h) * _SYMBOL_POINT_SIZE_RATIO)
        if cfg is not None:
            view.setSymbolConfiguration_(cfg)
        return view

    label = AppKit.NSTextField.alloc().initWithFrame_(frame)
    label.setEditable_(False)
    label.setSelectable_(False)
    label.setBordered_(False)
    label.setBezeled_(False)
    label.setDrawsBackground_(False)
    label.setAlignment_(AppKit.NSTextAlignmentCenter)
    label.setStringValue_(fallback_glyph)
    label.setTextColor_(tint)
    label.setFont_(AppKit.NSFont.systemFontOfSize_(max(8.0, min(w, h) * 0.85)))
    return label


def _make_copy_icon_view(rect, checkmark):
    """The copy icon (doc.on.doc), or a checkmark for ~COPY_FEEDBACK_SEC
    after a successful copy. Both symbols confirmed present via an
    on-machine probe against the installed AppKit."""
    if checkmark:
        return _make_icon_view(rect, "checkmark", "Copied", "\u2713")
    return _make_icon_view(rect, "doc.on.doc", "Copy", "\u29c9")


# state -> (SF Symbol, accessibility description, no-SF-Symbols fallback
# glyph) for the translate icon. None (idle) is the plain "translate"
# symbol -- Apple's own Translate glyph, confirmed present via an on-machine
# probe against the installed AppKit, alongside the three state symbols
# below. Shapes, not colors, distinguish the states: the tint is the same
# white for every icon in the panel (see _make_icon_view).
_TRANSLATE_ICON_SYMBOLS = {
    None: ("translate", "Translate", "\u6587"),
    "pending": ("ellipsis", "Translating", "\u2026"),
    "done": ("checkmark", "Translated", "\u2713"),
    "failed": ("exclamationmark.triangle", "Translation failed", "!"),
}


def _make_translate_icon_view(rect, state):
    """The translate icon in its `state` (None/idle, "pending", "done" or
    "failed" -- see RowDecor.translate_state). An unknown state falls back
    to the idle icon rather than raising: this runs inside a repaint, where
    a wrong-looking icon is a far better outcome than a broken panel."""
    symbol_name, description, fallback = _TRANSLATE_ICON_SYMBOLS.get(
        state, _TRANSLATE_ICON_SYMBOLS[None])
    return _make_icon_view(rect, symbol_name, description, fallback)


_CaptionStatusTarget = None


class _CaptionStatusItem:
    """A menu-bar button that survives hiding the caption window."""

    def __init__(self, on_show):
        import AppKit
        global _CaptionStatusTarget
        if _CaptionStatusTarget is None:
            class _CaptionMenuTarget(AppKit.NSObject):
                def showCaption_(self, sender):
                    if self.on_show is not None:
                        self.on_show()
            _CaptionStatusTarget = _CaptionMenuTarget
        self.target = _CaptionStatusTarget.alloc().init()
        self.target.on_show = on_show
        self.bar = AppKit.NSStatusBar.systemStatusBar()
        self.item = self.bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        button = self.item.button()
        icon = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "captions.bubble", "字幕を表示")
        if icon is not None:
            icon.setTemplate_(True)
            button.setImage_(icon)
        else:
            button.setTitle_("字幕")
        button.setToolTip_("字幕を表示（自動非表示しない）")
        button.setTarget_(self.target)
        button.setAction_("showCaption:")

    def close(self):
        if self.item is not None:
            self.item.button().setTarget_(None)
            self.bar.removeStatusItem_(self.item)
            self.item = None
        self.target.on_show = None


_AudioMeterView = None


def _make_audio_meter_view():
    global _AudioMeterView
    import AppKit
    if _AudioMeterView is None:
        class _CaptionAudioMeterView(AppKit.NSView):
            def drawRect_(self, rect):
                level = getattr(self, "level", 0.0)
                color = (AppKit.NSColor.systemGreenColor() if level > .02
                         else AppKit.NSColor.systemGrayColor())
                color.set()
                bounds = self.bounds()
                for x, y, w, h in _audio_bar_rects(level, bounds.size.width, bounds.size.height):
                    AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                        AppKit.NSMakeRect(x, y, w, h), w / 2, w / 2).fill()
        _AudioMeterView = _CaptionAudioMeterView
    view = _AudioMeterView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 20, 20))
    view.level = 0.0
    view.setToolTip_("マイク入力の音量")
    return view


class _IslandSurface:
    """Native, interruptible compact/expanded morph of a single NSPanel.

    A short-lived 60 Hz timer resizes the real window and lays out its children
    at each intermediate size. Text stays below the camera throughout, and
    the window doesn't leave an invisible expanded mouse target when compact.
    No timer or animation work remains after settling or closing.
    """

    def __init__(self, panel, content):
        self.panel, self.content = panel, content
        self.timer = None
        self.compact = True
        self.target = None
        self.body_views = []
        self.audio_level = 0.0

    @property
    def animating(self):
        return self.timer is not None

    def stop(self):
        if self.timer is not None:
            self.timer.invalidate()
            self.timer = None

    def show(self, rows, alphas, font_size, geometry, decor, scroll_frac=0.0):
        import AppKit
        screen = _caption_screen(self.panel, AppKit.NSScreen.mainScreen(),
                                 AppKit.NSScreen.screens(), geometry)
        if screen is None:
            self.stop()
            return []
        self.anchor = _island_anchor(screen)
        anchor = self.anchor
        decor = decor or RowDecor()
        max_width = max(10.0, min(640.0, anchor.screen_width * _MAX_WIDTH_FRACTION)
                        - 2 * _PAD_X - 2 * _icon_reserve_for(font_size)[1])
        labels, sizes = [], []
        for text, _live in rows:
            label = _make_row_label(font_size)
            w, h = _measure_row_size(label, text, max_width)
            labels.append(label)
            sizes.append((text, w, h))
        target, chips, boxes = _island_layout(sizes, font_size, decor.hover_row, anchor,
                                              scroll_frac=scroll_frac)
        was_visible = self.panel.isVisible()
        was_compact = self.compact
        old_frame = self.panel.frame()
        start = (old_frame.origin.x, old_frame.origin.y, old_frame.size.width, old_frame.size.height)
        self.stop()
        self.compact = not rows
        self.target = target
        for view in list(self.content.subviews()):
            view.removeFromSuperview()
        self.background = _get_rounded_background_view_class().alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, target[2], target[3]))
        self.background.background_alpha = 1.0
        self.background.attached_top = anchor.attached
        self.content.addSubview_(self.background)
        self.body_views = []
        for label, (_text, w, h), chip_layout, alpha in zip(labels, sizes, chips, alphas):
            cx, cy, cw, ch, icon_rect, translate_rect = chip_layout
            chip = AppKit.NSView.alloc().initWithFrame_(AppKit.NSMakeRect(cx, cy, cw, ch))
            chip.setAlphaValue_(alpha)
            reserve = _icon_reserve_for(font_size)[1]
            text_width = cw - 2 * _PAD_X - 2 * reserve
            label.setAlignment_(AppKit.NSTextAlignmentCenter)
            label.setFrame_(AppKit.NSMakeRect(_PAD_X + reserve, anchor.caption_padding, text_width, h))
            chip.addSubview_(label)
            if icon_rect:
                chip.addSubview_(_make_copy_icon_view(icon_rect, decor.copied_text == _text))
            if translate_rect:
                state = decor.translate_state if decor.translate_text == _text else None
                chip.addSubview_(_make_translate_icon_view(translate_rect, state))
            self.content.addSubview_(chip)
            self.body_views.append((chip, (cx, cy, cw, ch)))
        self.right_status = _make_audio_meter_view()
        self.right_status.level = self.audio_level
        self.content.addSubview_(self.right_status)
        self.content.control_views = []
        if rows:
            close, corners = _overlay_controls((0, 0, target[2], target[3]))
            for rect, kind in [(close, "close"), *[(r, "scale") for r in corners.values()]]:
                view = _make_overlay_control(rect, kind)
                view.setHidden_(not getattr(self.content, "controls_visible", False))
                self.content.addSubview_(view)
                self.content.control_views.append(view)
        self.content.controls_width = target[2]
        reduce_motion = getattr(AppKit.NSWorkspace.sharedWorkspace(),
                                "accessibilityDisplayShouldReduceMotion", lambda: False)()
        same_anchor = (abs(start[0] + start[2] / 2 - anchor.center_x) < 1
                       and abs(start[1] + start[3] - anchor.top) < 1)
        if _island_should_morph(was_visible, was_compact, self.compact,
                                reduce_motion, same_anchor, start, target):
            self.start_frame = start
            self.started_at = time.monotonic()
            self._draw(start)
            self.timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                1 / 60, True, lambda _timer: self.advance())
            AppKit.NSRunLoop.currentRunLoop().addTimer_forMode_(self.timer, AppKit.NSRunLoopCommonModes)
        else:
            self._draw(target)
        self.panel.orderFrontRegardless()
        return boxes

    def advance(self):
        p = (time.monotonic() - self.started_at) / .24
        self._draw(_island_tween(self.start_frame, self.target, p))
        if p >= 1:
            self.stop()

    def set_audio_level(self, level):
        self.audio_level = level
        view = getattr(self, "right_status", None)
        if view is not None and abs(view.level - level) > .001:
            view.level = level
            view.setNeedsDisplay_(True)
            self.panel.displayIfNeeded()

    def _draw(self, rect):
        import AppKit
        x, y, w, h = rect
        self.panel.setFrame_display_(AppKit.NSMakeRect(x, y, w, h), False)
        self.content.setFrame_(AppKit.NSMakeRect(0, 0, w, h))
        self.background.setFrame_(AppKit.NSMakeRect(0, 0, w, h))
        self.background.corner_radius = min(24.0, h / 2)
        self.background.setNeedsDisplay_(True)
        # Place content relative to the moving shell's TOP and CENTER. During
        # expansion it is revealed below the camera, never slid through it.
        dx, dy = (w - self.target[2]) / 2, h - self.target[3]
        for view, (cx, cy, cw, ch) in self.body_views:
            view.setFrame_(AppKit.NSMakeRect(cx + dx, cy + dy, cw, ch))
        # During a compact -> expanded morph, even intermediate icon frames
        # must fit the narrow wings on either side of the physical camera.
        wing_width = (w - self.anchor.camera_width) / 2
        inset = max(8.0, min(36.0, (wing_width - 20) / 2))
        sy = h - self.anchor.header_height / 2 - 10
        self.right_status.setFrame_(AppKit.NSMakeRect(w - inset - 20, sy, 20, 20))
        close, corners = _overlay_controls((0, 0, w, h))
        for view, frame in zip(self.content.control_views, [close, *corners.values()]):
            view.setFrame_(AppKit.NSMakeRect(*frame))
        self.content.setNeedsDisplay_(True)
        self.panel.displayIfNeeded()


class _AppKitRenderer:
    """The only AppKit-touching half of _OverlayController: owns the real
    NSPanel/content view and paints via _layout_stack()/orderOut_(). Kept
    as its own small object -- rather than inlined into tick() -- so
    _OverlayController's SHOW/HIDE DECISION logic (the part with real bug
    potential: see the "panel doesn't hide after idle_clear_sec of
    silence" investigation this class was extracted to make testable) can
    be driven headlessly in tests/test_caption_overlay.py against a fake
    renderer, with the real .panel/.content/_layout_stack call chain never
    touched there -- this file's AppKit-lazy-import invariant stays intact
    even though tick()'s decision logic has real regression coverage.

    Holds no font_size of its own beyond construction (only needed there to
    build the initial panel/content) -- .show() now receives font_size AND
    geometry fresh on every call, since both are user-adjustable at runtime
    (Cmd+scroll / drag) and owned by the controller, not this renderer."""

    def __init__(self, font_size):
        self.panel, self.content = _make_panel(font_size)  # _make_panel imports AppKit itself
        self.status_item = None
        self.island = None
        self.audio_timer = None
        self.audio_meter = AudioLevelMeter()

    def start_audio_meter(self, level_source):
        import AppKit
        self.stop_audio_meter()

        def update(_timer):
            now = time.monotonic()
            try:
                level = self.audio_meter.update(level_source(), now)
            except Exception:
                level = self.audio_meter.update(None, now)
            if self.island is not None:
                self.island.set_audio_level(level)

        self.audio_timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            1 / 30, True, update)
        AppKit.NSRunLoop.currentRunLoop().addTimer_forMode_(self.audio_timer, AppKit.NSRunLoopCommonModes)

    def stop_audio_meter(self):
        if self.audio_timer is not None:
            self.audio_timer.invalidate()
            self.audio_timer = None

    def install_status_item(self, on_show):
        self.status_item = _CaptionStatusItem(on_show)

    def remove_status_item(self):
        if self.status_item is not None:
            self.status_item.close()
            self.status_item = None

    def show(self, rows, alphas, font_size, geometry, decor=None, scroll_frac=0.0):
        if geometry.top_anchored:
            if self.island is None:
                self.island = _IslandSurface(self.panel, self.content)
                self.island.audio_level = self.audio_meter.level
            return self.island.show(rows, alphas, font_size, geometry, decor,
                                    scroll_frac=scroll_frac)
        if self.island is not None:
            self.island.stop()
            self.island = None
        boxes = _layout_stack(self.panel, self.content, rows, alphas, font_size, geometry,
                              decor=decor, scroll_frac=scroll_frac)
        return boxes

    def screen_state(self, geometry):
        import AppKit
        screen = _caption_screen(self.panel, AppKit.NSScreen.mainScreen(),
                                 AppKit.NSScreen.screens(), geometry)
        if screen is None:
            return None
        if geometry.top_anchored:
            return _island_anchor(screen)
        frame = _caption_visible_frame(screen, geometry)
        return (frame.origin.x, frame.origin.y, frame.size.width, frame.size.height)

    @property
    def animating(self):
        return self.island is not None and self.island.animating

    @property
    def control_frame(self):
        if self.island is not None and (self.island.compact or self.island.animating):
            return None
        # Drag handlers can move the panel between caption repaints.
        frame = self.panel.frame()
        return (frame.origin.x, frame.origin.y,
                getattr(self.content, "controls_width", frame.size.width), frame.size.height)

    def set_controls_visible(self, visible):
        changed = getattr(self.content, "controls_visible", False) != visible
        self.content.controls_visible = visible
        for control in getattr(self.content, "control_views", ()):
            if control.isHidden() == visible:
                control.setHidden_(not visible)
                changed = True
        if changed:
            # A hover-only change must reach the window even when the text
            # and layout have not changed and show() was skipped this tick.
            self.content.setNeedsDisplay_(True)
            self.panel.displayIfNeeded()

    def hide(self):
        if self.island is not None:
            self.island.stop()
        self.set_controls_visible(False)
        self.panel.orderOut_(None)


# Scroll-wheel tuning: a scroll distance of _PX_PER_ROW points counts as one
# history row. Precise trackpad deltas (scrollingDeltaY when
# hasPreciseScrollingDeltas() is true) already arrive in points; a
# non-precise (physical mouse wheel) event's delta is in small "line/notch"
# units instead (per NSEvent.h's own doc comment on hasPreciseScrollingDeltas
# -- see _row_delta_from_scroll_event), so it's scaled up by
# _WHEEL_LINE_SCALE first to land in the same points-ish range.
_PX_PER_ROW = 30.0
_WHEEL_LINE_SCALE = 10.0
# History-scroll sign, applied in _row_delta_from_scroll_event.
# CaptionLineModel.scroll_by: positive = older, negative = toward live.
# Natural scrolling moves the CONTENT with the fingers, and older rows sit
# ABOVE the live line, so swipe-DOWN (negative scrollingDeltaY) must yield
# a POSITIVE row_delta. The original +1 mapping (swipe-up = older, like a
# scrollbar) felt inverted on a real trackpad; this is the flip the
# function's own docstring used to invite. Cmd+scroll font-size uses
# _SCALE_SIGN instead, so it does not inherit this flip.
_SCROLL_SIGN = -1
_SCALE_SIGN = 1  # scroll-up grows the font (browser-zoom direction)


def _row_delta_from_scroll_event(event, accumulated_px, sign=_SCROLL_SIGN):
    """Convert one NSScrollWheel event into (row_delta, new_accumulated_px)
    -- row_delta is an int (usually -1/0/+1, occasionally more for a fast
    fling), new_accumulated_px is the leftover sub-row remainder to carry
    into the next event.

    Direction, from Apple's NSEvent.h doc comments PLUS a hands-on
    correction:
      - isDirectionInvertedFromDevice's doc comment: scrollingDeltaY's
        sign ALREADY reflects natural scrolling (moving the CONTENT).
      - Older caption rows sit ABOVE the pinned live row. Swiping DOWN
        moves the content down and reveals that older history -- the
        Messages/Slack direction. That gesture reports a NEGATIVE
        scrollingDeltaY, and CaptionLineModel.scroll_by wants a POSITIVE
        delta for older, so the default `sign` is _SCROLL_SIGN = -1.
        Cmd+scroll font-size passes _SCALE_SIGN = +1 instead (scroll-up
        grows), so the two gestures stay independent.

    hasPreciseScrollingDeltas()/scrollingDeltaY() scaling per NSEvent.h's
    own doc comment on scrollingDeltaX/Y: "When -hasPreciseScrollDeltas
    returns NO, multiply the returned value by line or row height. When
    ... returns YES, scroll by the returned value (in points)."
    """
    if event.hasPreciseScrollingDeltas():
        delta_px = event.scrollingDeltaY()
    else:
        delta_px = event.scrollingDeltaY() * _WHEEL_LINE_SCALE
    accumulated_px += sign * delta_px
    rows = int(accumulated_px / _PX_PER_ROW)  # truncates toward 0, keeping the sub-row remainder
    accumulated_px -= rows * _PX_PER_ROW
    return rows, accumulated_px


def _scroll_rows_from_event(event, px_per_row=_PX_PER_ROW):
    """Fractional history-row delta for one NSScrollWheel event.

    Unlike _row_delta_from_scroll_event (used for Cmd+scroll font steps,
    which WANT discrete notches), this applies every pixel immediately so
    a trackpad slides the history stack continuously. Sign is _SCROLL_SIGN
    (swipe-down = older). Non-precise mouse wheels are scaled by
    _WHEEL_LINE_SCALE first, same as the quantized helper."""
    if event.hasPreciseScrollingDeltas():
        delta_px = event.scrollingDeltaY()
    else:
        delta_px = event.scrollingDeltaY() * _WHEEL_LINE_SCALE
    return _SCROLL_SIGN * delta_px / px_per_row


# Reused (same accumulator/threshold math, own accumulator variable) for
# Cmd+scroll FONT-SIZE steps -- see _OverlayController._handle_scroll_wheel
# -- with sign=_SCALE_SIGN so scroll-up still grows the font (1.1**steps
# with steps > 0), independent of the history-scroll flip.

_NSEVENT_MODIFIER_FLAG_COMMAND = 1 << 20   # stable value of AppKit.NSEventModifierFlagCommand


def _is_command_scroll(event):
    """Whether `event` (an NSEvent handed to scrollWheel_) was received
    with the Command key held. Reads the raw modifierFlags() bitmask
    against a HARDCODED constant (AppKit.NSEventModifierFlagCommand's
    actual numeric value, per Apple's NSEvent.h -- unchanged since the
    original NSCommandKeyMask) rather than importing AppKit just to read
    one constant -- keeps this helper importable/callable with no AppKit
    dependency (this module's AppKit-lazy-import invariant) and directly
    unit-testable against a duck-typed fake event, same idiom
    _row_delta_from_scroll_event's own tests use."""
    return bool(event.modifierFlags() & _NSEVENT_MODIFIER_FLAG_COMMAND)


# Stable value of AppKit.NSPasteboardTypeString (verified against the
# installed pyobjc: AppKit.NSPasteboardTypeString == "public.utf8-plain-text",
# a UTI-based pasteboard type constant introduced in the 10.13 SDK to
# replace the legacy NSStringPboardType) -- hardcoded so
# _copy_text_to_pasteboard's fake-pasteboard path (used by every headless
# test in tests/test_caption_overlay.py, see that file's own "never
# imports AppKit" invariant) never needs to import AppKit just to read one
# constant, the same idiom _NSEVENT_MODIFIER_FLAG_COMMAND above already
# uses for _is_command_scroll.
_NS_PASTEBOARD_TYPE_STRING = "public.utf8-plain-text"


def _copy_text_to_pasteboard(text, pasteboard=None):
    """Copy `text` (a plain string) to the system general pasteboard, or to
    `pasteboard` if given -- a duck-typed stand-in exposing clearContents()
    and setString_forType_(text, type) -- see this module's headless tests
    (a plain fake, no AppKit involved at all) and the --demo smoke test (a
    real but NAMED pasteboard, NSPasteboard.pasteboardWithName_(...), so it
    never clobbers the user's actual clipboard). AppKit is only imported
    when `pasteboard` is None (the real production default) -- see
    _NS_PASTEBOARD_TYPE_STRING above for why the type constant itself never
    forces that import. Never raises: any failure (including a raising
    fake pasteboard) prints one warning line and returns False, the same
    never-raise contract save_geometry already uses, for the same reason --
    this runs from inside an AppKit mouseDown_ callback (see
    _OverlayController._handle_mouse_down), which must not propagate an
    exception into the event dispatch loop."""
    try:
        if pasteboard is not None:
            pb = pasteboard
        else:
            import AppKit
            pb = AppKit.NSPasteboard.generalPasteboard()
        pb.clearContents()
        return bool(pb.setString_forType_(text, _NS_PASTEBOARD_TYPE_STRING))
    except Exception as exc:
        print(f"[caption-overlay] warning: could not copy text to the clipboard: {exc}")
        return False


def _run_in_background(fn):
    """Run `fn` on a throwaway daemon thread -- the production half of
    _OverlayController's `run_async` seam (tests inject a synchronous one).
    Daemon on purpose: a translation still in flight must never hold up
    Ctrl-C, and its result is worthless once the overlay is gone."""
    threading.Thread(target=fn, name="caption-translate", daemon=True).start()


class _OverlayController:
    """Drains a display-event queue into a CaptionLineModel and repaints via
    `renderer` when the visible stack, font size, or panel position
    changes. Framework-agnostic driving: call .tick() from whatever timer
    the caller already has (rumps.Timer in menubar mode) or the NSTimer
    run_console_overlay sets up for itself.

    `renderer` defaults to a real _AppKitRenderer (touches AppKit); tests
    inject a fake exposing the same .show(rows, alphas, font_size,
    geometry)/.hide() shape instead, so tick()'s decision logic -- including
    the idle-hide transition -- runs headlessly and is actually exercised
    by an automated test (see tests/test_caption_overlay.py's "controller
    tick() decision logic" section).

    Hover expands the history stack; an active gesture keeps its current
    expansion state so resize grips do not jump away from the pointer.
    Hover and active drags postpone idle-hide so positioning and resizing
    can finish. Hover also toggles click-through (needed to receive
    scroll/mouse-down input), checked before the visibility decision while
    the panel is visible. _update_hover is frozen while a drag is in
    progress (AppKit keeps delivering mouseDragged_ to the mouseDown view
    even once the cursor has left its bounds, so click-through must not
    flip mid-drag).

    Position (geometry.cx_frac/bottom_px) and font size (geometry.font_size)
    are user-adjustable at runtime -- drag to move, Cmd+scroll to resize --
    and persisted to `settings_path` as JSON: debounced 1s after the last
    change while not dragging (see tick()), saved immediately on mouse-up
    (see _handle_mouse_up) so a drag's final position is never lost to the
    debounce window. settings_path=None disables persistence entirely."""

    def __init__(self, display_queue, font_size=_DEFAULT_FONT_SIZE, renderer=None,
                 settings_path=DEFAULT_SETTINGS_PATH, debug_input=False, copy_fn=None,
                 translate_fn=None, run_async=None, position=None, level_source=None):
        self.display_queue = display_queue
        self.font_size = font_size
        self.settings_path = settings_path
        self._debug_input = debug_input  # see _log_debug_input / main()'s --debug-input
        self._copy_fn = copy_fn if copy_fn is not None else _copy_text_to_pasteboard
        # Translate icon (see _start_translate): `translate_fn` is the
        # blocking text->translation call, `run_async` how it gets off the
        # UI thread. Tests inject a fake pair -- a synchronous run_async
        # keeps the queue/tick handoff itself real while removing the
        # concurrency, so nothing here needs a thread or a network.
        self._translate_fn = translate_fn if translate_fn is not None else translator.translate
        self._run_async = run_async if run_async is not None else _run_in_background
        self.model = CaptionLineModel(idle_clear_sec=_IDLE_CLEAR_SEC)
        self.renderer = renderer if renderer is not None else _AppKitRenderer(font_size)
        content = getattr(self.renderer, "content", None)
        if content is not None:
            content.on_scroll = self._handle_scroll_wheel
            content.on_mouse_down = self._handle_mouse_down
            content.on_mouse_drag = self._handle_mouse_drag
            content.on_mouse_up = self._handle_mouse_up
            content.debug_input = debug_input

        # settings_path=None: geometry is built straight from the font_size
        # argument (class defaults for position), never loaded or saved.
        # Otherwise, load from disk -- but an existing settings file's own
        # font_size wins over the font_size ARGUMENT (the CLI flag is only
        # a first-run default): only override it with the argument when the
        # file did NOT exist at all. That's a distinct condition from "did
        # not load" (load_geometry() also returns defaults for an unreadable
        # /corrupt file), so existence has to be checked separately, before
        # loading.
        if settings_path is None:
            self.geometry = OverlayGeometry(font_size=float(font_size))
        else:
            existed = os.path.exists(settings_path)
            self.geometry = load_geometry(settings_path)
            if not existed:
                self.geometry.font_size = float(font_size)
        if position is not None:
            self.geometry.set_position(position)
            if settings_path is not None:
                save_geometry(settings_path, self.geometry)

        # Starts as the "nothing shown" state, matching reality (the panel
        # is constructed hidden -- _make_panel never orders it front) --
        # NOT some sentinel like None. See _last_state's use in tick(): with
        # a None-vs-real-tuple mismatch, the very first tick() (even with
        # zero events ever having happened) would call renderer.hide() once
        # unnecessarily (harmless against the real panel, but it broke a
        # "hide() only fires on a real shown->hidden transition" test
        # invariant -- see git history around the idle-hide bug audit). The
        # geometry-derived fields must match self.geometry's own initial
        # values for the same reason -- built via the same helper tick()
        # uses so the two can never drift apart.
        self._last_state = self._geometry_state(((), (), 0, None, None, None, None))
        if self.geometry.top_anchored:
            self._last_state = None  # Draw the compact island before the first recognition.
        self._hovering = False
        self._expanded = False
        self._panel_visible = False
        self._scale_accum_px = 0.0
        self._dragging = False
        self._drag_base = None
        self._resize_base = None
        self._dismissed = False
        self._pinned = False
        self._close_hovered = False
        self._settings_dirty_at = None
        self._row_boxes = []          # [RowBox, ...] from the most recent repaint -- see tick()
        self._hover_row = None        # index into the current `rows`, or None -- see tick()
        self._copied = None           # (text, expire_at monotonic) or None -- see _handle_mouse_down/tick()
        self._icon_hover_started_at = None
        self._icon_hover_copied_text = None
        # Translate-icon state, all main-thread-only (worker threads touch
        # nothing but _translate_queue -- see _start_translate):
        #   _translate           (text, state, expire_at) or None -- the row
        #                        whose translate icon is not idle. expire_at
        #                        is None while "pending" (a slow API call
        #                        must not have its icon time out under it).
        #   _translate_queue     finished workers hand their result back here
        #                        for tick() to apply on the UI thread.
        #   _translate_inflight  texts with a request already running, so a
        #                        second click (or the dwell fallback) cannot
        #                        pay for the same row twice.
        self._translate = None
        self._translate_queue = queue.Queue()
        self._translate_inflight = set()
        self._translate_hover_started_at = None
        self._translate_hover_done_text = None
        self._global_mouse_monitor = None
        self._install_global_mouse_monitor()
        install_status_item = getattr(self.renderer, "install_status_item", None)
        if install_status_item is not None:
            install_status_item(self.show_from_menu)
        start_audio_meter = getattr(self.renderer, "start_audio_meter", None)
        if self.geometry.top_anchored and level_source is not None and start_audio_meter is not None:
            start_audio_meter(level_source)

    def _geometry_state(self, base):
        """`base` (a (rows, alphas, offset, hover_row, copied_text,
        translate_text, translate_state) tuple) extended with the rounded
        geometry fields tick()'s repaint-skip compare also watches --
        factored out so __init__'s sentinel and tick()'s real state are
        built identically and can never drift apart."""
        g = self.geometry
        screen_state = getattr(self.renderer, "screen_state", None)
        return base + (round(g.font_size, 2), round(g.cx_frac, 4), round(g.bottom_px, 1),
                       g.position, screen_state(g) if screen_state else None)

    def _handle_scroll_wheel(self, event):
        if _is_command_scroll(event):
            steps, self._scale_accum_px = _row_delta_from_scroll_event(
                event, self._scale_accum_px, sign=_SCALE_SIGN)
            if steps:
                self.geometry.scale_by(steps)
                self._settings_dirty_at = time.monotonic()
                if self._debug_input:
                    _log_debug_input(f"scale steps={steps} -> font_size={self.geometry.font_size:.1f}")
            return
        delta = _scroll_rows_from_event(event)
        if delta:
            self.model.scroll_by(delta, time.monotonic())
            # tick()'s own timer picks up the resulting state change on its
            # next cycle (<=_POLL_INTERVAL_SEC away, imperceptible) -- not
            # forced here, to avoid re-entering tick() from inside AppKit's
            # event dispatch.

    def _mouse_location(self):
        """Seam wrapping AppKit.NSEvent.mouseLocation() as a plain (x, y)
        tuple (screen coords) -- used by _update_hover and
        _handle_mouse_down so tests can monkeypatch this one method
        instead of needing a real AppKit/WindowServer cursor position."""
        import AppKit
        point = AppKit.NSEvent.mouseLocation()
        return (point.x, point.y)

    def _click_point(self, event):
        """The screen-coordinate (x, y) a mouseDown_/mouseUp_ `event`
        actually occurred at -- ROOT CAUSE FIX for "the copy icon does
        nothing when clicked" (see the diagnosis this fix came from:
        real-AppKit repro scripts that dispatched a synthetic mouseDown_
        via panel.sendEvent_ and compared its outcome against the
        previous _mouse_location()-only behavior).

        Prefers the event's OWN recorded location (event.locationInWindow(),
        a WINDOW-local point captured by AppKit at the moment the hardware
        click happened, converted to screen coordinates via
        event.window().convertPointToScreen_() -- verified against the
        installed macOS SDK's NSWindow.h, available since 10.12) over a
        fresh _mouse_location() re-poll of the CURRENT global cursor
        position. These are NOT the same thing: AppKit event dispatch is
        not instantaneous, and a real click/release inherently involves a
        little cursor motion (trackpad travel, mouse micro-drift, or the
        user's hand already starting to move toward the next thing) -- so
        by the time this callback runs, NSEvent.mouseLocation() can
        already read a few points off from where the mouseDown itself
        landed. Hands-on reproduction (real AppKit, synthetic mouseDown
        dispatched via panel.sendEvent_ -- see the diagnosis scripts):
        with the event's own location dead-center on the (deliberately
        generous, ~52x53pt) copy-icon hit rect but _mouse_location()
        simulating a cursor that had already drifted just 3pt past the
        icon's edge, the OLD code (which only ever called
        _mouse_location(), never the event) silently treated the click as
        a drag instead of a copy -- exactly the reported symptom, with no
        exception and no feedback. Hover-row detection (a whole-ROW
        target, see tick()) and drag repositioning (continuously
        re-sampled every mouseDragged_, self-correcting) are far less
        sensitive to this same gap, which is why only the icon click was
        affected.

        Falls back to _mouse_location() when `event` is None (every
        existing headless test drives this with event=None -- see
        test_handle_mouse_down_calls_mouse_location_exactly_once and
        friends -- and this is also simply the previous, only behavior) or
        when event.window() is None (defensive: should not happen for a
        real mouseDown_, which AppKit always associates with the window
        that received it)."""
        if event is not None:
            window = event.window()
            if window is not None:
                point = window.convertPointToScreen_(event.locationInWindow())
                return (point.x, point.y)
        return self._mouse_location()

    def _handle_mouse_down(self, event):
        """An icon click (see _hit_icon) copies that row's text and stops
        here -- it must NOT arm the drag gesture below. Any other click on
        the panel arms a drag exactly as before. Reads the click point via
        _click_point() ONCE (both branches need it) and reuses it -- not a
        second read.

        When self._debug_input is on (see _resolve_debug_input /
        CAPTION_OVERLAY_DEBUG_INPUT), logs ONE line before branching with
        everything needed to explain a miss by hand from a real session:
        the click point actually used (click=, from _click_point -- prefers
        the event's own location) alongside a FRESH, independent
        _mouse_location() poll (cursor=), so a real-world divergence
        between the two -- the "cursor drifted after the click" theory --
        is directly visible in the log, not just theorized; plus
        hover/row-box state and the _hit_icon result. Then either the
        existing "copied:" line (success), "copy FAILED for: <text>" (the
        real pasteboard write itself failed), or "no icon hit -> drag
        armed" (miss) -- covering every branch this method can take."""
        if getattr(self.renderer, "animating", False):
            return  # Target hit boxes are valid only after the shell settles.
        mouse = self._click_point(event)
        if (self.geometry.top_anchored
                and not self._row_boxes):
            self.show_from_menu()
            return
        control_frame = getattr(self.renderer, "control_frame", None)
        if control_frame is not None and self._panel_visible:
            close_rect, corners = _overlay_controls(control_frame)
            if _point_in_rect(mouse, close_rect):
                self._dismiss()
                return
            for corner, rect in corners.items():
                if _point_in_rect(mouse, rect):
                    self._resize_base = (mouse, corner, self.geometry.font_size, control_frame)
                    self._dragging = True
                    return
        box = _hit_icon(mouse, self._row_boxes)
        translate_box = _hit_translate_icon(mouse, self._row_boxes) if box is None else None
        if self._debug_input:
            cursor = self._mouse_location()
            hovered_icon = None
            if self._hover_row is not None:
                hovered_icon = next((b.icon for b in self._row_boxes if b.index == self._hover_row), None)
            print(f"[caption-overlay] mouseDown: click=({mouse[0]:.1f},{mouse[1]:.1f}) "
                  f"cursor=({cursor[0]:.1f},{cursor[1]:.1f}) hovering={self._hovering} "
                  f"hover_row={self._hover_row} boxes={len(self._row_boxes)} "
                  f"icon_rect={hovered_icon} hit={box.index if box is not None else None} "
                  f"translate_hit={translate_box.index if translate_box is not None else None}")
        if box is not None:
            self._copy_box(box, "mouseDown")
            return
        if translate_box is not None:
            self._start_translate(translate_box, "mouseDown")
            return
        if self.geometry.top_anchored:
            self.show_from_menu()
            return  # Docked positions stay anchored; resizing and actions still work.
        if self._debug_input:
            print("[caption-overlay] mouseDown: no icon hit -> drag armed")
        self._drag_base = (mouse[0], mouse[1], self.geometry.cx_frac, self.geometry.bottom_px)
        self._dragging = True

    def _copy_box(self, box, source):
        """Copy one hit-tested row and start the shared checkmark feedback."""
        if self._copy_fn(box.text):
            self._copied = (box.text, time.monotonic() + COPY_FEEDBACK_SEC)
            if self._debug_input:
                prefix = "" if source == "mouseDown" else f"{source}: "
                print(f"[caption-overlay] {prefix}copied: {box.text}")
            return True
        if self._debug_input:
            prefix = "" if source == "mouseDown" else f"{source}: "
            print(f"[caption-overlay] {prefix}copy FAILED for: {box.text}")
        return False

    def _start_translate(self, box, source):
        """Kick off one row's translation on a background worker and put its
        icon into the "pending" state. Returns False (changing nothing) when
        that exact text is already in flight -- a second click, or the dwell
        fallback firing on top of a click, must not pay for the same row
        twice or race two results into the clipboard.

        The worker ONLY calls _translate_fn and puts (text, result, error)
        on _translate_queue; everything with a side effect the user can see
        -- writing the clipboard, moving the icon out of "pending" -- happens
        back on the UI thread in _drain_translations. Exceptions are captured
        rather than raised: this can be reached from an AppKit mouseDown_
        callback, which must never see one propagate (same contract
        _copy_text_to_pasteboard already keeps)."""
        text = box.text
        if text in self._translate_inflight:
            if self._debug_input:
                print(f"[caption-overlay] {source}: translation already in flight for: {text}")
            return False

        self._translate_inflight.add(text)
        self._translate = (text, "pending", None)
        if self._debug_input:
            print(f"[caption-overlay] {source}: translating: {text}")

        def worker():
            try:
                self._translate_queue.put((text, self._translate_fn(text), None))
            except Exception as exc:      # noqa: BLE001 -- reported via the queue, see above
                self._translate_queue.put((text, None, exc))

        self._run_async(worker)
        return True

    def _drain_translations(self, now):
        """Apply every finished translation waiting on _translate_queue: put
        the TRANSLATION (not the original) on the clipboard and start the
        icon's ~COPY_FEEDBACK_SEC checkmark, or fall to the "failed" icon.

        Failures print one warning line unconditionally -- NOT only under
        --debug-input, unlike most logging here: a translate click that
        silently does nothing (bad/absent API key, no credit, no network) is
        otherwise unexplainable from the panel alone, and this line is the
        only place the reason exists."""
        while True:
            try:
                text, result, error = self._translate_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break   # a poisoned queue must never crash the UI loop
            self._translate_inflight.discard(text)
            if error is not None:
                print(f"[caption-overlay] warning: translation failed: {error}")
                self._translate = (text, "failed", now + COPY_FEEDBACK_SEC)
            elif self._copy_fn(result):
                self._translate = (text, "done", now + COPY_FEEDBACK_SEC)
                if self._debug_input:
                    print(f"[caption-overlay] translated -> clipboard: {result}")
            else:
                print("[caption-overlay] warning: the translation could not be copied to the clipboard")
                self._translate = (text, "failed", now + COPY_FEEDBACK_SEC)

    def _install_global_mouse_monitor(self):
        """Copy through a click that macOS routes below this utility panel.

        Some non-activating NSPanel setups visually accept hover while the
        WindowServer still sends the subsequent click to the frontmost app.
        A global monitor observes precisely those *other-app* mouse events;
        ordinary clicks delivered to this process still use mouseDown_ above,
        so they are not handled twice.  The monitor deliberately acts only
        inside a currently rendered copy-icon hit box.
        """
        if getattr(self.renderer, "panel", None) is None:
            return  # headless test renderer
        try:
            import AppKit
            self._global_mouse_monitor = AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                AppKit.NSEventMaskLeftMouseDown, self._handle_global_mouse_down)
        except Exception as exc:
            # The normal NSPanel event path remains available; do not make a
            # best-effort fallback capable of breaking caption display.
            if self._debug_input:
                print(f"[caption-overlay] global mouse monitor unavailable: {exc}")

    def _handle_global_mouse_down(self, event):
        """Handle an icon click observed by the global fallback monitor."""
        if getattr(self.renderer, "animating", False):
            return
        point = event.locationInWindow()  # global-monitor events have screen coordinates
        box = _hit_icon((point.x, point.y), self._row_boxes)
        if box is not None:
            self._copy_box(box, "global mouseDown")
            return
        translate_box = _hit_translate_icon((point.x, point.y), self._row_boxes)
        if translate_box is not None:
            self._start_translate(translate_box, "global mouseDown")

    def show_from_menu(self):
        """Restore the last caption and keep it shown until explicitly closed."""
        self._pinned = True
        self._dismissed = False
        self._last_state = None
        self.tick()

    def _dismiss(self):
        self._pinned = False
        self._dismissed = True
        self._last_state = None
        self._panel_visible = False
        self._hovering = False
        self._hover_row = None
        self._row_boxes = []
        if self.geometry.top_anchored:
            self.renderer.show([], (), self.geometry.font_size, self.geometry)
            self._panel_visible = True
        else:
            self.renderer.hide()

    def _resize_to(self, mouse):
        origin, (sx, sy), font_size, frame = self._resize_base
        dx, dy = mouse[0] - origin[0], mouse[1] - origin[1]
        # Project the pointer onto the corner diagonal for smooth scaling.
        w, h = frame[2:]
        scale = 1 + (sx * dx * w + sy * dy * h) / max(1, w * w + h * h)
        self.geometry.font_size = max(12.0, min(96.0, font_size * scale))
        self._settings_dirty_at = time.monotonic()

    def _handle_mouse_drag(self, event):
        if self._resize_base is not None:
            self._resize_to(self._click_point(event))
            self.tick()
            return
        if self._drag_base is None:
            return
        import AppKit
        # Resolved here, up front -- not only down at the reposition block
        # below where it's used again -- because _screen_for_panel needs it
        # immediately below. Assigning it only later, as this method used
        # to, makes `panel` local to the WHOLE function (Python's scoping
        # rule: a name assigned anywhere in a function body is local for
        # that entire body), which turned this very read into an
        # unconditional UnboundLocalError on every call -- drag-to-move was
        # completely dead.
        panel = getattr(self.renderer, "panel", None)
        screen = _screen_for_panel(panel, AppKit.NSScreen.mainScreen())
        if screen is None:
            return
        mouse = self._click_point(event)
        base_x, base_y, base_cx, base_bottom = self._drag_base
        dx = mouse[0] - base_x
        dy = mouse[1] - base_y   # screen coords are y-up: moving UP grows bottom_px
        visible = screen.visibleFrame()
        self.geometry.move_to(base_cx + dx / visible.size.width, base_bottom + dy, visible.size.height)

        # tick() only repaints every _POLL_INTERVAL_SEC (0.08s) -- imperceptible
        # for text/alpha changes, but a dragged panel visibly lagging the
        # cursor at that cadence would feel broken. Reposition the CURRENT
        # panel frame (size unchanged -- only a full _layout_stack repaint
        # resizes it) directly, right now, via _drag_reposition_origin --
        # the SAME text-width centering _compute_layout itself uses (see
        # that function's own docstring for the "panel jumps sideways after
        # a drag" regression this fixes), not frame.size.width directly, so
        # the two can never disagree; tick()'s next-cycle repaint then
        # reconciles everything else (a no-op here if nothing besides
        # position changed meanwhile). `panel` was already resolved at the
        # top of this method (needed early for _screen_for_panel).
        if panel is not None:
            frame = panel.frame()
            x, y = _drag_reposition_origin(visible, frame.size.width, frame.size.height,
                                            self.geometry.font_size, self.geometry)
            panel.setFrameOrigin_(AppKit.NSMakePoint(x, y))

        self._settings_dirty_at = time.monotonic()
        if self._debug_input:
            _log_debug_input(f"move cx_frac={self.geometry.cx_frac:.4f} bottom_px={self.geometry.bottom_px:.1f}")

    def _handle_mouse_up(self, event):
        """Only persist geometry when a drag was actually in progress -- an
        icon click (which never sets _dragging, see _handle_mouse_down)
        must not write the settings file even though AppKit still sends
        this view its mouseUp_ afterward."""
        was_dragging = self._dragging
        self._dragging = False
        self._drag_base = None
        self._resize_base = None
        if was_dragging and self.settings_path is not None:
            save_geometry(self.settings_path, self.geometry)
            self._settings_dirty_at = None

    def _update_hover(self):
        """Track whether the cursor is currently over the interactive panel. Only calls setIgnoresMouseEvents_ on an actual
        state change (never redundantly every tick). Never runs while a
        drag is in progress -- see this class's own docstring for why.
        Uses _mouse_location() (rather than reading AppKit.NSEvent.
        mouseLocation() directly) and the pure _point_in_rect (rather than
        AppKit.NSPointInRect) -- identical half-open contract (see
        _point_in_rect's own docstring), just without a second, redundant
        AppKit touch now that _mouse_location() already exists."""
        if self._dragging or getattr(self.renderer, "animating", False):
            return
        panel = getattr(self.renderer, "panel", None)
        if panel is None:  # fake renderer (tests) -- nothing to touch
            return
        frame = panel.frame()
        frame_tuple = (frame.origin.x, frame.origin.y, frame.size.width, frame.size.height)
        hovering = _point_in_rect(self._mouse_location(), frame_tuple)
        if hovering != self._hovering:
            self._hovering = hovering
            if hovering and self.geometry.top_anchored:
                self._dismissed = False
            panel.setIgnoresMouseEvents_(False)
            if hovering:
                # Being merely eligible to become key is insufficient:
                # this process normally runs as an accessory app, so macOS
                # otherwise continues routing the next physical click to
                # the app behind the overlay.  Make this *small panel* key
                # as soon as the cursor enters it.  The custom panel class
                # cannot become the main window, so this does not turn the
                # caption overlay into the application's document window.
                panel.makeKeyAndOrderFront_(None)
            if self._debug_input:
                _log_debug_input(
                    f"hovering -> {hovering} panel_frame="
                    f"({frame.origin.x:.1f}, {frame.origin.y:.1f}, {frame.size.width:.1f}, {frame.size.height:.1f})")
                if hovering:
                    print(f"[caption-overlay] panel key-window: {bool(panel.isKeyWindow())}")

    def _log_repaint_debug(self, rows, decor):
        """--debug-input instrumentation, called from tick() only on a
        repaint that carries a copy icon (decor.hover_row is not None) --
        NOT every text-change repaint, which would be far too noisy during
        active speech (a new word chunk repaints roughly every 0.15s).
        Logs the SAME icon_rect shape _handle_mouse_down's own
        "mouseDown:" line does (from self._row_boxes, just set by the
        caller from this exact repaint) so the two can be read side by
        side: if a row's icon_rect here (at PAINT time) differs from the
        icon_rect logged at CLICK time, the row's geometry moved between
        the two -- e.g. the live row below the hovered one is still
        growing and pushing the whole stack upward (see _compute_layout's
        bottom-up stacking) -- a real hands-on session with
        CAPTION_OVERLAY_DEBUG_INPUT=1 can now show this directly instead
        of it staying a theory. `panel` prefers the real renderer's
        panel.frame() (production); falls back to the bounding box of
        self._row_boxes' frames (fake-renderer tests, or no real panel,
        prefixed "~" since it is only an approximation); None if
        neither is available."""
        icon_rect = next((b.icon for b in self._row_boxes if b.index == decor.hover_row), None)
        panel = getattr(self.renderer, "panel", None)
        if panel is not None:
            frame = panel.frame()
            panel_desc = (f"({frame.origin.x:.1f}, {frame.origin.y:.1f}, "
                           f"{frame.size.width:.1f}, {frame.size.height:.1f})")
        elif self._row_boxes:
            xs0 = [b.frame[0] for b in self._row_boxes]
            ys0 = [b.frame[1] for b in self._row_boxes]
            xs1 = [b.frame[0] + b.frame[2] for b in self._row_boxes]
            ys1 = [b.frame[1] + b.frame[3] for b in self._row_boxes]
            panel_desc = (f"~({min(xs0):.1f}, {min(ys0):.1f}, "
                           f"{max(xs1) - min(xs0):.1f}, {max(ys1) - min(ys0):.1f})")
        else:
            panel_desc = None
        print(f"[caption-overlay] repaint: rows={len(rows)} hover_row={decor.hover_row} "
              f"icon_rect={icon_rect} panel={panel_desc}")

    def tick(self, now=None):
        """Drain every pending display event (never blocks: get_nowait until
        empty), repaint only if the visible stack, geometry, hover row, or
        copy-feedback actually changed, then (only while the panel IS
        visible) update hover/click-through state, then flush a debounced
        settings save if one is pending. Safe to call at any cadence; the
        caller owns the timer."""
        if now is None:
            now = time.monotonic()
        q = self.display_queue
        if q is not None:
            while True:
                try:
                    event = q.get_nowait()
                except queue.Empty:
                    break
                except Exception:
                    break  # a closed/poisoned queue must never crash the UI loop
                self.model.apply(event, now)
                if event[0] == "chunk" and event[1].strip():
                    self._dismissed = False

        # Read the current pointer before deciding whether to hide: entering
        # just as the idle timeout expires must still keep the panel visible.
        hover_updated = self._panel_visible
        if hover_updated:
            self._update_hover()

        over_controls = False
        control_frame = getattr(self.renderer, "control_frame", None)
        if control_frame is not None:
            close_rect, corners = _overlay_controls(control_frame)
            point = self._mouse_location()
            over_controls = any(_point_in_rect(point, rect) for rect in [close_rect, *corners.values()])
            close_hovered = _point_in_rect(point, close_rect)
            if (not self.geometry.top_anchored and self._panel_visible
                    and not self._dragging and close_hovered and not self._close_hovered):
                self._dismiss()
            # Edge-triggered: a cursor left on × must not hide new captions again.
            self._close_hovered = close_hovered

        if not over_controls and not self._dragging:
            self._expanded = self._hovering
        expanded = self._expanded
        hold = self._pinned or self._hovering or self._dragging
        idle = not self.model._live_visible_at(now) and not hold
        if self._dismissed or idle:
            self._hovering = False
            self._expanded = False
            self._row_boxes = []
            rows = []
        else:
            rows = self.model.visible_rows(now, expanded=expanded, hold=hold)
            if self._pinned and not self.model._text:
                rows = [("音声認識を待っています…", True)]
            elif self.geometry.top_anchored and self._hovering and not rows:
                rows = [("音声認識を待っています…", True)]
        alphas = _alphas_for(rows)
        offset = self.model.offset_at(now)

        # Hover-row hit-testing: only while actually hovering (not dragging
        # -- the mouse is busy moving the panel, showing a copy icon then
        # would be pointless/confusing) and only against _row_boxes from
        # the MOST RECENT repaint (one-tick-stale by construction, same
        # <=_POLL_INTERVAL_SEC lag `expanded` above already has relative to
        # _update_hover -- see that method).
        if getattr(self.renderer, "animating", False):
            new_hover_row = self._hover_row
        elif self._hovering and not self._dragging:
            hit = _hit_row(self._mouse_location(), self._row_boxes)
            new_hover_row = hit.index if hit is not None else None
        else:
            new_hover_row = None
        if new_hover_row != self._hover_row:
            self._hover_row = new_hover_row
            if self._debug_input:
                print(f"[caption-overlay] hover-row: {self._hover_row}")

        # Dwell fallback: on some macOS/overlay combinations a screen-saver
        # level non-activating panel can observe the pointer but never gets
        # the physical click.  Pointer polling is reliable there, so resting
        # on the visible copy icon for half a second still performs the
        # requested copy.  A normal click remains immediate above.
        dwell_point = (self._mouse_location() if self._hovering
                       and not getattr(self.renderer, "animating", False) else None)
        icon_hit = _hit_icon(dwell_point, self._row_boxes) if dwell_point is not None else None
        if icon_hit is None:
            self._icon_hover_started_at = None
            self._icon_hover_copied_text = None
        elif icon_hit.text != self._icon_hover_copied_text:
            if self._icon_hover_started_at is None:
                self._icon_hover_started_at = now
            elif now - self._icon_hover_started_at >= HOVER_COPY_DELAY_SEC:
                if self._copy_box(icon_hit, "icon dwell"):
                    self._icon_hover_copied_text = icon_hit.text
                self._icon_hover_started_at = None

        # The translate icon gets the same dwell fallback, for the same
        # reason (a click the WindowServer never delivers), with the same
        # per-text latch -- doubly important here: without it, a cursor left
        # resting on the icon would fire a paid API call every
        # HOVER_COPY_DELAY_SEC.
        translate_hit = _hit_translate_icon(dwell_point, self._row_boxes) if dwell_point is not None else None
        if translate_hit is None:
            self._translate_hover_started_at = None
            self._translate_hover_done_text = None
        elif translate_hit.text != self._translate_hover_done_text:
            if self._translate_hover_started_at is None:
                self._translate_hover_started_at = now
            elif now - self._translate_hover_started_at >= HOVER_COPY_DELAY_SEC:
                self._start_translate(translate_hit, "icon dwell")
                self._translate_hover_done_text = translate_hit.text
                self._translate_hover_started_at = None

        # Copy feedback (checkmark) expiry: purely wall-clock, independent
        # of hover/drag state -- see _handle_mouse_down, which sets
        # self._copied using the real clock (it has no `now` of its own;
        # AppKit calls it as cb(event)).
        if self._copied is not None and now >= self._copied[1]:
            self._copied = None
        copied_text = self._copied[0] if self._copied is not None else None

        # Finished translations land here (the workers only ever touch the
        # queue -- see _start_translate), so the clipboard write and the
        # icon state change both happen on this, the UI, thread.
        self._drain_translations(now)
        if self._translate is not None and self._translate[2] is not None and now >= self._translate[2]:
            self._translate = None
        translate_text = self._translate[0] if self._translate is not None else None
        translate_state = self._translate[1] if self._translate is not None else None

        decor = RowDecor(hover_row=self._hover_row, copied_text=copied_text,
                          translate_text=translate_text, translate_state=translate_state)
        state = self._geometry_state((tuple(rows), alphas, offset, self._hover_row, copied_text,
                                       translate_text, translate_state))
        if state != self._last_state:
            self._last_state = state
            self._panel_visible = bool(rows) or self.geometry.top_anchored
            if self._panel_visible:
                boxes = self.renderer.show(rows, alphas, self.geometry.font_size, self.geometry,
                                   decor=decor, scroll_frac=offset - math.floor(offset))
                self._row_boxes = list(boxes) if boxes else []
                if self._debug_input and decor.hover_row is not None:
                    self._log_repaint_debug(rows, decor)
            else:
                self.renderer.hide()
                self._row_boxes = []

        if self._panel_visible and not hover_updated:
            self._update_hover()

        set_controls_visible = getattr(self.renderer, "set_controls_visible", None)
        if set_controls_visible is not None:
            set_controls_visible(self._panel_visible and (self._hovering or self._dragging))

        # Debounced settings save: 1s of no further drag/scale change (and
        # not mid-drag -- _handle_mouse_up already saves immediately when a
        # drag ends, so this branch is really for Cmd+scroll and any
        # drag-adjacent settle) flushes geometry to disk. A no-op whenever
        # nothing is pending (_settings_dirty_at is None) or persistence is
        # off (settings_path is None).
        if (self.settings_path is not None and self._settings_dirty_at is not None
                and not self._dragging and now - self._settings_dirty_at >= 1.0):
            save_geometry(self.settings_path, self.geometry)
            self._settings_dirty_at = None

    def close(self):
        """Hide the panel and flush any pending debounced save. Without this,
        a Cmd+scroll resize in the last (<1.0s) moment before teardown would
        be lost: _handle_mouse_up already saves drags immediately on
        mouse-up, but the scale path only marks _settings_dirty_at and
        otherwise waits for tick()'s debounce window -- which never arrives
        once the run loop stops. Called from both run_console_overlay's
        `finally` and live_avatar.cmd_menubar's `finally`, i.e. on every
        normal and Ctrl-C teardown path."""
        if self.settings_path is not None and self._settings_dirty_at is not None:
            save_geometry(self.settings_path, self.geometry)
            self._settings_dirty_at = None
        if self._global_mouse_monitor is not None:
            try:
                import AppKit
                AppKit.NSEvent.removeMonitor_(self._global_mouse_monitor)
            except Exception:
                pass
            self._global_mouse_monitor = None
        remove_status_item = getattr(self.renderer, "remove_status_item", None)
        if remove_status_item is not None:
            remove_status_item()
        stop_audio_meter = getattr(self.renderer, "stop_audio_meter", None)
        if stop_audio_meter is not None:
            stop_audio_meter()
        self.renderer.hide()


def _require_appkit():
    """Import AppKit + PyObjCTools.AppHelper, or sys.exit() with the same
    friendly install hint main() prints under --demo -- instead of letting a
    bare ImportError traceback escape into whatever is driving this module.
    Used by both public entry points below (attach_timer_based_overlay /
    run_console_overlay), since a caller like live_avatar.py's
    cmd_transcribe/cmd_menubar may run in a venv where only
    requirements-transcribe.txt was installed and requirements-menubar.txt
    never was.

    sys.exit(message) raises SystemExit, a BaseException (NOT an Exception),
    so it is not caught by an `except KeyboardInterrupt:` -- or any plain
    `except Exception:` -- anywhere upstream: it unwinds straight through
    cmd_transcribe's inner try/except KeyboardInterrupt, still runs the
    `with tap:` block's __exit__ and the outer `finally:
    _finalize_transcriber(...)` on the way out (a bare `finally` always runs
    for any exception, SystemExit included), and only then reaches the
    interpreter, which prints this message to stderr and exits 1 with no
    traceback -- exactly like every other `sys.exit("...")` guard elsewhere
    in this codebase (e.g. live_avatar._start_transcriber's mlx_whisper/
    rumps install hints).
    """
    try:
        import AppKit
        from PyObjCTools import AppHelper
    except ImportError:
        sys.exit(
            "[caption-overlay] AppKit is not available. Install with:\n"
            "  pip install 'alwayswhisper[live]'"
        )
    return AppKit, AppHelper


def _resolve_debug_input(debug_input):
    """Whether --debug-input-style logging should be on: the explicit
    `debug_input` argument, OR -- when it was NOT explicitly requested --
    the CAPTION_OVERLAY_DEBUG_INPUT=1 environment variable. Lets a user
    turn on the same diagnostic logging main()'s --demo --debug-input flag
    already prints (see _log_debug_input / _OverlayController.
    _handle_mouse_down / _log_repaint_debug) in a REAL --transcribe-only/
    --menubar session -- neither of which has a --debug-input CLI flag of
    its own (see live_avatar.py's --captions wiring) -- with no code
    change, e.g.:
        CAPTION_OVERLAY_DEBUG_INPUT=1 .venv/bin/python scripts/avatar/live_avatar.py \\
            --transcribe-only --captions
    Only the literal string "1" counts (not "true"/"yes"/etc -- keeps the
    check trivially simple and unambiguous in a shell one-liner). Pure (no
    AppKit) -- directly unit-testable; see attach_timer_based_overlay, the
    only caller (run_console_overlay delegates to it, so this covers both
    public entry points)."""
    return debug_input or os.environ.get("CAPTION_OVERLAY_DEBUG_INPUT") == "1"


def attach_timer_based_overlay(display_queue, font_size=_DEFAULT_FONT_SIZE,
                                settings_path=DEFAULT_SETTINGS_PATH, debug_input=False, position=None,
                                level_source=None):
    """Build the panel + model on the CURRENT thread (must be the main
    thread -- AppKit UI objects are not thread-safe to create elsewhere) and
    return a controller with .tick(now=None). Does not start any timer of
    its own: the caller drives .tick() on its own schedule (e.g.
    live_avatar.cmd_menubar wires a second rumps.Timer to it, alongside its
    existing 0.15s gate-slider refresh timer). `settings_path` is where
    panel position + font size are loaded from / persisted to (see
    OverlayGeometry/load_geometry/save_geometry); pass None to disable
    persistence entirely. `debug_input=True` makes the panel print one
    stdout line per input event/geometry change/click-through flip -- see
    _log_debug_input and main()'s --debug-input flag; silent by default.
    Also turned on by the CAPTION_OVERLAY_DEBUG_INPUT=1 environment
    variable when this argument itself is left False -- see
    _resolve_debug_input -- which also covers run_console_overlay, since
    it delegates to this function."""
    debug_input = _resolve_debug_input(debug_input)
    _require_appkit()
    if position is None and (settings_path is None or not os.path.exists(settings_path)):
        position = "dynamic-island"
    return _OverlayController(display_queue, font_size=font_size, settings_path=settings_path,
                               debug_input=debug_input, position=position, level_source=level_source)


def _dispatch_pending_events(appkit):
    """Drain native mouse/keyboard events without starving caption timers."""
    app = appkit.NSApplication.sharedApplication()
    for _ in range(64):
        event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
            appkit.NSEventMaskAny, appkit.NSDate.distantPast(),
            appkit.NSDefaultRunLoopMode, True)
        if event is None:
            break
        app.sendEvent_(event)
    # Mirror the window-update step normally supplied by NSApplication.run.
    # https://developer.apple.com/documentation/appkit/nsapplication/updatewindows()
    app.updateWindows()


def _run_interactive_event_loop(app_helper):
    """Service native input while retaining the console loop's clean SIGINT.

    AppHelper.runConsoleEventLoop handles timers but never sends NSEvents.
    Its runEventLoop alternative terminates the process on stop, bypassing
    transcription cleanup. Pump queued input explicitly in the console loop.
    """
    import AppKit
    AppKit.NSApplication.sharedApplication().finishLaunching()
    input_timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
        0.01, True, lambda _timer: _dispatch_pending_events(AppKit))
    try:
        app_helper.runConsoleEventLoop(installInterrupt=True)
    finally:
        input_timer.invalidate()


def run_console_overlay(display_queue, font_size=_DEFAULT_FONT_SIZE, settings_path=DEFAULT_SETTINGS_PATH,
                        position=None, level_source=None):
    """For a caller with no event loop of its own (live_avatar.cmd_transcribe
    / --transcribe-only): build the panel, drive it from a private NSTimer,
    and block the calling (main) thread in AppKit's console run loop until
    Ctrl-C.

    Explicitly dispatches queued mouseDown/mouseDragged/mouseUp events
    alongside the console timer loop. SIGINT stops the loop; raise KeyboardInterrupt after it returns so the
    transcription caller still runs its normal shutdown path.

    Also sets the process's activation policy to Accessory before building
    the panel -- required (on top of the panel-level fullscreen fixes in
    _make_panel) for the overlay to be able to join ANOTHER app's native
    fullscreen Space at all: a caller reaching this function (--transcribe-
    only) never does any NSApplication activation setup of its own, so
    without this it runs with AppKit's default Regular activation policy
    (confirmed via an on-machine probe: a freshly-created
    NSApplication.sharedApplication() reports activationPolicy() ==
    NSApplicationActivationPolicyRegular == 0). rcaelers/workrave's
    MacOSOverlayWindow.mm -- a mature, real-world break-reminder app whose
    entire purpose is overlaying above other apps' fullscreen Spaces --
    independently does exactly this: saves the current activation policy,
    switches Regular -> Accessory only while an overlay is shown, restores
    it after. This overlay has no such "only while shown" window (it can
    be shown/hidden many times over the process's life -- see
    CaptionLineModel's idle-hide/revive), so the policy is simply set once,
    up front, for the process's whole run -- console/--transcribe-only mode
    has no Dock icon or app-switcher presence to preserve anyway. NOT done
    in attach_timer_based_overlay: --menubar mode's rumps already runs
    accessory (LSUIElement-equivalent) and owns this policy itself; setting
    it again here too would be redundant at best.
    """
    AppKit, AppHelper = _require_appkit()

    AppKit.NSApplication.sharedApplication().setActivationPolicy_(
        AppKit.NSApplicationActivationPolicyAccessory
    )

    controller = attach_timer_based_overlay(display_queue, font_size=font_size,
                                            settings_path=settings_path, position=position,
                                            level_source=level_source)
    timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
        _POLL_INTERVAL_SEC, True, lambda _t: controller.tick()
    )
    try:
        _run_interactive_event_loop(AppHelper)
    finally:
        timer.invalidate()
        controller.close()
    raise KeyboardInterrupt


# ------------------------------------------------------------------- demo ---
def _demo_script():
    """A flat list of (delay_sec_before_this_event, event) pairs: word
    chunks ~0.15s apart within a line, an eos right after each line's last
    word, then a per-line pause (`pre_delay` below) before the next line's
    first word. ~20s total (including the trailing hold in _run_demo).

    Each line is (pre_delay, words) so the pause before it can be
    overridden per line -- normally 2.0s, except the gap before the closing
    "Ctrl-C..." line, which is 7.0s specifically: idle_clear_sec is 5.0s
    (_IDLE_CLEAR_SEC), so that gap makes the PREVIOUS line's caption
    actually idle-hide (real hide, not just a paused display) before the
    closing line arrives and revives the panel fresh. Every --demo run now
    exercises the full show -> idle-hide -> revive path end to end, not
    just show/wrap -- this was previously untested even by --demo (see the
    "stays visible after 8+ seconds of silence" investigation: the longest
    gap here used to be 2.0s, well under idle_clear_sec, so the hide
    transition had never actually run through the real AppKit timer at all
    before this).

    "Hello." is the first line specifically as a short-text regression
    check: it must render COMPLETELY (no ellipsis, no wrapping), the panel
    hugging its text tightly -- the "which" -> "…hich" / "Hello." ->
    "…ello." truncation reports turned out to be caused by _layout_and_show
    sizing frames from a bare NSAttributedString measurement that
    undercounts NSTextFieldCell's own internal padding (now fixed to
    measure through the cell itself, see _layout_and_show), nothing to do
    with text length or the (separately removed) model char-trim.

    One long English line is included to make wrapping visually verifiable:
    it is wider than the panel's max width at the default font size, so it
    must wrap to multiple lines and grow the panel upward -- see
    _layout_stack -- rather than getting trimmed (the old rolling-trim
    behavior this demo showed before the "show the whole caption" change).

    Several short lines in a row (before the long one) are included so
    history visibly accumulates: past HISTORY_WINDOW (2) fresh lines,
    older ones stack above the live line, progressively more translucent
    with age (see _alphas_for) -- this demo now feeds enough distinct
    lines to fill and exceed that window, so the fade is visible, not just
    theoretical.

    Word tokens after the first in each line carry a leading space, the
    same shape live_transcriber's word-mode chunks do (see
    live_transcriber._transcribe_and_write / extract_timed_words), so
    concatenating them reproduces normal spacing with no separator logic
    needed here.
    """
    lines = [
        (0.0, ["Hello."]),
        (1.5, ["これは", "ライブ", "字幕", "の", "デモ", "です。"]),
        (1.5, ["画面", "下", "中央", "に", "一行", "ずつ", "表示", "されます。"]),
        (1.5, ["過去の", "行は", "上に", "積み重なり", "ます。"]),
        (1.5, ["古い", "行ほど", "薄く", "表示", "されます。"]),
        (1.5, ["パネルに", "カーソルを", "乗せると", "スクロールできます。"]),
        (1.5, ["So", " this", " is", " connected", " to", " the", " MCP", " server,",
               " which", " means", " an", " AI", " agent", " workflow", " that",
               " can", " read", " and", " write", " your", " workspace."]),
        (7.0, ["Ctrl-C", "で", "終了", "します。"]),  # >idle_clear_sec gap -> hide, then revive
    ]
    script = []
    for li, (pre_delay, words) in enumerate(lines):
        for wi, word in enumerate(words):
            if wi == 0:
                delay = pre_delay if li > 0 else 0.0   # very first word of the whole demo: no wait
            else:
                delay = 0.15    # steady word-by-word pacing within a line
            script.append((delay, ("chunk", word)))
        script.append((0.0, ("eos",)))
    return script


def _run_demo(font_size, hold_sec=0.0, debug_input=False, position=None):
    import AppKit
    from PyObjCTools import AppHelper

    print("[caption-overlay] --demo: showing a live-caption preview for ~25s "
          "(at the selected/saved position; several short lines feeding a "
          "history stack -- hover the panel to see it -- a long wrapping line, "
          "and a >5s silent gap so idle-hide-then-revive runs too) ...")
    # Uses the real DEFAULT_SETTINGS_PATH on purpose (attach_timer_based_overlay's
    # own default -- not overridden here): dragging/resizing the panel during
    # --demo tunes the SAME settings file real --captions runs read, so a
    # placement/size picked here carries over.
    print(f"[caption-overlay] --demo: position/size persist to {DEFAULT_SETTINGS_PATH}")
    if debug_input:
        print("[caption-overlay] --demo: --debug-input on -- logging every container "
              "mouse/scroll event received, ignoresMouseEvents flip, and geometry change.")
    display_queue = queue.Queue()
    controller = attach_timer_based_overlay(display_queue, font_size=font_size,
                                            debug_input=debug_input, position=position)

    # Log every real show()/hide() transition to the terminal. This is the
    # only way to CONFIRM the idle-hide-then-revive path (the 7.0s gap in
    # _demo_script) actually fired through the real AppKit timer/run loop --
    # a screenshot or eyeballing the screen isn't always available (e.g. a
    # sandboxed session with no Screen Recording permission), so don't rely
    # on it being the only evidence.
    real_show, real_hide = controller.renderer.show, controller.renderer.hide

    def _show_and_log(rows, alphas, font_size, geometry, decor=None, scroll_frac=0.0):
        live_text = rows[-1][0] if rows else ""
        preview = live_text if len(live_text) <= 60 else live_text[:57] + "..."
        print(f"[caption-overlay] --demo: SHOW {len(rows)} row(s), live={preview!r}, "
              f"alphas={tuple(round(a, 2) for a in alphas)}, font_size={font_size:.1f}, "
              f"cx_frac={geometry.cx_frac:.3f}, bottom_px={geometry.bottom_px:.1f}")
        return real_show(rows, alphas, font_size, geometry, decor=decor, scroll_frac=scroll_frac)

    def _hide_and_log():
        # Fires both from tick()'s idle-timeout branch (mid-script -- see
        # the 7.0s gap in _demo_script) and from close() at teardown; both
        # are legitimate "should be hidden now" moments, so this message
        # doesn't claim which one -- the mid-script occurrence is what
        # proves the idle-hide path actually ran.
        print("[caption-overlay] --demo: HIDE")
        real_hide()

    controller.renderer.show = _show_and_log
    controller.renderer.hide = _hide_and_log

    timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
        _POLL_INTERVAL_SEC, True, lambda _t: controller.tick()
    )
    script = _demo_script()

    def feed(i):
        if i >= len(script):
            if hold_sec > 0:
                print(f"[caption-overlay] --demo: scripted lines done -- holding {hold_sec:.0f}s "
                      "so you can try the interactions by hand: hover the panel to expand "
                      "history (collapses back to just the live line the instant the cursor "
                      "leaves), plain scroll while hovering to scroll through history (older "
                      "is down, like Messages; auto-returns to live 10s after you stop scrolling), Cmd+scroll "
                      "to resize the caption font, and drag the panel to move it anywhere on "
                      "screen. If you don't touch it, it idle-hides on its own after 5s -- "
                      "that's correct behavior, not a bug. Ctrl-C to exit early.")
            AppHelper.callLater(hold_sec if hold_sec > 0 else 2.0, finish)
            return
        delay, event = script[i]

        def put_and_advance():
            display_queue.put_nowait(event)
            feed(i + 1)

        AppHelper.callLater(delay, put_and_advance)

    def finish():
        timer.invalidate()
        controller.close()
        print("[caption-overlay] --demo: done.")
        AppHelper.stopEventLoop()

    AppHelper.callLater(0.0, lambda: feed(0))
    _run_interactive_event_loop(AppHelper)


def main():
    parser = argparse.ArgumentParser(
        description="macOS live-caption overlay (normally driven by live_avatar.py --captions)."
    )
    parser.add_argument("--demo", action="store_true",
                         help="preview the overlay with canned lines (short lines building a "
                              "history stack, a wrapping-long line, and a 5s+ silent gap to show "
                              "idle-hide), no mic/whisper needed; exits automatically after ~25s")
    parser.add_argument("--hold", type=float, default=0.0, metavar="SEC",
                         help="after the scripted demo lines finish, keep the overlay process "
                              "alive for SEC more seconds (default 0 = exit ~2s after the last "
                              "line) so you can hover over the panel and scroll through history "
                              "by hand -- e.g. --demo --hold 30")
    parser.add_argument("--font-size", type=int, default=_DEFAULT_FONT_SIZE,
                         help=f"caption font size in points (default {_DEFAULT_FONT_SIZE})")
    parser.add_argument("--position", choices=CAPTION_POSITIONS, default=None,
                         help="caption placement (default: saved position, initially dynamic-island; free for draggable captions)")
    parser.add_argument("--debug-input", action="store_true",
                         help="log one stdout line per container scrollWheel_/mouseDown_/"
                              "mouseDragged_/mouseUp_ actually received (with cursor screen "
                              "coords), each ignoresMouseEvents click-through flip, and each "
                              "geometry change (drag move / Cmd+scroll resize) -- for diagnosing "
                              "drag/scroll/hover input delivery by hand; --demo only, silent "
                              "otherwise")
    args = parser.parse_args()

    try:
        import AppKit  # noqa: F401
        from PyObjCTools import AppHelper  # noqa: F401
    except ImportError:
        print("[caption-overlay] AppKit is not available. Install with:\n"
              "  pip install 'alwayswhisper[live]'")
        sys.exit(1)

    if args.demo:
        _run_demo(args.font_size, hold_sec=args.hold, debug_input=args.debug_input, position=args.position)
        return

    print("[caption-overlay] nothing to do without --demo. This module is normally driven by "
          "live_avatar.py --captions; try: python caption_overlay.py --demo")


if __name__ == "__main__":
    main()
