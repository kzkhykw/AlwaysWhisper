"""Opt-in native event dispatch test (opens a temporary macOS caption panel).

RUN_CAPTION_APPKIT_TESTS=1 .venv/bin/python -m pytest tests/test_caption_overlay_appkit.py
Uses posted NSEvents, without moving the user's mouse or writing their settings.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("RUN_CAPTION_APPKIT_TESTS") != "1",
    reason="requires opt-in and a macOS WindowServer",
)
def test_native_drag_resize_and_sigint_cleanup():
    script = r'''
import os, signal, sys, time, queue, traceback
sys.path.insert(0, "src")
import AppKit
from PyObjCTools import AppHelper
from alwayswhisper.live import caption_overlay as co
app = AppKit.NSApplication.sharedApplication()
app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
q = queue.Queue()
q.put(("chunk", "字幕全体をドラッグで移動"))
c = co._OverlayController(q, settings_path=None)
c._mouse_location = lambda: (-9999, -9999)
c.tick()
p = c.renderer.panel
status_item = c.renderer.status_item
def control_pixels():
    content = c.renderer.content
    bitmap = content.bitmapImageRepForCachingDisplayInRect_(content.bounds())
    content.cacheDisplayInRect_toBitmapImageRep_(content.bounds(), bitmap)
    scale = bitmap.pixelsWide() / content.bounds().size.width
    counts = []
    for control in content.control_views:
        f = control.frame()
        top = content.bounds().size.height - f.origin.y - f.size.height
        white = dark = 0
        for x in range(int(f.origin.x * scale), int((f.origin.x + f.size.width) * scale)):
            for y in range(int(top * scale), int((top + f.size.height) * scale)):
                color = bitmap.colorAtX_y_(x, y).colorUsingColorSpace_(AppKit.NSColorSpace.genericRGBColorSpace())
                if color.alphaComponent() > .5:
                    white += color.redComponent() > .9
                    dark += color.redComponent() < .1
        counts.append((white, dark))
    return counts

def check_hover_and_caption_lifecycle():
    assert control_pixels() == [(0, 0), (0, 0)]
    f = p.frame()
    # Hover transparent header space: no row/icon or text change to force repaint.
    c._mouse_location = lambda: (f.origin.x + 100, f.origin.y + f.size.height - 14)
    c.tick()
    assert c._hover_row is None
    assert all(white > 20 and dark > 20 for white, dark in control_pixels())
    # Hovering the actual caption must keep both controls visible through a rebuild.
    c._mouse_location = lambda: (f.origin.x + 100, f.origin.y + 20)
    c.tick()
    q.put(("chunk", "できます"))
    c.tick()
    assert all(white > 20 and dark > 20 for white, dark in control_pixels())
    # A pause cannot remove a hovered caption. Closing stays explicit and temporary.
    c.model._last_event_at -= 30
    c.tick()
    assert c._panel_visible
    f = p.frame()
    c._mouse_location = lambda: (f.origin.x + 14, f.origin.y + f.size.height - 14)
    c.tick()
    assert not c._panel_visible
    q.put(("eos",))
    c.tick()
    assert not c._panel_visible
    q.put(("chunk", "字幕全体をドラッグで移動"))
    c.tick()
    c.tick()
    assert c._panel_visible, "pointer left on close must not hide new recognition"
    c._mouse_location = lambda: (-9999, -9999)
    c.tick()
    assert control_pixels() == [(0, 0), (0, 0)]
    # Exercise the real NSStatusBar button's Objective-C target/action route.
    button = status_item.item.button()
    assert button.image() is not None or button.title() == "字幕"
    c._dismiss()
    assert status_item.item is not None
    button.performClick_(None)
    assert c._pinned and c._panel_visible and p.isVisible()
    c.model._last_event_at -= 60
    c.tick()
    assert c._panel_visible and p.isVisible()
    c._dismiss()
    assert not c._pinned
    button.performClick_(None)
    assert c._pinned and c._panel_visible
    c._dismiss()
    q.put(("chunk", "字幕全体をドラッグで移動"))
    q.put(("eos",))
    c.tick()
    assert c._panel_visible and not c._pinned
    content = c.renderer.content
    bitmap = content.bitmapImageRepForCachingDisplayInRect_(content.bounds())
    content.cacheDisplayInRect_toBitmapImageRep_(content.bounds(), bitmap)
    assert bitmap.colorAtX_y_(2, 2).alphaComponent() < .005
    # Check the rendered subtitle too, not only the transparent outer area.
    chip = content.subviews()[0]
    chip_bitmap = chip.bitmapImageRepForCachingDisplayInRect_(chip.bounds())
    chip.cacheDisplayInRect_toBitmapImageRep_(chip.bounds(), chip_bitmap)
    white_pixels = dark_pixels = 0
    for x in range(0, chip_bitmap.pixelsWide(), 3):
        for y in range(0, chip_bitmap.pixelsHigh(), 3):
            color = chip_bitmap.colorAtX_y_(x, y).colorUsingColorSpace_(AppKit.NSColorSpace.genericRGBColorSpace())
            if color.alphaComponent() > .5:
                white_pixels += color.redComponent() > .9
                dark_pixels += color.redComponent() < .1
    assert white_pixels > 10, "subtitle text is missing"
    assert dark_pixels > 10, "subtitle's black background is missing"

initial = c.geometry.cx_frac
failures, completed = [], []
timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(.03, True, lambda _: c.tick())

def post(kind, x, y):
    event = AppKit.NSEvent.mouseEventWithType_location_modifierFlags_timestamp_windowNumber_context_eventNumber_clickCount_pressure_(
        kind, AppKit.NSMakePoint(x, y), 0, time.monotonic(), p.windowNumber(), None, 1, 1, 1.0)
    app.postEvent_atStart_(event, False)

def checked(fn):
    try:
        fn()
    except BaseException:
        failures.append(traceback.format_exc())
        AppHelper.stopEventLoop()


def resize_down():
    f = p.frame()
    post(AppKit.NSEventTypeLeftMouseDown, c.renderer.control_frame[2] - 14, f.size.height - 14)


def resize_drag():
    f = p.frame()
    post(AppKit.NSEventTypeLeftMouseDragged, f.size.width + 80, f.size.height + 20)


steps = [
    (lambda: post(AppKit.NSEventTypeLeftMouseDown, 100, 25), lambda: c._dragging, "move down"),
    (lambda: post(AppKit.NSEventTypeLeftMouseDragged, 180, 55), lambda: c.geometry.cx_frac > initial, "move drag"),
    (lambda: post(AppKit.NSEventTypeLeftMouseUp, 100, 25), lambda: not c._dragging, "move up"),
    (resize_down, lambda: c._resize_base is not None, "resize down"),
    (resize_drag, lambda: c.geometry.font_size > 28, "resize drag"),
    (lambda: post(AppKit.NSEventTypeLeftMouseUp, 100, 25), lambda: not c._dragging, "resize up"),
]


def run_step(index):
    if index == len(steps):
        completed.append(True)
        os.kill(os.getpid(), signal.SIGINT)
        return
    action, ready, label = steps[index]
    action()
    deadline = time.monotonic() + 2
    def verify():
        if ready():
            AppHelper.callLater(.03, lambda: checked(lambda: run_step(index + 1)))
        else:
            assert time.monotonic() < deadline, "native event not delivered: " + label
            AppHelper.callLater(.03, lambda: checked(verify))
    AppHelper.callLater(.03, lambda: checked(verify))


def start_checks():
    checked(check_hover_and_caption_lifecycle)
    if not failures:
        run_step(0)
AppHelper.callLater(.05, start_checks)
AppHelper.callLater(7, AppHelper.stopEventLoop)
try:
    co._run_interactive_event_loop(AppHelper)
finally:
    timer.invalidate()
    c.close()
assert not failures, failures
assert completed, "native event sequence did not complete"
assert not p.isVisible()
assert status_item.item is None
assert status_item.target.on_show is None
print("NATIVE_INPUT_AND_CLEANUP_OK")
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NATIVE_INPUT_AND_CLEANUP_OK" in result.stdout, result.stdout + result.stderr


@pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("RUN_CAPTION_APPKIT_TESTS") != "1",
    reason="requires opt-in and a macOS WindowServer",
)
@pytest.mark.parametrize("position", ["notch", "dynamic-island"])
def test_native_island_morph_lifecycle_and_camera_clearance(tmp_path, position):
    script = r'''
import sys, queue, time
sys.path.insert(0, "src")
import AppKit
from alwayswhisper.live import caption_overlay as co
AppKit.NSApplication.sharedApplication().setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
q = queue.Queue()
c = co._OverlayController(q, settings_path=None, position=sys.argv[2])
c._mouse_location = lambda: (-99999, -99999)

def pump(seconds):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        co._dispatch_pending_events(AppKit)
        AppKit.NSRunLoop.currentRunLoop().runUntilDate_(AppKit.NSDate.dateWithTimeIntervalSinceNow_(.01))

def frame():
    f = c.renderer.panel.frame()
    return f.origin.x, f.origin.y, f.size.width, f.size.height

def capture(name):
    content = c.renderer.content
    bitmap = content.bitmapImageRepForCachingDisplayInRect_(content.bounds())
    content.cacheDisplayInRect_toBitmapImageRep_(content.bounds(), bitmap)
    data = bitmap.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {})
    assert data.writeToFile_atomically_(sys.argv[1] + "/" + name + ".png", True)

try:
    c.tick(now=0)
    surface = c.renderer.island
    assert surface.compact and c.renderer.panel.isVisible()
    anchor = surface.anchor
    compact = frame()
    assert abs(compact[1] + compact[3] - anchor.top) < 1
    capture("compact")
    q.put(("chunk", "字幕をこの島の中に表示します"))
    c.tick(now=1)
    assert not surface.compact
    target = surface.target
    assert target[3] > compact[3]
    # Exercise a real NSTimer morph, not only the interpolation helper.
    if surface.animating:
        pump(.08)
        middle = frame()
        assert compact[3] < middle[3] < target[3]
        assert abs(middle[1] + middle[3] - anchor.top) < 1
        assert c.renderer.control_frame is None  # no stale action targets in flight
    pump(.3)
    assert not surface.animating
    expanded = frame()
    assert abs(expanded[3] - target[3]) < 1
    assert abs(expanded[1] + expanded[3] - anchor.top) < 1
    live = c._row_boxes[-1]
    assert live.frame[1] + live.frame[3] <= anchor.top - anchor.camera_height
    # The label is centered INSIDE the full black surface, not shifted by a gutter.
    chip = surface.body_views[0][0]
    label = chip.subviews()[0]
    assert label.alignment() == AppKit.NSTextAlignmentCenter
    lf, cf = label.frame(), chip.frame()
    assert abs(cf.origin.x + lf.origin.x + lf.size.width / 2 - expanded[2] / 2) < 1
    label_bottom = expanded[1] + cf.origin.y + lf.origin.y
    label_top = label_bottom + lf.size.height
    assert abs((anchor.top - anchor.header_height - label_top) * anchor.scale - 2) < .01
    assert abs((label_bottom - expanded[1]) * anchor.scale - 4) < .01
    capture("expanded")
    # Hovering changes only the actions: neither the centered text rectangle
    # nor the wrapping/row geometry may move, including when history is shown.
    def text_frames():
        result = []
        px, py, _, _ = frame()
        for view, _ in surface.body_views:
            vf, lf = view.frame(), view.subviews()[0].frame()
            result.append((px + vf.origin.x + lf.origin.x,
                           py + vf.origin.y + lf.origin.y, lf.size.width, lf.size.height))
        return result

    for texts in [
        ["字幕をこの島の中に表示します"],
        ["長い字幕もホバー前後で同じ位置と改行を維持します。" * 4],
        ["前の字幕", "Long captions keep their position when hovering over the actions.", "最新の字幕"],
    ]:
        rows = [(text, i == len(texts) - 1) for i, text in enumerate(texts)]
        alphas = [1] * len(rows)
        c.renderer.show(rows, alphas, 28, c.geometry)
        pump(.3)
        base_frame, base_text_frames = frame(), text_frames()
        for hovered in range(len(rows)):
            boxes = c.renderer.show(rows, alphas, 28, c.geometry, co.RowDecor(hover_row=hovered))
            assert frame() == base_frame
            assert text_frames() == base_text_frames, (text_frames(), base_text_frames, frame(), surface.target)
            text_rect = text_frames()[hovered]
            box = boxes[hovered]
            assert box.icon is not None and box.translate_icon is not None
            assert text_rect[0] + text_rect[2] <= box.icon[0]
            # Visual icons and their screen-coordinate click targets agree.
            chip = surface.body_views[hovered][0]
            for view, hit in zip(list(chip.subviews())[1:], [box.icon, box.translate_icon]):
                vf, cf = view.frame(), chip.frame()
                icon_center = (frame()[0] + cf.origin.x + vf.origin.x + vf.size.width / 2,
                               frame()[1] + cf.origin.y + vf.origin.y + vf.size.height / 2)
                assert co._point_in_rect(icon_center, hit)
        c.renderer.show(rows, alphas, 28, c.geometry)
        assert text_frames() == base_text_frames
    c.tick(now=7)
    assert surface.compact
    pump(.3)
    assert frame() == compact
    assert c.renderer.panel.isVisible()
    # Hover the accessible wing: expands the last caption from the compact shell.
    c._mouse_location = lambda: (compact[0] + 20, compact[1] + compact[3] / 2)
    c.tick(now=8)
    assert not surface.compact
    pump(.3)
    assert c._row_boxes[-1].text == "字幕をこの島の中に表示します"
    c._dismiss()
    pump(.3)
    assert surface.compact and c.renderer.panel.isVisible()
    # Incoming speech interrupts collapse/expansion safely.
    c._mouse_location = lambda: (-99999, -99999)
    q.put(("chunk", "次の字幕"))
    c.tick(now=9)
    c._dismiss()
    assert surface.compact
    c.close()  # closes even mid-animation
    assert surface.timer is None
    assert not c.renderer.panel.isVisible()
    print("NATIVE_ISLAND_MORPH_OK", anchor)
finally:
    c.close()
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), position],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NATIVE_ISLAND_MORPH_OK" in result.stdout


@pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("RUN_CAPTION_APPKIT_TESTS") != "1",
    reason="requires opt-in and a macOS WindowServer",
)
def test_native_meter_tracks_audio_capture_without_rebuilding_captions(tmp_path):
    script = r'''
import sys, queue, time
sys.path.insert(0, "src")
import numpy as np
import AppKit
from alwayswhisper.live import live_transcriber as lt
from alwayswhisper.live import caption_overlay as co
AppKit.NSApplication.sharedApplication().setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
q = queue.Queue()
t = lt.LiveTranscriber(16000, "unused", "ja", sys.argv[1], display_queue=q)
# No mic, ASR process, or network: feed known PCM through the production capture path.
c = co._OverlayController(q, settings_path=None, position="dynamic-island", level_source=t.get_audio_level)
c._mouse_location = lambda: (-99999, -99999)
def pump(seconds):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        co._dispatch_pending_events(AppKit)
        AppKit.NSRunLoop.currentRunLoop().runUntilDate_(AppKit.NSDate.dateWithTimeIntervalSinceNow_(.01))
def set_volume(amplitude):
    t.feed(np.tile(np.array([amplitude, -amplitude], dtype=np.float32), 128))
    pump(.2)
    return c.renderer.island.right_status.level
try:
    c.tick()
    surface = c.renderer.island
    assert surface.compact
    assert not hasattr(surface, "left_status")
    idle = set_volume(0)
    quiet = set_volume(.004)
    loud = set_volume(.1)
    assert idle == 0 and idle < quiet < loud
    view = surface.right_status
    # Native drawing produces a different raster as the real RMS value changes.
    def pixels():
        bitmap = view.bitmapImageRepForCachingDisplayInRect_(view.bounds())
        view.cacheDisplayInRect_toBitmapImageRep_(view.bounds(), bitmap)
        return bytes(bitmap.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {}))
    loud_pixels = pixels()
    pump(1.2)  # no new audio: stale capture must fade down even without new captions
    assert view.level == 0
    assert loud_pixels != pixels()
    q.put(("chunk", "音量が動いても字幕は固定"))
    c.tick()
    pump(.3)
    chip = surface.body_views[0][0]
    meter_view = surface.right_status
    panel_frame = c.renderer.panel.frame()
    assert set_volume(.1) > .5
    assert surface.body_views[0][0] is chip
    assert surface.right_status is meter_view
    assert c.renderer.panel.frame() == panel_frame
    assert q.empty()
    timer = c.renderer.audio_timer
    c.close()
    assert not timer.isValid() and c.renderer.audio_timer is None
    assert not c.renderer.panel.isVisible()
    print("NATIVE_AUDIO_METER_OK")
finally:
    c.close()
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NATIVE_AUDIO_METER_OK" in result.stdout
