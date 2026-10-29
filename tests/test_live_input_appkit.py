"""Opt-in native check using a temporary text field in a child process."""

import os
from pathlib import Path
import queue
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("RUN_CAPTION_APPKIT_TESTS") != "1",
    reason="requires opt-in and a macOS WindowServer",
)
def test_accessibility_focus_unicode_input_and_safe_correction():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from alwayswhisper.live.live_input import InsertedSpan, LiveInputSession, MacTextInput, _utf16_length
    import AppKit
    from alwayswhisper.live import caption_overlay as co

    previous_app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    controller = None

    child = r'''
import AppKit, os, sys, objc, Quartz
class TestApplication(AppKit.NSApplication):
    def sendEvent_(self, event):
        # An already-running dictation app can target this temporary field too.
        # Accept keyboard events only from the test process, never other apps.
        if event.type() in (AppKit.NSEventTypeKeyDown, AppKit.NSEventTypeKeyUp):
            cg_event = event.CGEvent()
            if cg_event is not None and Quartz.CGEventGetIntegerValueField(
                    cg_event, Quartz.kCGEventSourceUnixProcessID) != int(sys.argv[1]):
                return
        objc.super(TestApplication, self).sendEvent_(event)
app = TestApplication.sharedApplication()
app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
    AppKit.NSMakeRect(200, 200, 400, 100), AppKit.NSWindowStyleMaskTitled,
    AppKit.NSBackingStoreBuffered, False)
field = AppKit.NSTextField.alloc().initWithFrame_(AppKit.NSMakeRect(20, 35, 360, 28))
window.contentView().addSubview_(field)
window.makeKeyAndOrderFront_(None)
app.activateIgnoringOtherApps_(True)
window.makeFirstResponder_(field)
print("READY", flush=True)
app.run()
'''
    process = subprocess.Popen([sys.executable, "-c", child, str(os.getpid())], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, bufsize=1)
    try:
        assert process.stdout.readline().strip() == "READY"
        writer = MacTextInput()
        focused_target = writer.focused_target
        def isolated_target():
            candidate = focused_target()
            return candidate if candidate is not None and candidate.pid == process.pid else None
        writer.focused_target = isolated_target
        target = None
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and target is None:
            target = writer.focused_target()
            if target is not None and target.pid != process.pid:
                target = None
            if target is None:
                AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
                    AppKit.NSDate.dateWithTimeIntervalSinceNow_(.05))
        assert target is not None
        assert writer.snapshot(target) == ("", (0, 0))

        def wait_text(text):
            end = time.monotonic() + 2
            expected = (text, (_utf16_length(text), 0))
            while time.monotonic() < end and writer.snapshot(target) != expected:
                AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
                    AppKit.NSDate.dateWithTimeIntervalSinceNow_(.02))
            assert writer.snapshot(target) == expected

        assert writer.type_text(target, "Hello 日本語😀")
        wait_text("Hello 日本語😀")
        assert writer.replace_if_unchanged(InsertedSpan(target, 0, "Hello 日本語😀"), "修正しました")
        wait_text("修正しました")

        session = LiveInputSession(lambda: writer)
        assert session.toggle()
        session.feed(("chunk", "続き"), fresh=True, current_text="続き")
        session.flush()
        wait_text("修正しました続き")
        session.feed(("replace", "追記"), current_text="追記")
        session.flush()
        wait_text("修正しました追記")

        # Exercise the actual island-to-field route: hover makes the panel key,
        # Fn must return input to this temporary native field.
        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        session.toggle()
        q = queue.Queue()
        controller = co._OverlayController(q, settings_path=None,
                                           position="dynamic-island", live_input_session=session)
        controller._mouse_location = lambda: (-99999, -99999)
        controller.tick()

        def pump(seconds):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                co._dispatch_pending_events(AppKit)
                AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
                    AppKit.NSDate.dateWithTimeIntervalSinceNow_(.01))

        def press_fn():
            for flags in (AppKit.NSEventModifierFlagFunction, 0):
                event = AppKit.NSEvent.keyEventWithType_location_modifierFlags_timestamp_windowNumber_context_characters_charactersIgnoringModifiers_isARepeat_keyCode_(
                    AppKit.NSEventTypeFlagsChanged, AppKit.NSMakePoint(0, 0), flags,
                    time.monotonic(), controller.renderer.panel.windowNumber(), None, "", "", False, 63)
                app.postEvent_atStart_(event, False)
            pump(.1)

        panel_frame = controller.renderer.panel.frame()
        rect = co._island_live_input_rect(
            (panel_frame.origin.x, panel_frame.origin.y, panel_frame.size.width, panel_frame.size.height),
            controller.renderer.island.anchor)
        assert controller.renderer.live_input_frame is None  # pencil has been removed
        # Establish the isolated field as the active app before exercising the
        # panel's temporary focus and return path (AppKit was initialized above).
        AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(process.pid).activateWithOptions_(
            AppKit.NSApplicationActivateIgnoringOtherApps)
        pump(.1)
        assert writer.focused_target() == target
        controller._mouse_location = lambda: (rect[0] + 10, rect[1] + 10)
        controller.tick()
        pump(.3)
        controller.tick()
        press_fn()
        assert session.active
        assert not controller.renderer.panel.isKeyWindow()
        assert writer.focused_target() == target, (
            "frontmost pid", AppKit.NSWorkspace.sharedWorkspace().frontmostApplication().processIdentifier(),
            "test pid", os.getpid(), "field pid", process.pid)
        def press_live_shortcut(key="l", flags=0):
            panel = controller.renderer.panel
            event = AppKit.NSEvent.keyEventWithType_location_modifierFlags_timestamp_windowNumber_context_characters_charactersIgnoringModifiers_isARepeat_keyCode_(
                AppKit.NSEventTypeKeyDown, AppKit.NSMakePoint(0, 0), flags, time.monotonic(),
                panel.windowNumber(), None, key, key, False, 37)
            app.postEvent_atStart_(event, False)
            pump(.1)
        controller.show_from_menu()
        pump(.3)
        assert controller.renderer.panel.isKeyWindow()
        press_live_shortcut()
        assert not session.active
        press_live_shortcut("L", AppKit.NSEventModifierFlagShift)
        assert session.active and not controller.renderer.panel.isKeyWindow()
        assert writer.focused_target() == target
        q.put(("chunk", "最初の\n文。", 1))
        q.put(("eos",))
        controller.tick()
        pump(.3)
        assert writer.snapshot(target)[0] == "修正しました追記最初の文。"
        controller._dismiss()
        pump(.3)
        assert session.active
        assert writer.focused_target() == target, "focus changed after dismissing the island"
        q.put(("chunk", "次の\r\n文。", 2))
        q.put(("eos",))
        controller.tick()
        pump(.4)
        assert writer.snapshot(target)[0] == "修正しました追記最初の文。次の文。", (
            session.error, writer.focused_target() == target,
            controller.renderer.panel.isKeyWindow())
        # Both translation directions use the real native insertion path.
        workers = []
        controller._async_translation_enabled = True
        controller._run_async = workers.append
        controller._auto_translate_fn = {
            "Let's start the meeting.": "会議を\n始めましょう。",
            "今日は晴れです。": "It is\nsunny today.",
        }.__getitem__
        expected = "修正しました追記最初の文。次の文。"
        for language, source, translated, utterance in (
            ("en-ja", "Let's start the meeting.", "会議を始めましょう。", 4),
            ("ja-en", "今日は晴れです。", "It is sunny today.", 5),
        ):
            controller._apply_language(language)
            q.put(("chunk", source, utterance)); q.put(("eos",))
            controller.tick(); pump(.1)
            assert writer.snapshot(target)[0] == expected  # no source preview entered
            # Translation is asynchronous: establish the intended caret again
            # before completing it, without relying on desktop focus staying idle.
            AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(process.pid).activateWithOptions_(
                AppKit.NSApplicationActivateIgnoringOtherApps)
            pump(.1)
            assert writer.focused_target() == target
            workers.pop(0)()
            controller.tick(); pump(.2)
            expected += translated
            wait_text(expected)
        controller._mouse_location = lambda: (rect[0] + 10, rect[1] + 10)
        controller.tick(); pump(.3); controller.tick()
        press_fn()
        assert not session.active
        q.put(("chunk", "オフの間は入力しない", 3))
        controller.tick()
        pump(.2)
        assert writer.snapshot(target)[0] == expected
    finally:
        if controller is not None:
            controller.close()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if previous_app is not None:
            previous_app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps)
