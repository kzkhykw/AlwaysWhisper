"""Fn gestures and AppKit wiring without observing or injecting user keystrokes."""
import os
import sys
from types import SimpleNamespace

import pytest

from alwayswhisper.live import fn_shortcut as fn


def event(kind=fn.FLAGS_CHANGED, code=fn.FN_KEY, flags=0, source_pid=None):
    # Deliberately no characters()/charactersIgnoringModifiers(): this shortcut
    # must never need the contents of what the user types in another app.
    return SimpleNamespace(type=lambda: kind, keyCode=lambda: code, modifierFlags=lambda: flags,
                           CGEvent=lambda: source_pid)


def test_bare_fn_toggles_on_release_once_and_can_repeat():
    gesture = fn.FnTap()
    assert not gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, 0)  # startup with Fn held
    for _ in range(2):
        assert not gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, fn.FN_FLAG)
        assert not gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, fn.FN_FLAG)
        assert gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, 0)
        assert not gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, 0)


@pytest.mark.parametrize("kind, code", [
    (fn.KEY_DOWN, 123),  # Fn + left arrow (Home)
    (fn.KEY_DOWN, 117),  # Fn + delete
    (fn.KEY_DOWN, 122),  # Fn + F1
    (fn.KEY_DOWN, 8),    # Fn + C (Control Center)
    (fn.KEY_UP, 0),     # another key was already held
    (fn.SYSTEM_DEFINED, None),  # media/brightness keys
    (fn.FLAGS_CHANGED, 56),     # Shift added after Fn
])
def test_fn_chords_do_not_toggle(kind, code):
    gesture = fn.FnTap()
    gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, fn.FN_FLAG)
    gesture.feed(kind, code, fn.FN_FLAG)
    assert not gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, 0)


@pytest.mark.parametrize("modifier", [1 << 17, 1 << 18, 1 << 19, 1 << 20])
def test_modifiers_already_held_cancel_fn_even_if_released_first(modifier):
    gesture = fn.FnTap()
    gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, fn.FN_FLAG | modifier)
    gesture.feed(fn.FLAGS_CHANGED, 56, fn.FN_FLAG)
    assert not gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, 0)


def test_function_flag_on_other_keys_is_not_physical_fn():
    gesture = fn.FnTap()
    for key in (122, 123, 124, 125, 126):
        assert not gesture.feed(fn.KEY_DOWN, key, fn.FN_FLAG)
        assert not gesture.feed(fn.KEY_UP, key, fn.FN_FLAG)
    assert not gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, 0)


def test_missed_release_cancels_stale_gesture_and_next_tap_works():
    gesture = fn.FnTap()
    gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, fn.FN_FLAG)
    gesture.feed(fn.KEY_DOWN, 0, 0)
    assert not gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, 0)
    gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, fn.FN_FLAG)
    assert gesture.feed(fn.FLAGS_CHANGED, fn.FN_KEY, 0)


def frameworks(monkeypatch, *, trusted=True, fail_local=False):
    installed, removed = {}, []
    def add(scope, mask, handler):
        if scope == "local" and fail_local:
            raise RuntimeError("registration failed")
        installed[scope] = (mask, handler)
        return scope
    appkit = SimpleNamespace(
        NSEventMaskFlagsChanged=1 << fn.FLAGS_CHANGED,
        NSEventMaskKeyDown=1 << fn.KEY_DOWN, NSEventMaskKeyUp=1 << fn.KEY_UP,
        NSEventMaskSystemDefined=1 << fn.SYSTEM_DEFINED,
        NSEvent=SimpleNamespace(
            addGlobalMonitorForEventsMatchingMask_handler_=lambda mask, cb: add("global", mask, cb),
            addLocalMonitorForEventsMatchingMask_handler_=lambda mask, cb: add("local", mask, cb),
            removeMonitor_=removed.append))
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    monkeypatch.setitem(sys.modules, "ApplicationServices", SimpleNamespace(AXIsProcessTrusted=lambda: trusted))
    monkeypatch.setitem(sys.modules, "Quartz", SimpleNamespace(
        kCGEventSourceUnixProcessID=41, CGEventGetIntegerValueField=lambda native, _field: native))
    return installed, removed


def test_global_and_local_routes_share_gesture_preserve_events_and_close(monkeypatch):
    installed, removed = frameworks(monkeypatch)
    toggles = []
    shortcut = fn.FnShortcut(lambda: toggles.append(True))
    shortcut.start()
    shortcut.start()  # do not register twice
    assert set(installed) == {"global", "local"}
    down, up = event(flags=fn.FN_FLAG), event()
    global_handler, local_handler = installed["global"][1], installed["local"][1]
    for press, release in ((global_handler, global_handler),
                           (local_handler, local_handler),
                           (global_handler, local_handler)):
        press(down)
        release(up)
    assert toggles == [True] * 3
    assert local_handler(up) is up  # do not swallow keys
    ordinary = event(fn.KEY_DOWN, 0, 0)
    assert local_handler(ordinary) is ordinary
    assert len(toggles) == 3
    shortcut.close()
    shortcut.close()
    assert removed == ["global", "local"]
    global_handler(down); global_handler(up)
    assert len(toggles) == 3  # no late callback after shutdown


def test_partial_install_failure_removes_monitor(monkeypatch):
    _, removed = frameworks(monkeypatch, fail_local=True)
    shortcut = fn.FnShortcut(lambda: None)
    with pytest.raises(RuntimeError, match="registration failed"):
        shortcut.start()
    assert removed == ["global"]
    shortcut.close()
    assert removed == ["global"]


def test_live_input_synthetic_typing_does_not_cancel_fn_stop(monkeypatch):
    frameworks(monkeypatch)
    toggles = []
    shortcut = fn.FnShortcut(lambda: toggles.append(True))
    shortcut.start()
    shortcut._global_event(event(flags=fn.FN_FLAG))
    for kind in (fn.KEY_DOWN, fn.KEY_UP):
        shortcut._global_event(event(kind, 0, 0, source_pid=os.getpid()))
    shortcut._global_event(event())
    assert toggles == [True]
    shortcut.close()


def test_missing_permission_gives_actionable_message_without_requesting_access(monkeypatch, capsys):
    frameworks(monkeypatch, trusted=False)
    shortcut = fn.FnShortcut(lambda: None)
    shortcut.start()
    assert "アクセシビリティ" in capsys.readouterr().err
    shortcut.close()
