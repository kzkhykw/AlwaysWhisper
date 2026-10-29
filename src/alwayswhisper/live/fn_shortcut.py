"""A bare Fn/Globe tap, observed without consuming another app's events.

Only event kinds, virtual key codes and modifier flags are inspected. No typed
characters are read or retained. AppKit delivers both monitors on the main thread.
"""
import os
import sys

# NSEventType / NSEventModifierFlags and Carbon's kVK_Function (0x3f).
KEY_DOWN, KEY_UP, FLAGS_CHANGED, SYSTEM_DEFINED = 10, 11, 12, 14
FN_KEY = 63
FN_FLAG = 1 << 23
CHORD_FLAGS = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 20)


class FnTap:
    """Toggle once on release; Fn chords and duplicate events never toggle."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.down = False
        self.used_in_chord = False

    def feed(self, kind, key_code, flags):
        if kind == FLAGS_CHANGED and key_code == FN_KEY:
            if flags & FN_FLAG:
                if not self.down:
                    self.down = True
                    self.used_in_chord = bool(flags & CHORD_FLAGS)
                else:
                    self.used_in_chord |= bool(flags & CHORD_FLAGS)
                return False
            toggle = self.down and not self.used_in_chord and not (flags & CHORD_FLAGS)
            self.reset()
            return bool(toggle)
        if self.down:
            self.used_in_chord = True
            if not (flags & FN_FLAG):
                # Recover from a missed release (e.g. secure keyboard input).
                # An orphaned release must never turn dictation on later.
                self.reset()
        return False


class FnShortcut:
    """Own local + global monitors for one overlay lifetime.

    NSEvent's two scopes are disjoint, so this works with the island focused,
    another application focused, and when focus moves while Fn is held.
    https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/EventOverview/MonitoringEvents/MonitoringEvents.html
    """

    def __init__(self, toggle):
        self._toggle = toggle
        self._gesture = FnTap()
        self._appkit = None
        self._quartz = None
        self._monitors = []
        self._closed = False

    def start(self):
        if self._closed or self._monitors:
            return
        import AppKit
        import ApplicationServices as AX
        import Quartz
        self._appkit = AppKit
        self._quartz = Quartz
        mask = (AppKit.NSEventMaskFlagsChanged | AppKit.NSEventMaskKeyDown
                | AppKit.NSEventMaskKeyUp | AppKit.NSEventMaskSystemDefined)
        try:
            for install, handler in (
                    (AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_, self._global_event),
                    (AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_, self._local_event)):
                token = install(mask, handler)
                if token is None:
                    raise RuntimeError("macOSがFnキーの監視を開始できませんでした")
                self._monitors.append(token)
        except Exception:
            self.close()
            raise
        if not AX.AXIsProcessTrusted():
            print("[caption-overlay] FnでLive Inputを切り替えるには、macOSのアクセシビリティで"
                  "起動元のターミナル等を許可してalwayswhisper liveを再起動してください。"
                  "アイランドを選択してLキーからも権限を確認できます。", file=sys.stderr)

    def _observe(self, event):
        if self._closed:
            return
        kind = int(event.type())
        if kind in (KEY_DOWN, KEY_UP) and self._quartz is not None:
            # Live Input emits Unicode key events into the focused app. Those
            # must not cancel a physical Fn tap used to stop that very input.
            native = event.CGEvent()
            if native is not None and self._quartz.CGEventGetIntegerValueField(
                    native, self._quartz.kCGEventSourceUnixProcessID) == os.getpid():
                return
        key_code = int(event.keyCode()) if kind in (KEY_DOWN, KEY_UP, FLAGS_CHANGED) else None
        if self._gesture.feed(kind, key_code, int(event.modifierFlags())):
            self._toggle()

    def _global_event(self, event):
        self._observe(event)

    def _local_event(self, event):
        self._observe(event)
        return event  # keep all other applications' / the island's shortcuts intact

    def close(self):
        self._closed = True
        self._gesture.reset()
        for token in self._monitors:
            try:
                self._appkit.NSEvent.removeMonitor_(token)
            except Exception as exc:
                print(f"[caption-overlay] Fnキー監視の終了に失敗しました: {exc}", file=sys.stderr)
        self._monitors.clear()
