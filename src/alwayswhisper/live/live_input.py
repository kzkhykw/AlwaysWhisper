"""Opt-in macOS dictation into the currently focused editable text field.

Accessibility is used only to identify a non-secure text target and to check
whether a later ASR correction still matches our own insertion. Text itself is
delivered as Unicode keyboard events, so browser and native fields receive
their normal input events. Frameworks are imported lazily: ordinary captions
do not need Accessibility permission or the ApplicationServices wrapper.
"""

from dataclasses import dataclass
import os


def _utf16_length(text):
    return len(text.encode("utf-16-le")) // 2


def _utf16_slice(text, start, length):
    data = text.encode("utf-16-le")
    return data[2 * start:2 * (start + length)].decode("utf-16-le", errors="replace")


def _separator(previous, following):
    if not previous or not following or following[0].isspace() or previous[-1].isspace():
        return ""
    return " " if previous[-1].isascii() and following[0].isascii() and (
        previous[-1].isalnum() or previous[-1] in ".!?") and following[0].isalnum() else ""


def _single_line(text):
    """Never send line/paragraph breaks, including ones inside ASR corrections."""
    result = ""
    for part in text.splitlines():
        result += _separator(result, part) + part
    return result


@dataclass(frozen=True)
class TextTarget:
    pid: int
    element: object


@dataclass(frozen=True)
class InsertedSpan:
    target: TextTarget
    start: int  # UTF-16 code units, as used by AXSelectedTextRange
    text: str


class MacTextInput:
    """The small AppKit/Accessibility boundary used by LiveInputSession."""

    def __init__(self):
        import AppKit
        import ApplicationServices as AX
        import Quartz
        self.appkit, self.ax, self.quartz = AppKit, AX, Quartz

    def request_access(self):
        ax, quartz = self.ax, self.quartz
        trusted = ax.AXIsProcessTrustedWithOptions({ax.kAXTrustedCheckOptionPrompt: True})
        if not trusted:
            return False, "macOSのアクセシビリティ権限を許可してから、Fnキーをもう一度押してください"
        if not quartz.CGPreflightPostEventAccess():
            quartz.CGRequestPostEventAccess()
            return False, "macOSの入力操作権限を許可してから、Fnキーをもう一度押してください"
        return True, None

    def _attribute(self, element, name):
        error, value = self.ax.AXUIElementCopyAttributeValue(element, name, None)
        return value if error == 0 else None

    def focused_target(self):
        app = self.appkit.NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None or app.processIdentifier() == os.getpid():
            return None
        pid = app.processIdentifier()
        ax = self.ax
        root = ax.AXUIElementCreateApplication(pid)
        ax.AXUIElementSetMessagingTimeout(root, 0.3)
        element = self._attribute(root, ax.kAXFocusedUIElementAttribute)
        if element is None:
            return None
        role = self._attribute(element, ax.kAXRoleAttribute)
        if self._attribute(element, ax.kAXSubroleAttribute) == ax.kAXSecureTextFieldSubrole:
            return None
        editable = self._attribute(element, ax.kAXIsEditableAttribute)
        text_roles = (ax.kAXTextFieldRole, ax.kAXTextAreaRole,
                      getattr(ax, "kAXComboBoxRole", "AXComboBox"))
        if editable is False or (role not in text_roles and editable is not True):
            return None
        return TextTarget(pid, element)

    def snapshot(self, target):
        ax = self.ax
        value = self._attribute(target.element, ax.kAXValueAttribute)
        selection = self._attribute(target.element, ax.kAXSelectedTextRangeAttribute)
        if not isinstance(value, str) or selection is None:
            return None
        ok, pair = ax.AXValueGetValue(selection, ax.kAXValueCFRangeType, None)
        return (value, pair) if ok else None

    def type_text(self, target, text):
        if not text:
            return True
        if self.focused_target() != target:
            return False
        quartz = self.quartz
        # Small events are accepted by both native fields and WebKit/Chromium.
        parts, part, units = [], "", 0
        for char in text:
            count = _utf16_length(char)
            if part and units + count > 32:
                parts.append(part)
                part, units = "", 0
            part += char
            units += count
        if part:
            parts.append(part)
        for part in parts:
            for down in (True, False):
                event = quartz.CGEventCreateKeyboardEvent(None, 0, down)
                quartz.CGEventKeyboardSetUnicodeString(event, _utf16_length(part), part)
                quartz.CGEventPostToPid(target.pid, event)
        return True

    def replace_if_unchanged(self, span, replacement):
        target, ax = span.target, self.ax
        current = self.focused_target()
        if current != target:
            return False
        snapshot = self.snapshot(target)
        length = _utf16_length(span.text)
        if (snapshot is None or snapshot[1] != (span.start + length, 0)
                or _utf16_slice(snapshot[0], span.start, length) != span.text):
            return False
        error, settable = ax.AXUIElementIsAttributeSettable(
            target.element, ax.kAXSelectedTextRangeAttribute, None)
        if error or not settable:
            return False
        selection = ax.AXValueCreate(ax.kAXValueCFRangeType,
                                     (span.start, length))
        if ax.AXUIElementSetAttributeValue(target.element,
                                           ax.kAXSelectedTextRangeAttribute, selection):
            return False
        return self.type_text(target, replacement)


class LiveInputSession:
    """Batches one UI poll's recognition chunks and mirrors future speech."""

    def __init__(self, input_factory=MacTextInput):
        self._input_factory = input_factory
        self.input = None
        self.active = False
        self.error = None
        self._pending = []
        self._current_pending = None
        self._current_line_sent = False
        self._complete_line = False
        self._current_text = ""
        self._span = None
        self._replace = None
        self._last_char = ""
        self._last_target = None

    def toggle(self):
        if self.active:
            self.active = False
            self.error = None
            self._reset()
            return False
        try:
            self.input = self.input or self._input_factory()
            allowed, error = self.input.request_access()
        except Exception as exc:
            allowed, error = False, f"Live Inputを開始できません: {exc}"
        self.active, self.error = bool(allowed), error
        self._reset()
        return self.active

    def _reset(self):
        self._pending.clear()
        self._current_pending = None
        self._current_line_sent = False
        self._complete_line = False
        self._current_text = ""
        self._span = None
        self._replace = None
        self._last_char = ""
        self._last_target = None

    def feed(self, event, *, fresh=False, current_text="", accepted=True):
        if not self.active or not accepted:
            return
        kind = event[0]
        current_text = _single_line(current_text)
        if kind == "chunk" and len(event) > 1 and event[1]:
            text = _single_line(event[1])
            if fresh:
                self._current_line_sent = False
                self._complete_line = True
                self._span = None
                self._replace = None
                self._current_pending = [True, current_text]
                self._pending.append(self._current_pending)
            elif self._current_pending is not None:
                self._current_pending[1] += text
            else:
                self._current_pending = [False, text]
                self._pending.append(self._current_pending)
            self._current_text = current_text
        elif kind == "replace" and current_text and current_text != self._current_text:
            self._current_text = current_text
            # Toggling on mid-utterance or missing part of it must never replay
            # the earlier words when the recognizer later corrects the whole line.
            if not self._complete_line:
                return
            if self._current_pending is not None and not self._current_line_sent:
                self._current_pending[1] = current_text
            else:
                if self._current_pending is not None:
                    self._pending.pop()  # current line is always the final segment
                    self._current_pending = None
                self._replace = current_text

    def flush(self):
        if not self.active or self.input is None:
            return
        if self._pending:
            try:
                target = self.input.focused_target()
                snapshot = self.input.snapshot(target) if target is not None else None
                previous = self._last_char if target == self._last_target else ""
                output, last_start, last_text = "", None, None
                for segment in self._pending:
                    fresh, text = segment
                    prefix = _separator(previous, text) if fresh else ""
                    if segment is self._current_pending:
                        last_start = _utf16_length(output + prefix)
                        last_text = text
                    output += prefix + text
                    previous = text[-1:] if text else previous
                if target is not None and self.input.type_text(target, output):
                    self._last_char = output[-1:] or self._last_char
                    self._last_target = target
                    if last_text is not None:
                        was_sent = self._current_line_sent
                        self._current_line_sent = True
                        if snapshot is not None:
                            start = snapshot[1][0] + last_start
                            if (self._span is not None and self._span.target == target
                                    and not self._current_pending[0]
                                    and snapshot[1] == (self._span.start + _utf16_length(self._span.text), 0)
                                    and _utf16_slice(snapshot[0], self._span.start,
                                                     _utf16_length(self._span.text)) == self._span.text):
                                self._span = InsertedSpan(target, self._span.start,
                                                          self._span.text + last_text)
                            else:
                                self._span = InsertedSpan(target, start, last_text)
                                if was_sent:
                                    self._complete_line = False
                        else:
                            self._span = None
                else:
                    self._span = None
                    self._complete_line = False
            except Exception as exc:
                self.error = f"Live Inputへの入力に失敗しました: {exc}"
                self._span = None
                self._complete_line = False
            finally:
                self._pending.clear()
                self._current_pending = None
        if self._replace is not None and self._span is not None and self._complete_line:
            try:
                if self.input.replace_if_unchanged(self._span, self._replace):
                    self._span = InsertedSpan(self._span.target, self._span.start, self._replace)
                    self._last_char = self._replace[-1:]
            except Exception as exc:
                self.error = f"Live Inputの修正に失敗しました: {exc}"
            finally:
                self._replace = None
        else:
            self._replace = None
