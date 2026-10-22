"""Live Input's text batching and correction rules, without OS input events."""

import os
import sys

import pytest

from alwayswhisper.live import live_input as li  # noqa: E402


class FakeInput:
    def __init__(self):
        self.current = "a"
        self.values = {"a": "", "b": ""}
        self.carets = {"a": 0, "b": 0}
        self.allowed = True

    def request_access(self):
        return (self.allowed, None if self.allowed else "permission needed")

    def focused_target(self):
        return self.current

    def snapshot(self, target):
        return self.values[target], (self.carets[target], 0)

    def type_text(self, target, text):
        value, caret = self.values[target], self.carets[target]
        self.values[target] = value[:caret] + text + value[caret:]
        self.carets[target] += len(text)
        return True

    def replace_if_unchanged(self, span, replacement):
        target = span.target
        if target != self.current:
            return False
        value = self.values[target]
        if (self.carets[target] != span.start + len(span.text)
                or value[span.start:span.start + len(span.text)] != span.text):
            return False
        self.values[target] = value[:span.start] + replacement + value[span.start + len(span.text):]
        self.carets[target] = span.start + len(replacement)
        return True


def test_streams_chunks_and_corrects_only_its_own_text():
    field = FakeInput()
    session = li.LiveInputSession(lambda: field)
    assert session.toggle()
    session.feed(("chunk", "Hello"), fresh=True, current_text="Hello")
    session.flush()
    session.feed(("chunk", " world"), current_text="Hello world")
    session.flush()
    assert field.values["a"] == "Hello world"
    session.feed(("replace", "Hello there"), current_text="Hello there")
    session.flush()
    assert field.values["a"] == "Hello there"
    # The user changed our span. A late ASR correction must leave it alone.
    field.values["a"] = "User edited this"
    session.feed(("replace", "Hello again"), current_text="Hello again")
    session.flush()
    assert field.values["a"] == "User edited this"


def test_new_focus_gets_future_speech_without_copying_earlier_text():
    field = FakeInput()
    session = li.LiveInputSession(lambda: field)
    session.toggle()
    session.feed(("chunk", "Hello"), fresh=True, current_text="Hello")
    session.flush()
    field.current = "b"
    session.feed(("chunk", "world"), fresh=True, current_text="world")
    session.flush()
    assert field.values == {"a": "Hello", "b": "world"}


def test_late_correction_does_not_move_a_user_selected_caret():
    field = FakeInput()
    session = li.LiveInputSession(lambda: field)
    session.toggle()
    session.feed(("chunk", "Hello"), fresh=True, current_text="Hello")
    session.flush()
    field.carets["a"] = 0
    session.feed(("replace", "Hallo"), current_text="Hallo")
    session.flush()
    assert field.values["a"] == "Hello"
    assert field.carets["a"] == 0


def test_same_poll_batches_words_and_uses_final_correction():
    field = FakeInput()
    session = li.LiveInputSession(lambda: field)
    session.toggle()
    session.feed(("chunk", "Helo"), fresh=True, current_text="Helo")
    session.feed(("replace", "Hello"), current_text="Hello")
    session.feed(("eos",))
    session.feed(("chunk", "world"), fresh=True, current_text="world")
    session.flush()
    assert field.values["a"] == "Hello world"


def test_permission_failure_keeps_mode_off_and_toggle_off_stops_input():
    field = FakeInput()
    field.allowed = False
    session = li.LiveInputSession(lambda: field)
    assert not session.toggle()
    assert session.error == "permission needed"
    field.allowed = True
    assert session.toggle()
    assert not session.toggle()
    session.feed(("chunk", "ignored"), fresh=True, current_text="ignored")
    session.flush()
    assert field.values["a"] == ""


def test_utf16_span_counts_non_bmp_characters():
    assert li._utf16_length("A😀B") == 4
    assert li._utf16_slice("A😀B", 1, 2) == "😀"


@pytest.mark.parametrize("newline", ["\n", "\r", "\r\n", "\u2028", "\u2029"])
def test_chunks_sentences_and_corrections_never_insert_line_breaks(newline):
    field = FakeInput()
    session = li.LiveInputSession(lambda: field)
    session.toggle()
    text = f"最初の{newline}文。"
    session.feed(("chunk", text), fresh=True, current_text=text)
    session.flush()
    session.feed(("eos",))
    text = f"次の{newline}文。"
    session.feed(("chunk", text), fresh=True, current_text=text)
    session.flush()
    text = f"修正した{newline}文。"
    session.feed(("replace", text), current_text=text)
    session.flush()
    assert field.values["a"] == "最初の文。修正した文。"
    assert li._single_line(f"Hello{newline}world") == "Hello world"


def test_enabling_mid_sentence_does_not_replay_earlier_words_on_correction():
    field = FakeInput()
    session = li.LiveInputSession(lambda: field)
    session.toggle()
    session.feed(("chunk", "続き"), current_text="前半と続き")
    session.flush()
    session.feed(("replace", "前半と後半"), current_text="前半と後半")
    session.flush()
    assert field.values["a"] == "続き"


def test_switching_fields_mid_sentence_does_not_copy_earlier_words_on_correction():
    field = FakeInput()
    session = li.LiveInputSession(lambda: field)
    session.toggle()
    session.feed(("chunk", "前半"), fresh=True, current_text="前半")
    session.flush()
    field.current = "b"
    session.feed(("chunk", "と続き"), current_text="前半と続き")
    session.flush()
    session.feed(("replace", "前半と後半"), current_text="前半と後半")
    session.flush()
    assert field.values == {"a": "前半", "b": "と続き"}


def test_correction_from_previous_utterance_cannot_replace_the_next_one():
    field = FakeInput()
    session = li.LiveInputSession(lambda: field)
    session.toggle()
    session.feed(("chunk", "最初"), fresh=True, current_text="最初")
    session.flush()
    session.feed(("replace", "修正"), current_text="修正")
    session.feed(("eos",))
    session.feed(("chunk", "次の文"), fresh=True, current_text="次の文")
    session.flush()
    assert field.values["a"] == "最初次の文"


def test_no_focused_field_drops_text_without_replaying_it_later():
    field = FakeInput()
    session = li.LiveInputSession(lambda: field)
    session.toggle()
    field.current = None
    session.feed(("chunk", "取りこぼし"), fresh=True, current_text="取りこぼし")
    session.flush()
    field.current = "a"
    session.feed(("chunk", "続き"), current_text="取りこぼし続き")
    session.flush()
    session.feed(("replace", "取りこぼしと続き"), current_text="取りこぼしと続き")
    session.flush()
    assert field.values["a"] == "続き"
