"""Tests for word-level SRT realignment."""

from alwayswhisper.core.srt_parser import SrtFile
from alwayswhisper.core.srt_realigner import _build_edit_converters, realign_srt_by_words


SAMPLE_SRT = """1
00:00:00,500 --> 00:00:01,400
皆さんこんにちは

2
00:00:01,400 --> 00:00:03,500
今日はいい天気ですね

3
00:00:03,500 --> 00:00:05,000
ありがとうございます
"""


# Word timestamps simulate Whisper output: entry starts are 100-200ms earlier
# than the SRT claims (i.e., SRT is drifted late).
WORDS = [
    {"word": "皆さん", "start": 0.640, "end": 1.000},
    {"word": "こんにちは", "start": 1.000, "end": 1.400},
    {"word": "今日", "start": 1.520, "end": 1.800},
    {"word": "は", "start": 1.800, "end": 1.900},
    {"word": "いい天気", "start": 1.900, "end": 2.700},
    {"word": "ですね", "start": 2.700, "end": 3.350},
    {"word": "ありがとう", "start": 3.700, "end": 4.300},
    {"word": "ございます", "start": 4.300, "end": 4.900},
]


class TestRealigner:
    def test_shifts_start_to_word_boundary(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        stats = realign_srt_by_words(srt, WORDS, edit_plan={})
        assert stats["adjusted"] == 3
        assert stats["failed"] == 0
        # Entry 1 should start at 640ms (皆さん)
        assert srt.entries[0].start.to_ms() == 640
        # Entry 2 should start at 1520ms (今日)
        assert srt.entries[1].start.to_ms() == 1520
        # Entry 3 should start at 3700ms (ありがとう)
        assert srt.entries[2].start.to_ms() == 3700

    def test_preserves_duration_when_no_overlap(self):
        # Use widely-spaced entries so overlap clamp doesn't kick in.
        srt = SrtFile.parse(
            "1\n00:00:00,500 --> 00:00:01,300\n皆さんこんにちは\n\n"
            "2\n00:00:05,000 --> 00:00:06,000\nありがとうございます\n"
        )
        original_durations = [e.end.to_ms() - e.start.to_ms() for e in srt.entries]
        realign_srt_by_words(srt, WORDS, edit_plan={})
        new_durations = [e.end.to_ms() - e.start.to_ms() for e in srt.entries]
        assert new_durations == original_durations

    def test_no_overlap_after_realign(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        realign_srt_by_words(srt, WORDS, edit_plan={})
        for i in range(len(srt.entries) - 1):
            assert srt.entries[i].end <= srt.entries[i + 1].start

    def test_applies_edit_plan_offset(self):
        # Simulate a pause removed at 1.2s pre-edit that cuts 500ms.
        # A word at pre=2.0s maps to post=1.5s.
        srt = SrtFile.parse(
            "1\n00:00:01,000 --> 00:00:02,000\n今日\n\n"
        )
        edit_plan = {
            "pause_removals": [
                {"start": "00:00:01,200", "end": "00:00:01,700",
                 "removed_ms": 500}
            ]
        }
        # Word 今日 is at pre=2.0s → post expected 1.5s
        words = [{"word": "今日", "start": 2.0, "end": 2.5}]
        stats = realign_srt_by_words(srt, words, edit_plan)
        assert stats["adjusted"] == 1
        assert srt.entries[0].start.to_ms() == 1500

    def test_empty_srt(self):
        srt = SrtFile(entries=[])
        stats = realign_srt_by_words(srt, WORDS, edit_plan={})
        assert stats["adjusted"] == 0

    def test_empty_words(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        stats = realign_srt_by_words(srt, [], edit_plan={})
        assert stats["failed"] == 3
        # Timestamps unchanged
        assert srt.entries[0].start.to_ms() == 500

    def test_no_matching_text_keeps_original(self):
        # Text completely absent from word transcript — should be marked failed
        # and timestamp preserved.
        srt = SrtFile.parse(
            "1\n00:00:10,000 --> 00:00:12,000\n全く違うテキスト\n\n"
        )
        original_start = srt.entries[0].start.to_ms()
        stats = realign_srt_by_words(srt, WORDS, edit_plan={})
        assert stats["failed"] == 1
        assert srt.entries[0].start.to_ms() == original_start


# FIX 2 (DRY): _build_edit_converters used to re-implement removal parsing a
# third time (alongside s06_srt_sync and edit_plan.parse_removals), with a
# subtly different filler condition (`removed_ms is None` vs
# parse_removals's sentinel compare). It now builds its interval list via
# parse_removals() and only owns the pre_to_post/post_to_pre closures. These
# hand-computed points pin that the converters still behave identically.
SYNTHETIC_EDIT_PLAN = {
    "filler_removals": [
        # No removed_ms field -> parse_removals derives 500ms from end-start.
        {"text": "えー", "start": "00:00:01,000", "end": "00:00:01,500"},
    ],
    "pause_removals": [
        # removed_ms present (a minimum pause is kept, so it's less than the
        # full 2000ms gap_ms) -> used directly.
        {
            "between": [1, 2], "gap_ms": 2000, "removed_ms": 1700,
            "start": "00:00:05,000", "end": "00:00:06,700",
        },
    ],
}


class TestBuildEditConvertersUsesParseRemovals:
    def test_pre_to_post_hand_computed_points(self):
        pre_to_post, _ = _build_edit_converters(SYNTHETIC_EDIT_PLAN)

        assert pre_to_post(500) == 500     # before first removal: unaffected
        assert pre_to_post(1000) == 1000   # exactly at first removal start
        assert pre_to_post(1200) == 1000   # inside first removal: clamped to its start
        assert pre_to_post(1500) == 1000   # exactly at first removal end
        assert pre_to_post(3000) == 2500   # between removals: shifted by 500ms
        assert pre_to_post(5000) == 4500   # exactly at second removal start
        assert pre_to_post(6000) == 4500   # inside second removal: clamped
        assert pre_to_post(6700) == 4500   # exactly at second removal end
        assert pre_to_post(8000) == 5800   # after both: shifted by 500+1700=2200ms

    def test_post_to_pre_hand_computed_points(self):
        _, post_to_pre = _build_edit_converters(SYNTHETIC_EDIT_PLAN)

        assert post_to_pre(500) == 500
        assert post_to_pre(1000) == 1000
        assert post_to_pre(1500) == 2000
        assert post_to_pre(4500) == 5000
        assert post_to_pre(5800) == 8000

    def test_round_trips_outside_removed_spans(self):
        pre_to_post, post_to_pre = _build_edit_converters(SYNTHETIC_EDIT_PLAN)
        for pre_ms in [0, 500, 3000, 4999, 8000, 20000]:
            assert post_to_pre(pre_to_post(pre_ms)) == pre_ms

    def test_empty_plan_is_identity(self):
        pre_to_post, post_to_pre = _build_edit_converters({})
        assert pre_to_post(12345) == 12345
        assert post_to_pre(12345) == 12345

    def test_malformed_entries_are_skipped_not_erroring(self):
        # parse_removals skips entries missing required fields; the
        # converters should simply ignore them rather than raising.
        plan = {
            "filler_removals": [{"text": "no timestamps"}],
            "pause_removals": [{"between": [1, 2]}],  # missing "start"
        }
        pre_to_post, post_to_pre = _build_edit_converters(plan)
        assert pre_to_post(1000) == 1000
        assert post_to_pre(1000) == 1000
