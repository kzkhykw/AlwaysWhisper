"""Tests for SRT parser."""

import pytest
from alwayswhisper.core.srt_parser import SrtFile, SrtEntry
from alwayswhisper.core.timestamp import Timestamp


SAMPLE_SRT = """1
00:00:00,000 --> 00:00:03,000
皆さんこんにちは

2
00:00:03,000 --> 00:00:06,000
こんにちは、開発チームです

3
00:00:06,000 --> 00:00:10,000
今日は音声認識について話します
"""


class TestSrtFile:
    def test_parse(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        assert len(srt.entries) == 3
        assert srt.entries[0].text == "皆さんこんにちは"
        assert srt.entries[1].text == "こんにちは、開発チームです"
        assert srt.entries[2].text == "今日は音声認識について話します"

    def test_parse_timestamps(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        assert srt.entries[0].start.to_ms() == 0
        assert srt.entries[0].end.to_ms() == 3000
        assert srt.entries[2].end.to_ms() == 10000

    def test_to_string_roundtrip(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        output = srt.to_string()
        srt2 = SrtFile.parse(output)
        assert len(srt2.entries) == 3
        for e1, e2 in zip(srt.entries, srt2.entries):
            assert e1.text == e2.text
            assert e1.start.to_ms() == e2.start.to_ms()
            assert e1.end.to_ms() == e2.end.to_ms()

    def test_reindex(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        srt.entries.pop(1)
        srt.reindex()
        assert srt.entries[0].index == 1
        assert srt.entries[1].index == 2

    def test_merge_entries(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        srt.merge_entries(0, 1)
        assert len(srt.entries) == 2
        assert srt.entries[0].text == "皆さんこんにちはこんにちは、開発チームです"
        assert srt.entries[0].end.to_ms() == 6000

    def test_split_entry(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        # Split "皆さんこんにちは" at position 4 -> "皆さんこ" "んにちは"
        srt.split_entry(0, 4)
        assert len(srt.entries) == 4
        assert srt.entries[0].text == "皆さんこ"
        assert srt.entries[1].text == "んにちは"
        # Check timestamps are proportional
        assert srt.entries[0].start.to_ms() == 0
        assert srt.entries[1].end.to_ms() == 3000

    def test_offset_all(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        srt.offset_all(1000)
        assert srt.entries[0].start.to_ms() == 1000
        assert srt.entries[0].end.to_ms() == 4000

    def test_remove_entry(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        srt.remove_entry(1)
        assert len(srt.entries) == 2
        assert srt.entries[0].text == "皆さんこんにちは"
        assert srt.entries[1].text == "今日は音声認識について話します"

    def test_entries_longer_than(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        long_entries = srt.entries_longer_than(10)
        assert 2 in long_entries  # "今日は音声認識について話します" = 15 chars

    def test_entries_shorter_than(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        short = srt.entries_shorter_than(10)
        assert 0 in short  # "皆さんこんにちは" = 8 chars

    def test_duration_ms(self):
        srt = SrtFile.parse(SAMPLE_SRT)
        assert srt.entries[0].duration_ms == 3000
        assert srt.entries[2].duration_ms == 4000


class TestSrtEntry:
    def test_to_srt_block(self):
        entry = SrtEntry(
            index=1,
            start=Timestamp.from_ms(0),
            end=Timestamp.from_ms(3000),
            text="テスト",
        )
        block = entry.to_srt_block()
        assert "1\n" in block
        assert "00:00:00,000 --> 00:00:03,000\n" in block
        assert "テスト\n" in block
