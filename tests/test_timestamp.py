"""Tests for timestamp utilities."""

import pytest
from alwayswhisper.core.timestamp import Timestamp, offset_timestamp, proportional_split


class TestTimestamp:
    def test_from_ms_zero(self):
        ts = Timestamp.from_ms(0)
        assert ts.hours == 0
        assert ts.minutes == 0
        assert ts.seconds == 0
        assert ts.milliseconds == 0

    def test_from_ms_basic(self):
        ts = Timestamp.from_ms(3661500)  # 1h 1m 1s 500ms
        assert ts.hours == 1
        assert ts.minutes == 1
        assert ts.seconds == 1
        assert ts.milliseconds == 500

    def test_from_ms_negative(self):
        ts = Timestamp.from_ms(-100)
        assert ts.to_ms() == 0

    def test_from_srt_string(self):
        ts = Timestamp.from_srt_string("01:23:45,678")
        assert ts.hours == 1
        assert ts.minutes == 23
        assert ts.seconds == 45
        assert ts.milliseconds == 678

    def test_to_ms(self):
        ts = Timestamp(hours=1, minutes=2, seconds=3, milliseconds=456)
        assert ts.to_ms() == 3723456

    def test_to_srt_string(self):
        ts = Timestamp(hours=0, minutes=1, seconds=23, milliseconds=456)
        assert ts.to_srt_string() == "00:01:23,456"

    def test_to_srt_string_zero_pad(self):
        ts = Timestamp(hours=0, minutes=0, seconds=5, milliseconds=10)
        assert ts.to_srt_string() == "00:00:05,010"

    def test_sub(self):
        ts1 = Timestamp.from_ms(5000)
        ts2 = Timestamp.from_ms(3000)
        assert ts1 - ts2 == 2000

    def test_add(self):
        ts = Timestamp.from_ms(5000)
        result = ts + 2000
        assert result.to_ms() == 7000

    def test_comparison(self):
        ts1 = Timestamp.from_ms(5000)
        ts2 = Timestamp.from_ms(3000)
        ts3 = Timestamp.from_ms(5000)
        assert ts1 > ts2
        assert ts2 < ts1
        assert ts1 >= ts3
        assert ts1 <= ts3
        assert ts1 == ts3

    def test_roundtrip(self):
        original = "00:12:34,567"
        ts = Timestamp.from_srt_string(original)
        assert ts.to_srt_string() == original


class TestOffsetTimestamp:
    def test_positive_offset(self):
        ts = Timestamp.from_ms(5000)
        result = offset_timestamp(ts, 2000)
        assert result.to_ms() == 7000

    def test_negative_offset(self):
        ts = Timestamp.from_ms(5000)
        result = offset_timestamp(ts, -2000)
        assert result.to_ms() == 3000


class TestProportionalSplit:
    def test_equal_split(self):
        start = Timestamp.from_ms(10000)
        end = Timestamp.from_ms(15000)
        mid = proportional_split(start, end, 5, 5)
        assert mid.to_ms() == 12500

    def test_unequal_split(self):
        start = Timestamp.from_ms(10000)
        end = Timestamp.from_ms(15000)
        mid = proportional_split(start, end, 8, 15)
        # 10000 + 5000 * 8/23 = 10000 + 1739 = 11739
        assert mid.to_ms() == 11739

    def test_zero_length(self):
        start = Timestamp.from_ms(10000)
        end = Timestamp.from_ms(15000)
        mid = proportional_split(start, end, 0, 0)
        assert mid.to_ms() == 10000
