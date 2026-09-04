"""Tests for typewriter animation."""

import pytest
from alwayswhisper.caption.typewriter import calculate_visible_chars, generate_char_timestamps


class TestCalculateVisibleChars:
    def test_zero_elapsed(self):
        assert calculate_visible_chars(0, 3000, 10) == 0

    def test_full_elapsed(self):
        assert calculate_visible_chars(3000, 3000, 10) == 10

    def test_past_duration(self):
        assert calculate_visible_chars(5000, 3000, 10) == 10

    def test_at_400ms_all_visible(self):
        # At 400ms (default completion), all 10 chars should be visible
        assert calculate_visible_chars(400, 3000, 10) == 10

    def test_halfway_through_400ms(self):
        # At 200ms (50% of 400ms), 5 of 10 chars visible
        result = calculate_visible_chars(200, 3000, 10)
        assert result == 5

    def test_short_text_same_speed(self):
        # Short text also completes in 400ms
        assert calculate_visible_chars(400, 3000, 3) == 3
        assert calculate_visible_chars(200, 3000, 3) == 1

    def test_negative_elapsed(self):
        assert calculate_visible_chars(-100, 3000, 10) == 0

    def test_zero_duration(self):
        assert calculate_visible_chars(100, 0, 10) == 10

    def test_zero_chars(self):
        assert calculate_visible_chars(100, 3000, 0) == 0

    def test_always_at_least_one(self):
        result = calculate_visible_chars(1, 3000, 10)
        assert result >= 1

    def test_custom_completion_ms(self):
        # With 200ms completion, all visible at 200ms
        assert calculate_visible_chars(200, 3000, 10, completion_ms=200) == 10
        # At 100ms, half visible
        assert calculate_visible_chars(100, 3000, 10, completion_ms=200) == 5


class TestGenerateCharTimestamps:
    def test_basic(self):
        timestamps = generate_char_timestamps(3000, 10)
        assert len(timestamps) == 10
        assert timestamps[0] == 0
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_empty(self):
        assert generate_char_timestamps(3000, 0) == []

    def test_zero_duration(self):
        timestamps = generate_char_timestamps(0, 5)
        assert timestamps == [0, 0, 0, 0, 0]

    def test_completion_within_400ms(self):
        timestamps = generate_char_timestamps(3000, 10)
        # Last char should appear before 400ms
        assert timestamps[-1] <= 400
