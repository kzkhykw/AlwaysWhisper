"""Tests for alwayswhisper.core.edit_plan.parse_removals.

parse_removals is the single shared derivation of "what time ranges were
physically cut from the audio" from edit_plan.json.
"""

from alwayswhisper.core.edit_plan import parse_removals


class TestParseRemovals:
    def test_filler_derives_removed_ms_from_start_end(self):
        plan = {
            "filler_removals": [
                {"text": "えー", "start": "00:00:01,000", "end": "00:00:01,500"},
            ],
        }
        assert parse_removals(plan) == [{"start_ms": 1000, "removed_ms": 500}]

    def test_pause_uses_its_own_removed_ms_field(self):
        plan = {
            "pause_removals": [
                {
                    "between": [1, 2],
                    "gap_ms": 2000,
                    "removed_ms": 1700,
                    "start": "00:00:03,000",
                    "end": "00:00:05,000",
                },
            ],
        }
        assert parse_removals(plan) == [{"start_ms": 3000, "removed_ms": 1700}]

    def test_combined_plan_sorted_by_start_ms(self):
        plan = {
            "filler_removals": [
                {"text": "えー", "start": "00:00:01,000", "end": "00:00:01,500"},
            ],
            "pause_removals": [
                {
                    "between": [1, 2],
                    "gap_ms": 2000,
                    "removed_ms": 1700,
                    "start": "00:00:03,000",
                    "end": "00:00:05,000",
                },
            ],
        }
        result = parse_removals(plan)
        assert result == [
            {"start_ms": 1000, "removed_ms": 500},
            {"start_ms": 3000, "removed_ms": 1700},
        ]

    def test_result_always_sorted_regardless_of_input_order(self):
        plan = {
            "filler_removals": [
                {"text": "b", "start": "00:00:05,000", "end": "00:00:05,200"},
                {"text": "a", "start": "00:00:01,000", "end": "00:00:01,200"},
            ],
        }
        result = parse_removals(plan)
        assert [r["start_ms"] for r in result] == [1000, 5000]

    def test_pause_without_start_is_skipped(self):
        plan = {"pause_removals": [{"between": [1, 2], "gap_ms": 500}]}
        assert parse_removals(plan) == []

    def test_pause_without_removed_ms_or_end_is_skipped(self):
        plan = {"pause_removals": [{"start": "00:00:01,000"}]}
        assert parse_removals(plan) == []

    def test_pause_derives_removed_ms_from_end_when_field_missing(self):
        plan = {
            "pause_removals": [
                {"start": "00:00:01,000", "end": "00:00:01,300"},
            ],
        }
        assert parse_removals(plan) == [{"start_ms": 1000, "removed_ms": 300}]

    def test_malformed_filler_entry_is_skipped(self):
        plan = {"filler_removals": [{"text": "no timestamps here"}]}
        assert parse_removals(plan) == []

    def test_empty_plan_returns_empty_list(self):
        assert parse_removals({}) == []
