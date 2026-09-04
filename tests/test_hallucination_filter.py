"""Tests for alwayswhisper.core.hallucination_filter -- mechanical removal of
Whisper's stock outro hallucination ("ご視聴" / "ありがとうございました")
from word-level timestamps, plain SRT text, the AV QA re-transcription, and
the packaged default config.
"""

import copy
import random
from pathlib import Path

import pytest
import yaml

import alwayswhisper
import alwayswhisper.core.av_qa as av_qa
from alwayswhisper.core.av_qa import run_av_qa
from alwayswhisper.core.hallucination_filter import (
    DEFAULT_STRIP_PHRASES,
    strip_hallucination_srt,
    strip_hallucination_text,
    strip_hallucination_words,
)
from alwayswhisper.core.srt_parser import SrtEntry, SrtFile
from alwayswhisper.core.timestamp import Timestamp


CONFIG_PATH = Path(alwayswhisper.__file__).resolve().parent / "data" / "default_config.yaml"


def _word(text, start, end):
    return {"word": text, "start": start, "end": end}


def _entry(index, start_ms, end_ms, text):
    return SrtEntry(
        index=index,
        start=Timestamp.from_ms(start_ms),
        end=Timestamp.from_ms(end_ms),
        text=text,
    )


# ---------------------------------------------------------------------------
# strip_hallucination_words
# ---------------------------------------------------------------------------


class TestStripHallucinationWords:
    def test_lone_tail_token_with_punctuation_dropped_with_matching_span(self):
        words = [
            _word("と思います", 10.0, 10.5),
            _word("ありがとうございました。", 10.5, 10.74),
        ]
        out, spans = strip_hallucination_words(words)

        assert [w["word"] for w in out] == ["と思います"]
        assert spans == [(10.5, 10.74)]

    def test_similar_but_non_target_phrase_is_preserved(self):
        # NOTE: bare "ありがとうございました" is now itself a default target
        # phrase (see DEFAULT_STRIP_PHRASES expansion), so this fixture was
        # changed to "どうもありがとう" -- thematically similar but not a
        # substring of any target phrase -- to keep covering genuinely
        # non-matching content.
        words = [
            _word("こんにちは", 0.0, 0.5),
            _word("どうもありがとう", 5.0, 5.3),
        ]
        out, spans = strip_hallucination_words(words)

        assert [w["word"] for w in out] == ["こんにちは", "どうもありがとう"]
        assert spans == []

    def test_split_tokens_dropped_as_one_span(self):
        words = [
            _word("参考にしてください", 0.0, 1.0),
            _word("ご", 1.0, 1.1),
            _word("視聴", 1.1, 1.3),
            _word("ありがとう", 1.3, 1.6),
            _word("ございました", 1.6, 1.9),
        ]
        out, spans = strip_hallucination_words(words)

        assert [w["word"] for w in out] == ["参考にしてください"]
        assert spans == [(1.0, 1.9)]

    def test_fused_token_dropped(self):
        words = [
            _word("それでは", 0.0, 0.4),
            _word("ご視聴ありがとうございました", 0.4, 1.2),
        ]
        out, spans = strip_hallucination_words(words)

        assert [w["word"] for w in out] == ["それでは"]
        assert spans == [(0.4, 1.2)]

    def test_partial_fused_token_kept_with_original_timestamps_not_a_span(self):
        words = [_word("と思いますご視聴ありがとうございました", 100.0, 101.0)]
        out, spans = strip_hallucination_words(words)

        assert len(out) == 1
        assert out[0]["word"] == "と思います"
        assert out[0]["start"] == 100.0
        assert out[0]["end"] == 101.0
        assert spans == []

    def test_mid_transcript_occurrence_removed_neighbours_untouched_order_preserved(self):
        words = [
            _word("こんにちは", 0.0, 0.5),
            _word("ご", 1.0, 1.1),
            _word("視聴", 1.1, 1.3),
            _word("ありがとう", 1.3, 1.6),
            _word("ございました", 1.6, 1.9),
            _word("さようなら", 2.0, 2.5),
        ]
        out, spans = strip_hallucination_words(words)

        assert [w["word"] for w in out] == ["こんにちは", "さようなら"]
        assert out[0]["start"] == 0.0 and out[0]["end"] == 0.5
        assert out[1]["start"] == 2.0 and out[1]["end"] == 2.5
        assert spans == [(1.0, 1.9)]

    def test_no_match_returns_equal_content_and_empty_spans_input_not_mutated(self):
        words = [
            _word("こんにちは", 0.0, 0.5),
            _word("元気ですか", 0.5, 1.0),
        ]
        snapshot = copy.deepcopy(words)

        out, spans = strip_hallucination_words(words)

        assert out == words
        assert spans == []
        assert words == snapshot  # input list/dicts untouched

    def test_empty_phrases_tuple_strips_nothing(self):
        words = [_word("ありがとうございました。", 0.0, 0.24)]

        out, spans = strip_hallucination_words(words, phrases=())

        assert [w["word"] for w in out] == ["ありがとうございました。"]
        assert spans == []

    def test_non_dict_entries_pass_through_unchanged(self):
        words = [
            _word("こんにちは", 0.0, 0.5),
            "not-a-dict",
            _word("ありがとうございました。", 1.0, 1.3),
            None,
        ]

        out, spans = strip_hallucination_words(words)

        assert out[0]["word"] == "こんにちは"
        assert out[1] == "not-a-dict"
        assert out[2] is None
        assert len(out) == 3
        assert spans == [(1.0, 1.3)]

    def test_returns_new_list_object(self):
        words = [_word("こんにちは", 0.0, 0.5)]
        out, _spans = strip_hallucination_words(words)
        assert out is not words

    def test_none_phrases_behaves_like_defaults(self):
        # A YAML `strip_phrases:` with nothing after it parses to None; core
        # functions must treat that the same as the omitted default, not
        # raise TypeError trying to iterate None.
        words = [_word("ありがとうございました。", 0.0, 0.24)]
        out, spans = strip_hallucination_words(words, phrases=None)

        assert out == []
        assert spans == [(0.0, 0.24)]

    def test_non_target_tokens_pass_through_unchanged(self):
        # Real Whisper output contains pre-existing empty ("") and
        # punctuation-only ("・") tokens mid-video that no phrase touches at
        # all -- these must never be folded into a "hallucination" span or
        # dropped just because their OWN content happens to have no
        # letter/digit. Only words actually touched by a phrase match may
        # be dropped/trimmed.
        #
        # NOTE: the trailing real-content tokens were changed from a
        # split-token spelling of bare "ありがとうございました" to
        # "おつかれさまでした" -- the former is now itself a target phrase
        # (see DEFAULT_STRIP_PHRASES expansion) and would legitimately be
        # removed, so it no longer serves as a non-matching contrast case.
        empty = _word("", 7.22, 7.24)
        naka_guro = _word("・", 8.0, 8.1)
        words = [
            _word("こんにちは", 0.0, 0.5),
            empty,
            naka_guro,
            _word("おつ", 10.0, 10.2),
            _word("かれ", 10.2, 10.4),
            _word("さまで", 10.4, 10.6),
            _word("した", 10.6, 10.8),
        ]

        out, spans = strip_hallucination_words(words)

        assert [w["word"] for w in out] == ["こんにちは", "", "・", "おつ", "かれ", "さまで", "した"]
        assert out[1] == empty
        assert out[2] == naka_guro
        assert spans == []

    def test_untouched_word_between_two_runs_splits_them_into_two_spans(self):
        # An untouched word (even an empty "") sitting between two
        # hallucination runs is a KEPT word -- it must close the first run
        # rather than let the two runs merge into one span.
        words = [
            _word("ご視聴", 0.0, 0.4),
            _word("", 0.4, 0.4),
            _word("ありがとうございました", 1.0, 1.3),
        ]

        out, spans = strip_hallucination_words(words)

        assert [w["word"] for w in out] == [""]
        assert spans == [(0.0, 0.4), (1.0, 1.3)]

    def test_bare_form_without_punctuation_removed_by_default(self):
        # "ありがとうございました" with no trailing "。" must also be treated
        # as a hallucination phrase now (previously only ご視聴-prefixed or
        # period-suffixed forms were stripped).
        words = [
            _word("本日は", 0.0, 0.5),
            _word("ありがとうございました", 0.5, 0.8),
        ]
        out, spans = strip_hallucination_words(words)

        assert [w["word"] for w in out] == ["本日は"]
        assert spans == [(0.5, 0.8)]

    def test_hai_prefix_and_period_suffix_tokens_fully_removed_default_phrases(self):
        # "はい、" + "ありがとうございました。" split across two tokens must be
        # fully removed as a single span -- no residual "。" token left
        # behind.
        words = [
            _word("はい、", 0.0, 0.3),
            _word("ありがとうございました。", 0.3, 0.6),
        ]
        out, spans = strip_hallucination_words(words)

        assert out == []
        assert spans == [(0.0, 0.6)]


# ---------------------------------------------------------------------------
# strip_hallucination_text
# ---------------------------------------------------------------------------


class TestStripHallucinationText:
    def test_basic_replace_removes_each_phrase(self):
        assert strip_hallucination_text("ご視聴ありがとうございました") == ""

    def test_replace_leaves_surrounding_text_intact(self):
        assert strip_hallucination_text("本日はご視聴ありがとうございました") == "本日は"

    def test_collapses_whitespace_runs_left_by_removal(self):
        assert (
            strip_hallucination_text("本日は  ご視聴ありがとうございました")
            == "本日は"
        )

    def test_no_match_returns_input_unchanged(self):
        assert strip_hallucination_text("こんにちは") == "こんにちは"

    def test_empty_phrases_strips_nothing(self):
        assert (
            strip_hallucination_text("ありがとうございました", phrases=())
            == "ありがとうございました"
        )

    def test_none_phrases_behaves_like_defaults(self):
        assert strip_hallucination_text("ご視聴ありがとうございました", phrases=None) == ""

    def test_hai_prefix_with_default_phrases_leaves_no_residue(self):
        # Regression: strip_hallucination_text("はい、ありがとうございました",
        # phrases=["ありがとうございました"]) used to leave "はい、" behind
        # because the only (short) phrase available matched inside the
        # longer run without anything to remove the "はい、" prefix. The
        # default phrase list now includes the "はい、"-prefixed form
        # outright, so the whole thing disappears.
        assert strip_hallucination_text("はい、ありがとうございました") == ""

    def test_order_independent_when_short_phrase_listed_before_long_phrase(self):
        # Longest-match-first must not depend on the caller's phrase order:
        # passing the short phrase before the long one still removes the
        # whole long phrase atomically instead of leaving "ご視聴" behind.
        phrases = ["ありがとうございました", "ご視聴ありがとうございました"]
        assert (
            strip_hallucination_text("本日はご視聴ありがとうございました", phrases=phrases)
            == "本日は"
        )

    def test_bare_form_without_punctuation_removed_by_default(self):
        assert strip_hallucination_text("本日はありがとうございました") == "本日は"


# ---------------------------------------------------------------------------
# strip_hallucination_srt
# ---------------------------------------------------------------------------


class TestStripHallucinationSrt:
    def test_drops_dedicated_entries_and_trims_mixed_entry(self):
        # NOTE: intentional spec change -- DEFAULT_STRIP_PHRASES now also
        # includes the bare form ("ありがとうございました", no punctuation),
        # so the entry that used to be preserved as-is (bare form) and the
        # entry that used to be preserved as-is ("。ありがとうございました",
        # leading punctuation only) are now both dropped too, not just the
        # ご視聴-prefixed / trailing-period ones.
        srt = SrtFile(
            entries=[
                _entry(1, 0, 1000, "こんにちは"),
                _entry(2, 1000, 2000, "ありがとうございました"),
                _entry(3, 2000, 3000, "。ありがとうございました"),
                _entry(4, 3000, 4000, "ご視聴ありがとうございました。"),
                _entry(5, 4000, 5000, "本日はご視聴ありがとうございました"),
                _entry(6, 5000, 6000, "また今度お会いしましょう"),
            ]
        )

        dropped, trimmed = strip_hallucination_srt(srt)

        assert dropped == 3
        assert trimmed == 1
        texts = [e.text for e in srt.entries]
        assert texts == [
            "こんにちは",
            "本日は",
            "また今度お会いしましょう",
        ]
        # reindexed contiguously from 1
        assert [e.index for e in srt.entries] == [1, 2, 3]

    def test_unrelated_entries_completely_unchanged(self):
        srt = SrtFile(entries=[_entry(1, 0, 1000, "今日はいい天気ですね")])
        dropped, trimmed = strip_hallucination_srt(srt)

        assert dropped == 0
        assert trimmed == 0
        assert srt.entries[0].text == "今日はいい天気ですね"

    def test_empty_phrases_drops_and_trims_nothing(self):
        srt = SrtFile(entries=[_entry(1, 0, 1000, "ありがとうございました。")])
        dropped, trimmed = strip_hallucination_srt(srt, phrases=())

        assert dropped == 0
        assert trimmed == 0
        assert len(srt.entries) == 1
        assert srt.entries[0].text == "ありがとうございました。"

    def test_none_phrases_behaves_like_defaults(self):
        srt = SrtFile(entries=[_entry(1, 0, 1000, "ありがとうございました。")])
        dropped, trimmed = strip_hallucination_srt(srt, phrases=None)

        assert dropped == 1
        assert trimmed == 0
        assert srt.entries == []

    def test_untouched_entries_kept_byte_identical_even_if_punctuation_only(self):
        # An entry containing NONE of the phrases must be left completely
        # unchanged -- not even whitespace-normalised -- and never dropped,
        # even if its own text is punctuation-only ("・") or has trailing
        # whitespace ("。 ") that strip_hallucination_text's own whitespace
        # collapse would otherwise strip away.
        naka_guro_entry = _entry(1, 0, 500, "・")
        trailing_space_entry = _entry(2, 500, 1000, "。 ")
        srt = SrtFile(entries=[naka_guro_entry, trailing_space_entry])

        dropped, trimmed = strip_hallucination_srt(srt)

        assert (dropped, trimmed) == (0, 0)
        assert srt.entries[0].text == "・"
        assert srt.entries[1].text == "。 "
        assert srt.entries[0] is naka_guro_entry
        assert srt.entries[1] is trailing_space_entry


# ---------------------------------------------------------------------------
# av_qa integration
# ---------------------------------------------------------------------------


class TestAvQaStripsHallucination:
    def test_hallucinated_tail_stripped_before_ratio_and_report(self, monkeypatch):
        def fake_extract(video_path, output_wav, start_sec, end_sec):
            Path(output_wav).write_bytes(b"")

        monkeypatch.setattr(av_qa, "extract_audio_clip", fake_extract)

        entries = [_entry(1, 0, 2000, "今日はいい天気ですね")]

        report = run_av_qa(
            "video.mp4",
            entries,
            samples=1,
            min_ratio=0.9,
            rng=random.Random(0),
            transcriber=lambda wav_path: "今日はいい天気ですねご視聴ありがとうございました",
        )

        sample = report["samples"][0]
        assert "ご視聴" not in sample["asr_text"]
        assert "ありがとうございました" not in sample["asr_text"]
        assert sample["ratio"] == pytest.approx(1.0)
        assert sample["ok"] is True
        assert report["passed"] is True


# ---------------------------------------------------------------------------
# DEFAULT_STRIP_PHRASES
# ---------------------------------------------------------------------------


class TestDefaultStripPhrasesConstant:
    def test_default_strip_phrases_cover_hai_and_bare_forms_longest_first(self):
        assert DEFAULT_STRIP_PHRASES == (
            "ご視聴ありがとうございました。",
            "はい、ありがとうございました。",
            "ご視聴ありがとうございました",
            "はい、ありがとうございました",
            "ありがとうございました。",
            "ありがとうございました",
        )


# ---------------------------------------------------------------------------
# packaged data/default_config.yaml
# ---------------------------------------------------------------------------


class TestDefaultConfigStripPhrases:
    def test_default_yaml_strip_phrases_is_present_and_null(self):
        # null in the packaged config means "use DEFAULT_STRIP_PHRASES when
        # transcribe.language is 'ja'" (see hallucination_filter._resolve_
        # phrases and pipeline.resolve_strip_phrases) -- the key must be
        # present (not simply absent) so the contract is documented in the
        # yaml itself, and its value must be None rather than a literal copy
        # of the phrase list (which would fork the source of truth in two;
        # AlwaysWhisper's config deep-merges, unlike the origin pipeline this
        # was ported from, so there is no need to spell the list out here).
        cfg = yaml.safe_load(CONFIG_PATH.read_text())
        assert "strip_phrases" in cfg["transcribe"]
        assert cfg["transcribe"]["strip_phrases"] is None
