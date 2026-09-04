"""Tests for alwayswhisper.prompt: Whisper bias-prompt token limiting and
glossary-file loading.

Whisper's decoder reserves 224 tokens for the prompt; anything longer is
silently dropped from the *front*, so glossary terms placed early stop biasing
the model without any error. These tests pin the conservative token estimate
and the tail-preserving truncation that surfaces/avoids that failure.
"""

from alwayswhisper.prompt import (
    estimate_whisper_tokens,
    load_glossary_text,
    truncate_prompt_to_tokens,
)


class TestEstimateWhisperTokens:
    def test_empty_is_zero(self):
        assert estimate_whisper_tokens("") == 0

    def test_ascii_roughly_quarter_of_char_count(self):
        # ~1 token per 4 ASCII chars (usual BPE ratio).
        est = estimate_whisper_tokens("a" * 40)
        assert 8 <= est <= 12

    def test_cjk_counts_about_two_per_char(self):
        # Conservative: Whisper's multilingual vocab splits kana/kanji into
        # 1-2 tokens each; we over-estimate at ~2.
        est = estimate_whisper_tokens("あいうえおかきくけこ")  # 10 chars
        assert est >= 18

    def test_cjk_heavier_than_equal_length_ascii(self):
        assert estimate_whisper_tokens("あ" * 20) > estimate_whisper_tokens("a" * 20)

    def test_longer_text_never_estimates_fewer_tokens(self):
        assert estimate_whisper_tokens("hello world foo bar") >= estimate_whisper_tokens("hello")


class TestTruncatePromptToTokens:
    def test_short_prompt_is_unchanged(self):
        p = "Acme, Widget Pro, Foobar SDK, Gizmo"
        out, before, after = truncate_prompt_to_tokens(p, 224)
        assert out == p
        assert before == after
        assert before <= 224

    def test_long_prompt_truncated_under_limit(self):
        p = ", ".join(f"term{i}" for i in range(500))
        out, before, after = truncate_prompt_to_tokens(p, 224)
        assert before > 224
        assert after <= 224
        assert len(out) < len(p)

    def test_keeps_the_tail_drops_the_head(self):
        # Whisper keeps the final 224 tokens, so the tail term must survive
        # and the head term must be dropped.
        p = "HEADWORD " + ("filler " * 400) + "TAILWORD"
        out, before, after = truncate_prompt_to_tokens(p, 224)
        assert out.endswith("TAILWORD")
        assert "HEADWORD" not in out

    def test_does_not_cut_mid_term(self):
        # No leading partial-word fragment after truncation.
        p = ("supercalifragilistic " * 100).strip()
        out, before, after = truncate_prompt_to_tokens(p, 20)
        assert all(tok == "supercalifragilistic" for tok in out.split())

    def test_truncated_output_estimate_within_limit_cjk(self):
        p = "あ" * 500  # ~1000 tokens
        out, before, after = truncate_prompt_to_tokens(p, 224)
        assert after <= 224
        assert estimate_whisper_tokens(out) <= 224

    def test_reported_before_matches_estimate_of_input(self):
        p = ", ".join(f"用語{i}" for i in range(300))
        out, before, after = truncate_prompt_to_tokens(p, 224)
        assert before == estimate_whisper_tokens(p)

    def test_prints_warning_only_when_truncating(self, capsys):
        short = truncate_prompt_to_tokens("hello world", 224)
        assert capsys.readouterr().out == ""

        long_prompt = ", ".join(f"term{i}" for i in range(500))
        truncate_prompt_to_tokens(long_prompt, 224)
        assert "WARNING" in capsys.readouterr().out


class TestLoadGlossaryText:
    def test_collapses_whitespace_and_strips(self, tmp_path):
        path = tmp_path / "glossary.txt"
        path.write_text("  Widget Pro 5\nGizmo\n\tFoobar  \n\n", encoding="utf-8")

        assert load_glossary_text(path) == "Widget Pro 5 Gizmo Foobar"

    def test_accepts_str_path(self, tmp_path):
        path = tmp_path / "glossary.txt"
        path.write_text("term one\nterm two", encoding="utf-8")

        assert load_glossary_text(str(path)) == "term one term two"

    def test_reads_utf8(self, tmp_path):
        path = tmp_path / "glossary.txt"
        path.write_text("ご視聴  ありがとう", encoding="utf-8")

        assert load_glossary_text(path) == "ご視聴 ありがとう"
