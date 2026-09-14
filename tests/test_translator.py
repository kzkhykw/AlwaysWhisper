"""Unit tests for alwayswhisper.live.translator.py -- the JA<->EN translation
helper behind caption_overlay.py's translate icon.

Headless and offline by construction: every test that exercises translate()
injects its own `post_json` (a plain callable taking (url, headers, payload,
timeout) and returning the already-decoded response dict), so no test in
this file ever opens a socket, needs an OPENROUTER_API_KEY, or spends money
-- the same dependency-injection idiom tests/test_caption_overlay.py already
uses for copy_fn/the fake pasteboard. The real urllib-based transport
(_post_json) is deliberately the ONLY thing here without direct coverage;
everything above it (direction detection, prompt/payload shape, header and
model resolution, response parsing, and every error path the overlay's
"failed" icon state depends on) is tested.
"""
import os
import sys

import pytest

from alwayswhisper.live import translator as tr  # noqa: E402

# Captured BEFORE the autouse fixture below starts redirecting it, so the
# "where does it point" test can still see the real, shipped value.
_REAL_DEFAULT_ENV_FILE = tr.DEFAULT_ENV_FILE


@pytest.fixture(autouse=True)
def _no_real_dotenv(monkeypatch, tmp_path):
    """No test may fall back to the developer's REAL .env: with a live
    OPENROUTER_API_KEY sitting in it, the "missing key" tests below would
    silently stop testing anything. Every test in this file therefore runs
    against a .env path that does not exist unless it makes one itself."""
    monkeypatch.setattr(tr, "DEFAULT_ENV_FILE", str(tmp_path / "does-not-exist.env"))


# ------------------------------------------------------------ detect_direction ---
@pytest.mark.parametrize("text", [
    "こんにちは",                       # hiragana
    "コンピューター",                    # katakana
    "会議",                             # kanji
    "AI エージェントの話",               # mixed latin + Japanese
    "AI、API。",                        # latin words, Japanese punctuation only
    "ｱｲｳｴｵ",                           # halfwidth katakana
])
def test_detect_direction_japanese_text_translates_to_english(text):
    assert tr.detect_direction(text) == tr.JA_TO_EN


@pytest.mark.parametrize("text", [
    "Thank you for your time today.",
    "OpenRouter API",
    "12345 - 67%",                      # no script at all: nothing to translate INTO English
    "",
])
def test_detect_direction_non_japanese_text_translates_to_japanese(text):
    assert tr.detect_direction(text) == tr.EN_TO_JA


def test_detect_direction_single_kana_in_a_long_latin_line_still_counts_as_japanese():
    """Any Japanese script at all wins: a live caption is Japanese speech
    with borrowed English words far more often than the reverse, and the
    JA->EN direction is the useful one for such a mixed line."""
    assert tr.detect_direction("We use Notion で全部管理しています") == tr.JA_TO_EN


# ---------------------------------------------------------------- build_payload ---
def test_build_payload_carries_model_messages_and_the_source_text_verbatim():
    payload = tr.build_payload("今日はありがとう", tr.JA_TO_EN, "vendor/model-x")

    assert payload["model"] == "vendor/model-x"
    assert payload["stream"] is False
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user"]
    assert payload["messages"][1]["content"] == "今日はありがとう"


def test_build_payload_system_prompt_names_the_target_language_per_direction():
    """Both prompts mention both languages ("translate the user's Japanese
    into English"), so the assertion has to be on the DIRECTIONAL phrase --
    which language is the target -- not on either word's mere presence."""
    ja_to_en = tr.build_payload("こんにちは", tr.JA_TO_EN, "m")["messages"][0]["content"]
    en_to_ja = tr.build_payload("hello", tr.EN_TO_JA, "m")["messages"][0]["content"]

    assert "Japanese text into natural, spoken English" in ja_to_en
    assert "English text into natural, spoken Japanese" in en_to_ja


def test_build_payload_rejects_an_unknown_direction():
    with pytest.raises(ValueError):
        tr.build_payload("hi", "fr->de", "m")


# -------------------------------------------------------------------- translate ---
def _ok_response(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class _RecordingPost:
    """Fake transport: records the one call it gets, returns `response`."""

    def __init__(self, response=None, raises=None):
        self.response = response if response is not None else _ok_response("ok")
        self.raises = raises
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        if self.raises is not None:
            raise self.raises
        return self.response


def test_translate_returns_the_assistant_content_stripped():
    post = _RecordingPost(_ok_response("  Thank you for your time today.  \n"))

    out = tr.translate("今日はお時間ありがとうございます", api_key="k", post_json=post)

    assert out == "Thank you for your time today."


def test_translate_posts_to_the_openrouter_endpoint_with_a_bearer_key():
    post = _RecordingPost()

    tr.translate("こんにちは", api_key="sk-test-123", post_json=post)

    call = post.calls[0]
    assert call["url"] == tr.DEFAULT_API_URL
    assert call["headers"]["Authorization"] == "Bearer sk-test-123"
    assert call["headers"]["Content-Type"] == "application/json"


def test_translate_detects_the_direction_from_the_text_it_is_given():
    post = _RecordingPost()

    tr.translate("Good morning", api_key="k", post_json=post)

    assert "Japanese" in post.calls[0]["payload"]["messages"][0]["content"]


def test_translate_prefers_an_explicit_model_over_env_over_the_default(monkeypatch):
    monkeypatch.delenv(tr.MODEL_ENV, raising=False)
    post = _RecordingPost()

    tr.translate("hi", api_key="k", post_json=post)
    assert post.calls[-1]["payload"]["model"] == tr.DEFAULT_MODEL

    monkeypatch.setenv(tr.MODEL_ENV, "vendor/from-env")
    tr.translate("hi", api_key="k", post_json=post)
    assert post.calls[-1]["payload"]["model"] == "vendor/from-env"

    tr.translate("hi", api_key="k", model="vendor/explicit", post_json=post)
    assert post.calls[-1]["payload"]["model"] == "vendor/explicit"


def test_translate_reads_the_api_key_from_the_environment_when_not_passed(monkeypatch):
    monkeypatch.setenv(tr.API_KEY_ENV, "sk-from-env")
    post = _RecordingPost()

    tr.translate("hi", post_json=post)

    assert post.calls[0]["headers"]["Authorization"] == "Bearer sk-from-env"


def test_translate_without_any_api_key_raises_a_named_actionable_error(monkeypatch):
    monkeypatch.delenv(tr.API_KEY_ENV, raising=False)
    post = _RecordingPost()

    with pytest.raises(tr.TranslationError) as excinfo:
        tr.translate("hi", post_json=post)

    assert tr.API_KEY_ENV in str(excinfo.value)   # the message must say WHICH var to set
    assert post.calls == []                        # and must not have hit the network


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_translate_refuses_blank_text_without_calling_the_api(text):
    post = _RecordingPost()

    with pytest.raises(tr.TranslationError):
        tr.translate(text, api_key="k", post_json=post)

    assert post.calls == []


def test_translate_wraps_a_transport_failure_in_translation_error():
    post = _RecordingPost(raises=OSError("connection reset"))

    with pytest.raises(tr.TranslationError) as excinfo:
        tr.translate("hi", api_key="k", post_json=post)

    assert "connection reset" in str(excinfo.value)


def test_translate_surfaces_an_openrouter_error_payload_as_translation_error():
    post = _RecordingPost({"error": {"message": "insufficient credits", "code": 402}})

    with pytest.raises(tr.TranslationError) as excinfo:
        tr.translate("hi", api_key="k", post_json=post)

    assert "insufficient credits" in str(excinfo.value)


@pytest.mark.parametrize("response", [
    {},                                                   # no choices at all
    {"choices": []},                                      # empty choices
    {"choices": [{"message": {}}]},                       # no content key
    {"choices": [{"message": {"content": "   "}}]},       # blank content
    {"choices": [{"message": {"content": None}}]},        # null content
])
def test_translate_raises_on_a_response_with_no_usable_text(response):
    with pytest.raises(tr.TranslationError):
        tr.translate("hi", api_key="k", post_json=_RecordingPost(response))


@pytest.mark.parametrize("raw, expected", [
    ('"Thank you."', "Thank you."),
    ("「ありがとう」", "ありがとう"),
    ("'Thank you.'", "Thank you."),
    ('"Say "hi" to them"', 'Say "hi" to them'),   # only the OUTERMOST pair is stripped
])
def test_translate_strips_a_wrapping_quote_pair_the_model_added(raw, expected):
    out = tr.translate("なにか", api_key="k", post_json=_RecordingPost(_ok_response(raw)))
    assert out == expected


def test_translate_keeps_unpaired_quotes_that_are_part_of_the_translation():
    out = tr.translate("なにか", api_key="k", post_json=_RecordingPost(_ok_response('He said "go"')))
    assert out == 'He said "go"'


def test_translate_passes_the_timeout_through_to_the_transport():
    post = _RecordingPost()

    tr.translate("hi", api_key="k", timeout=3.5, post_json=post)

    assert post.calls[0]["timeout"] == 3.5


def test_translate_uses_an_override_url_when_given():
    post = _RecordingPost()

    tr.translate("hi", api_key="k", url="https://example.test/v1/chat/completions", post_json=post)

    assert post.calls[0]["url"] == "https://example.test/v1/chat/completions"


# --------------------------------------------------------------------- _env_value ---
# live_avatar.py is normally started from a shell that has NOT sourced .env,
# so a key the user put in the repo's .env (the convention CLAUDE.md
# documents for OPENAI_API_KEY) has to be picked up from the file itself.
def test_env_value_prefers_the_process_environment_over_the_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-environ")

    assert tr._env_value("OPENROUTER_API_KEY", str(env_file)) == "from-environ"


def test_env_value_falls_back_to_the_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert tr._env_value("OPENROUTER_API_KEY", str(env_file)) == "from-file"


@pytest.mark.parametrize("line, expected", [
    ("KEY=plain", "plain"),
    ("export KEY=exported", "exported"),
    ('KEY="double quoted"', "double quoted"),
    ("KEY='single quoted'", "single quoted"),
    ("KEY=  spaced  ", "spaced"),
    ("KEY=with=equals=inside", "with=equals=inside"),
])
def test_env_value_parses_the_common_dotenv_line_shapes(monkeypatch, tmp_path, line, expected):
    env_file = tmp_path / ".env"
    env_file.write_text(f"# a comment\n\nOTHER=ignored\n{line}\n", encoding="utf-8")
    monkeypatch.delenv("KEY", raising=False)

    assert tr._env_value("KEY", str(env_file)) == expected


def test_env_value_ignores_a_commented_out_key(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("#KEY=commented\n", encoding="utf-8")
    monkeypatch.delenv("KEY", raising=False)

    assert tr._env_value("KEY", str(env_file)) == ""


def test_env_value_returns_empty_for_a_missing_file(monkeypatch, tmp_path):
    monkeypatch.delenv("KEY", raising=False)
    assert tr._env_value("KEY", str(tmp_path / "nope.env")) == ""


def test_translate_finds_an_api_key_that_only_exists_in_the_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{tr.API_KEY_ENV}=sk-from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv(tr.API_KEY_ENV, raising=False)
    monkeypatch.setattr(tr, "DEFAULT_ENV_FILE", str(env_file))
    post = _RecordingPost()

    tr.translate("hi", post_json=post)

    assert post.calls[0]["headers"]["Authorization"] == "Bearer sk-from-dotenv"


def test_default_env_file_points_at_launch_directory_dotenv():
    assert _REAL_DEFAULT_ENV_FILE == os.path.abspath(".env")
