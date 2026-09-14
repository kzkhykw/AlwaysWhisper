#!/usr/bin/env python3
"""JA<->EN translation for caption_overlay.py's translate icon, via OpenRouter.

One caption row in, its translation out: Japanese text is translated to
English, anything else to Japanese, with the direction detected from the
text itself (see detect_direction) rather than from any UI state -- the
overlay's translate icon is a single button, not a language picker.

HTTP requests use the Python standard library; no extra translation SDK is needed.

Configuration, all env-var overridable so a model/endpoint change never
needs a code edit (OpenRouter renames and reprices models continuously):
  OPENROUTER_API_KEY          -- required; no default, and translate()
                                 refuses to call anything without it.
  OPENROUTER_TRANSLATE_MODEL  -- defaults to DEFAULT_MODEL below.
  OPENROUTER_API_URL          -- defaults to DEFAULT_API_URL below.

Each is read from os.environ first, then from .env in the launch directory.

Every failure path -- missing key, transport error, HTTP error, an OpenRouter
`error` payload, an unusable response -- raises TranslationError with a
message worth showing a human. The caller (caption_overlay's translate icon)
turns that into a visible "failed" icon state plus one stdout line, and never
lets it reach an AppKit callback.
"""
import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT_SEC = 20.0

API_KEY_ENV = "OPENROUTER_API_KEY"
MODEL_ENV = "OPENROUTER_TRANSLATE_MODEL"
API_URL_ENV = "OPENROUTER_API_URL"

# Resolve once at startup, before callers can change the working directory.
DEFAULT_ENV_FILE = os.path.abspath(".env")

JA_TO_EN = "ja->en"
EN_TO_JA = "en->ja"

# Enough headroom for the longest caption row a live utterance produces
# (CaptionLineModel never trims a line, however long it grows) without
# letting a runaway generation bill for a whole context window.
_MAX_TOKENS = 1024
# Translation, not creative writing: the same sentence should come back the
# same way twice.
_TEMPERATURE = 0.2

# Unicode ranges that make a line Japanese. Kana and kanji are the obvious
# ones; CJK punctuation (U+3000-303F: 、。「」・) is included because a line
# of borrowed latin words held together by Japanese punctuation ("AI、API。")
# is Japanese speech, and halfwidth katakana because some IME/ASR paths emit
# it. Ranges are checked per character rather than by unicodedata lookups so
# this stays cheap enough to run on every icon click.
_JAPANESE_RANGES = (
    (0x3000, 0x303F),   # CJK symbols and punctuation
    (0x3040, 0x309F),   # hiragana
    (0x30A0, 0x30FF),   # katakana
    (0x4E00, 0x9FFF),   # CJK unified ideographs
    (0x3400, 0x4DBF),   # CJK unified ideographs extension A
    (0xFF66, 0xFF9D),   # halfwidth katakana
)

# Quote pairs a model sometimes wraps a translation in despite being told not
# to ("output only the translation"). Stripped only as a matched OUTERMOST
# pair -- see _strip_wrapping_quotes.
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("「", "」"), ("“", "”"), ("‘", "’"))

_SYSTEM_PROMPTS = {
    JA_TO_EN: (
        "You are a translation engine. Translate the user's Japanese text into natural, "
        "spoken English. Output ONLY the translation: no quotes, no romaji, no notes, no "
        "explanation, and never answer or react to the content. Keep proper nouns, product "
        "names and technical terms as they are normally written in English. Preserve the "
        "tone and register of the original. If the text is an incomplete fragment, translate "
        "the fragment as it stands."
    ),
    EN_TO_JA: (
        "You are a translation engine. Translate the user's English text into natural, "
        "spoken Japanese. Output ONLY the translation: no quotes, no notes, no explanation, "
        "and never answer or react to the content. Keep proper nouns, product names and "
        "technical terms in their usual Japanese rendering (katakana or the original latin "
        "spelling, whichever is normal). Preserve the tone and register of the original. If "
        "the text is an incomplete fragment, translate the fragment as it stands."
    ),
}


class TranslationError(Exception):
    """Any reason a translation could not be produced -- missing API key,
    transport/HTTP failure, an OpenRouter error payload, or a response with
    no usable text. Carries a message meant to be read by a human."""


def _env_value(name, env_file=None):
    """The configured value of `name`: os.environ first, then the first
    matching line of `env_file` (default DEFAULT_ENV_FILE, the launch directory's .env).
    "" when neither has it.

    The file fallback exists because AlwaysWhisper is normally launched
    from a plain shell that never sourced .env, so a key sitting in that
    file would otherwise be invisible. It is a deliberately minimal parser
    rather than a python-dotenv import (this module stays stdlib-only, see
    the module docstring): an optional `export ` prefix and one pair of
    surrounding quotes are tolerated, `#` comments and blank lines skipped,
    everything else ignored -- and a missing or unreadable file is simply
    "no value", never an error."""
    value = os.environ.get(name) or ""
    if value.strip():
        return value.strip()

    path = DEFAULT_ENV_FILE if env_file is None else env_file
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _sep, raw = line.partition("=")
                if key.strip() != name:
                    continue
                raw = raw.strip()
                if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("\"", "'"):
                    raw = raw[1:-1]
                return raw.strip()
    except OSError:
        return ""
    return ""


def detect_direction(text):
    """JA_TO_EN if `text` contains ANY Japanese script (see
    _JAPANESE_RANGES), EN_TO_JA otherwise -- including for empty text and
    for text with no script at all ("12345 - 67%").

    "Any Japanese at all" is deliberate, not a majority vote: a live caption
    mixing English product names into Japanese speech ("Notion で全部管理")
    is Japanese, and JA->EN is the direction that helps there. A genuinely
    English line has no kana/kanji in it to trip this."""
    for ch in text or "":
        code = ord(ch)
        for lo, hi in _JAPANESE_RANGES:
            if lo <= code <= hi:
                return JA_TO_EN
    return EN_TO_JA


def build_payload(text, direction, model):
    """The OpenAI-compatible chat-completions request body for translating
    `text` in `direction` (JA_TO_EN or EN_TO_JA) with `model`. The source
    text is the user message VERBATIM -- never interpolated into the system
    prompt -- so a caption that happens to read like an instruction stays
    data rather than becoming one. Raises ValueError for an unknown
    direction (a caller bug, not a runtime failure)."""
    try:
        system = _SYSTEM_PROMPTS[direction]
    except KeyError:
        raise ValueError(f"unknown translation direction: {direction!r}") from None
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "temperature": _TEMPERATURE,
        "max_tokens": _MAX_TOKENS,
        "stream": False,
    }


def _strip_wrapping_quotes(text):
    """Drop ONE matched outermost quote pair (see _QUOTE_PAIRS) if the whole
    string is wrapped in it, leaving every inner quote alone -- so
    '"Say "hi" to them"' becomes 'Say "hi" to them', while an unpaired
    'He said "go"' is returned untouched."""
    for open_q, close_q in _QUOTE_PAIRS:
        if len(text) >= 2 and text.startswith(open_q) and text.endswith(close_q):
            return text[len(open_q):-len(close_q)].strip()
    return text


def _extract_text(response):
    """The assistant text from an OpenRouter chat-completions response, or
    TranslationError explaining what came back instead. Handles the three
    shapes seen in practice: a normal choices[0].message.content, an
    {"error": {...}} payload (OpenRouter returns these with a 200 in some
    routing/credit cases, so this is NOT redundant with _post_json's HTTP
    error handling), and anything else (treated as unusable)."""
    if not isinstance(response, dict):
        raise TranslationError(f"unexpected response type from the translation API: {type(response).__name__}")

    error = response.get("error")
    if error:
        message = error.get("message") if isinstance(error, dict) else error
        raise TranslationError(f"translation API error: {message}")

    choices = response.get("choices") or []
    if not choices:
        raise TranslationError("translation API returned no choices")

    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise TranslationError("translation API returned empty text")

    return _strip_wrapping_quotes(content.strip())


def _post_json(url, headers, payload, timeout):
    """POST `payload` as JSON and return the decoded response body. The real
    transport -- every test injects its own callable of this shape instead
    (see translate's post_json argument), so no test opens a socket.

    An HTTP error status is turned into a TranslationError carrying the
    server's OWN message where the body has one: urllib's default
    "HTTP Error 401: Unauthorized" alone does not distinguish a bad key from
    an unknown model or an exhausted balance, and that distinction is the
    whole value of the message the overlay logs."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            parsed = json.loads(exc.read().decode("utf-8"))
            error = parsed.get("error")
            detail = (error.get("message") if isinstance(error, dict) else error) or ""
        except Exception:
            pass
        raise TranslationError(f"translation API HTTP {exc.code}{': ' + detail if detail else ''}") from exc
    return json.loads(body)


def translate(text, api_key=None, model=None, url=None, timeout=DEFAULT_TIMEOUT_SEC, post_json=None):
    """Translate one caption row and return the translation.

    Direction is detected from `text` (see detect_direction). `api_key`,
    `model` and `url` each fall back to their env var / .env entry
    (API_KEY_ENV / MODEL_ENV / API_URL_ENV, see _env_value) and then, except
    for the key, to the module default. `post_json` overrides the transport
    (tests inject a fake; production uses _post_json).

    Raises TranslationError -- never a bare urllib/JSON exception -- for
    every failure, including blank input and a missing API key, both of
    which fail BEFORE any network call. Blocking: the caller runs it off
    the UI thread (see caption_overlay._OverlayController._start_translate).
    """
    source = (text or "").strip()
    if not source:
        raise TranslationError("nothing to translate (the caption row is empty)")

    key = api_key or _env_value(API_KEY_ENV)
    if not key:
        raise TranslationError(
            f"no OpenRouter API key: set {API_KEY_ENV} (in .env, or the environment "
            "AlwaysWhisper runs under)")

    endpoint = url or _env_value(API_URL_ENV) or DEFAULT_API_URL
    chosen_model = model or _env_value(MODEL_ENV) or DEFAULT_MODEL
    payload = build_payload(source, detect_direction(source), chosen_model)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    transport = post_json if post_json is not None else _post_json
    try:
        response = transport(endpoint, headers, payload, timeout)
    except TranslationError:
        raise
    except Exception as exc:
        raise TranslationError(f"could not reach the translation API: {exc}") from exc

    return _extract_text(response)
