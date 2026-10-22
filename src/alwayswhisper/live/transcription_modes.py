"""The four live transcription modes, independent of UI language."""

LANGUAGE_MODES = ("ja", "en", "en-ja", "ja-en")
TRANSLATION_MODES = ("en-ja", "ja-en")
MODE_LABELS = {"ja": "JA", "en": "EN", "en-ja": "EN→JA", "ja-en": "JA→EN"}
MODE_DESCRIPTIONS = {"ja": "日本語→日本語", "en": "英語→英語",
                     "en-ja": "英語→日本語", "ja-en": "日本語→英語"}


def recognition_language(mode):
    # Keep the existing CLI-only auto mode available for older launch scripts.
    if mode in (None, "auto"):
        return None
    return {"en-ja": "en", "ja-en": "ja"}.get(mode, mode)


def next_language_mode(mode):
    if mode not in LANGUAGE_MODES:
        return LANGUAGE_MODES[0]
    return LANGUAGE_MODES[(LANGUAGE_MODES.index(mode) + 1) % len(LANGUAGE_MODES)]
