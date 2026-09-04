from __future__ import annotations

"""High-level pipeline: transcribe, segment, caption, QA -- and the glue
between them.

This module is AlwaysWhisper's public API surface for embedding in other
Python code (the CLI in cli.py is a thin wrapper around these functions).
Every function that can burn captions or run QA imports the functions it
calls (create_backend, add_captions_fast, add_captions_to_video, run_av_qa)
as bare names at module scope specifically so tests can monkeypatch them on
this module's namespace without touching the real backends/caption/av_qa
modules.
"""

import json
import shutil
from pathlib import Path

import yaml

from .backends import create_backend
from .caption.overlay import add_captions_to_video
from .caption.overlay_fast import add_captions_fast
from .config import resolve_style_path
from .core.av_qa import run_av_qa
from .core.ffmpeg import extract_audio
from .core.hallucination_filter import DEFAULT_STRIP_PHRASES, strip_hallucination_words
from .core.srt_parser import SrtFile
from .core.srt_realigner import realign_srt_by_words
from .core.word_segmenter import words_to_srt, words_to_srt_en
from .prompt import WHISPER_PROMPT_TOKEN_LIMIT, load_glossary_text, truncate_prompt_to_tokens


# --- Helpers ---------------------------------------------------------------


def resolve_strip_phrases(cfg: dict) -> tuple:
    """Resolve transcribe.strip_phrases to a concrete tuple of phrases.

    None (absent, or an explicit YAML `null`) -> DEFAULT_STRIP_PHRASES when
    transcribe.language is "ja", else no phrases at all. A list (including
    an empty one, which explicitly disables stripping) is used verbatim.

    This is language-scoped by design -- a deliberate change from the
    origin repo, which always defaulted to the (Japanese-only)
    DEFAULT_STRIP_PHRASES regardless of language. AlwaysWhisper supports
    multiple languages, and Japanese hallucination phrases have no business
    being stripped from, say, an English transcript by default.
    """
    transcribe_cfg = cfg.get("transcribe", {}) if isinstance(cfg, dict) else {}
    phrases = transcribe_cfg.get("strip_phrases")
    if phrases is None:
        return DEFAULT_STRIP_PHRASES if transcribe_cfg.get("language") == "ja" else ()
    return tuple(phrases)


def resolve_prompt(cfg: dict, glossary_path: str | Path | None = None) -> str | None:
    """Resolve the Whisper bias prompt to use for transcription.

    Precedence: an explicit transcribe.prompt in cfg, else the text of
    glossary_path (if given), else no prompt at all. The resolved prompt is
    then truncated to transcribe.prompt_max_tokens (default
    WHISPER_PROMPT_TOKEN_LIMIT) so callers never have to think about
    Whisper's prompt token budget themselves.
    """
    transcribe_cfg = cfg.get("transcribe", {}) if isinstance(cfg, dict) else {}
    prompt = transcribe_cfg.get("prompt")
    if prompt is None and glossary_path is not None:
        prompt = load_glossary_text(glossary_path)
    if not prompt:
        return None

    max_tokens = transcribe_cfg.get("prompt_max_tokens", WHISPER_PROMPT_TOKEN_LIMIT)
    truncated, _before, _after = truncate_prompt_to_tokens(prompt, max_tokens)
    return truncated


# Languages whose script isn't space-delimited -- mirrors faster-whisper's
# own tokenizer no-space language set (the languages its tokenizer joins
# word-piece tokens for without inserting a space, matching how Whisper
# itself groups word timestamps for these languages: ja/zh/yue/th/lo/my).
# These get the character-based segmenter (words_to_srt): its sentence
# rules are Japanese-specific, but degrade gracefully to plain char-limit
# splitting for the others. Everything else -- including an unset/unknown
# language (None) -- gets the space-delimited generic segmenter
# (words_to_srt_en).
NO_SPACE_LANGUAGES = {"ja", "zh", "yue", "th", "lo", "my"}


def _dispatch_segmenter(words: list[dict], language: str | None, srt_cfg: dict) -> SrtFile:
    """Pick the language-appropriate word segmenter.

    language in NO_SPACE_LANGUAGES (ja/zh/yue/th/lo/my) -> words_to_srt
    (char-based; default max_chars=35, min_chars=5). Anything else,
    including an unset language (None) -> words_to_srt_en (space-delimited;
    default max_chars=42, min_chars=10). Previously this dispatched on
    `language == "en"` vs. everything else, which wrongly ran Japanese
    grammar rules (particle/conjunction split points, no inter-word spaces)
    against other space-delimited languages such as German or Korean.

    Mirrors the origin repo's exact per-language max_chars/min_chars
    fallback semantics: max_chars uses `or` (so an explicit 0/null/empty
    value falls back too), min_chars uses a plain dict.get default (only an
    ABSENT key falls back -- an explicit value, even 0, is honored). Always
    dispatching on the actual transcribe.language (rather than always using
    the Japanese segmenter) fixes a bug present in the origin repo's
    `resegment` command.
    """
    srt_cfg = srt_cfg or {}
    if language in NO_SPACE_LANGUAGES:
        max_chars = srt_cfg.get("max_chars") or 35
        min_chars = srt_cfg.get("min_chars", 5)
        return words_to_srt(words, max_chars=max_chars, min_chars=min_chars)
    max_chars = srt_cfg.get("max_chars") or 42
    min_chars = srt_cfg.get("min_chars", 10)
    return words_to_srt_en(words, max_chars=max_chars, min_chars=min_chars)


# --- Public pipeline functions ----------------------------------------------


def transcribe_file(
    input_path: str | Path,
    workdir: str | Path,
    cfg: dict,
    glossary_path: str | Path | None = None,
) -> dict:
    """Extract audio, transcribe to word timestamps, and segment to SRT.

    Writes workdir/transcript_words.json and workdir/transcript_raw.srt.
    The extracted intermediate WAV is always cleaned up, even on failure.
    Returns a dict of the artifact paths plus word/entry counts.
    """
    input_path = Path(input_path)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    transcribe_cfg = cfg.get("transcribe", {}) if isinstance(cfg, dict) else {}
    language = transcribe_cfg.get("language")

    # Resolve the backend and prompt before doing any real work: an unknown
    # backend name or a missing glossary file should fail immediately,
    # rather than after paying for a (possibly slow) audio extraction.
    backend = create_backend(cfg)
    prompt = resolve_prompt(cfg, glossary_path)

    audio_path = workdir / "audio_for_whisper.wav"
    print("  Extracting audio...")
    extract_audio(input_path, audio_path)

    try:
        print(f"  [{language}] Transcribing (word-level timestamps)...")
        words = backend.transcribe_words(audio_path, language=language, prompt=prompt)

        strip_phrases = resolve_strip_phrases(cfg)
        words, hallu_spans = strip_hallucination_words(words, strip_phrases)
        if hallu_spans:
            spans_str = ", ".join(f"{s:.2f}-{e:.2f}s" for s, e in hallu_spans)
            print(
                f"  [{language}] Stripped {len(hallu_spans)} hallucination "
                f"span(s) from words: {spans_str}"
            )

        words_path = workdir / "transcript_words.json"
        words_path.write_text(
            json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  [{language}] Got {len(words)} words")

        srt_cfg = cfg.get("srt", {}) if isinstance(cfg, dict) else {}
        srt = _dispatch_segmenter(words, language, srt_cfg)
        srt.reindex()

        srt_path = workdir / "transcript_raw.srt"
        srt.save(srt_path)
        print(f"  [{language}] Generated {len(srt.entries)} entries -> {srt_path}")
    finally:
        audio_path.unlink(missing_ok=True)

    return {
        "words_path": words_path,
        "srt_path": srt_path,
        "word_count": len(words),
        "entry_count": len(srt.entries),
        "hallucination_spans_removed": len(hallu_spans),
    }


def segment_words(words_json: str | Path, out_srt: str | Path, cfg: dict) -> Path:
    """Segment a transcript_words.json file into an SRT, saved to out_srt.

    Uses the same language-dispatching segmenter as transcribe_file (see
    _dispatch_segmenter) -- unlike the origin repo's `resegment` command,
    this always honors transcribe.language rather than always using the
    Japanese segmenter.
    """
    words_json = Path(words_json)
    out_srt = Path(out_srt)
    out_srt.parent.mkdir(parents=True, exist_ok=True)

    words = json.loads(words_json.read_text(encoding="utf-8"))

    cfg = cfg or {}
    transcribe_cfg = cfg.get("transcribe", {}) if isinstance(cfg, dict) else {}
    language = transcribe_cfg.get("language")
    srt_cfg = cfg.get("srt", {}) if isinstance(cfg, dict) else {}

    srt = _dispatch_segmenter(words, language, srt_cfg)
    srt.reindex()
    srt.save(out_srt)
    return out_srt


def caption_video(
    video: str | Path,
    srt_path: str | Path,
    out_path: str | Path,
    cfg: dict,
    words_json: str | Path | None = None,
    edit_plan_json: str | Path | None = None,
) -> dict:
    """Burn `srt_path`'s captions onto `video`, writing `out_path`.

    Optionally realigns SRT start times against word-level timestamps
    first (caption.realign + words_json), and always runs the AV QA spot
    check before burning unless qa.enabled is false -- in which case no
    backend is created at all (a --no-qa caption run must never import or
    instantiate a whisper backend). QA runs on the current entry
    timing/text, then gaps are filled (so captions are always visible)
    right before the burn; a failed QA raises RuntimeError and no burn
    happens (the qa_report.json is still written either way).

    Output is at the source video's resolution -- there is deliberately no
    finalize/resize step here.
    """
    video = Path(video)
    srt_path = Path(srt_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = cfg or {}
    caption_cfg = cfg.get("caption", {}) if isinstance(cfg, dict) else {}
    qa_cfg = cfg.get("qa", {}) if isinstance(cfg, dict) else {}
    transcribe_cfg = cfg.get("transcribe", {}) if isinstance(cfg, dict) else {}

    srt = SrtFile.from_file(srt_path)
    report: dict = {"video": video, "srt_path": srt_path, "out_path": out_path}

    if caption_cfg.get("realign") and words_json is not None and Path(words_json).exists():
        words = json.loads(Path(words_json).read_text(encoding="utf-8"))
        edit_plan = (
            json.loads(Path(edit_plan_json).read_text(encoding="utf-8"))
            if edit_plan_json is not None and Path(edit_plan_json).exists()
            else {}
        )
        stats = realign_srt_by_words(srt, words, edit_plan)
        print(
            f"  Realigned starts: adjusted={stats['adjusted']} "
            f"unchanged={stats['unchanged']} failed={stats['failed']} "
            f"max_shift={stats['max_shift_ms']}ms mean_shift={stats['mean_shift_ms']:+.1f}ms"
        )
        realigned_srt_path = out_path.with_suffix(".realigned.srt")
        srt.save(realigned_srt_path)
        report["realign_stats"] = stats
        report["realigned_srt_path"] = realigned_srt_path

    if qa_cfg.get("enabled", True):
        backend = create_backend(cfg)
        qa_report = run_av_qa(
            video,
            srt.entries,
            samples=qa_cfg.get("samples", 5),
            min_ratio=qa_cfg.get("min_ratio", 0.5),
            pad_ms=qa_cfg.get("pad_ms", 300),
            min_entry_ms=qa_cfg.get("min_entry_ms", 500),
            language=transcribe_cfg.get("language"),
            transcriber=backend.transcribe_text,
            strip_phrases=resolve_strip_phrases(cfg),
        )
        qa_report_path = out_path.parent / "qa_report.json"
        qa_report_path.write_text(
            json.dumps(qa_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["qa_report"] = qa_report
        report["qa_report_path"] = qa_report_path

        for s in qa_report["samples"]:
            print(f"  QA #{s['index']}: ratio={s['ratio']:.2f} ok={s['ok']}")
        print(
            f"  QA avg_ratio={qa_report['avg_ratio']:.2f} "
            f"({qa_report['sampled']}/{qa_report['eligible']} sampled) -> "
            f"{'PASS' if qa_report['passed'] else 'FAIL'}"
        )
        if not qa_report["passed"]:
            # Pass/fail is decided per-sample (every sample must individually
            # meet min_ratio) -- avg_ratio can sit above min_ratio while one
            # sample still fails, so the message must not read as an
            # average-vs-threshold comparison.
            n_failed = sum(1 for s in qa_report["samples"] if not s["ok"])
            raise RuntimeError(
                f"AV QA failed: {n_failed}/{qa_report['sampled']} sample(s) "
                f"below min_ratio={qa_report['min_ratio']} "
                f"(avg_ratio={qa_report['avg_ratio']:.2f}) -- see {qa_report_path}"
            )
    else:
        print("  AV QA disabled (qa.enabled: false), skipping...")

    filled = srt.fill_gaps(min_gap_ms=50)
    if filled:
        print(f"  Filled {filled} timestamp gaps")

    style_path = resolve_style_path(caption_cfg.get("style"))
    style = yaml.safe_load(style_path.read_text(encoding="utf-8")) or {}

    fast_mode = bool(caption_cfg.get("fast_mode", False))
    mode_label = "fast (libass)" if fast_mode else "standard (MoviePy/PIL)"
    print(f"  Adding captions to {len(srt.entries)} entries [{mode_label}]...")
    if fast_mode:
        add_captions_fast(video, srt, out_path, style)
    else:
        add_captions_to_video(video, srt, out_path, style)
    print(f"  Output: {out_path}")

    report["entry_count"] = len(srt.entries)
    return report


def qa_check(video: str | Path, srt_path: str | Path, cfg: dict) -> dict:
    """Standalone AV QA: spot-check srt_path's captions against video's audio.

    Writes qa_report.json next to srt_path. Unlike caption_video, this never
    raises on a failed check -- the caller (typically the `qa` CLI
    subcommand) decides what to do with report["passed"] (e.g. the process
    exit code).
    """
    video = Path(video)
    srt_path = Path(srt_path)
    cfg = cfg or {}
    qa_cfg = cfg.get("qa", {}) if isinstance(cfg, dict) else {}
    transcribe_cfg = cfg.get("transcribe", {}) if isinstance(cfg, dict) else {}

    srt = SrtFile.from_file(srt_path)
    backend = create_backend(cfg)

    report = run_av_qa(
        video,
        srt.entries,
        samples=qa_cfg.get("samples", 5),
        min_ratio=qa_cfg.get("min_ratio", 0.5),
        pad_ms=qa_cfg.get("pad_ms", 300),
        min_entry_ms=qa_cfg.get("min_entry_ms", 500),
        language=transcribe_cfg.get("language"),
        transcriber=backend.transcribe_text,
        strip_phrases=resolve_strip_phrases(cfg),
    )

    report_path = srt_path.parent / "qa_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {"report": report, "report_path": report_path}


def auto_run(
    input_path: str | Path,
    out_path: str | Path,
    cfg: dict,
    workdir: str | Path | None = None,
    glossary_path: str | Path | None = None,
) -> dict:
    """End-to-end: transcribe input_path, then burn captions to out_path.

    A convenience wrapper around transcribe_file + caption_video for the
    common "I have a video, give me a captioned video" case. The SRT that
    was actually burned (the post-realign snapshot if caption.realign ran,
    else the raw transcript SRT) is copied alongside out_path as a plain
    `.srt` sibling.
    """
    input_path = Path(input_path)
    out_path = Path(out_path)
    if workdir is None:
        workdir = out_path.parent / f"{out_path.stem}_work"
    workdir = Path(workdir)

    transcribe_report = transcribe_file(
        input_path, workdir, cfg, glossary_path=glossary_path
    )

    caption_report = caption_video(
        video=input_path,
        srt_path=transcribe_report["srt_path"],
        out_path=out_path,
        cfg=cfg,
        words_json=transcribe_report["words_path"],
    )

    burned_srt_src = caption_report.get("realigned_srt_path") or transcribe_report["srt_path"]
    final_srt_path = out_path.with_suffix(".srt")
    shutil.copy2(burned_srt_src, final_srt_path)

    return {
        "transcribe": transcribe_report,
        "caption": caption_report,
        "srt_path": final_srt_path,
    }
