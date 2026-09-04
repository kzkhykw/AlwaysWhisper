from __future__ import annotations

"""AlwaysWhisper command-line interface.

Every subcommand's flags become an `overrides` dict, deep-merged over the
loaded config (packaged defaults <- --config YAML <- these overrides) --
only flags the user actually passed are set, so an unset flag never
clobbers a value from --config or the packaged defaults. Pipeline functions
are imported as bare names (not accessed via `pipeline.xxx`) specifically so
tests can monkeypatch them directly on this module's namespace.
"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .pipeline import (
    auto_run,
    caption_video,
    qa_check,
    segment_words,
    transcribe_file,
)

# Errors a user can actually act on (bad path, bad backend name, failed QA,
# missing API key, missing optional dependency) get a clean one-line
# message instead of a traceback.
_USER_FACING_ERRORS = (FileNotFoundError, RuntimeError, ValueError, ImportError)


def _set(overrides: dict, path: tuple, value) -> None:
    """overrides[path[0]][path[1]]... = value, skipped entirely if value is
    None -- so an unset CLI flag never overwrites a --config/default value
    during the deep_merge in load_config().
    """
    if value is None:
        return
    node = overrides
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _print_paths(**paths) -> None:
    for label, path in paths.items():
        if path is not None:
            print(f"  {label}: {path}")


def _require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


# --- subcommand handlers -----------------------------------------------------


def cmd_transcribe(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    _require_exists(input_path, "Input file")

    overrides: dict = {}
    _set(overrides, ("transcribe", "backend"), args.backend)
    _set(overrides, ("transcribe", "model"), args.model)
    _set(overrides, ("transcribe", "language"), args.language)
    _set(overrides, ("transcribe", "device"), args.device)
    _set(overrides, ("transcribe", "compute_type"), args.compute_type)
    _set(overrides, ("transcribe", "prompt"), args.prompt)
    _set(overrides, ("transcribe", "vad_filter"), args.vad_filter)
    _set(overrides, ("srt", "max_chars"), args.max_chars)
    _set(overrides, ("srt", "min_chars"), args.min_chars)

    cfg = load_config(args.config, overrides)

    output_dir = (
        Path(args.output)
        if args.output
        else input_path.parent / f"{input_path.stem}_alwayswhisper"
    )

    result = transcribe_file(input_path, output_dir, cfg, glossary_path=args.glossary)

    print(
        f"Transcribed {result['word_count']} words -> "
        f"{result['entry_count']} caption entries"
    )
    _print_paths(words=result["words_path"], srt=result["srt_path"])


def cmd_segment(args: argparse.Namespace) -> None:
    words_json = Path(args.words_json)
    _require_exists(words_json, "Words JSON file")

    overrides: dict = {}
    _set(overrides, ("transcribe", "language"), args.language)
    _set(overrides, ("srt", "max_chars"), args.max_chars)
    _set(overrides, ("srt", "min_chars"), args.min_chars)

    cfg = load_config(args.config, overrides)

    out_srt = segment_words(words_json, args.output, cfg)

    print("Segmented words into SRT")
    _print_paths(srt=out_srt)


def cmd_caption(args: argparse.Namespace) -> None:
    video = Path(args.video)
    srt_path = Path(args.srt)
    _require_exists(video, "Video file")
    _require_exists(srt_path, "SRT file")

    overrides: dict = {}
    _set(overrides, ("transcribe", "backend"), args.backend)
    _set(overrides, ("transcribe", "model"), args.model)
    _set(overrides, ("transcribe", "language"), args.language)
    _set(overrides, ("caption", "style"), args.style)
    _set(overrides, ("caption", "fast_mode"), args.fast)
    _set(overrides, ("caption", "realign"), args.realign)
    _set(overrides, ("qa", "samples"), args.qa_samples)
    _set(overrides, ("qa", "min_ratio"), args.qa_min_ratio)
    _set(overrides, ("qa", "enabled"), False if args.no_qa else None)

    cfg = load_config(args.config, overrides)

    report = caption_video(
        video,
        srt_path,
        args.output,
        cfg,
        words_json=args.words,
        edit_plan_json=args.edit_plan,
    )

    print(f"Captioned {report['entry_count']} entries")
    _print_paths(
        output=report["out_path"],
        qa_report=report.get("qa_report_path"),
        realigned_srt=report.get("realigned_srt_path"),
    )


def cmd_qa(args: argparse.Namespace) -> None:
    video = Path(args.video)
    srt_path = Path(args.srt)
    _require_exists(video, "Video file")
    _require_exists(srt_path, "SRT file")

    overrides: dict = {}
    _set(overrides, ("transcribe", "backend"), args.backend)
    _set(overrides, ("transcribe", "model"), args.model)
    _set(overrides, ("transcribe", "language"), args.language)
    _set(overrides, ("qa", "samples"), args.samples)
    _set(overrides, ("qa", "min_ratio"), args.min_ratio)

    cfg = load_config(args.config, overrides)

    print(f"Video: {video}")
    print(f"SRT: {srt_path}")
    result = qa_check(video, srt_path, cfg)
    report = result["report"]

    for s in report["samples"]:
        print(f"  #{s['index']} [{s['start']} --> {s['end']}] ratio={s['ratio']:.2f} ok={s['ok']}")
        print(f"    SRT: {s['srt_text']}")
        print(f"    ASR: {s['asr_text']}")
    print(
        f"avg_ratio={report['avg_ratio']:.2f} "
        f"({report['sampled']}/{report['eligible']} sampled, min_ratio={report['min_ratio']})"
    )
    _print_paths(qa_report=result["report_path"])

    if not report["passed"]:
        print("QA FAILED")
        sys.exit(1)
    print("QA PASSED")


def cmd_auto(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    _require_exists(input_path, "Input file")

    overrides: dict = {}
    _set(overrides, ("transcribe", "backend"), args.backend)
    _set(overrides, ("transcribe", "model"), args.model)
    _set(overrides, ("transcribe", "language"), args.language)
    _set(overrides, ("transcribe", "device"), args.device)
    _set(overrides, ("transcribe", "compute_type"), args.compute_type)
    _set(overrides, ("transcribe", "vad_filter"), args.vad_filter)
    _set(overrides, ("srt", "max_chars"), args.max_chars)
    _set(overrides, ("srt", "min_chars"), args.min_chars)
    _set(overrides, ("caption", "style"), args.style)
    _set(overrides, ("caption", "fast_mode"), args.fast)
    _set(overrides, ("caption", "realign"), args.realign)
    _set(overrides, ("qa", "enabled"), False if args.no_qa else None)

    cfg = load_config(args.config, overrides)

    report = auto_run(
        input_path,
        args.output,
        cfg,
        workdir=args.workdir,
        glossary_path=args.glossary,
    )

    print(f"Auto-captioned {report['caption']['entry_count']} entries")
    _print_paths(
        output=report["caption"]["out_path"],
        srt=report["srt_path"],
        qa_report=report["caption"].get("qa_report_path"),
    )


def cmd_prefetch(args: argparse.Namespace) -> None:
    from faster_whisper import download_model

    print(f"Downloading model {args.model!r}...")
    destination = download_model(args.model)
    print(f"Model {args.model!r} downloaded successfully to {destination}")


# --- argument parser ---------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alwayswhisper",
        description=(
            "Local-Whisper transcription and stylish burned-in captions "
            "(word-level timestamps, language-aware segmentation, "
            "typewriter effect, automatic AV QA)."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"alwayswhisper {__version__}"
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="Path to a YAML config file, layered over the packaged defaults.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- transcribe -----------------------------------------------------
    p = subparsers.add_parser(
        "transcribe", help="Transcribe a video/audio file to word timestamps + SRT."
    )
    p.add_argument("input", help="Input video or audio file.")
    p.add_argument(
        "-o", "--output", metavar="DIR",
        help="Output directory (default: <INPUT stem>_alwayswhisper next to INPUT).",
    )
    p.add_argument("--backend", help="Transcription backend: faster-whisper or openai-api.")
    p.add_argument("--model", help="Model name (faster-whisper only; openai-api always uses whisper-1).")
    p.add_argument("--language", help="Language code, e.g. ja, en.")
    p.add_argument("--device", help="faster-whisper device: cpu, cuda, or auto.")
    p.add_argument("--compute-type", dest="compute_type", help="faster-whisper compute type.")
    p.add_argument("--prompt", metavar="TEXT", help="Whisper bias prompt (overrides --glossary).")
    p.add_argument("--glossary", metavar="FILE", help="Text file of vocabulary to bias transcription toward.")
    p.add_argument("--max-chars", dest="max_chars", type=int, help="Max characters per caption entry.")
    p.add_argument("--min-chars", dest="min_chars", type=int, help="Min characters per caption entry.")
    p.add_argument(
        "--vad-filter", dest="vad_filter", action="store_true", default=None,
        help="Enable faster-whisper's voice-activity-detection filter.",
    )
    p.set_defaults(func=cmd_transcribe)

    # -- segment ----------------------------------------------------------
    p = subparsers.add_parser(
        "segment", help="Segment a transcript_words.json file into an SRT."
    )
    p.add_argument("words_json", help="Path to a transcript_words.json file.")
    p.add_argument("-o", "--output", required=True, metavar="OUT_SRT", help="Output SRT path.")
    p.add_argument("--language", help="Language code -- selects the segmenter (char-based for ja/zh/yue/th/lo/my, space-delimited otherwise).")
    p.add_argument("--max-chars", dest="max_chars", type=int, help="Max characters per caption entry.")
    p.add_argument("--min-chars", dest="min_chars", type=int, help="Min characters per caption entry.")
    p.set_defaults(func=cmd_segment)

    # -- caption ------------------------------------------------------------
    p = subparsers.add_parser("caption", help="Burn captions onto a video.")
    p.add_argument("video", help="Source video file.")
    p.add_argument("srt", help="SRT file to burn in.")
    p.add_argument("-o", "--output", required=True, metavar="OUT_MP4", help="Output video path.")
    p.add_argument("--words", metavar="FILE", help="transcript_words.json, required for --realign.")
    p.add_argument("--edit-plan", dest="edit_plan", metavar="FILE", help="edit_plan.json, used with --realign.")
    p.add_argument("--style", metavar="NAME_OR_PATH", help="Caption style: 'default', 'en', or a YAML file path.")
    p.add_argument("--fast", action="store_true", default=None, help="Fast libass/ffmpeg burn-in instead of MoviePy/PIL.")
    p.add_argument("--no-qa", dest="no_qa", action="store_true", default=None, help="Skip the AV QA spot check.")
    p.add_argument("--qa-samples", dest="qa_samples", type=int, help="Number of AV QA samples.")
    p.add_argument("--qa-min-ratio", dest="qa_min_ratio", type=float, help="Minimum AV QA match ratio.")
    p.add_argument("--realign", action="store_true", default=None, help="Snap SRT starts to word timestamps before burning.")
    p.add_argument("--backend", help="Transcription backend used for the AV QA re-transcription.")
    p.add_argument("--model", help="Model name for the AV QA backend.")
    p.add_argument("--language", help="Language code for the AV QA backend.")
    p.set_defaults(func=cmd_caption)

    # -- qa -------------------------------------------------------------
    p = subparsers.add_parser(
        "qa", help="Spot-check an SRT's captions against a video's audio."
    )
    p.add_argument("video", help="Source video file.")
    p.add_argument("srt", help="SRT file to check.")
    p.add_argument("--samples", type=int, help="Number of samples to check.")
    p.add_argument("--min-ratio", dest="min_ratio", type=float, help="Minimum match ratio to pass.")
    p.add_argument("--backend", help="Transcription backend used for re-transcription.")
    p.add_argument("--model", help="Model name for the QA backend.")
    p.add_argument("--language", help="Language code for the QA backend.")
    p.set_defaults(func=cmd_qa)

    # -- auto -----------------------------------------------------------
    p = subparsers.add_parser(
        "auto", help="End-to-end: transcribe + caption in one step."
    )
    p.add_argument("input", help="Input video file.")
    p.add_argument("-o", "--output", required=True, metavar="OUT_MP4", help="Output video path.")
    p.add_argument("--workdir", metavar="DIR", help="Working directory for intermediate transcription artifacts.")
    p.add_argument("--style", metavar="NAME_OR_PATH", help="Caption style: 'default', 'en', or a YAML file path.")
    p.add_argument("--fast", action="store_true", default=None, help="Fast libass/ffmpeg burn-in instead of MoviePy/PIL.")
    p.add_argument("--no-qa", dest="no_qa", action="store_true", default=None, help="Skip the AV QA spot check.")
    p.add_argument("--realign", action="store_true", default=None, help="Snap SRT starts to word timestamps before burning.")
    p.add_argument("--backend", help="Transcription backend: faster-whisper or openai-api.")
    p.add_argument("--model", help="Model name (faster-whisper only).")
    p.add_argument("--language", help="Language code, e.g. ja, en.")
    p.add_argument("--device", help="faster-whisper device: cpu, cuda, or auto.")
    p.add_argument("--compute-type", dest="compute_type", help="faster-whisper compute type.")
    p.add_argument("--glossary", metavar="FILE", help="Text file of vocabulary to bias transcription toward.")
    p.add_argument("--max-chars", dest="max_chars", type=int, help="Max characters per caption entry.")
    p.add_argument("--min-chars", dest="min_chars", type=int, help="Min characters per caption entry.")
    p.add_argument(
        "--vad-filter", dest="vad_filter", action="store_true", default=None,
        help="Enable faster-whisper's voice-activity-detection filter.",
    )
    p.set_defaults(func=cmd_auto)

    # -- prefetch ---------------------------------------------------------
    p = subparsers.add_parser(
        "prefetch", help="Pre-download a faster-whisper model (no transcription)."
    )
    p.add_argument("--model", default="large-v3", help="Model name to download (default: large-v3).")
    p.set_defaults(func=cmd_prefetch)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
    except _USER_FACING_ERRORS as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
