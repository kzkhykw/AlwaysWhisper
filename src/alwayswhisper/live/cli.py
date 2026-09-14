"""Microphone captions on Apple Silicon Macs; optional dependencies load at runtime."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import multiprocessing
from pathlib import Path
import platform
import sys
import time

from . import caption_overlay

DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_TRANSCRIPTS = Path.home() / "Library/Application Support/AlwaysWhisper/transcripts"


def positive_float(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def configure_parser(parser):
    parser.add_argument("--captions", action=argparse.BooleanOptionalAction, default=True,
                        help="show live captions (default on; --no-captions for terminal only)")
    parser.add_argument("--captions-position", choices=caption_overlay.CAPTION_POSITIONS, default=None,
                        help="saved position, initially dynamic-island; free restores the draggable overlay")
    parser.add_argument("--font-size", type=int, choices=range(12, 97), metavar="12..96", default=28,
                        help="initial caption font size; saved size wins on later runs")
    parser.add_argument("--device", help="microphone name or device index")
    parser.add_argument("--list-devices", action="store_true", help="list microphone devices and exit")
    parser.add_argument("--sample-rate", type=positive_float, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MLX Whisper model repository")
    parser.add_argument("--language", default="ja", help="language code, or auto for detection")
    parser.add_argument("--glossary", type=Path, help="UTF-8 vocabulary/bias prompt file")
    parser.add_argument("--transcript-dir", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--gate", type=positive_float, default=.006, help="speech detection RMS threshold")
    parser.add_argument("--max-sec", type=positive_float, default=8, help="maximum speech segment duration")
    parser.add_argument("--silence-sec", type=positive_float, default=.7, help="pause duration to end a segment")
    parser.add_argument("--demo", action="store_true", help="preview captions without a microphone or ASR model")
    parser.set_defaults(func=run)
    return parser


@contextmanager
def session_lock():
    """Prevent two writers from appending to the same daily transcript."""
    import fcntl
    path = Path.home() / ".config/alwayswhisper/live.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("AlwaysWhisper live is already running; stop it before starting another session.") from None
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def run(args):
    if sys.platform != "darwin":
        raise RuntimeError("Live captions require macOS. File transcription remains available on this platform.")
    if args.demo:
        caption_overlay._require_appkit()
        caption_overlay._run_demo(args.font_size, position=args.captions_position)
        return
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError("Install microphone support with: pip install 'alwayswhisper[live]'") from None
    if args.list_devices:
        print(sd.query_devices())
        return
    if platform.machine() != "arm64":
        raise RuntimeError("Live microphone transcription uses MLX Whisper and requires Apple Silicon.")
    if importlib.util.find_spec("mlx_whisper") is None:
        raise RuntimeError("Install local ASR with: pip install 'alwayswhisper[live]'")
    if args.captions:
        caption_overlay._require_appkit()
    device = int(args.device) if args.device and args.device.isdecimal() else args.device
    info = sd.query_devices(device, "input")
    sr = int(args.sample_rate or info["default_samplerate"])
    prompt = args.glossary.read_text(encoding="utf-8").strip() if args.glossary else None
    from .live_transcriber import LiveTranscriber
    with session_lock():
        display_queue = multiprocessing.get_context("spawn").Queue(maxsize=512) if args.captions else None
        transcriber = LiveTranscriber(
            sr, args.model, None if args.language == "auto" else args.language,
            str(args.transcript_dir.expanduser()), prompt=prompt,
            display_queue=display_queue, gate=args.gate,
            max_sec=args.max_sec, silence_sec=args.silence_sec,
        )
        def callback(indata, frames, timing, status):
            transcriber.feed(indata[:, 0])
        try:
            transcriber.start()
            print(f"Microphone: {info['name']} | transcripts: {args.transcript_dir}")
            print("Press Ctrl-C to stop and finish writing the transcript.")
            with sd.InputStream(device=device, channels=1, samplerate=sr,
                                dtype="float32", callback=callback):
                if args.captions:
                    caption_overlay.run_console_overlay(
                        display_queue, font_size=args.font_size, position=args.captions_position,
                        level_source=transcriber.get_audio_level,
                    )
                else:
                    while True:
                        time.sleep(.25)
        except KeyboardInterrupt:
            pass
        finally:
            summary = transcriber.stop()
            if display_queue is not None:
                display_queue.close()
            print(f"Transcript: {summary['jsonl_path']}")


def main(argv=None):
    parser = configure_parser(argparse.ArgumentParser(description=__doc__))
    try:
        run(parser.parse_args(argv))
    except (RuntimeError, OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
