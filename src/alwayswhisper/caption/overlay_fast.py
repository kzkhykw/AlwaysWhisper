from __future__ import annotations

"""Fast caption burn-in using ffmpeg's libass-backed `ass` filter.

Drop-in alternative to overlay.add_captions_to_video that avoids the
MoviePy + PIL per-frame render path. Generates an .ass file from the
parsed SRT and runs a single ffmpeg pass with hardware-accelerated
encoding by default (on macOS).
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..core.srt_parser import SrtFile
from .ass_writer import srt_to_ass


# Homebrew's regular `ffmpeg` formula no longer enables libass; the
# `ffmpeg-full` formula does. We prefer that build (keg-only path) and fall
# back to the PATH ffmpeg so users on other distros / installs aren't blocked.
_FFMPEG_FULL_DIRS = [
    "/opt/homebrew/opt/ffmpeg-full/bin",  # Apple Silicon Homebrew
    "/usr/local/opt/ffmpeg-full/bin",     # Intel Homebrew
]


def _resolve_bin(name: str, env_override: str | None) -> str:
    if env_override and Path(env_override).exists():
        return env_override
    for ffmpeg_full_dir in _FFMPEG_FULL_DIRS:
        candidate = Path(ffmpeg_full_dir) / name
        if candidate.exists():
            return str(candidate)
    path_bin = shutil.which(name)
    if path_bin:
        return path_bin
    raise FileNotFoundError(
        f"Could not locate {name} (looked in {', '.join(_FFMPEG_FULL_DIRS)} and $PATH)"
    )


def _ffmpeg_bin() -> str:
    return _resolve_bin("ffmpeg", os.environ.get("FFMPEG_LIBASS_BIN"))


def _ffprobe_bin() -> str:
    return _resolve_bin("ffprobe", os.environ.get("FFPROBE_LIBASS_BIN"))


def _has_libass(ffmpeg: str) -> bool:
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-h", "filter=ass"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    # When libass is missing, ffmpeg prints "Unknown filter 'ass'."
    return "Unknown filter" not in out and "AVOptions" in out


def _libass_install_hint() -> str:
    """Per-OS guidance for installing/locating a libass-enabled ffmpeg."""
    if sys.platform == "darwin":
        return (
            "Install a libass-enabled ffmpeg: `brew install ffmpeg-full` "
            "(Homebrew's plain `ffmpeg` formula disables libass), or set "
            "FFMPEG_LIBASS_BIN to a libass-enabled ffmpeg binary."
        )
    if sys.platform.startswith("linux"):
        return (
            "Install ffmpeg with libass support: on Debian/Ubuntu, "
            "`apt install ffmpeg` normally ships libass already; on other "
            "distros install a libass-enabled ffmpeg build, or set "
            "FFMPEG_LIBASS_BIN to one."
        )
    if sys.platform.startswith("win"):
        return (
            "Install a full ffmpeg build with libass support (e.g. from "
            "https://www.gyan.dev/ffmpeg/builds/), or set "
            "FFMPEG_LIBASS_BIN to a libass-enabled ffmpeg.exe."
        )
    return (
        "Install a libass-enabled ffmpeg build, or set FFMPEG_LIBASS_BIN "
        "to a libass-enabled ffmpeg binary."
    )


def _probe_video_size(path: Path) -> tuple[int, int]:
    cmd = [
        _ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    w, h = out.split(",")
    return int(w), int(h)


def _default_encoder() -> str:
    """Auto-pick an encoder: hardware H.264 on macOS, libx264 elsewhere."""
    return "h264_videotoolbox" if sys.platform == "darwin" else "libx264"


def _quality_args(encoder: str, quality: int) -> list[str]:
    """Build the quality-control ffmpeg args for the chosen encoder."""
    if encoder == "libx264":
        return ["-crf", "18", "-preset", "medium"]
    # videotoolbox (and anything else) uses a -q:v target.
    return ["-q:v", str(quality)]


def add_captions_fast(
    video_path: str | Path,
    srt: SrtFile,
    output_path: str | Path,
    style: dict | None = None,
    encoder: str | None = None,
    quality: int = 60,
) -> None:
    """Burn captions into video via libass + a fast encoder.

    Args:
        video_path: Source video (mkv or mp4).
        srt: SRT entries (with timestamps already aligned).
        output_path: Output .mp4 path.
        style: Caption style dict (same shape as caption_style.yaml).
        encoder: ffmpeg video codec. Default None -> auto: h264_videotoolbox
            on macOS (sys.platform == "darwin"), libx264 elsewhere.
        quality: -q:v target used for videotoolbox (~0-100, higher = better).
            libx264 ignores this and always uses -crf 18 -preset medium.
    """
    style = style or {}
    video_path = Path(video_path)
    output_path = Path(output_path)
    encoder = encoder or _default_encoder()

    ffmpeg = _ffmpeg_bin()
    if not _has_libass(ffmpeg):
        raise RuntimeError(
            f"{ffmpeg} was built without libass — the 'ass' filter is unavailable.\n"
            f"{_libass_install_hint()}"
        )

    width, height = _probe_video_size(video_path)
    ass_content = srt_to_ass(srt, style, width, height)

    with tempfile.TemporaryDirectory(prefix="captionfast_") as td:
        ass_path = Path(td) / "captions.ass"
        ass_path.write_text(ass_content, encoding="utf-8")

        # `ass` filter takes the .ass path directly. Wrap in single quotes
        # so ffmpeg's filter parser treats it literally.
        subtitle_filter = f"ass='{ass_path}'"

        source_ext = video_path.suffix.lower()
        if source_ext == ".mkv":
            # mkv typically carries FLAC; mp4 needs AAC.
            audio_args = ["-c:a", "aac", "-b:a", "192k"]
        else:
            audio_args = ["-c:a", "copy"]

        cmd = [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-vf", subtitle_filter,
            "-c:v", encoder,
            *_quality_args(encoder, quality),
            *audio_args,
            "-movflags", "+faststart",
            str(output_path),
        ]
        print("  ffmpeg fast burn-in:")
        print("    " + " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg fast caption burn-in failed (rc={result.returncode})"
            )


__all__ = ["add_captions_fast"]
