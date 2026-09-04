from __future__ import annotations

"""FFmpeg wrapper for video/audio processing."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoMetadata:
    """Video file metadata."""
    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str
    audio_codec: str
    file_size_bytes: int


class FFmpegError(Exception):
    """FFmpeg command execution error."""
    pass


def run_ffmpeg(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run ffmpeg command."""
    cmd = ["ffmpeg", "-y"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise FFmpegError(f"ffmpeg failed: {result.stderr[-500:]}")
    return result


def run_ffprobe(args: list[str]) -> subprocess.CompletedProcess:
    """Run ffprobe command."""
    cmd = ["ffprobe"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {result.stderr[-500:]}")
    return result


def get_metadata(video_path: str | Path) -> VideoMetadata:
    """Extract video metadata using ffprobe."""
    path = str(video_path)
    result = run_ffprobe([
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ])
    data = json.loads(result.stdout)

    video_stream = None
    audio_stream = None
    for stream in data.get("streams", []):
        if stream["codec_type"] == "video" and video_stream is None:
            video_stream = stream
        elif stream["codec_type"] == "audio" and audio_stream is None:
            audio_stream = stream

    fmt = data.get("format", {})

    fps = 30.0
    if video_stream and "r_frame_rate" in video_stream:
        num, den = video_stream["r_frame_rate"].split("/")
        if int(den) > 0:
            fps = int(num) / int(den)

    return VideoMetadata(
        duration_seconds=float(fmt.get("duration", 0)),
        width=int(video_stream.get("width", 0)) if video_stream else 0,
        height=int(video_stream.get("height", 0)) if video_stream else 0,
        fps=fps,
        codec=video_stream.get("codec_name", "") if video_stream else "",
        audio_codec=audio_stream.get("codec_name", "") if audio_stream else "",
        file_size_bytes=int(fmt.get("size", 0)),
    )


def extract_audio(input_path: str | Path, output_path: str | Path) -> None:
    """Extract audio from video as WAV."""
    run_ffmpeg([
        "-i", str(input_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(output_path),
    ])


def extract_audio_clip(
    video_path: str | Path,
    output_wav: str | Path,
    start_sec: float,
    end_sec: float,
) -> None:
    """Extract a short [start_sec, end_sec) audio clip as 16kHz mono WAV.

    Used by the AV QA spot check to pull a few seconds of audio around a
    sampled caption. `-ss` is placed as an INPUT option (before `-i`): per
    `man ffmpeg`, seeking there jumps to the nearest seek point and then --
    because we're transcoding (not stream-copying) and `-accurate_seek` is
    on by default -- ffmpeg decodes and discards the small remainder up to
    the exact position, so the seek is both fast and accurate. `-t`
    (duration, always relative regardless of placement) selects the clip
    length; `-to` is deliberately avoided here since combined with an
    input-side `-ss` its position is measured on the original timeline,
    which is a well-known footgun for short-clip extraction.
    """
    duration = max(0.0, end_sec - start_sec)
    run_ffmpeg([
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(output_wav),
    ])
