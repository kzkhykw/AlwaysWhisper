from __future__ import annotations

"""Video caption overlay using MoviePy."""

import subprocess
import tempfile

import numpy as np
from pathlib import Path

from moviepy import VideoFileClip, VideoClip, CompositeVideoClip
from PIL import Image

from ..core.srt_parser import SrtFile, SrtEntry
from .renderer import render_caption_frame
from .typewriter import apply_cursor, calculate_visible_chars


def _create_caption_clip(
    entry: SrtEntry,
    frame_width: int,
    frame_height: int,
    style: dict,
) -> VideoClip:
    """Create a VideoClip for a single caption entry with typewriter effect."""
    total_duration_ms = entry.duration_ms
    total_chars = len(entry.text)
    start_sec = entry.start.to_ms() / 1000
    duration_sec = total_duration_ms / 1000

    anim_config = style.get("animation", {})
    completion_ms = anim_config.get("completion_ms", 400)

    def make_frame(t):
        elapsed_ms = int(t * 1000)
        visible = calculate_visible_chars(
            elapsed_ms, total_duration_ms, total_chars,
            completion_ms,
        )
        visible_text = entry.text[:visible]
        visible_text = apply_cursor(visible_text, visible, total_chars)
        frame = render_caption_frame(
            visible_text, frame_width, frame_height, style,
            full_text=entry.text,
        )
        return np.array(frame)

    clip = VideoClip(make_frame, duration=duration_sec)
    clip = clip.with_start(start_sec)
    return clip


def add_captions_to_video(
    video_path: str | Path,
    srt: SrtFile,
    output_path: str | Path,
    style: dict | None = None,
) -> None:
    """Add captions with typewriter effect to video.

    Writes video-only with MoviePy, then muxes the original audio
    via ffmpeg stream copy to prevent A/V sync drift.

    Args:
        video_path: Input video file
        srt: SRT subtitle data
        output_path: Output video file
        style: Caption style configuration
    """
    if style is None:
        style = {}

    video = VideoFileClip(str(video_path))
    frame_width = video.w
    frame_height = video.h
    fps = video.fps

    # Create caption clips
    caption_clips = []
    for entry in srt.entries:
        if entry.text.strip():
            clip = _create_caption_clip(entry, frame_width, frame_height, style)
            caption_clips.append(clip)

    if not caption_clips:
        # No captions, just copy
        import shutil
        shutil.copy2(video_path, output_path)
        video.close()
        return

    # Composite and write video-only (no audio) to a temp file.
    # This avoids MoviePy re-encoding audio which introduces A/V drift.
    final = CompositeVideoClip([video] + caption_clips)

    video_only_path = Path(output_path).with_suffix(".video_only.mp4")
    try:
        final.write_videofile(
            str(video_only_path),
            codec="libx264",
            fps=fps,
            audio=False,
            logger="bar",
        )
        video.close()
        final.close()

        # Mux: take video from captioned file, audio from original.
        # If source is .mkv (FLAC audio), encode to AAC for .mp4 output.
        # Otherwise stream copy audio.
        source_ext = Path(video_path).suffix.lower()
        if source_ext == ".mkv":
            audio_args = ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
        else:
            audio_args = ["-c", "copy"]
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_only_path),
            "-i", str(video_path),
            "-map", "0:v",
            "-map", "1:a",
            *audio_args,
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed: {result.stderr[-500:]}")
    finally:
        if video_only_path.exists():
            video_only_path.unlink()
