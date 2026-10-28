"""macOS system audio -> bounded, clock-aligned PCM -> existing local ASR.

The optional microphone is mixed on the same timeline, never concatenated.
ScreenCaptureKit leaves headphone/output routing alone. This module neither
transcribes audio itself nor contacts a service.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

SAMPLE_RATE = 48000
HEADER = struct.Struct("<4sId")
BLOCK_FRAMES = 960  # one 20 ms timeline, even when two devices are active


def build_helper():
    if sys.platform != "darwin" or int(platform.mac_ver()[0].split(".")[0]) < 13:
        raise RuntimeError("内部音声の取り込みには macOS 13 以降が必要です")
    source = Path(__file__).with_name("system_audio_capture.swift")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = Path.home() / "Library/Caches/AlwaysWhisper"
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / "system-audio-capture"
    version = cache / "system-audio-capture.sha256"
    if binary.exists() and version.exists() and version.read_text() == digest:
        return binary
    fd, temporary = tempfile.mkstemp(prefix="system-audio-", dir=cache)
    os.close(fd)
    try:
        result = subprocess.run(
            ["xcrun", "swiftc", "-O", "-parse-as-library", str(source), "-o", temporary],
            capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError("内部音声ヘルパーをビルドできません。Xcode Command Line Toolsを確認してください。\n"
                               + result.stderr[-3000:])
        os.replace(temporary, binary)
        version.write_text(digest)
    except FileNotFoundError as exc:
        raise RuntimeError("内部音声の初回起動には Xcode Command Line Tools（xcrun/swiftc）が必要です") from exc
    finally:
        Path(temporary).unlink(missing_ok=True)
    return binary


def _read_exact(stream, size, *, allow_eof=False):
    chunks = bytearray()
    while len(chunks) < size:
        block = stream.read(size - len(chunks))
        if not block:
            if allow_eof and not chunks:
                return None
            raise ValueError("内部音声のPCMパケットが途中で切れました")
        chunks.extend(block)
    return chunks


def read_packet(stream):
    header = _read_exact(stream, HEADER.size, allow_eof=True)
    if header is None:
        return None
    magic, frames, timestamp = HEADER.unpack(header)
    if magic != b"AUD0" or not 0 < frames <= SAMPLE_RATE or not math.isfinite(timestamp):
        raise ValueError("内部音声のPCMパケット形式が不正です")
    pcm = np.frombuffer(_read_exact(stream, frames * 4), dtype="<f4").copy()
    if not np.isfinite(pcm).all():
        raise ValueError("内部音声に不正なサンプル値があります")
    return timestamp, pcm


class TimedAudioMixer:
    """Two-second ring buffer indexed by capture time, with one output clock."""

    def __init__(self, sources, *, sr=SAMPLE_RATE, origin=None, seconds=2):
        self.sr = sr
        self.origin = time.monotonic() if origin is None else origin
        self.capacity = int(sr * seconds)
        self.buffers = {name: np.zeros(self.capacity, np.float32) for name in sources}
        self.cursor = 0
        self.lock = threading.Lock()
        self.dropped_frames = 0

    def push(self, source, samples, started_at):
        samples = np.asarray(samples, dtype=np.float32).ravel()
        if not math.isfinite(started_at) or not np.isfinite(samples).all():
            return
        index = round((started_at - self.origin) * self.sr)
        # The first native packet can straddle READY / our timeline origin.
        # Trimming that pre-start portion is expected, not a capture dropout.
        if index < 0:
            samples = samples[min(-index, len(samples)):]
            index = 0
        with self.lock:
            lo, hi = max(index, self.cursor), min(index + len(samples), self.cursor + self.capacity)
            self.dropped_frames += len(samples) - max(0, hi - lo)
            if hi <= lo:
                return
            positions = np.arange(lo, hi) % self.capacity
            self.buffers[source][positions] = samples[lo - index:hi - index]

    def pull(self, frames):
        if not 0 <= frames <= self.capacity:
            raise ValueError("mixer read exceeds its bounded buffer")
        with self.lock:
            positions = np.arange(self.cursor, self.cursor + frames) % self.capacity
            mixed = np.zeros(frames, np.float32)
            for buffer in self.buffers.values():
                mixed += buffer[positions]
                buffer[positions] = 0
            self.cursor += frames
        return np.clip(mixed, -1, 1)

    def skip_to(self, frame):
        """After a stall/sleep, discard stale audio instead of replaying a burst."""
        with self.lock:
            if frame <= self.cursor:
                return
            positions = np.arange(self.cursor, min(frame, self.cursor + self.capacity)) % self.capacity
            for buffer in self.buffers.values():
                buffer[positions] = 0
            self.dropped_frames += frame - self.cursor
            self.cursor = frame


class SystemAudioTap:
    """Context-managed input source with the same on_chunk contract as _AudioTap."""

    def __init__(self, *, microphone_factory=None, app_ids=(), helper=None, latency=0.15):
        self.sr = SAMPLE_RATE
        self.on_chunk = None
        self.microphone_factory = microphone_factory
        self.app_ids = tuple(app_ids)
        self.helper = Path(helper) if helper else build_helper()
        self.latency = latency
        self.mixer = TimedAudioMixer(("system", "mic") if microphone_factory else ("system",))
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._process = None
        self._mic = None
        self._threads = []

    def _fail(self, detail):
        if not self._stop.is_set() and self._error is None:
            self._error = f"内部音声の取り込みを停止しました: {detail}"
            print(f"[system-audio] {self._error}", file=sys.stderr, flush=True)
        self._ready.set()

    def raise_if_failed(self):
        if self._error:
            raise RuntimeError(self._error)

    def _read_audio(self):
        try:
            while not self._stop.is_set():
                packet = read_packet(self._process.stdout)
                if packet is None:
                    self._fail("キャプチャプロセスが終了しました。権限と音声出力を確認してください")
                    return
                timestamp, samples = packet
                self.mixer.push("system", samples, timestamp)
        except Exception as exc:
            self._fail(str(exc))

    def _read_status(self):
        for raw in self._process.stderr:
            line = raw.decode("utf-8", errors="replace").strip()
            if line == "READY":
                self._ready.set()
            elif line.startswith("ERROR:"):
                self._fail(line)
            elif line:
                print(f"[system-audio] {line}", file=sys.stderr, flush=True)

    def _microphone_chunk(self, samples):
        if not self._stop.is_set():
            self.mixer.push("mic", samples, time.monotonic() - len(samples) / self.sr)

    def _emit(self, samples):
        if self.on_chunk is not None and len(samples):
            self.on_chunk(samples)

    def _pump(self):
        try:
            while not self._stop.is_set() and self._error is None:
                due = self.mixer.origin + (self.mixer.cursor + BLOCK_FRAMES) / self.sr + self.latency
                if self._stop.wait(max(0, due - time.monotonic())):
                    break
                lag = time.monotonic() - due
                if lag > 0.5:
                    self.mixer.skip_to(int((time.monotonic() - self.mixer.origin - self.latency) * self.sr))
                self._emit(self.mixer.pull(BLOCK_FRAMES))
        except Exception as exc:
            self._fail(str(exc))

    def __enter__(self):
        print("[system-audio] 内部音声を取り込みます。初回はmacOSの「画面収録とシステムオーディオ録音」を許可してください。",
              flush=True)
        try:
            self._process = subprocess.Popen(
                [str(self.helper), *self.app_ids], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for fn in (self._read_audio, self._read_status):
                thread = threading.Thread(target=fn, daemon=True, name=fn.__name__)
                self._threads.append(thread)
                thread.start()
            if not self._ready.wait(20):
                raise RuntimeError("内部音声の開始がタイムアウトしました。macOSの録音権限を確認し、再起動してください")
            self.raise_if_failed()
            self.mixer = TimedAudioMixer(self.mixer.buffers)
            if self.microphone_factory:
                self._mic = self.microphone_factory()
                self._mic.on_chunk = self._microphone_chunk
                self._mic.__enter__()
            thread = threading.Thread(target=self._pump, daemon=True, name="system-audio-mixer")
            self._threads.append(thread)
            thread.start()
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        ended_at = time.monotonic()
        self._stop.set()
        try:
            if self._mic is not None:
                self._mic.__exit__(exc_type, exc_val, exc_tb)
        finally:
            process = self._process
            if process is not None:
                try:
                    process.stdin.close()  # EOF asks the native stream to stop
                except OSError:
                    pass  # the helper may already have failed or closed stdin
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
                for thread in self._threads:
                    thread.join(timeout=1)
                process.stdout.close()
                process.stderr.close()
        # Preserve the latency buffer on normal shutdown before ASR finalizes.
        if exc_type in (None, KeyboardInterrupt) and self._error is None:
            remaining = min(self.mixer.capacity, max(0, round(
                (ended_at - self.mixer.origin) * self.sr) - self.mixer.cursor))
            self._emit(self.mixer.pull(remaining))
        if self.mixer.dropped_frames:
            print(f"[system-audio] 遅延・範囲外の音声サンプル: {self.mixer.dropped_frames}", file=sys.stderr)
        return False
