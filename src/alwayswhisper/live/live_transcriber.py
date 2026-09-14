"""Local microphone transcription in a spawned worker process.

The parent feeds timestamped PCM and exposes a latest-value RMS sample for
the live meter. The worker segments speech, runs MLX Whisper, streams caption
events, and appends session JSONL and daily Markdown transcripts.
"""
import datetime
import json
import multiprocessing
import os
import queue
import re
import sys
import time
import unicodedata
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- pure logic ---
def resample_to_16k(x, sr):
    """Linear-resample 1-D float32 `x` at `sr` Hz to 16000 Hz (mlx-whisper's
    expected rate). Passthrough (exact copy) when already 16k."""
    x = np.asarray(x, dtype=np.float32)
    if sr == 16000:
        return x.copy()
    n_in = len(x)
    if n_in == 0:
        return np.zeros(0, dtype=np.float32)
    n_out = max(1, int(round(n_in * 16000.0 / sr)))
    src_idx = np.arange(n_in, dtype=np.float64)
    dst_idx = np.arange(n_out, dtype=np.float64) * (sr / 16000.0)
    y = np.interp(dst_idx, src_idx, x.astype(np.float64))
    return y.astype(np.float32)


# ---------------------------------------------------------------- speech gate ---
# Absolute floor of SegmentBuffer's speech-ENTER threshold, in RMS. push() takes
# max(adaptive_floor * 4.0, GATE_DEFAULT), and in a quiet room the adaptive floor
# collapses toward the room tone -- so this constant, not the adaptive part, is
# what a quiet voice actually has to clear before a segment can open at all.
#
# It was 0.015 (~-36.5 dBFS) until 2026-09-03, which is loud-conversational:
# softly spoken Japanese sits nearer 0.006-0.010 RMS and never opened a segment,
# no matter how quiet the room was (reported as "小さい声だと反応しない").
# Lowered to 0.006 (~-44 dBFS). The false speech starts that admits are cheap:
# they still have to clear min_speech_sec to be emitted at all, and whatever
# noise does reach the model is dropped again by clean_transcription's
# no_speech_prob gate. Override per run with alwayswhisper live --gate.
GATE_DEFAULT = 0.006

# Longest a single speech segment may grow before it is cut anyway, in seconds.
# Nobody pausing means nothing to transcribe yet, so this is what bounds how
# long the screen can stay empty while someone talks: the lag on continuous
# speech is essentially this number (inference measured at ~20x realtime on
# large-v3 -- 1.26s for a 25s clip -- so the model is never the wait). The cost
# of lowering it is that a long sentence gets cut mid-way, with less context
# for that decode. Override per run with alwayswhisper live --max-sec.
#
# 8 since 2026-09-03 (was 25, owner request): a live-caption run should never
# show nothing for 25 seconds while someone talks through their point. Measured
# segment lengths put p75 at 6.2s, so most utterances end on their own pause
# well before this and are unaffected.
MAX_SEC_DEFAULT = 8.0

# How long a speaker has to stay quiet before their segment is cut and sent to
# the model: the fixed part of the gap between finishing a sentence and seeing
# it on screen. Lower is snappier but splits sentences at ordinary mid-thought
# pauses, giving each decode less context. Override with
# alwayswhisper live --silence-sec.
#
# 0.4 since 2026-09-03 (was 0.8, owner request): with inference measured at
# 0.3-0.5s for a typical utterance, this was most of what stood between
# finishing a sentence and reading it -- the ask was for it to fire as soon as
# the speaker stops.
SILENCE_SEC_DEFAULT = 0.4

# Ceiling on the tokens Whisper may sample for one segment, as
# DECODE_TOKENS_PER_SEC * audio_seconds + DECODE_TOKENS_BASE.
#
# Whisper loops on non-speech ("サイト、サイト、サイト…") and keeps generating
# until it runs out of room: measured 2026-09-03 on a live session, near-silent
# 1.3-1.8s segments cost 8-20 SECONDS of inference each -- versus ~0.3s for
# real speech of the same length -- and while the worker sat in one of those,
# incoming audio queued up and pushed the lag on the next real utterance to
# 20-29s. Capping the sample length bounds the loop instead of paying for it
# (measured on a 1.5s noise clip: 11.2s -> 0.86s, with real speech byte-
# identical and no faster or slower).
#
# The rate is the margin: real Japanese speech measured 2.9-5.8 tokens/s across
# six clips, so 16/s is about 3x the fastest measured rate. A hit can still
# mean truncated speech or a hallucination; retry without the glossary once
# when possible, then skip persistent hits with a diagnostic ("token cap").
DECODE_TOKENS_PER_SEC = 16
DECODE_TOKENS_BASE = 32


def decode_sample_len(audio_sec):
    """Token ceiling for `audio_sec` of audio -- see DECODE_TOKENS_PER_SEC."""
    return int(DECODE_TOKENS_PER_SEC * audio_sec) + DECODE_TOKENS_BASE


def reached_token_cap(result, sample_len):
    """Conservative cap check from the tokens exposed by mlx-whisper.

    sample_len applies to each decode window (seek), not the entire result.
    Sub-segments from the same window share that budget. The backend may
    discard unfinished tokens, so this detects visible hits, not every hit.
    """
    counts = {}
    for segment in result.get("segments") or []:
        seek = segment.get("seek", 0)
        counts[seek] = counts.get(seek, 0) + len(segment.get("tokens") or [])
    return any(count >= sample_len for count in counts.values())

# The speech-STAY threshold's absolute floor, as a fraction of the enter floor.
# Hysteresis: once speech is under way a lower bar holds it, so a trailing-off
# sentence end or the quiet gap between syllables doesn't chop one utterance
# into fragments. Scales with the gate so tuning one number tunes both.
STAY_GATE_RATIO = 0.5


@dataclass
class AudioSegment:
    samples: np.ndarray  # float32, device sr (not yet resampled)
    start_s: float        # session-relative seconds
    end_s: float
    # Unix timestamps (time.time() basis), filled in by SegmentBuffer._emit()
    # via wall_of() -- None when no push() call has ever carried a wall
    # stamp yet (legacy callers that never pass wall=, or a segment emitted
    # before the very first stamped chunk arrived). See SegmentBuffer.wall_of
    # and TranscriptWriter.write's wall_start/wall_end parameters, which this
    # feeds (via _worker -- see live_transcriber's module docstring on the
    # wall-clock-primary redesign).
    wall_start: float | None = None
    wall_end: float | None = None


class SegmentBuffer:
    """Chunk-granular speech segmenter: push() arbitrary-size audio chunks,
    get back zero or more AudioSegments cut on trailing silence.

    Each pushed chunk is classified as a whole (its own RMS vs. an adaptive
    noise floor) -- there is no within-chunk analysis -- so segment
    boundaries resolve to within one chunk's duration. That is the deliberate
    trade-off that keeps this deterministic and independent of chunk size.

    pos_s exposes the running total of samples pushed so far, in seconds, on
    the exact same absolute basis as every emitted AudioSegment's start_s/
    end_s -- see pos_s's own docstring below -- for callers (namely _worker)
    that need a live position between segments, e.g. to measure idle silence
    against the last emitted segment's end_s.

    push(chunk, wall=...) additionally accepts the real wall-clock instant
    (time.time() basis, stamped by the caller at capture time -- see
    LiveTranscriber.feed) that chunk was captured at. wall_of()/wall_now
    derive the wall-clock time of any absolute sample position from
    whichever chunk was stamped most recently, so a caller can recover real
    wall time even across a gap where the audio clock itself stalled (e.g.
    the Mac sleeping: CoreAudio delivers no frames, so self._total stops
    advancing while wall time keeps going -- see the module docstring). A
    caller that never passes wall= (every existing test, and any other
    legacy use) leaves wall_of()/wall_now returning None forever, exactly as
    if this feature did not exist.
    """

    def __init__(self, sr, silence_sec=SILENCE_SEC_DEFAULT, min_speech_sec=0.2, max_sec=MAX_SEC_DEFAULT, pre_roll_sec=0.3,
                 gate=GATE_DEFAULT):
        self.sr = sr
        self.silence_sec = silence_sec
        # 0.2s (was 0.3s until 2026-09-03): 0.3s discarded whole one-word
        # replies ("はい", "うん"), and quiet speech loses further chunks to
        # the stay threshold before this even gets counted.
        self.min_speech_sec = min_speech_sec
        self.max_sec = max_sec
        self.pre_roll_sec = pre_roll_sec
        self.gate = float(gate)
        self._max_samples = int(round(max_sec * sr))
        self._preroll_n = int(round(pre_roll_sec * sr))

        self._floor = 0.005
        self._speaking = False   # hysteresis: are we currently "in speech"?
        self._active = False     # are we buffering a candidate segment?
        self._total = 0          # absolute samples pushed so far

        # Most recent wall-clock stamp seen by push(chunk, wall=...), and the
        # absolute sample position at the end of that same chunk -- together
        # the one reference point wall_of() extrapolates every other
        # position from. _last_wall stays None (and wall_of()/wall_now keep
        # returning None) until the first stamped chunk arrives; see wall_of.
        self._last_wall = None
        self._last_wall_pos = 0

        self._buf_chunks = []
        self._buf_len = 0
        self._buf_start = 0      # absolute sample index of buf_chunks[0]
        self._speech_len = 0     # samples classified as speaking within buf
        self._silence_len = 0    # trailing samples classified as not-speaking

        self._pre_chunks = []
        self._pre_len = 0

    @property
    def pos_s(self):
        """Absolute position, in seconds, of every sample pushed so far
        (self._total / self.sr) -- silence and speech alike, since
        self._total accumulates unconditionally at the top of push(), before
        any speech/silence classification happens. Shares the exact same
        coordinate system as every emitted AudioSegment's start_s/end_s: both
        derive from the same cumulative sample count on this same native sr,
        never resampled or reset independently of it, so a caller can compare
        a live pos_s directly against a previously emitted segment's end_s --
        e.g. _worker calls writer.maybe_flush(buf.pos_s) to measure idle
        silence against the last transcribed segment's end_s -- with no unit
        conversion or offset needed."""
        return self._total / self.sr

    def wall_of(self, pos):
        """Wall-clock time (time.time() basis) of absolute sample position
        `pos` (same coordinate system as pos_s/_total, but in samples, not
        seconds), derived from the single most recently stamped chunk:
        self._last_wall - (self._last_wall_pos - pos) / self.sr. That
        subtraction is what makes this self-correcting across a gap where
        the stream clock stalled but wall time didn't (e.g. the Mac
        sleeping -- see the module docstring): each new stamp replaces the
        reference point outright, so a position is always extrapolated from
        the *nearest* known wall time rather than accumulating drift from a
        stamp that predates a sleep. Returns None when no chunk has ever
        been stamped yet (self._last_wall is None) -- in particular, for
        every caller that never passes wall= to push(), forever, matching
        the legacy no-wall behavior exactly."""
        if self._last_wall is None:
            return None
        return self._last_wall - (self._last_wall_pos - pos) / self.sr

    @property
    def wall_now(self):
        """wall_of(self._total): the wall-clock instant of the most recently
        pushed sample ("now"). Exposed for the idle-flush path -- _worker
        passes this to writer.maybe_flush() the same way it passes pos_s,
        see TranscriptWriter.maybe_flush."""
        return self.wall_of(self._total)

    def _pre_push(self, chunk):
        self._pre_chunks.append(chunk)
        self._pre_len += len(chunk)
        while len(self._pre_chunks) > 1 and self._pre_len - len(self._pre_chunks[0]) >= self._preroll_n:
            removed = self._pre_chunks.pop(0)
            self._pre_len -= len(removed)

    def _take_preroll(self):
        if not self._pre_chunks:
            return np.zeros(0, dtype=np.float32)
        pre = self._pre_chunks[0] if len(self._pre_chunks) == 1 else np.concatenate(self._pre_chunks)
        if len(pre) > self._preroll_n:
            pre = pre[-self._preroll_n:]
        return pre

    def _emit(self, force):
        samples = self._buf_chunks[0] if len(self._buf_chunks) == 1 else np.concatenate(self._buf_chunks)
        start_s = self._buf_start / self.sr
        end_s = (self._buf_start + self._buf_len) / self.sr
        keep = force or (self._speech_len / self.sr >= self.min_speech_sec)
        # wall_of() as of right now (before the buffer/position bookkeeping
        # below runs) -- self._buf_start/_buf_len are still the emitted
        # segment's own, so these are exactly that segment's wall_start/end.
        # Both are None together whenever no chunk has ever been stamped,
        # i.e. every legacy no-wall caller -- see wall_of's own docstring.
        seg = AudioSegment(samples=samples, start_s=start_s, end_s=end_s,
                            wall_start=self.wall_of(self._buf_start),
                            wall_end=self.wall_of(self._buf_start + self._buf_len)) if keep else None

        if force:
            # Ongoing speech flows straight into the next segment: no pre-roll,
            # buffer resumes exactly where this one ended (stays contiguous).
            self._buf_chunks = []
            self._buf_len = 0
            self._buf_start = self._buf_start + len(samples)
        else:
            self._buf_chunks = []
            self._buf_len = 0
            self._pre_chunks = []
            self._pre_len = 0
        self._speech_len = 0
        self._silence_len = 0
        return seg

    def push(self, chunk, wall=None):
        """wall, if given, is the wall-clock instant (time.time() basis)
        this chunk was captured at -- see LiveTranscriber.feed(), which is
        the real caller that supplies it. Recorded as the new "most recently
        stamped chunk" reference point (self._last_wall/_last_wall_pos) that
        wall_of() extrapolates from; a call with wall=None (the default --
        every existing caller/test) leaves that reference point untouched,
        so wall_of()/wall_now keep returning whatever they already were
        (None, if no call has ever passed wall=)."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if len(chunk) == 0:
            return []
        chunk_start = self._total
        self._total += len(chunk)
        if wall is not None:
            self._last_wall = wall
            self._last_wall_pos = self._total

        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < self._floor:
            self._floor = 0.9 * self._floor + 0.1 * rms
        else:
            self._floor = 0.999 * self._floor + 0.001 * rms

        enter_thr = max(self._floor * 4.0, self.gate)
        stay_thr = max(self._floor * 2.2, self.gate * STAY_GATE_RATIO)
        self._speaking = (rms >= stay_thr) if self._speaking else (rms >= enter_thr)

        if not self._active:
            if self._speaking:
                pre = self._take_preroll()
                self._pre_chunks = []  # consumed into buf below; idle accumulation restarts fresh
                self._pre_len = 0
                self._buf_chunks = [pre, chunk] if len(pre) else [chunk]
                self._buf_len = len(pre) + len(chunk)
                self._buf_start = chunk_start - len(pre)
                self._speech_len = len(chunk)
                self._silence_len = 0
                self._active = True
            else:
                self._pre_push(chunk)
                return []
        else:
            self._buf_chunks.append(chunk)
            self._buf_len += len(chunk)
            if self._speaking:
                self._speech_len += len(chunk)
                self._silence_len = 0
            else:
                self._silence_len += len(chunk)

        if self._buf_len >= self._max_samples:
            seg = self._emit(force=True)
            return [seg] if seg is not None else []
        if self._silence_len / self.sr >= self.silence_sec:
            seg = self._emit(force=False)
            self._active = False
            return [seg] if seg is not None else []
        return []

    def flush(self):
        """Emit whatever is currently buffered (e.g. at shutdown), still
        subject to min_speech_sec. Safe to call repeatedly (returns None once
        drained)."""
        if not self._active or self._buf_len == 0:
            self._active = False
            return None
        seg = self._emit(force=False)
        self._active = False
        return seg


# The "## " session-header line a fresh day's md file gets is written in two
# shapes: the plain one (TranscriptWriter.__init__, a session that actually
# started that day) and the "continuation" one below (TranscriptWriter.write,
# a session that's rolling its output over from an earlier day). Both the H1
# and the continuation header are also needed, byte-for-byte, by
# backfill_transcript_days.py when it splits an *old* elapsed-timestamp file
# into per-day files after the fact -- so both are module-level functions
# here (not inlined into the class) and that script imports them rather than
# re-deriving the format. See TranscriptWriter's docstring and
# backfill_transcript_days.py's module docstring for the full story.
def format_day_h1(date_str):
    """The single H1 heading written once at the top of a fresh/empty day's
    md file: '# {date} 文字起こし'."""
    return f"# {date_str} 文字起こし\n"


CONTINUATION_MARKER = "から継続"  # substring that marks a header as a rollover continuation


def format_continuation_header(session_hms, model_and_lang, start_date_str):
    """The '## ' header written when a session's md output rolls over past
    local midnight into a new day's file: same shape as the plain session
    header, plus a trailing note identifying which earlier day the session
    actually started on, so a reader opening this file understands the
    session is a holdover, not a fresh start. `model_and_lang` is the exact
    "MODEL, lang=LANG" text that goes inside the parens -- callers already
    have it in that combined shape (TranscriptWriter builds it from
    self._model/self._language; backfill_transcript_days.py lifts it
    verbatim from the old header via SESSION_RE) so this function doesn't
    need to know how it was assembled, only reuse it unchanged."""
    return f"\n## {session_hms} セッション ({model_and_lang}) — {start_date_str}{CONTINUATION_MARKER}\n\n"


def _wall_dt(ts):
    """Convert a time.time()-basis Unix timestamp to a local, timezone-aware
    datetime -- the one place TranscriptWriter.write() turns a wall_start/
    wall_end float into the datetime it stamps bullets/decides rollover
    with and serializes into the jsonl "wall" field. datetime.fromtimestamp
    (no tz arg) already interprets `ts` in the system's local timezone;
    .astimezone() only attaches that same offset (making the result
    tz-aware) without changing the moment it represents -- consistent with
    meta["started_at"], which LiveTranscriber also writes via
    started.astimezone().isoformat()."""
    return datetime.datetime.fromtimestamp(ts).astimezone()


class TranscriptWriter:
    """Appends transcript segments to a per-session .jsonl (machine-readable,
    source of truth, always session-relative seconds) and to human-readable
    per-day .md file(s) opened in APPEND mode. Normally that's one file, but
    a session that keeps running past local midnight does NOT keep
    appending to its start-day file forever: write() rolls the md output
    over to <next-day>.md (and, if the session keeps going, the day after
    that, and so on -- see write()/_rollover()) the moment a segment's
    wall-clock time crosses into a new calendar day, so a day's file only
    ever holds that day's speech. Flushes after every write so a tail -f /
    concurrent reader always sees the latest text.

    The md side does not write one bulleted line per recognized segment --
    that read as choppy, sentence-fragment-per-line output (the complaint
    that prompted this redesign, 2026-07-27). A paragraph is one burst of
    speech: write() keeps at most one md line "open" (not yet newline-
    terminated) at a time, and starts a NEW line only when the silence gap
    since the previous segment written to the open line -- the new
    segment's start_s minus the previous one's end_s -- is >=
    paragraph_gap_sec (default 60s/1min; see write()), a day rolls over
    (_rollover(), see below), or the writer closes (close()). A fourth
    trigger closes the line even when no next segment ever arrives to make
    write() discover the gap: maybe_flush(now_s), called by _worker once per
    queue item with now_s=buf.pos_s (see SegmentBuffer.pos_s), closes the
    line the instant paragraph_gap_sec of *idle* silence has elapsed, so a
    long pause (or simply the absence of any further speech) still reaches
    Notion on the next sync tick instead of waiting -- possibly indefinitely
    -- for whatever utterance eventually breaks the silence (see
    maybe_flush's own docstring below). Otherwise the
    new segment's text is appended to the open line with a single
    half-width space -- no new stamp, no new line; only a paragraph's first
    segment gets the "- [HH:MM:SS]" stamp. This is gap-based, not
    wall-clock-bucketed (an earlier design this one replaced outright): two
    segments a few seconds apart still join even if they straddle a
    15-minute, hourly, or any other wall-clock mark, since it's the actual
    silence between them that's compared against paragraph_gap_sec, not
    which window either one's timestamp falls in. A segment that
    clean_transcription() drops (silence/hallucination) never reaches
    write(), so the audio time it covered is correctly counted as part of
    the surrounding silence when the next real segment's gap is computed.
    self._last_end_s holds the session-relative end_s of the last segment
    written to the currently-open line, or None when no line is open in the
    currently-open md file; write()/_rollover()/close()/maybe_flush() are the
    only four places that touch it, and whichever of them last terminated a
    line always did so by writing a bare "\\n". That keeps two properties
    sync_transcripts_notion.py's incremental line cursor depends on intact
    (see that module's docstring): the file's already-written bytes are
    never rewritten, only ever appended to (an unterminated trailing line is
    simply not consumed yet), and every terminated line keeps the exact
    "- [HH:MM:SS] text" shape it always had.

    Every wall-clock value this class produces -- both the rollover decision
    and each paragraph line's "- [HH:MM:SS]" stamp -- prefers the real
    wall-clock instant a segment was captured at, write(..., wall_start=,
    wall_end=), when the caller supplies it (see _worker, which passes
    AudioSegment.wall_start/wall_end -- themselves from SegmentBuffer.
    wall_of(), stamped at capture time by LiveTranscriber.feed(); see the
    module docstring). That's the PRIMARY source, added 2026-09 to fix a
    real bug: the original design below derived every wall-clock value from
    self._session_start + start_s, a STREAM clock (cumulative captured-
    sample count) that stops advancing whenever the audio callback stops
    receiving frames -- e.g. the Mac sleeping -- while wall time keeps
    going, so a post-sleep utterance was stamped and filed as if no time had
    passed at all. self._session_start + start_s (self._session_start parsed
    *once*, in __init__, from meta["started_at"]) remains as the fallback
    whenever a caller omits wall_start/wall_end (every existing caller
    predating this fix, and every test that doesn't pass them), reproducing
    the original behavior byte for byte -- deliberately, since it's also
    what keeps this class deterministic/testable without a wall clock: a
    test fixes started_at and start_s and knows in advance exactly which
    file and which stamp will result, no matter when the test itself runs.

    meta must include (on top of the existing started_at/samplerate/model/
    language keys written verbatim to the jsonl meta line):
      - date_str: "YYYY-MM-DD" for the H1 written when the file this session
        starts on (self.md_path) is fresh/empty. Matches the calendar date
        of started_at (LiveTranscriber derives both from the same moment).
      - session_hms: "HH:MM:SS" session-start time for the very first "## "
        session header (written once, in __init__, on self.md_path). A
        later midnight rollover writes its own continuation "## " header
        instead (see write()), derived from self._session_start rather than
        this field -- the two always agree, session_hms just saves __init__
        from reformatting started_at itself.

    paragraph_gap_sec (default 60.0) is the silence-gap threshold described
    above. write() compares it against the wall-clock gap (wall_start -
    self._last_wall_end) when both are available, else falls back to the
    session-relative stream gap (start_s - self._last_end_s) -- the wall gap
    wins specifically so a sleep gap still forces a new paragraph even
    though it barely moved the stream clock at all (a few seconds of actual
    audio silence around the sleep, however many real hours it spanned).
    self._last_wall_end mirrors self._last_end_s exactly (same four places
    read/write it: write()/_rollover()/close()/maybe_flush()), holding the
    last segment-on-the-open-line's wall_end, or None whenever
    self._last_end_s is None OR that segment didn't carry a wall_end.
    maybe_flush(now_s, wall_now=None) applies this exact same preference the
    same way, just with now_s/wall_now (in practice the caller's current
    SegmentBuffer.pos_s/wall_now) standing in for what would otherwise be
    the next segment's start_s/wall_start -- see maybe_flush below.
    """

    def __init__(self, jsonl_path, md_path, meta, paragraph_gap_sec=60.0):
        self.jsonl_path = jsonl_path
        self.md_path = md_path
        self._jsonl_f = open(jsonl_path, "w", encoding="utf-8")
        self._jsonl_f.write(json.dumps({"type": "meta", **meta}, ensure_ascii=False) + "\n")
        self._jsonl_f.flush()

        # Parsed once, here -- see the class docstring: every later
        # wall-clock computation derives from this instead of datetime.now().
        self._session_start = datetime.datetime.fromisoformat(meta["started_at"])
        self._model = meta["model"]
        self._language = meta.get("language")
        self._out_dir = os.path.dirname(md_path)
        self._md_day = meta["date_str"]  # calendar date of the currently-open md file
        self._paragraph_gap_sec = paragraph_gap_sec  # silence-gap threshold, see class docstring
        self._last_end_s = None  # end_s of the last segment on the open md line, or None if none is open
        self._last_wall_end = None  # wall_end of that same segment, or None -- see class docstring

        is_fresh = not os.path.exists(md_path) or os.path.getsize(md_path) == 0
        self._md_f = open(md_path, "a", encoding="utf-8")
        if is_fresh:
            self._md_f.write(format_day_h1(meta["date_str"]))
        language = meta.get("language") or "auto"
        # The leading "\n" here also terminates any dangling unterminated
        # line a crashed earlier session left open in this same (appended-
        # to) file -- see the class docstring's append-only/line-cursor note.
        self._md_f.write(f"\n## {meta['session_hms']} セッション ({meta['model']}, lang={language})\n\n")
        self._md_f.flush()

    def write(self, start_s, end_s, text, wall_start=None, wall_end=None):
        """wall_start/wall_end (time.time() basis -- see AudioSegment.
        wall_start/wall_end and _worker, which passes them straight through)
        are the real capture-time wall clock for this segment. When given,
        they are the PRIMARY source for the bullet stamp, the rollover-day
        decision, and the paragraph-gap decision below; omitting them (the
        default) reproduces the original session_start + start_s behavior
        exactly -- see the class docstring."""
        wall = _wall_dt(wall_start) if wall_start is not None else (
            self._session_start + datetime.timedelta(seconds=start_s))

        rec = {"type": "segment", "start": round(float(start_s), 2),
               "end": round(float(end_s), 2), "text": text}
        if wall_start is not None:
            rec["wall"] = wall.isoformat(timespec="seconds")
        self._jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._jsonl_f.flush()

        day = wall.strftime("%Y-%m-%d")
        if day != self._md_day:
            self._rollover(day)  # terminates any open line in the OLD file first, see _rollover()

        # A new paragraph starts when no line is open yet (first write to
        # this file, or right after a rollover) or the gap since the last
        # segment on the open line is >= paragraph_gap_sec -- the wall gap
        # when both wall_start and a previous wall_end are available (this
        # is what forces a new paragraph across a sleep gap the stream clock
        # barely saw), else the legacy stream gap -- see the class
        # docstring's paragraph_gap_sec paragraph.
        if self._last_end_s is None:
            new_line = True
        elif wall_start is not None and self._last_wall_end is not None:
            new_line = (wall_start - self._last_wall_end) >= self._paragraph_gap_sec
        else:
            new_line = (start_s - self._last_end_s) >= self._paragraph_gap_sec
        if new_line:
            if self._last_end_s is not None:
                self._md_f.write("\n")  # terminate the previous paragraph before starting a new one
            self._md_f.write(f"- [{wall.strftime('%H:%M:%S')}] {text}")
        else:
            self._md_f.write(f" {text}")  # gap below threshold: join, don't start a new paragraph
        self._last_end_s = end_s
        self._last_wall_end = wall_end
        self._md_f.flush()

    def maybe_flush(self, now_s, wall_now=None):
        """Close the currently-open md paragraph line if it's been idle for
        >= paragraph_gap_sec, even though no new segment has arrived to
        trigger write()'s own gap check. `now_s` is the caller's current
        position on the same session-relative-seconds clock as start_s/
        end_s -- in practice buf.pos_s (see SegmentBuffer.pos_s); `wall_now`
        is the same "now", on the wall clock (in practice buf.wall_now).
        Compared here exactly the way write() compares an incoming segment's
        start_s/wall_start: the wall gap (wall_now - self._last_wall_end)
        when both are available, else the stream gap (now_s -
        self._last_end_s) -- see the class docstring's paragraph_gap_sec
        paragraph -- split at >= paragraph_gap_sec either way.

        A no-op, returning False, when no line is open (self._last_end_s is
        None) or the idle gap hasn't reached the threshold yet. Otherwise
        appends a bare "\\n" (the same terminator write()/_rollover()/
        close() use -- see the class docstring's append-only/line-cursor
        invariant, which this preserves: only ever appends, never rewrites,
        and the newly-terminated line keeps the exact "- [HH:MM:SS] text"
        shape it already had), flushes, resets self._last_end_s and
        self._last_wall_end to None, and returns True.

        Writes only to the md file -- the jsonl side has nothing to add on
        an idle tick, since jsonl records one line per actually-transcribed
        segment, not per paragraph. Reads no wall clock of its own either
        way: now_s/wall_now are supplied entirely by the caller, keeping
        this exactly as deterministic/testable as write() when it too is
        called with no wall_start/wall_end -- see the class docstring's note
        on self._session_start.
        """
        if self._last_end_s is None:
            return False
        if wall_now is not None and self._last_wall_end is not None:
            gap = wall_now - self._last_wall_end
        else:
            gap = now_s - self._last_end_s
        if gap < self._paragraph_gap_sec:
            return False
        self._md_f.write("\n")
        self._md_f.flush()
        self._last_end_s = None
        self._last_wall_end = None
        return True

    def _rollover(self, day):
        """Switch the currently-open md file to <out_dir>/<day>.md: the
        segment write() is about to record fell on a later calendar day than
        the file that's open right now. day is always the segment's own
        wall-clock date (not "the next day after self._md_day"), so a
        session silent across more than one midnight jumps straight to the
        right file in one call -- no empty file is created for any day
        nobody actually spoke on in between. Always writes a *continuation*
        header (never the plain __init__ one), since by definition this
        session didn't start on `day`. Terminates the OLD file's currently-
        open paragraph line (if any) with a bare "\\n" before closing it --
        this file is done being written to, so nothing may be left dangling
        unterminated -- then resets self._last_end_s (and self._last_wall_end)
        to None so the new file starts with no line open (and no stale
        end_s/wall_end to gap against). This happens unconditionally on a
        day change, regardless of how short the actual silence gap to the
        next segment is -- crossing midnight always forces a new
        paragraph."""
        if self._last_end_s is not None:
            self._md_f.write("\n")
        self._md_f.close()
        new_path = os.path.join(self._out_dir, f"{day}.md")
        is_fresh = not os.path.exists(new_path) or os.path.getsize(new_path) == 0
        self._md_f = open(new_path, "a", encoding="utf-8")
        if is_fresh:
            self._md_f.write(format_day_h1(day))
        language = self._language or "auto"
        self._md_f.write(format_continuation_header(
            self._session_start.strftime("%H:%M:%S"),
            f"{self._model}, lang={language}",
            self._session_start.strftime("%Y-%m-%d"),
        ))
        self._md_f.flush()
        self._md_day = day
        self._last_end_s = None
        self._last_wall_end = None

    def close(self):
        """Terminate any still-open paragraph line with a bare "\\n" (so the
        md file always ends newline-terminated and its final paragraph gets
        picked up by sync_transcripts_notion.py's line cursor), then close
        both files."""
        if self._last_end_s is not None:
            self._md_f.write("\n")
            self._md_f.flush()
        self._jsonl_f.close()
        self._md_f.close()


def fmt_hms(seconds):
    """Format seconds (session-relative) as HH:MM:SS. Truncates sub-second
    remainders (3661.5 -> '01:01:01'), not rounds, so it agrees with the
    already-elapsed whole second rather than rounding into the next one."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_text(result):
    """Pull the transcript text out of an mlx_whisper.transcribe() result
    dict (or any dict-alike), trimmed. No mlx import needed -- takes a plain
    dict so it's testable with a fake result."""
    return (result.get("text") or "").strip()


def extract_words(result):
    """Return Whisper word-timestamp strings in recognition order.

    mlx-whisper stores words below each returned segment.  Keep this parser
    deliberately tolerant of both dict and attribute-shaped results so the
    display path cannot make a successful transcription fail merely because a
    backend changed its result container.
    """
    words = []
    for segment in result.get("segments") or []:
        segment_words = segment.get("words", []) if isinstance(segment, dict) else getattr(segment, "words", [])
        for item in segment_words or []:
            word = item.get("word") if isinstance(item, dict) else getattr(item, "word", None)
            if word and word.strip():
                words.append(word.strip())
    return words


def extract_timed_words(result, offset_s=0.0):
    """Return ``(text, start_s, end_s)`` tuples from Whisper segments.

    ``offset_s`` converts timestamps relative to one speech buffer into the
    session-relative clock used by ``AudioSegment``.  A word can be missing a
    timestamp in a malformed backend response; skip it here so the caller can
    use its normal text fallback rather than inventing a timing boundary.
    """
    words = []
    for segment in result.get("segments") or []:
        segment_words = segment.get("words", []) if isinstance(segment, dict) else getattr(segment, "words", [])
        for item in segment_words or []:
            if isinstance(item, dict):
                word, start, end = item.get("word"), item.get("start"), item.get("end")
            else:
                word = getattr(item, "word", None)
                start, end = getattr(item, "start", None), getattr(item, "end", None)
            if word and word.strip() and start is not None and end is not None:
                words.append((word, float(start) + offset_s, float(end) + offset_s))
    return words


def ends_display_sentence(word):
    """Whether a timestamp unit closes a natural terminal display line."""
    return word.rstrip().endswith(("。", "！", "？", "!", "?"))


# --- per-segment timing (diagnostic) -----------------------------------------
def format_timing_line(audio_sec, infer_sec, lag_sec, dropped=False):
    """One `--transcribe-timing` line: where a segment's seconds went.

    audio = how long the speech itself was, infer = how long the model took on
    it (with the realtime multiple, so >1x means the model keeps up), lag = how
    long after the speaker stopped talking this segment finished -- the number
    the person watching the terminal actually feels. lag includes the fixed
    silence gate that ends a segment, any audio backlog from earlier segments
    the worker was still busy with, and infer itself; comparing it against
    infer is what separates "the model is slow" from "we are behind".
    """
    speed = f"{audio_sec / infer_sec:.1f}x" if infer_sec > 0 else "-x"
    return (f"[timing] audio {audio_sec:.1f}s  infer {infer_sec:.2f}s ({speed})  "
            f"lag {lag_sec:.1f}s" + ("  (dropped)" if dropped else ""))


# --- anti-hallucination ------------------------------------------------------
# Whisper fabricates text on silence/non-speech: fixed "outro" phrases (Thank
# you. / ご視聴ありがとうございました / 다음 영상에서 만나요.) and degenerate
# repetition loops (剛剛剛… / チェックチェック… / "Let's do it." x4). Its own
# guards don't stop them: a *confident* hallucination has high no_speech_prob
# but avg_logprob > logprob_threshold, and Whisper's skip rule needs BOTH, so it
# emits anyway; repetition only triggers a temperature *retry*, whose best-of is
# still emitted. So we re-judge each returned segment on the very signals
# Whisper already computed, plus a language-agnostic repetition test.
NO_SPEECH_MAX = 0.6      # drop a segment Whisper itself thinks is >60% likely silence
LOGPROB_MIN = -1.1       # drop very low-confidence decodes (lenient: keep quiet speech)
COMPRESSION_MAX = 2.4    # drop over-compressible (= repetitive) text; Whisper's own default

# The signal-based gates above only catch a hallucination when Whisper's own
# no_speech_prob/avg_logprob/compression_ratio give it away. Some sail through
# all three anyway with normal-looking confidence -- a bare symbol ("!" on a
# music beat) or a canonical outro phrase ("Thank you for watching!") -- so we
# also pattern-match the *text itself*, independent of any confidence signal.
def _normalize_for_filter(text):
    """NFKC-fold (full-width "！" -> "!", etc.), casefold, and keep only
    letters/digits (Unicode category L*/N*). Strips whitespace/punctuation so
    phrase variants ("Thank you for watching!" vs "thank you for watching.")
    and symbol-only text ("!", "♪〜", "。。。") normalize to a comparable
    (or, for symbol-only text, empty) form."""
    norm = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in norm if unicodedata.category(ch)[0] in ("L", "N"))


# Canonical music/outro hallucinations Whisper emits verbatim on non-speech
# audio, matched after _normalize_for_filter so case/punctuation variants
# collapse onto the same key.
HALLUCINATION_EXACT = frozenset(_normalize_for_filter(p) for p in (
    "Thank you for watching",
    "Thanks for watching",
    "Thank you so much for watching",
    "Please subscribe",
    "Please subscribe to my channel",
    "See you in the next video",
    "See you next time",
    "Subtitles by the Amara.org community",
    # Standalone JA thanks: silence hallucinations AND real meeting sign-offs
    # alike are deliberately dropped (owner request, 2026-08-24) -- they carry
    # no content and flooded the daily transcripts. Both tenses (past
    # ございました / present ございます, added 2026-09-03 -- Whisper fabricates
    # either one on non-speech). Exact match only, so thanks embedded in a
    # longer utterance ("先週ありがとうございました") still survives.
    "ありがとうございました",
    "はい、ありがとうございました",
    "ありがとうございます",
    "はい、ありがとうございます",
))

# Japanese outro hallucinations vary too much in wording for an exact-match
# blacklist (ご視聴いただき/本当に insertions, etc.), so these match as a
# conjunctive substring pair instead: both sides must appear somewhere in the
# (normalized) text, in either order.
HALLUCINATION_PAIRS = (
    ("ご視聴", "ありがとう"),
    ("チャンネル登録", "お願い"),
    ("チャンネル登録", "高評価"),
)


def is_hallucination_text(text):
    """True if `text` itself -- independent of Whisper's confidence signals
    -- looks like a music/silence hallucination: empty after normalization
    (symbol-only, e.g. "!" / "♪〜" / "。。。"), a canonical English outro
    phrase (e.g. "Thank you for watching!"), or a Japanese ご視聴/
    チャンネル登録 outro pair (e.g. "ご視聴ありがとうございました",
    "チャンネル登録お願いします")."""
    norm = _normalize_for_filter(text)
    if not norm:
        return True  # punctuation/symbol-only (e.g. "!", "♪〜") -- music hallucination
    if norm in HALLUCINATION_EXACT:
        return True
    return any(a in norm and b in norm for a, b in HALLUCINATION_PAIRS)


def _cycle_repeat(tokens, min_repeats_multi=3, min_repeats_single=5, coverage=0.8, max_period=6):
    """True if `tokens` is (mostly) a short block repeated: the first `p` tokens
    repeated `r` times covering >= `coverage` of the list. A multi-token block
    needs >= `min_repeats_multi` repeats, a single-token block >= `min_repeats_single`
    (so real emphasis like ['no','no','no'] survives but a 30x loop doesn't)."""
    n = len(tokens)
    if n < 2:
        return False
    for p in range(1, min(max_period, n // 2) + 1):
        r = n // p
        if tokens[:p] * r != tokens[:p * r]:
            continue
        if p * r < coverage * n:
            continue
        if (p >= 2 and r >= min_repeats_multi) or (p == 1 and r >= min_repeats_single):
            return True
    return False


def looks_like_repetition(text):
    """A Whisper repetition-loop hallucination: a short word phrase or CJK-char
    run repeated over and over. Language-agnostic. The char-level check runs only
    for space-free text (CJK); when there are word boundaries the word split is
    authoritative, so 'no no no' isn't mis-read as the char cycle 'no'x3."""
    t = (text or "").strip()
    if not t:
        return False
    words = t.split()
    if _cycle_repeat(words):
        return True
    if len(words) <= 1 and _cycle_repeat(list(t)):
        return True
    return False


# Characters that end a phrase: collapse_immediate_repeat() only ever compares
# and removes whole phrases cut on these, never a slice of one. Scanning raw
# characters instead (tried 2026-09-03, reverted the same day after a 32k-
# segment replay) mangles words whose own letters repeat -- "illallah" ->
# "illah", "ごちゃごちゃ" -> "ごちゃ", "Yeah, yeah, yeah!" -> "Yeah, y eah!".
PHRASE_DELIMITERS = "、。，．,.!?！？…‥\n\t\u3000 "

# For the one case with no phrase boundary to cut on -- a whole phrase that is
# itself the same block twice over, "前提として前提として" -- how long that block
# must be, in characters. Japanese reduplications are real words and are all
# two-character blocks (いろいろ / そろそろ / はいはい) or three (ごちゃごちゃ /
# もじもじ), so a doubling needs four. Three or more copies are much stronger
# evidence of a decoder loop, so those need only two ("はいはいはい").
MIN_REPEAT_BLOCK_CHARS = 4
MIN_REPEAT_BLOCK_CHARS_MANY = 2


def _split_phrases(text):
    """Cut `text` after every run of PHRASE_DELIMITERS, keeping them attached.

    "あ、いいです。 いいです。" -> ["あ、", "いいです。 ", "いいです。"]. Every
    character of the input lands in exactly one piece, in order, so "".join()
    of the result reproduces the input exactly.
    """
    phrases = []
    cur = []
    i = 0
    while i < len(text):
        cur.append(text[i])
        if text[i] in PHRASE_DELIMITERS:
            while i + 1 < len(text) and text[i + 1] in PHRASE_DELIMITERS:
                i += 1
                cur.append(text[i])
            phrases.append("".join(cur))
            cur = []
        i += 1
    if cur:
        phrases.append("".join(cur))
    return phrases


def _phrase_key(phrase):
    """What makes two phrases "the same" for dedup: content only -- the
    surrounding punctuation and spacing Whisper re-punctuates freely between
    copies ("Yeah, " vs "yeah!") is not part of the comparison."""
    return phrase.strip(PHRASE_DELIMITERS).casefold()


def _collapse_repeated_phrases(phrases):
    """Drop every immediately-repeated run of phrases, keeping one copy.

    Handles a repeated single phrase ("いいです。いいです。") and a repeated
    GROUP of them ("A。B。A。B。"), any number of copies; the shortest repeating
    group wins, so "A。A。A。" collapses to one "A。" rather than to two. The
    kept copy is the first, except for its trailing punctuation, which comes
    from the run's LAST copy -- that is the one the speaker actually ended on
    ("Yeah, yeah, yeah!" -> "Yeah!", not "Yeah,").
    """
    keys = [_phrase_key(p) for p in phrases]
    n = len(phrases)
    out = []
    i = 0
    while i < n:
        run = None
        for k in range(1, (n - i) // 2 + 1):
            if any(not keys[j] for j in range(i, i + k)):
                continue  # punctuation-only piece: never the seed of a "repeat"
            r = 1
            while keys[i + r * k:i + (r + 1) * k] == keys[i:i + k]:
                r += 1
            if r >= 2:
                run = (k, r)
                break
        if run is None:
            out.append(phrases[i])
            i += 1
            continue
        k, r = run
        block = list(phrases[i:i + k])
        last = phrases[i + k * r - 1]
        block[-1] = block[-1].rstrip(PHRASE_DELIMITERS) + last[len(last.rstrip(PHRASE_DELIMITERS)):]
        out.extend(block)
        i += k * r
    return out


def _collapse_within_phrase(phrase, min_block_chars, min_block_chars_many):
    """Collapse a phrase whose whole content is one block repeated back-to-back
    ("前提として前提として" -> "前提として"), keeping its trailing punctuation.

    Whole-content only: a repeat that covers just part of the phrase
    ("日本の日本の最大の顧客") is left alone, because at these block lengths
    telling one apart from a word's own repeated letters is guesswork.
    """
    core = phrase.rstrip(PHRASE_DELIMITERS)
    tail = phrase[len(core):]
    n = len(core)
    for p in range(min(min_block_chars, min_block_chars_many), n // 2 + 1):
        if n % p:
            continue
        copies = n // p
        if p < (min_block_chars if copies == 2 else min_block_chars_many):
            continue
        if core == core[:p] * copies:
            return core[:p] + tail
    return phrase


def collapse_immediate_repeat(text, min_block_chars=MIN_REPEAT_BLOCK_CHARS,
                              min_block_chars_many=MIN_REPEAT_BLOCK_CHARS_MANY):
    """Collapse phrases Whisper emitted twice in a row: "…A A…" -> "…A…".

    Catches the decoder repeat looks_like_repetition() deliberately doesn't: a
    phrase repeated only two or three times, which its cycle test lets through
    because that needs >= 3 repeats for a multi-token block and only looks at
    periods of <= 6 characters for space-free CJK. The repeat does not have to
    be the whole utterance -- Whisper doubles a leading or trailing phrase just
    as often ("あ、いいです。いいです。").

    Works on whole phrases cut at PHRASE_DELIMITERS (see
    _collapse_repeated_phrases), plus the one delimiter-less case of a phrase
    that is the same block twice over (_collapse_within_phrase). It therefore
    never removes a slice of a word, and never touches text without a
    back-to-back repeat -- beyond folding whitespace orphaned by a removal
    (runs collapse to one space) and stripping the ends.
    """
    t = (text or "").strip()
    if not t:
        return t
    phrases = _split_phrases(t)
    # To a fixed point: removing one repeat can expose another that wasn't
    # back-to-back before ("コスプレ、コスプレ、写真…。コスプレ、写真…。" only
    # becomes a repeated PAIR once the doubled "コスプレ、" is gone), and
    # collapsing inside a phrase can leave it identical to its neighbour.
    # Every pass strictly shortens the list or stops, so this terminates.
    while True:
        collapsed = _collapse_repeated_phrases(
            [_collapse_within_phrase(p, min_block_chars, min_block_chars_many)
             for p in phrases])
        if collapsed == phrases:
            break
        phrases = collapsed
    return re.sub(r"\s+", " ", "".join(phrases)).strip()


# A prompt echo can only be a chunk of the glossary, so it is short by nature.
# The cap keeps a long real sentence that happens to be built out of glossary
# words from ever being mistaken for one.
PROMPT_ECHO_MAX_CHARS = 24


def _normalized_prompt(prompt):
    """_normalize_for_filter(prompt), cached -- is_prompt_echo() runs per
    segment against the same (long, unchanging) session bias prompt."""
    global _PROMPT_CACHE
    if _PROMPT_CACHE[0] != prompt:
        _PROMPT_CACHE = (prompt, _normalize_for_filter(prompt))
    return _PROMPT_CACHE[1]


_PROMPT_CACHE = (None, "")


def is_prompt_echo(text, prompt):
    """True when `text` is just the Whisper bias prompt read back as speech.

    Given an initial_prompt (our glossary of proper nouns), Whisper sometimes
    decodes part of that prompt as if it had been spoken when the audio is
    noise or near-silence -- the 2026-09-03 session emitted "データベース,
    Notion" and "AIエージェント, Notion", verbatim tails of the glossary.
    Matched after _normalize_for_filter, so the separators and case Whisper
    re-punctuates with don't matter, and only for text up to
    PROMPT_ECHO_MAX_CHARS: a real sentence that merely mentions glossary terms
    is longer than any echo and is never a substring of the prompt anyway. A
    longer run of glossary entries is caught phrase by phrase instead (see
    below). No prompt (None/"") -> nothing to echo, always False.
    """
    if not prompt:
        return False
    norm_prompt = _normalized_prompt(prompt)

    def _echoes(s):
        norm = _normalize_for_filter(s)
        return bool(norm) and len(norm) <= PROMPT_ECHO_MAX_CHARS and norm in norm_prompt

    if _echoes(text):
        return True
    # Whisper often reads several glossary entries back in a row ("データベース,
    # Notionデータベース, Notion"): too long to be one echo, but every phrase in
    # it is one. Real speech doesn't survive that test -- its phrases carry
    # particles and verbs that appear nowhere in a comma-separated term list.
    phrases = [p for p in _split_phrases(text) if _phrase_key(p)]
    return len(phrases) > 1 and all(_echoes(p) for p in phrases)


def clean_segments(result, no_speech_max=NO_SPEECH_MAX, logprob_min=LOGPROB_MIN,
                   compression_max=COMPRESSION_MAX, prompt=None):
    """The sub-segments of an mlx_whisper result worth keeping, in order.

    Each kept segment is returned as a dict with its "text" already cleaned
    (collapse_immediate_repeat applied) and every other key -- "words",
    "start", "end", the confidence signals -- carried through untouched.
    Dropped are: empty text, anything Whisper's own signals call silence
    (no_speech_prob), low confidence (avg_logprob) or over-compressible
    (compression_ratio), a repetition loop (judged before the collapse, so a
    loop is dropped rather than tidied into one copy), a text-level
    hallucination (is_hallucination_text) and a bias-prompt echo
    (is_prompt_echo) -- the last two judged after it, so a doubled echo
    ("データベース, Notion" twice) is recognized.

    Split out of clean_transcription so the DISPLAY path can show exactly the
    segments the transcript keeps (see display_units): before this existed it
    printed words straight off the raw result, so a hallucination dropped from
    the .md/.jsonl still reached the terminal and the caption overlay.
    """
    kept = []
    for s in result.get("segments") or []:
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        if s.get("no_speech_prob", 0.0) > no_speech_max:
            continue
        if s.get("avg_logprob", 0.0) < logprob_min:
            continue
        if s.get("compression_ratio", 0.0) > compression_max:
            continue
        # Order matters: a repetition LOOP is judged on the raw text and
        # dropped whole -- collapsing first would hand looks_like_repetition a
        # single tidy copy of the loop and let the whole segment through.
        if looks_like_repetition(txt):
            continue
        txt = collapse_immediate_repeat(txt)
        if is_hallucination_text(txt):
            continue
        if is_prompt_echo(txt, prompt):
            continue
        kept.append(dict(s, text=txt))
    return kept


def display_units(segments, text, offset_s=0.0):
    """(unit, start_s, end_s) tuples to print for a cleaned result.

    `segments` are clean_segments() output and `text` the final
    clean_transcription() text -- what the transcript file records. Word
    timestamps are used only when those words reconstruct `text` exactly
    (whitespace ignored): the words are the raw decode, so once anything has
    been dropped or collapsed after them, streaming them would put back on
    screen precisely what the transcript removed. In that case the whole
    cleaned `text` goes out as one unit spanning the kept segments' time
    range instead. Same tuple shape as extract_timed_words(), so the caller's
    display loop doesn't care which of the two it got.
    """
    words = extract_timed_words({"segments": list(segments)}, offset_s=offset_s)
    squash = lambda s: "".join(s.split())  # noqa: E731
    if words and squash("".join(w for w, _, _ in words)) == squash(text):
        return words
    if not text:
        return []
    starts = [s.get("start") for s in segments if s.get("start") is not None]
    ends = [s.get("end") for s in segments if s.get("end") is not None]
    start = (min(starts) if starts else 0.0) + offset_s
    end = (max(ends) if ends else 0.0) + offset_s
    return [(text, float(start), float(end))]


def clean_transcription(result, no_speech_max=NO_SPEECH_MAX, logprob_min=LOGPROB_MIN,
                        compression_max=COMPRESSION_MAX, prompt=None):
    """Turn an mlx_whisper.transcribe() result into text to keep, or '' to drop.

    Joins the sub-segments clean_segments() kept (see it for every gate) and
    re-runs the whole-text checks on the join: the repetition-loop test, then
    collapse_immediate_repeat -- two identical sub-segments only become a
    visible "A A" duplicate at this point, once the join has put a space
    between them. Falls back to the top-level text (same loop test, collapse,
    then hallucination-/echo-filtered) when a result carries no `segments`.
    `prompt` is the Whisper bias prompt this result was decoded with, so the
    glossary can be recognized when Whisper reads it back as speech; None
    (the default) disables that check alone.
    """
    segs = result.get("segments")
    if segs:
        text = " ".join(
            s["text"] for s in clean_segments(
                result, no_speech_max=no_speech_max, logprob_min=logprob_min,
                compression_max=compression_max, prompt=prompt)
        ).strip()
    else:
        text = (result.get("text") or "").strip()
        if looks_like_repetition(text):
            return ""
        text = collapse_immediate_repeat(text)
        if is_hallucination_text(text) or is_prompt_echo(text, prompt):
            text = ""
    if not text or looks_like_repetition(text):
        return ""
    return collapse_immediate_repeat(text)


# --------------------------------------------------------------- process glue ---
# Whisper HF repo used ONLY when engine="qwen" fails to load and
# _make_transcribe_fn falls back to the whisper path: cfg["model"] at that
# point holds the QWEN model id (resolved by
# live_avatar._resolve_transcribe_model for the engine the caller actually
# asked for), which is not a valid whisper checkpoint -- so the fallback uses
# this known-good default instead of cfg["model"]. Mirrors
# live_avatar.WHISPER_DEFAULT_MODEL.
_WHISPER_FALLBACK_MODEL = "mlx-community/whisper-large-v3-mlx"


def _make_transcribe_fn(cfg):
    """Build (fn, engine_name) for _worker: fn(audio16k_float32) -> a result
    dict with at least "text". engine_name is the engine actually wired up --
    "whisper" or "qwen" -- which can differ from cfg.get("engine") if a qwen
    load failure fell back.

    Must be called from the CHILD process (inside _worker), same constraint
    as the plain mlx_whisper import this replaces: the returned `fn` closes
    over a loaded model object that is not picklable, so it can never be
    constructed in the parent and handed across the multiprocessing boundary
    (see live_avatar._start_transcriber, which only ever passes cfg -- plain
    data -- to the child).

    engine="whisper" (the default) calls mlx_whisper.transcribe(), retaining
    its raw segments and adding _token_cap_reached for the worker's output
    gate. A capped, otherwise acceptable result is retried once without the
    glossary when one was supplied. Missing mlx_whisper prints an install
    hint and exits.

    engine="qwen" tries qwen3_asr_mlx first (Qwen3ASR.from_pretrained(cfg["model"])).
    If the import OR the from_pretrained() call raises ANY exception, this
    prints a one-line warning and falls back to the whisper path instead
    (using _WHISPER_FALLBACK_MODEL, not cfg["model"] -- see its own
    docstring above) -- a LOAD-TIME fallback only; a later per-segment
    transcribe() failure is handled by _transcribe_and_write's own
    try/except, same as always. On success, qwen's TranscriptionResult is
    normalized to {"text": result.text} -- clean_transcription() already
    handles a segments-less dict via its existing fallback branch (see
    tests/test_live_transcriber.py's qwen-shape tests).
    """
    engine = cfg.get("engine", "whisper")

    def _whisper_fn(model_id):
        try:
            import mlx_whisper
        except ImportError:
            print("[transcribe] mlx_whisper is not installed. Install with:\n"
                  "  pip install 'alwayswhisper[live]'", flush=True)
            sys.exit(1)

        def fn(audio16k):
            sample_len = decode_sample_len(len(audio16k) / 16000.0)
            options = dict(
                path_or_hf_repo=model_id,
                language=cfg["language"],   # "ja" default; None = auto-detect
                initial_prompt=cfg.get("prompt"),  # glossary/vocab bias (None = off)
                condition_on_previous_text=False,
                word_timestamps=cfg.get("display_mode") == "word",
                # Greedy only. Whisper's default retries a decode its own
                # thresholds dislike at six rising temperatures -- exactly what
                # a non-speech loop trips, turning one slow decode into six
                # (measured: 11.2s on a 1.5s noise clip, 2.1s with this off).
                # A loop is dropped by clean_transcription either way, so the
                # retries only ever bought latency here.
                temperature=0.0,
                sample_len=sample_len,
            )
            result = mlx_whisper.transcribe(audio16k, **options)
            hit_cap = reached_token_cap(result, sample_len)
            # A short, capped decode may be inventing speech from the glossary.
            # Give otherwise acceptable text one bounded retry without that
            # bias. Never enlarge the budget or reintroduce temperature loops.
            if hit_cap and options["initial_prompt"] and clean_transcription(
                result,
                no_speech_max=cfg.get("no_speech_max", NO_SPEECH_MAX),
                logprob_min=cfg.get("logprob_min", LOGPROB_MIN),
                compression_max=cfg.get("compression_max", COMPRESSION_MAX),
                prompt=cfg.get("prompt"),
            ):
                options["initial_prompt"] = None
                result = mlx_whisper.transcribe(audio16k, **options)
                hit_cap = reached_token_cap(result, sample_len)
            return dict(result, _token_cap_reached=hit_cap)
        return fn

    if engine == "qwen":
        try:
            from qwen3_asr_mlx import Qwen3ASR
            qmodel = Qwen3ASR.from_pretrained(cfg["model"])
        except Exception as e:
            print(f"[transcribe] qwen engine unavailable ({e}); falling back to whisper", flush=True)
            return _whisper_fn(_WHISPER_FALLBACK_MODEL), "whisper"

        # "auto" is the CLI's spelling of "no language forced" (see
        # live_avatar._start_transcriber, which already turns it into None
        # before it ever reaches cfg) -- treated the same way here defensively,
        # for any cfg built directly (e.g. tests) rather than via that path.
        language = cfg["language"]
        if language in (None, "auto"):
            language = None

        def fn(audio16k):
            result = qmodel.transcribe(audio16k, language=language, context=cfg.get("prompt"))
            return {"text": result.text}
        return fn, "qwen"

    return _whisper_fn(cfg["model"]), "whisper"


def _emit_display(display_q, event):
    """Best-effort mirror of a display print to the parent process, for the
    caption overlay (scripts/avatar/caption_overlay.py). `event` is
    ("chunk", text) or ("eos",) -- see _worker's word/sentence display
    branches below, which call this at exactly the points they print(). A
    no-op when display_q is None (--captions not requested -- see
    live_avatar._start_transcriber). Never blocks and never raises: display
    is a best-effort UI channel, not part of the transcript contract, so a
    full queue (an overlay consumer that's fallen behind) just silently
    drops the event rather than stalling transcription."""
    if display_q is None:
        return
    try:
        display_q.put_nowait(event)
    except queue.Full:
        pass


def _worker(q, cfg, display_q=None):
    """Child-process entrypoint (must be a module-level function so
    multiprocessing's spawn context can pickle it by reference).

    cfg keys: sr, model, language, engine, jsonl_path, md_path, meta,
    silence_sec, min_speech_sec, max_sec, pre_roll_sec, gate,
    paragraph_gap_sec. engine defaults to "whisper" when the key is absent
    (see _make_transcribe_fn) and gate to GATE_DEFAULT, so older/hand-built
    cfg dicts without them behave exactly as before those flags existed.

    Each queue item is either a (chunk, wall) tuple -- wall the time.time()
    instant that chunk was captured at, put there by LiveTranscriber.feed()
    -- or, for backward compatibility with every caller/test that pushes
    raw chunks directly onto a queue, a bare chunk (treated exactly as
    (chunk, None)); either shape is accepted, and the None shutdown
    sentinel is still checked first, before this unpacking. See the module
    docstring for why wall matters (the sleep-gap timestamp bug) and
    SegmentBuffer.push/wall_of for how it flows onward.

    display_q, if given (see LiveTranscriber.start(), which passes
    self.display_queue here as this same third positional argument -- an mp
    queue, like the audio queue, must ride Process args under spawn rather
    than cfg), receives a live mirror of everything _worker prints for
    on-screen display: every print() below that writes display text or
    closes a display line has a matching _emit_display() call, so a caption-
    overlay consumer sees byte-for-byte what the terminal does. This never
    changes what actually gets printed to stdout or written to the
    transcript files -- display_q is purely an additional, best-effort
    output channel.

    After handling every non-shutdown queue item -- whether or not it
    produced a segment worth transcribing -- calls
    writer.maybe_flush(buf.pos_s, wall_now=buf.wall_now), so a long stretch
    of silence with no new utterance still closes (and eventually syncs to
    Notion) the currently-open md paragraph line on its own; see
    TranscriptWriter.maybe_flush and SegmentBuffer.pos_s/wall_now. Not
    called for the shutdown item itself (item is None): writer.close(), in
    the finally block below, already terminates whatever line is still open
    at that point.
    """
    print(f"[transcribe] loading model {cfg['model']} (engine: {cfg.get('engine', 'whisper')}) ...",
          flush=True)
    transcribe_fn, engine = _make_transcribe_fn(cfg)

    buf = SegmentBuffer(cfg["sr"], silence_sec=cfg["silence_sec"], min_speech_sec=cfg["min_speech_sec"],
                         max_sec=cfg["max_sec"], pre_roll_sec=cfg["pre_roll_sec"],
                         gate=cfg.get("gate", GATE_DEFAULT))
    writer = TranscriptWriter(
        cfg["jsonl_path"], cfg["md_path"], cfg["meta"],
        paragraph_gap_sec=cfg.get("paragraph_gap_sec", 60.0),
    )
    model_ready = False
    last_display_end_s = None
    display_line_open = False
    # Wall clock at the audio clock's zero point (the first chunk pushed), so a
    # segment's session-relative end_s can be turned back into "how long ago
    # did the speaker stop talking" for the timing line. Stays None until the
    # first chunk arrives; a timing line printed before that reports lag 0.
    audio_t0 = None

    def _transcribe_and_write(seg):
        nonlocal model_ready, last_display_end_s, display_line_open
        try:
            audio16k = resample_to_16k(seg.samples, cfg["sr"])
            infer_start = time.time()
            result = transcribe_fn(audio16k)
            infer_sec = time.time() - infer_start
            if not model_ready:
                model_ready = True
                print("[transcribe] model ready", flush=True)
            gates = {
                "no_speech_max": cfg.get("no_speech_max", NO_SPEECH_MAX),
                "logprob_min": cfg.get("logprob_min", LOGPROB_MIN),
                "compression_max": cfg.get("compression_max", COMPRESSION_MAX),
                # What Whisper was biased with this session, so clean_segments
                # can recognize the glossary being read back as speech.
                "prompt": cfg.get("prompt"),
            }
            # The same gates decide both what is written and what is shown:
            # kept feeds the word display below, text feeds the files -- see
            # display_units. (clean_transcription re-runs clean_segments; that
            # is pure dict work on a handful of segments, and keeping it as the
            # single entry point for "the text" is worth more than the saving.)
            kept = clean_segments(result, **gates)
            text = clean_transcription(result, **gates)
            hit_cap = engine == "whisper" and result.get(
                "_token_cap_reached",
                reached_token_cap(result, decode_sample_len(len(audio16k) / 16000.0)),
            )
            if text and hit_cap:
                # Do not publish a truncated hallucination as a transcript.
                # Keep the loss visible in diagnostics, without echoing the
                # unreliable text into stdout, captions, or transcript files.
                print(f"[transcribe] {seg.end_s - seg.start_s:.1f}秒の音声で生成上限に到達"
                      "（token cap）。不確かな認識結果をスキップしました",
                      file=sys.stderr, flush=True)
                text = ""
            if cfg.get("timing"):
                # stderr, so `2>timing.log` leaves stdout a clean transcript.
                # Printed for a dropped segment too: it cost the same seconds.
                lag = time.time() - (audio_t0 + seg.end_s) if audio_t0 else 0.0
                print(format_timing_line(seg.end_s - seg.start_s, infer_sec, lag,
                                         dropped=not text),
                      file=sys.stderr, flush=True)
            if not text:
                return  # silence/repetition hallucination -> drop, don't write
            writer.write(seg.start_s, seg.end_s, text, wall_start=seg.wall_start, wall_end=seg.wall_end)
            # Console echo is the bare transcript only -- no "[transcribe]" label
            # or timestamp. The jsonl/md files keep full timing (writer.write above).
            # display_q mirrors this exact stream as ("chunk", text)/("eos",)
            # events (see _emit_display) for a caption-overlay consumer -- every
            # print() below that writes display text or closes a display line
            # has a matching _emit_display() call.
            if cfg.get("display_mode") == "word":
                # Word timestamps are requested only for this mode. The
                # persisted transcript remains one record per speech segment;
                # this option changes terminal presentation only. Keep words
                # on one line and break only at sentence punctuation or a
                # meaningful pause between timestamp units. The units come
                # from the CLEANED segments (display_units), never the raw
                # result: the screen must not show what the transcript
                # dropped -- until 2026-09-03 it did, printing filtered
                # "ご視聴ありがとうございました" that never reached the .md.
                words = display_units(kept, text, offset_s=seg.start_s)
                if words:
                    for word, start_s, end_s in words:
                        if display_line_open and last_display_end_s is not None and start_s - last_display_end_s >= 0.7:
                            print(flush=True)
                            _emit_display(display_q, ("eos",))
                            display_line_open = False
                        print(word, end="", flush=True)
                        _emit_display(display_q, ("chunk", word))
                        display_line_open = True
                        last_display_end_s = end_s
                        if ends_display_sentence(word):
                            print(flush=True)
                            _emit_display(display_q, ("eos",))
                            display_line_open = False
                else:
                    # Graceful fallback for an unexpected backend response.
                    print(text, flush=True)
                    _emit_display(display_q, ("chunk", text))
                    _emit_display(display_q, ("eos",))
                    last_display_end_s = seg.end_s
                    display_line_open = False
            else:
                print(text, flush=True)
                _emit_display(display_q, ("chunk", text))
                _emit_display(display_q, ("eos",))
        except Exception as e:  # one bad segment must not kill the transcript
            print(f"[transcribe] segment failed: {e}", flush=True)

    try:
        while True:
            item = q.get()
            if item is None:
                final = buf.flush()
                if final is not None:
                    _transcribe_and_write(final)
                break
            # (chunk, wall) from LiveTranscriber.feed(), or a bare chunk --
            # see this function's docstring -- from any caller/test that
            # still pushes raw chunks directly.
            if isinstance(item, tuple):
                chunk, wall = item
            else:
                chunk, wall = item, None
            if audio_t0 is None:
                audio_t0 = time.time()
            for seg in buf.push(chunk, wall=wall):
                _transcribe_and_write(seg)
            # Passing wall_now= only when there's an actual stamp to give
            # (rather than unconditionally, e.g. wall_now=buf.wall_now) is
            # behaviorally identical -- maybe_flush already defaults
            # wall_now=None and treats a None exactly like an omitted one --
            # but keeps this call's shape byte-for-byte the same as before
            # this feature existed on every legacy/no-wall path, including a
            # 2-arg-only maybe_flush stand-in a caller might monkeypatch in.
            wall_now = buf.wall_now
            if wall_now is not None:
                writer.maybe_flush(buf.pos_s, wall_now=wall_now)
            else:
                writer.maybe_flush(buf.pos_s)
    except KeyboardInterrupt:
        # Ctrl-C reaches the whole foreground process group, so this child gets
        # it too -- usually while blocked in q.get() above. Shutdown is the
        # parent's job (LiveTranscriber.stop()); all this process could add is a
        # multiprocessing traceback printed over the parent's shutdown messages.
        # Exit quietly instead and let the finally block close the transcript.
        pass
    finally:
        if display_line_open:
            print(flush=True)  # cleanly terminate a final unterminated phrase at shutdown
            _emit_display(display_q, ("eos",))
        writer.close()


class LiveTranscriber:
    """Parent-side handle: owns the child transcription process and its
    input queue. Call start(), feed() chunks from the audio callback, stop()
    at shutdown."""

    def __init__(self, sr, model, language, out_dir, prompt=None, engine="whisper", display_mode="word",
                 display_queue=None, timing=False, gate=GATE_DEFAULT,
                 max_sec=MAX_SEC_DEFAULT, silence_sec=SILENCE_SEC_DEFAULT):
        os.makedirs(out_dir, exist_ok=True)
        started = datetime.datetime.now()
        stamp = started.strftime("%Y%m%d_%H%M%S")
        date_str = started.strftime("%Y-%m-%d")
        self.jsonl_path = os.path.join(out_dir, f"live_{stamp}.jsonl")
        # Shared per-day file (local date at init), appended to by every
        # session that runs that day -- not per-session like the jsonl.
        # This path only ever names "today's file as of process start",
        # though: if the process is still running when local midnight
        # passes, TranscriptWriter.write() rolls the *actual* md output over
        # to <next-day>.md itself (and the day after that, and so on -- see
        # TranscriptWriter's docstring), so a long-running session's
        # transcript ends up split across multiple day files even though
        # this attribute (used only for the stop()-time summary print, and
        # as the file the very first "## " session header lands on) never
        # changes. The writer -- not this class -- tracks and switches the
        # file that's actually current.
        self.md_path = os.path.join(out_dir, f"{date_str}.md")
        meta = {
            "started_at": started.astimezone().isoformat(),
            "samplerate": sr,
            "model": model,
            "language": language,
            "prompt": prompt,   # bias prompt used this session (provenance)
            "date_str": date_str,
            "session_hms": started.strftime("%H:%M:%S"),
        }
        self.cfg = {
            "sr": sr,
            "model": model,
            "language": language,
            "engine": engine,   # "whisper" (default) or "qwen" -- see _make_transcribe_fn
            "display_mode": display_mode,  # terminal-only "word" (default) or "sentence"
            # Diagnostic per-segment timing on stderr (--transcribe-timing).
            "timing": timing,
            "prompt": prompt,   # Whisper initial_prompt: glossary/vocab bias (None = off)
            "jsonl_path": self.jsonl_path,
            "md_path": self.md_path,
            "meta": meta,
            "silence_sec": silence_sec,
            "min_speech_sec": 0.2,
            "max_sec": max_sec,
            "pre_roll_sec": 0.3,
            # speech-detection noise gate (RMS) -- see GATE_DEFAULT
            "gate": gate,
            # anti-hallucination gates applied to each transcribe() result
            "no_speech_max": NO_SPEECH_MAX,
            "logprob_min": LOGPROB_MIN,
            "compression_max": COMPRESSION_MAX,
            # TranscriptWriter's md paragraph silence-gap threshold, seconds
            # (see its class docstring); no CLI flag, so this default is the
            # only value.
            "paragraph_gap_sec": 60.0,
        }
        # display_queue (None unless --captions is on -- see
        # live_avatar._start_transcriber, which creates it and passes it in
        # here) is deliberately NOT part of self.cfg: an mp queue rides
        # Process args directly under spawn (same reason the audio queue
        # itself is a start()-time Process arg, not cfg data), not through a
        # picklable config dict. start() forwards it to _worker as that same
        # third positional argument (display_q).
        self.display_queue = display_queue
        # Parent-process-only latest sample: the UI reads this directly, so
        # recognition backlog never delays the meter or fills the caption queue.
        self._audio_level = None
        self.dropped = 0
        self._queue = None
        self._proc = None

    def start(self):
        ctx = multiprocessing.get_context("spawn")
        self._queue = ctx.Queue(maxsize=4096)
        self._proc = ctx.Process(target=_worker, args=(self._queue, self.cfg, self.display_queue), daemon=True)
        self._proc.start()

    def feed(self, chunk):
        """Called from the PortAudio callback thread: must never block or
        raise. Stamps the chunk with time.time() -- the real wall-clock
        instant it was captured at, taken here rather than in _worker
        because this call happens right as PortAudio hands the chunk over,
        with none of the queueing/scheduling delay _worker's q.get() can
        pick up -- and puts (chunk_copy, wall) on the queue rather than the
        bare array, so _worker can recover real wall time even across a gap
        where the audio clock itself stalls (e.g. sleep; see the module
        docstring). _worker accepts either shape (see its own docstring),
        so this is the only change needed to switch production capture over
        to wall-stamped chunks."""
        if self.display_queue is not None:
            try:
                samples = np.asarray(chunk, dtype=np.float32).ravel()
                rms = float(np.sqrt(np.dot(samples, samples) / samples.size)) if samples.size else 0.0
                self._audio_level = (time.monotonic(), rms if np.isfinite(rms) else 0.0)
            except Exception:
                self._audio_level = None
        q = self._queue
        if q is None:
            return
        try:
            q.put_nowait((np.asarray(chunk, dtype=np.float32).copy(), time.time()))
        except queue.Full:
            self.dropped += 1
        except Exception:
            self.dropped += 1  # must never raise from the audio callback thread

    def get_audio_level(self):
        """Latest (monotonic capture time, RMS) tuple, atomically replaced by feed()."""
        return self._audio_level

    def stop(self, timeout=90):
        summary = {"jsonl_path": self.jsonl_path, "md_path": self.md_path,
                   "dropped": self.dropped, "exitcode": None}
        if self._proc is None:
            return summary
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(5)
        summary["exitcode"] = self._proc.exitcode
        return summary
