# AlwaysWhisper

[日本語版はこちら (README.ja.md)](README.ja.md)

AlwaysWhisper takes a video or audio file and hands you back two things: the
same video with readable captions burned right into the picture, and a plain
text caption file (called an **SRT** file) that you can reuse anywhere else.
It listens to the audio, writes down every word that's said, works out
exactly when each word happens, splits that into caption-sized lines, and
draws them onto the video for you. Everything runs **on your own computer**
— no video or audio leaves your machine, unless you deliberately switch on
the optional hosted backend (running the model on someone else's server
instead of yours) described below.

## What it does, in plain words

- **Listens and writes it down.** It transcribes (turns speech into text)
  your video or audio file, and records the exact moment each word was
  spoken.
- **Cuts the text into caption-sized lines.** It splits the transcript into
  short lines that fit on screen, using rules tuned for Japanese by default
  (and a simpler rule for languages like English that use spaces between
  words).
- **Cleans up known mistakes automatically.** For example, Whisper (the
  underlying speech-to-text engine) sometimes invents a fake "thanks for
  watching" during silence — AlwaysWhisper strips that out before it ever
  reaches your captions.
- **Draws the captions onto your video**, one line at a time, with a
  typewriter-style reveal effect (the letters appear one after another,
  like someone typing).
- **Double-checks its own work.** It picks a few random caption lines,
  re-listens to just that bit of audio, and compares the result to what it
  drew. If something doesn't match well enough, it stops before finishing
  the video so you can look into it.
- **Runs on your own computer by default** — no account, no API key (a
  secret password for a paid service), and nothing sent over the internet,
  unless you turn on the optional hosted OpenAI backend yourself.

## Words you will see

You don't need to memorize this table before you start — it's here so you
can look a word up the moment you hit it.

| Term | What it means |
|---|---|
| Model | The trained "brain" file that turns sound into text. A bigger model is usually more accurate, but slower and needs more memory. |
| GPU / VRAM | GPU = a graphics chip built for fast, repetitive math, separate from your computer's main processor. It can run this kind of work much faster than the main processor. VRAM = the GPU's own private memory, separate from your computer's regular memory. |
| CPU | Your computer's main, general-purpose processor — every computer has one. AlwaysWhisper can run entirely on the CPU; it's just slower than using a GPU. |
| RAM | Your computer's main short-term memory, used to hold the running program and its data (as opposed to VRAM, which belongs only to the GPU). |
| int8 / float16 (precision) | Different ways of storing the model's numbers. `float16` keeps more decimal detail; `int8` rounds those numbers into a smaller, faster-to-compute form. Lower precision uses less memory and runs faster, at a small, usually barely noticeable, accuracy cost. |
| SRT | A plain text file that lists each caption line together with a start time and an end time. It's the standard subtitle file format, and it's what AlwaysWhisper produces and burns onto your video. |
| Word-level timestamps | The exact start and end moment of every single word, not just every sentence. This is what lets AlwaysWhisper line captions up precisely, and re-snap them later if the timing drifts. |
| ffmpeg | A free command-line program that reads, converts, and writes video and audio files. AlwaysWhisper uses it behind the scenes to pull the audio track out of your video and to draw captions onto the picture. |
| Virtual environment | A private folder of Python packages set up just for this project, so the exact versions AlwaysWhisper needs don't clash with anything else already on your machine. |
| Environment variable | A named setting that your operating system's shell keeps and that programs can read when they start. It's how you hand a program a value (a folder path, a secret token) without typing it into the command itself. |
| faster-whisper | A faster, lighter re-implementation of OpenAI's Whisper speech-to-text engine. AlwaysWhisper uses it by default to transcribe locally, with no internet connection or API key required. |
| Whisper | The original speech-to-text engine from OpenAI (it listens to audio and writes down what's said). AlwaysWhisper is built around it — through faster-whisper by default, or through OpenAI's own hosted API as an optional extra. |
| Glossary / bias prompt | A short list of names or words (like product names or people's names) that you hand the transcriber ahead of time, so it's more likely to spell them correctly. |
| QA | Short for "quality assurance" — AlwaysWhisper's automatic double-check that catches captions that don't actually match the audio, before you trust the finished video. |

## Features

- **Runs fully local** using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  — no API key needed, and models download themselves automatically the
  first time you use them. If you'd rather use OpenAI's hosted Whisper API
  instead of running the model yourself, AlwaysWhisper calls this choice of
  transcription engine a **backend** — the OpenAI API backend is available
  as an opt-in extra you install separately (see [Install](#install)
  below).
- **Japanese-grammar-aware caption splitting** — instead of just counting
  characters, it recognizes actual Japanese sentence endings (like です/ます
  forms and connective particles — the small grammatical words that join
  clauses) to decide where one caption should end and the next begin. Other
  space-separated languages (English, German, and so on) get a simpler
  segmenter that splits on spaces instead of following Japanese grammar
  rules.
- **Strips out Whisper's fake "thank you for watching"** — Whisper has a
  habit of hallucinating (confidently making up something that was never
  actually said) stock phrases like "ご視聴ありがとうございました" ("thanks
  for watching") into silent or trailing audio. AlwaysWhisper mechanically
  removes these known phrases (configurable, with Japanese defaults) so they
  never make it into your captions or transcript.
- **Automatic QA** — random caption samples get re-extracted from the
  video's own audio, re-transcribed independently (without any bias prompt,
  so nothing nudges the result toward matching), and fuzzy-matched (compared
  for close-enough similarity, not an exact character match) against the
  caption text. If any sample scores below the match threshold, the
  pipeline stops before burning captions in.
- **Typewriter-effect burn-in**, with two ways to render it: a fast path
  (using a library called libass together with ffmpeg) for quick iteration,
  and a portable path (using MoviePy and PIL, two Python image/video
  libraries) that needs nothing but a plain ffmpeg install.
- **Word-level SRT realignment** (opt-in, meaning it's off unless you turn
  it on) — if your subtitle timing has drifted out of sync (for example,
  after you hand-edited the text or ran an external correction pass), this
  snaps each caption's start time back to the nearest word-level timestamp
  from the original transcript.

## Requirements

- **Python ≥ 3.10** — the programming language AlwaysWhisper is written in;
  you need it installed to run the tool at all.
- **ffmpeg** — the video/audio tool described in the glossary above. It's
  required today for every AlwaysWhisper command that touches a video or
  audio file: `transcribe`/`auto` shell out to it (call it as an external
  program) to pull out a WAV audio file before handing that to the
  transcription engine, and `caption`/`qa` use it to cut out short audio
  clips for QA and, in standard (non-fast) mode, to put the original audio
  track back onto the captioned output. (faster-whisper itself can actually
  decode audio directly through a library called PyAV, without needing
  ffmpeg — but AlwaysWhisper's own audio-extraction step doesn't currently
  take advantage of that, so plan on installing ffmpeg no matter which
  transcription backend you use.)
- **Fast burn-in mode** (`--fast`) additionally needs a **libass-enabled**
  ffmpeg build — specifically, one that includes the `ass` filter (libass
  is the library that draws styled subtitles; not every ffmpeg build
  includes it):
  - **macOS**: Homebrew's plain `ffmpeg` formula ships with libass disabled
    — install `brew install ffmpeg-full` instead. AlwaysWhisper
    automatically looks for it at `/opt/homebrew/opt/ffmpeg-full/bin`
    (Apple Silicon Macs) and `/usr/local/opt/ffmpeg-full/bin` (Intel Macs);
    or you can point the `FFMPEG_LIBASS_BIN` (and `FFPROBE_LIBASS_BIN`)
    environment variable straight at a libass-enabled binary yourself.
  - **Debian/Ubuntu**: the standard `apt install ffmpeg` normally already
    ships with libass.
  - **Windows**: use a full build such as the ones at
    [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
  - Standard (non-fast) burn-in doesn't need libass — any ffmpeg on your
    `PATH` (the list of folders your system searches for programs) works
    fine.

## Install

Not yet published to PyPI (Python's public package index — the usual place
you'd `pip install` something from by name alone). Install straight from
GitHub instead:

```bash
pip install "git+https://github.com/kzkhykw/AlwaysWhisper.git"
```

This downloads AlwaysWhisper straight from its GitHub repository (a
project's stored copy of its own code and history) and installs it, the
same way `pip install` would from PyPI.

If you want the OpenAI Whisper API backend (transcribing using OpenAI's
hosted service instead of running a model on your own machine), add the
`[api]` extra (an optional add-on dependency group):

```bash
pip install "alwayswhisper[api] @ git+https://github.com/kzkhykw/AlwaysWhisper.git"
```

This backend needs `OPENAI_API_KEY` set in your environment or in a `.env`
file in your working directory (a `.env` file is just a plain text file of
`KEY=value` lines that tools like this one read automatically).

Or, from a local clone (a copy of the repository on your own disk):

```bash
git clone https://github.com/kzkhykw/AlwaysWhisper.git
cd AlwaysWhisper
pip install .
# or: pip install ".[api]"
```

This downloads the full source code into a folder, then installs it from
that local copy instead of from GitHub directly.

Alternatively, from a local clone, use the bundled `setup.sh` script, which
creates a **virtual environment** for you (see the glossary above):

```bash
./setup.sh          # creates .venv, installs alwayswhisper in editable mode
./setup.sh small     # also prefetches the "small" model afterward
```

"Editable mode" means changes you make to the source code take effect
immediately, without reinstalling — handy if you're planning to modify
AlwaysWhisper itself.

## Models

AlwaysWhisper uses faster-whisper's models, which are Whisper models
converted to a format called CTranslate2 (a library built for running this
kind of model faster and lighter). A model downloads itself automatically
the first time you use its name — there's no separate install step — and
gets cached (saved locally so it doesn't re-download next time) under
Hugging Face's Hub cache, a shared download folder used by many machine
learning tools. `large-v3` is the default model (`transcribe.model` in
`src/alwayswhisper/data/default_config.yaml`, the packaged settings file
described in [Configuration](#configuration) below).

### Choosing a model for your hardware

| Model | Parameters | Download | Languages | Notes |
|---|---|---|---|---|
| `tiny` | 39 M | ~76 MB | Multilingual | Smoke-testing the pipeline only |
| `base` | 74 M | ~145 MB | Multilingual | Still low accuracy; smoke-testing only |
| `small` | 244 M | ~484 MB | Multilingual | Good for quick local iteration |
| `medium` | 769 M | ~1.53 GB | Multilingual | Solid mid-tier accuracy |
| `large-v3` | 1550 M | ~3.09 GB | Multilingual | **Best accuracy — the default, recommended for Japanese** |
| `large-v3-turbo` (alias `turbo`) | 809 M | ~1.62 GB | Multilingual | Optimized version of `large-v3`: faster transcription with "minimal degradation in accuracy" per openai/whisper's README — check on your own material before trusting it over `large-v3` |
| `*.en` sizes, `distil-*` family | varies | varies | **English-only** | `tiny.en`...`medium.en` and the Distil-Whisper models (`distil-large-v3`, etc.) only work for English — never use them for Japanese |

"Parameters" is a rough size measure for a model — millions (M) of internal
numbers it learned during training; more parameters generally means more
accuracy, but also more memory and compute needed. "Download" is how much
disk space and network transfer fetching that model costs you.

Parameter counts and the `turbo` description are from
[openai/whisper's model table](https://github.com/openai/whisper#available-models-and-languages);
download sizes are the `model.bin` file size on Hugging Face for the
`Systran/faster-whisper-*` repos (`mobiuslabsgmbh/faster-whisper-large-v3-turbo`
for turbo).

**Reference numbers.** If you're planning to run this on a GPU, here's how
much of its VRAM (the GPU's own memory, see the glossary above) a model
needs. openai/whisper's README documents the original PyTorch
implementation's VRAM and speed, measured transcribing English speech on an
A100 GPU, relative to the `large` model (real-world speed varies by
language, speaking speed and hardware):

| Size | Params | Required VRAM | Relative speed |
|---|---|---|---|
| tiny | 39 M | ~1 GB | ~10x |
| base | 74 M | ~1 GB | ~7x |
| small | 244 M | ~2 GB | ~4x |
| medium | 769 M | ~5 GB | ~2x |
| large | 1550 M | ~10 GB | 1x |
| turbo | 809 M | ~6 GB | ~8x |

faster-whisper (the CTranslate2 reimplementation AlwaysWhisper actually
runs) needs substantially less than the table above. From its own README
benchmark, transcribing 13 minutes of audio at beam size 5 (a setting that
controls how many alternative transcriptions the model weighs before
picking one — higher is more thorough but slower):

| Model | Device | Precision | Time | Memory |
|---|---|---|---|---|
| large-v2 | RTX 3070 Ti (GPU) | fp16 | 1m03s | 4525 MB VRAM |
| large-v2 | RTX 3070 Ti (GPU) | int8 | 59s | 2926 MB VRAM |
| small | i7-12700K, 8 threads (CPU) | fp32 | 2m37s | 2257 MB RAM |
| small | i7-12700K, 8 threads (CPU) | int8 | 1m42s | 1477 MB RAM |

In other words: large-v2's fp16 (a higher-precision number format — see the
glossary's int8/float16 entry) GPU run above (4525 MB) is already well
under half the ~10 GB openai/whisper lists for the original `large` model,
and switching to int8 (a smaller, faster number format) nearly halves it
again — the same pattern applies to `large-v3`, which is the same size as
`large-v2`.

**Decision guide, by hardware:**

| Your hardware | Try | Why |
|---|---|---|
| NVIDIA GPU, ≥ 8 GB VRAM | `large-v3`, `compute_type: float16` (or leave `auto`, which picks the fastest type your GPU supports) | Best accuracy at GPU speed |
| NVIDIA GPU, 4-6 GB VRAM | `large-v3` with `compute_type: int8_float16` (~3 GB, from the large-v2 `int8` row above — large-v2/v3 are stored in float16, so a CTranslate2 `int8` request actually executes as `int8_float16`), or `large-v3-turbo` | Fits a tighter VRAM budget |
| Apple Silicon Mac, or CPU-only PC with ≥ 16 GB RAM | `large-v3` on CPU with `compute_type: int8`, or `large-v3-turbo` for more speed | Works, but expect it several times slower than the `small` CPU benchmark above — **estimate**: no CPU benchmark exists for `large-v3` in these docs; it has ~6x `small`'s parameters (1550 M vs 244 M) |
| ~8 GB RAM machine | `small` (or `medium` if it fits) | Reserve `large-v3` for short clips only |
| English-only source material | `.en` or `distil-large-v3` | Smaller/faster; never for Japanese (see the Languages column above) |

Two settings control which model runs where, and how precisely: `device`
(CPU vs. GPU) and `compute_type` (the number format used internally, like
int8 or float16).

**`device`** (`--device` / `transcribe.device` in config, default `auto`):
`cpu`, `cuda`, or `auto`. GPU acceleration in CTranslate2 only works on
NVIDIA GPUs (specifically ones with "Compute Capability" ≥ 3.5 — Compute
Capability is NVIDIA's own version number for what a given GPU chip
supports), and the current release needs CUDA 12 + cuDNN 9 (CUDA and cuDNN
are NVIDIA's own GPU-programming toolkits; more on installing them in GPU
setup below). Apple Silicon, AMD and Intel GPUs aren't accelerated by
CTranslate2 at all, so on those machines `auto`/`cpu`/`cuda` all end up
running the CPU path regardless of which one you pick.

**`compute_type`** (`--compute-type` / `transcribe.compute_type`, default
`auto`):
- `auto` — "use the fastest computation type that is supported on this
  system and device" (CTranslate2 docs)
- `default` — keep the type the model was converted with. Systran's
  official conversions (what AlwaysWhisper downloads for a size name like
  `large-v3`) are stored in float16 — their model card states "the model
  weights are saved in FP16"
- On **CPU**, CTranslate2 silently substitutes types it can't run natively:
  `float16` becomes `float32`, and the `int8_float16`/`int8_bfloat16`/`int16`
  family becomes `int8_float32` (CTranslate2's implicit-type-conversion
  table). So on CPU the practical choices are `int8` (fastest, least
  memory) or `float32` (full precision) — confirmed on this Mac (Apple
  Silicon, ctranslate2 4.8.1) by running
  `ctranslate2.get_supported_compute_types("cpu")`, which prints
  `{'int8_float32', 'float32', 'int8'}`; `float16` and plain `int8_float16`
  aren't in that set.
- On **GPU**, the practical choices are `float16` or `int8_float16` (int8
  on GPU needs Compute Capability ≥ 7.0 or 6.1 per CTranslate2's docs;
  older cards should stay on `float16`).

**CPU threads.** faster-whisper's `cpu_threads` setting (how many CPU
threads, i.e. parallel lines of execution, it's allowed to use) defaults to
0, which its docs describe as "4 by default, a non zero value overrides the
OMP_NUM_THREADS environment variable" — but AlwaysWhisper's
`FasterWhisperBackend` doesn't expose `cpu_threads` as an option (it only
forwards `model`, `device` and `compute_type` to `WhisperModel`). To use
more than 4 CPU cores, set `OMP_NUM_THREADS` (an environment variable, a
setting your shell passes to the program when it starts) before running:

```bash
OMP_NUM_THREADS=8 alwayswhisper transcribe clip.mp4 --model large-v3 --device cpu --compute-type int8
```

(Use your machine's physical core count, not thread count.)

**Benchmark it yourself.** Every number above is from someone else's
hardware, so treat it as a starting point, not a guarantee. On a 1-2 minute
clip of your own, `time` (a command that measures how long a program takes
to run) is enough to compare:

```bash
time alwayswhisper transcribe clip.mp4 --model small
time alwayswhisper transcribe clip.mp4 --model large-v3
```

`caption`/`qa`'s automatic AV QA re-transcription reuses `transcribe.model`
(the same model you transcribed with) unless you pass a separate `--model`
to `caption` or `qa`.

Set the model/device/compute type per command:

```bash
alwayswhisper transcribe clip.mp4 --model large-v3 --device cpu --compute-type int8
```

or once, in a `--config config.yaml` file (a text file of settings you can
reuse across runs — also picked up by `auto`; see
[Configuration](#configuration) below):

```yaml
transcribe:
  model: large-v3
  device: cpu
  compute_type: int8
```

### Installing and pre-downloading models

Nothing to install ahead of time — the first `transcribe`/`auto`/`caption`/
`qa` run downloads whatever model it's told to use. To fetch a model
without transcribing anything (for example, to warm a cache — pre-fill it
so nothing needs the network later — before going offline, or before a CI
run, an automated test run that typically has no internet access set up):

```bash
alwayswhisper prefetch --model large-v3
```

or, right after cloning, let `setup.sh` do it as part of environment setup:

```bash
./setup.sh large-v3
```

**Where models land.** Hugging Face's Hub cache, `~/.cache/huggingface/hub`
by default (a hidden folder in your home directory). Override it with the
`HF_HOME` environment variable (moves the whole Hugging Face cache root) or
`HF_HUB_CACHE` (just the hub cache) before running any AlwaysWhisper
command.

**Offline use.** Once a model is cached, set `HF_HUB_OFFLINE=1` (another
environment variable) so it's never checked against the Hub again (no
network needed). To seed an offline machine, copy the cache directory
across (or just the model's `models--<org>--<name>` subfolder inside it).

**Accepted `--model` values** (`transcribe`, `auto`, `caption`, `qa`, and
`prefetch`):
- A size name from the table above (`tiny`, `small`, `large-v3`,
  `large-v3-turbo`, ...)
- Any CTranslate2-converted Whisper repo ID on the Hugging Face Hub (e.g.
  `Systran/faster-whisper-large-v3`)
- A path to a local converted-model directory — accepted by
  `transcribe`/`auto`/`caption`/`qa`, but not `prefetch` (there's nothing
  to download for a model that's already on disk)

### Using a custom or fine-tuned model

Fine-tuned your own Whisper checkpoint (a saved, trained version of a
model), or want one faster-whisper hasn't already converted? Convert it to
CTranslate2 format first (from faster-whisper's README, "Model
conversion"):

```bash
pip install "transformers[torch]>=4.23"

ct2-transformers-converter --model openai/whisper-large-v3 --output_dir whisper-large-v3-ct2 \
    --copy_files tokenizer.json preprocessor_config.json --quantization float16
```

Swap `openai/whisper-large-v3` for your own checkpoint (a Hub repo ID or a
local directory), then point AlwaysWhisper at the converted output
directory:

```bash
alwayswhisper transcribe talk.mp4 --model ./whisper-large-v3-ct2
```

or [upload the converted directory to the Hugging Face
Hub](https://huggingface.co/docs/transformers/model_sharing#upload-with-the-web-interface)
and reference it by name, the same way you'd write `large-v3`:

```bash
alwayswhisper transcribe talk.mp4 --model username/whisper-large-v3-ct2
```

### GPU setup (NVIDIA)

`--device cuda` only accelerates NVIDIA GPUs (Compute Capability ≥ 3.5);
Apple Silicon, AMD and Intel GPUs always fall back to the CPU path
regardless of this flag. The current CTranslate2 release needs **CUDA 12 +
cuDNN 9** (NVIDIA's GPU-programming toolkits, mentioned above).

On Linux, the NVIDIA libraries can be installed with pip instead of a
system CUDA install — just make sure `LD_LIBRARY_PATH` (an environment
variable telling the system where to find shared libraries) is set before
launching Python:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*

export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

On Windows, see Purfview's
[whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win)
library archive, or install the CUDA Toolkit + cuDNN yourself.

Stuck on an older driver? Downgrade `ctranslate2` itself instead:
`pip install --force-reinstall ctranslate2==3.24.0` for CUDA 11 + cuDNN 8,
or `ctranslate2==4.4.0` for CUDA 12 + cuDNN 8.

## Quickstart

The fastest way, doing everything in one command:

```bash
alwayswhisper auto talk.mp4 -o final.mp4 --max-chars 21 --fast
```

This transcribes `talk.mp4` and writes a fully captioned `final.mp4`,
start to finish, with no stops along the way.

The recommended workflow, though, is to review and fix the transcript text
before burning captions in (Whisper's word-level timestamps stay valid even
if you edit the text, as long as you don't add or remove entries):

```bash
# 1. Transcribe to word timestamps + a raw SRT
alwayswhisper transcribe talk.mp4 -o talk_alwayswhisper --language ja --max-chars 21

# 2. Hand-edit talk_alwayswhisper/transcript_raw.srt: fix names, homophones,
#    misrecognitions -- keep the timestamps as they are.

# 3. Burn the corrected captions in
alwayswhisper caption talk.mp4 talk_alwayswhisper/transcript_raw.srt -o final.mp4 --fast
```

Step 1 listens to the audio and writes out a folder of raw transcript files
(an SRT plus word-level timestamps) — nothing is burned into the video yet.
Step 2 is you, reading that SRT file in a text editor and fixing anything
it misheard. Step 3 takes your corrected SRT and burns it onto the video as
the final captioned file.

**Japanese recipe**: `--language ja --max-chars 21` keeps most captions to a
single line at typical font sizes (see [Caption styles](#caption-styles)
below for the sizing math). Bias the transcription toward proper nouns or
domain vocabulary with `--glossary terms.txt` (a plain text file of correct
spellings/terms — collapsed into a single bias-prompt line and truncated to
Whisper's prompt token budget automatically; a "token" here is one of the
small text chunks a language model reads at a time — not quite a word
count, but close enough to think of it that way).

## CLI reference

This section lists every command AlwaysWhisper offers, and every flag (an
option starting with `--` that you type after the command, like
`--language ja`) it accepts. Every subcommand
also accepts the global `--config FILE` option (a YAML file — a plain text
settings file, see [Configuration](#configuration) below — deep-merged over
the packaged defaults). `alwayswhisper --version` prints the installed
version and exits. Flag defaults shown as "packaged default" come from
`data/default_config.yaml` (the settings file that ships inside
AlwaysWhisper itself) unless a flag explicitly documents its own default
below.

### `alwayswhisper transcribe INPUT`

Listens to `INPUT` and writes out the transcript: word timestamps + an SRT
caption file. This is step 1 of the recommended workflow above.

| Flag | Description |
|---|---|
| `-o, --output DIR` | Output directory (default: `<INPUT stem>_alwayswhisper` next to INPUT) |
| `--backend BACKEND` | Transcription backend: `faster-whisper` or `openai-api` |
| `--model MODEL` | Model name (faster-whisper only; openai-api always uses `whisper-1`) |
| `--language LANGUAGE` | Language code, e.g. `ja`, `en` |
| `--device DEVICE` | faster-whisper device: `cpu`, `cuda`, or `auto` |
| `--compute-type COMPUTE_TYPE` | faster-whisper compute type |
| `--prompt TEXT` | Whisper bias prompt (overrides `--glossary`) |
| `--glossary FILE` | Text file of vocabulary to bias transcription toward |
| `--max-chars MAX_CHARS` | Max characters per caption entry |
| `--min-chars MIN_CHARS` | Min characters per caption entry |
| `--vad-filter` | Enable faster-whisper's voice-activity-detection filter |

Outputs: `transcript_words.json` (word-level timestamps) and
`transcript_raw.srt` in the output directory.

### `alwayswhisper segment WORDS_JSON -o OUT_SRT`

Turns a `transcript_words.json` file back into an SRT caption file —
useful for re-segmenting (re-splitting into caption lines) after
hand-fixing word text, or after changing `--max-chars`.

| Flag | Description |
|---|---|
| `-o, --output OUT_SRT` | Output SRT path (required) |
| `--language LANGUAGE` | Language code — selects the segmenter (char-based for ja/zh/yue/th/lo/my, space-delimited otherwise) |
| `--max-chars MAX_CHARS` | Max characters per caption entry |
| `--min-chars MIN_CHARS` | Min characters per caption entry |

### `alwayswhisper caption VIDEO SRT -o OUT_MP4`

Draws the captions from `SRT` onto `VIDEO` and writes the finished,
captioned video. This is step 3 of the recommended workflow above.

| Flag | Description |
|---|---|
| `-o, --output OUT_MP4` | Output video path (required) |
| `--words FILE` | `transcript_words.json`, required for `--realign` |
| `--edit-plan FILE` | `edit_plan.json`, used with `--realign` |
| `--style NAME_OR_PATH` | Caption style: `default`, `en`, or a YAML file path |
| `--fast` | Fast libass/ffmpeg burn-in instead of MoviePy/PIL |
| `--no-qa` | Skip the AV QA spot check |
| `--qa-samples QA_SAMPLES` | Number of AV QA samples |
| `--qa-min-ratio QA_MIN_RATIO` | Minimum AV QA match ratio |
| `--realign` | Snap SRT starts to word timestamps before burning |
| `--backend BACKEND` | Transcription backend used for the AV QA re-transcription |
| `--model MODEL` | Model name for the AV QA backend |
| `--language LANGUAGE` | Language code for the AV QA backend |

Outputs: the burned-in video, and (unless `--no-qa`) a `qa_report.json`
next to it. With `--realign`, also `<output stem>.realigned.srt`.

### `alwayswhisper qa VIDEO SRT`

Runs just the QA spot-check (see [QA](#qa) below) against a video and an
SRT, on its own — no burn-in. Useful for validating an externally-edited
SRT before committing to a render (the final burn-in pass, which can take a
while on longer videos).

| Flag | Description |
|---|---|
| `--samples SAMPLES` | Number of samples to check |
| `--min-ratio MIN_RATIO` | Minimum match ratio to pass |
| `--backend BACKEND` | Transcription backend used for re-transcription |
| `--model MODEL` | Model name for the QA backend |
| `--language LANGUAGE` | Language code for the QA backend |

Prints a per-sample report to stdout (your terminal window), writes
`qa_report.json` next to the SRT, and exits with an error status if QA
fails.

### `alwayswhisper auto INPUT -o OUT_MP4`

Does the whole job in one step: transcribe, then caption, back to back —
the command behind the one-line Quickstart example above.

| Flag | Description |
|---|---|
| `-o, --output OUT_MP4` | Output video path (required) |
| `--workdir DIR` | Working directory for intermediate transcription artifacts (default: `<OUTPUT stem>_work` next to OUTPUT) |
| `--style NAME_OR_PATH` | Caption style: `default`, `en`, or a YAML file path |
| `--fast` | Fast libass/ffmpeg burn-in instead of MoviePy/PIL |
| `--no-qa` | Skip the AV QA spot check |
| `--realign` | Snap SRT starts to word timestamps before burning |
| `--backend BACKEND` | Transcription backend: `faster-whisper` or `openai-api` |
| `--model MODEL` | Model name (faster-whisper only) |
| `--language LANGUAGE` | Language code, e.g. `ja`, `en` |
| `--device DEVICE` | faster-whisper device: `cpu`, `cuda`, or `auto` |
| `--compute-type COMPUTE_TYPE` | faster-whisper compute type |
| `--glossary FILE` | Text file of vocabulary to bias transcription toward |
| `--max-chars MAX_CHARS` | Max characters per caption entry |
| `--min-chars MIN_CHARS` | Min characters per caption entry |
| `--vad-filter` | Enable faster-whisper's voice-activity-detection filter |

Outputs: the captioned video, a `.srt` sibling next to it (the SRT that was
actually burned), and (unless `--no-qa`) `qa_report.json`.

### `alwayswhisper prefetch`

Downloads a faster-whisper model ahead of time, without transcribing
anything (see [Installing and pre-downloading
models](#installing-and-pre-downloading-models) above).

| Flag | Description |
|---|---|
| `--model MODEL` | Model name to download (default: `large-v3`) |

Run any subcommand with `--help` for the exact, current flag list.

## Configuration

A config file is just a plain text file of settings you can reuse across
runs, instead of retyping the same flags every time. `--config config.yaml`
is **deep-merged** over AlwaysWhisper's packaged defaults: unlike some
pipelines, a partial config file is fine — any key you don't set keeps its
packaged default, section by section, key by key. CLI flags are merged on
top of that (only the flags you actually pass; an unset flag never
overwrites your `--config` value).

Full key table, matching `src/alwayswhisper/data/default_config.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `transcribe.backend` | `faster-whisper` | `faster-whisper` (local) or `openai-api` (hosted, needs the `[api]` extra) |
| `transcribe.model` | `large-v3` | faster-whisper model name; ignored by the openai-api backend (always `whisper-1`) |
| `transcribe.device` | `auto` | faster-whisper device: `cpu`, `cuda`, or `auto` |
| `transcribe.compute_type` | `auto` | faster-whisper compute type (e.g. `int8`, `float16`) |
| `transcribe.language` | `ja` | Language code passed to the backend; also selects the caption segmenter (see [Language support](#language-support)) |
| `transcribe.prompt` | `null` | Optional Whisper bias prompt text; `null` = none (or use `--glossary FILE`) |
| `transcribe.prompt_max_tokens` | `224` | Bias prompt is truncated (from the front) to this many estimated tokens, matching Whisper's own prompt budget |
| `transcribe.strip_phrases` | `null` | `null` = use the built-in Japanese hallucination phrases when `language` is `ja` (none otherwise); `[]` disables stripping; a non-empty list replaces the defaults wholesale |
| `srt.max_chars` | `null` | Max characters per caption entry; `null` resolves to 35 (ja/zh/yue/th/lo/my) or 42 (other languages) |
| `srt.min_chars` | `4` | Min characters per caption entry |
| `caption.style` | `null` | `null` = packaged default style; also accepts `"default"`, `"en"`, or a path to a style YAML |
| `caption.fast_mode` | `false` | `true` = libass/ffmpeg burn-in (needs a libass-enabled ffmpeg, see [Requirements](#requirements)) |
| `caption.realign` | `false` | Snap SRT start times to word-level timestamps before burning (see [Word-level realignment](#word-level-realignment)) |
| `qa.enabled` | `true` | Run the automatic AV QA spot check before burning |
| `qa.samples` | `5` | Number of caption entries to spot-check |
| `qa.min_ratio` | `0.5` | Minimum fuzzy-match ratio for a sample to pass |
| `qa.pad_ms` | `300` | Padding (ms) added around each sampled entry before re-extracting its audio |
| `qa.min_entry_ms` | `500` | Entries shorter than this are excluded from sampling entirely (see [QA](#qa)) |

One recognized key isn't in the packaged YAML: `transcribe.vad_filter`
(settable via `--vad-filter` or your own `config.yaml`) defaults to `false`
in code when absent.

## Caption styles

A caption style controls how the captions actually look on screen — font,
size, color, background box, and so on. `--style default|en|/path/to/style.yaml`
selects which one to use. Two styles ship with AlwaysWhisper
(`data/styles/default.yaml`, tuned for Japanese with a CJK — Chinese/
Japanese/Korean — font stack and `font_size: 64`; `data/styles/en.yaml`, a
Latin font stack at `font_size: 44`) — pass a path to any YAML with the
same shape to fully customize it. Annotated shape:

```yaml
position:
  align: "center_bottom"     # only value currently supported
  margin_bottom: 60          # px from the bottom edge

background:
  color: [0, 0, 0, 200]      # [R, G, B, A]
  corner_radius: 12          # px; standard mode only (see table below)
  padding: 16                # px around the text

text:
  color: "#FFFFFF"
  font_family:                # tried in order; paths and installed family
    - "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"   # names both work
    - "Hiragino Sans"
  font_size: 64
  font_weight: "bold"        # fast mode only (see table below)

shadow:
  enabled: true
  offset: [2, 2]              # [x, y] px
  color: "#333333"

animation:
  type: "typewriter"          # only value currently supported
  completion_ms: 400          # time to reveal a full caption entry
```

**Fast vs. standard rendering differences** — both paths read the same
style file, but they're drawn by different software, and that software
doesn't have identical capabilities: standard mode draws captions itself
with PIL/MoviePy (two Python image/video libraries), while fast mode hands
the job to libass, a library that draws subtitles in a format called ASS
(Advanced SubStation Alpha) — which is why some differences below are
described in ASS's own terms:

| Style key | Standard (MoviePy/PIL) | Fast (libass/ffmpeg) |
|---|---|---|
| `background.corner_radius` | Honored — rounded box | **Ignored** — libass boxes are always rectangular |
| `shadow.color` | True drop-shadow color, drawn behind the text | Reused as the box **border/outline** color instead |
| `shadow.offset` | Both X and Y honored | Only X honored (ASS shadow distance is a single scalar) — Y is **ignored** |
| `text.font_weight` | **Ignored** — boldness comes entirely from which font file `font_family` resolves to | Honored — libass synthesizes/selects bold from the `Bold` flag |

**Font sizing**: the two settings that matter most are `font_size` (how big
the text is) and how many characters fit on one line before it runs out of
room. At 720p (a common video resolution, 1280 pixels wide), max characters
per line ≈ `font_size × 0.20` (96px ≈ 19 chars, 64px ≈ 28 chars, 48px ≈ 38
chars). Pick a `font_size` where your longest caption fits on one line —
this matters more than it sounds: standard mode has **no line-wrapping at
all** (an overlong caption just extends past both frame edges and gets
clipped by the frame boundary), and fast mode's own line-wrapping setting
(libass's `WrapStyle: 0`) is unreliable for Japanese/CJK text since there
are no spaces for it to break on, so in practice captions that are too wide
get clipped in both modes, not wrapped. The Japanese segmenter's
`--max-chars` is also a soft target, not a hard cap: to avoid cutting
mid-sentence, it can run a caption a few characters over the limit when
that lands on a full grammatical sentence ending — so size your font for a
bit more than `max_chars`, not exactly `max_chars`.

## Word-level realignment

`--realign` (with `--words transcript_words.json`) fixes captions whose
timing has **drifted** — meaning the text is right, but the start time no
longer quite lines up with when it's actually spoken (this can happen after
you or an LLM hand-edit an SRT: the text changes but the clock doesn't
automatically follow along). It snaps each SRT entry's start time to the
nearest word-level timestamp in the original transcript, preserving each
entry's duration and clamping away (trimming back) any resulting overlap
with the next entry (`prev.end <= next.start`). It's opt-in, and meant for
subtitles whose *text* you trust but whose *timing* has drifted — e.g.
after a manual or LLM-assisted correction pass introduced small cumulative
drift between entries and the audio.

If the SRT you're realigning was also produced from *cut* audio (some
external edit removed fillers/pauses between the original transcript and
this SRT), pass `--edit-plan edit_plan.json` too — a JSON file (a common
structured-text format for data) with `filler_removals`/`pause_removals`
lists, each entry an SRT-timestamp `start`/`end` pair and optionally an
explicit `removed_ms` — so realignment can account for what was physically
cut. The realigned copy is saved alongside the output as `<output
stem>.realigned.srt`.

## QA

Before burning captions in (unless `qa.enabled: false` / `--no-qa`),
AlwaysWhisper randomly samples `qa.samples` caption entries that are at
least `qa.min_entry_ms` long, re-extracts each one's audio (padded by
`qa.pad_ms`), re-transcribes it **without the bias prompt** (independence
from the prompt is the point — a biased re-transcription could "confirm" a
drifted caption instead of catching it), strips known hallucination phrases
from the result, normalizes both texts (NFKC — a standard way of making
equivalent character forms consistent — plus lowercase and punctuation
stripped), and fuzzy-matches them with `difflib.SequenceMatcher` (a
standard Python tool for scoring how similar two pieces of text are). Any
sample below `qa.min_ratio` **fails** the whole check; `caption`/`auto`
raise an error and never burn the video if that happens (the
`qa_report.json` is still written either way, so you can inspect what
failed).

In plain words: a FAIL means at least one caption sample didn't sound close
enough to what AlwaysWhisper actually heard when it double-checked, so it
refuses to guess and stops for you to look at `qa_report.json` first.

Tuning:

- `qa.samples` — how many entries to check (more = slower but more coverage)
- `qa.min_ratio` — how forgiving the fuzzy match is (lower = more forgiving)
- `qa.min_entry_ms` — **short entries can false-positive-fail** (fail the
  check even though they're actually correct): a half-second caption where
  the re-transcription's word/segment boundaries land slightly differently
  can score a low ratio despite being correct. Raise `qa.min_entry_ms`
  (e.g. `2000`) to exclude short entries from sampling entirely if you're
  seeing spurious failures.
- `--no-qa` (or `qa.enabled: false`) skips the check entirely — no backend
  is even created in that case.

## Python API

Most of the time the commands above are all you need. Reach for the Python
API instead when you want to call AlwaysWhisper from your own Python script
— for example, to fold it into a bigger automated pipeline instead of
running it by hand:

```python
from alwayswhisper import load_config, transcribe_file, caption_video, auto_run

cfg = load_config(overrides={
    "transcribe": {"language": "ja"},
    "srt": {"max_chars": 21},
    "caption": {"fast_mode": True},
})

# One-shot: transcribe + caption
report = auto_run("talk.mp4", "final.mp4", cfg)
print(report["srt_path"], report["caption"]["qa_report_path"])

# Or drive the two steps separately (e.g. to hand-edit the SRT in between)
transcribe_report = transcribe_file("talk.mp4", "talk_work", cfg)
# ... edit transcribe_report["srt_path"] on disk here if you want ...
caption_report = caption_video(
    "talk.mp4",
    transcribe_report["srt_path"],
    "final.mp4",
    cfg,
    words_json=transcribe_report["words_path"],
)
```

`load_config()` with no arguments returns the packaged defaults untouched;
pass `config_path=` for a YAML file and/or `overrides=` for a dict (a
Python object of key/value pairs), both deep-merged the same way the CLI
does it.

## Language support

AlwaysWhisper is tuned for **Japanese** first. Its segmenter dispatch
(which caption-splitting logic gets used) mirrors faster-whisper's own
tokenizer no-space language set:

- `ja`, `zh`, `yue`, `th`, `lo`, `my` → the character-based segmenter. Its
  sentence-ending detection rules (です/ます forms, connective particles,
  ...) are Japanese-specific — for the other languages in this set they
  degrade gracefully to plain character-limit splitting, which is still the
  right *kind* of segmenter for a script with no inter-word spaces.
- Every other language, **including an unset/unrecognized language**, gets
  a simpler space-delimited segmenter tuned for languages like English.

Hallucination-phrase stripping (`transcribe.strip_phrases`) is Japanese by
default and only auto-applies when `transcribe.language` is `ja`; set your
own list for other languages, or leave it as `[]`.

## Optional: saving results to Notion (a spec for coding agents)

**This is not built into AlwaysWhisper.** Nothing below runs today. This
section is a complete, ready-to-hand-over specification: everything a
coding agent (Claude Code, Codex, or similar) needs to build a "push this
transcript to Notion" feature locally, on top of AlwaysWhisper's existing
output files, without the agent having to go research the Notion API
itself first. If you want this feature, paste the prompt at the bottom of
this section to a coding agent.

Notion (https://www.notion.so) is a note-taking and database app; this spec
assumes you already have a Notion account and workspace, and want each
transcribed video to become one page inside a Notion database (a Notion
collection of pages that share the same set of properties, like columns in
a spreadsheet).

### 1. What a human has to set up first

Before any code can talk to Notion, a person needs to do this once, by
hand:

1. Create an **internal connection** (Notion's name for an API integration
   scoped to your own workspace) at
   [notion.so/profile/integrations](https://www.notion.so/profile/integrations).
2. Copy its access token from the connection's Configuration tab. Treat
   this token like a password — anyone who has it can read and write
   whatever you share with the connection.
3. **Share the target database with the connection** from Notion's own UI
   — open the database, then use its page menu (Connections, or "Connect
   to") to grant the connection access. The API cannot see a database it
   hasn't been explicitly shared with, even with a valid token.
4. Open that database in Notion and copy its **database ID** out of the
   page's URL.
5. Make both of those available to the script as environment variables —
   for example `NOTION_TOKEN` (the access token from step 2) and
   `NOTION_DATABASE_ID` (the ID from step 4).

### 2. What to build

A `notion-push` subcommand (e.g. `alwayswhisper notion-push <workdir>
...`) or a small standalone script that:

1. Takes a finished AlwaysWhisper output as input — either a working
   directory from `transcribe`/`auto` (which contains
   `transcript_raw.srt` and `transcript_words.json`), or a Python
   `transcribe_file(...)` result (whose returned dict includes
   `srt_path`; see [Python API](#python-api) above).
2. Creates **one Notion page per video**, with these suggested properties:
   - **title** — the video's file name
   - **date** — when it was recorded
   - **number** — duration, in seconds
   - **select** — language code (e.g. `ja`, `en`)
   - **url** (optional) — a link to the video itself, if you host it
     somewhere
3. Writes the transcript into the page body as a sequence of paragraph
   blocks (Notion's name for one block of body text) — one block per SRT
   entry, or grouped a few entries at a time, whichever reads better.

### 3. The API calls, in order

Every request needs these headers:

```http
Authorization: Bearer <NOTION_TOKEN>
Notion-Version: 2026-03-11
Content-Type: application/json
```

`2026-03-11` is the current API version as of this writing; the older
`2022-06-28` still works but predates the "data source" concept used below.
Base URL for every call: `https://api.notion.com/v1`.

**Step 1 — discover the data source.** Since API version 2025-09-03, a
database can contain one or more *data sources*, and pages are created
under a data source, not the database directly — so look it up first:

```http
GET /v1/databases/{database_id}
```

The response includes `data_sources: [{id, name}, ...]`. Store the `id`
you want (usually the only one) — you'll need it as `data_source_id` in
the next step.

**Step 2 — create the page**, with its properties:

```http
POST /v1/pages
```

```json
{
  "parent": { "type": "data_source_id", "data_source_id": "<data_source_id from step 1>" },
  "properties": {
    "Name": { "title": [{ "text": { "content": "talk.mp4" } }] },
    "Recorded": { "date": { "start": "2026-09-04" } },
    "Duration (s)": { "number": 613.2 },
    "Language": { "select": { "name": "ja" } },
    "Video": { "url": "https://example.com/talk.mp4" }
  }
}
```

Property names (`Name`, `Recorded`, ...) must match whatever property names
your database actually uses — these are examples, not fixed requirements.
If you also want a short plain-text property instead of (or alongside) the
title, use `rich_text`, shaped like this: `{"Summary": {"rich_text":
[{"text": {"content": "..."}}]}}`.

**Step 3 — append the transcript as body text**, in batches:

```http
PATCH /v1/blocks/{page_id}/children
```

```json
{
  "children": [
    {
      "object": "block",
      "type": "paragraph",
      "paragraph": { "rich_text": [{ "text": { "content": "One caption or paragraph of transcript text." } }] }
    }
  ]
}
```

Repeat this call once per batch until the whole transcript has been
appended (see the batch-size limit below). Because this always adds to the
end, you don't need the `position` field that this API version introduced
to replace the older `after` parameter — that only matters if you're
inserting blocks somewhere other than the end.

**Handling rate limits.** If any call returns HTTP 429 (Too Many Requests),
read the `Retry-After` response header (how many seconds to wait), sleep
for that long, then retry the same request. Don't just retry immediately
in a loop.

**Hard limits to respect** (exceeding these makes the request fail
outright, so batch to stay under them):

- Max **100 blocks** per `children` append request — split a long
  transcript into batches of 100 paragraph blocks or fewer, and call step 3
  once per batch.
- Max **2000 characters** in any single `text.content` string — split a
  longer caption/paragraph across multiple blocks or multiple `text`
  entries.
- Max **100 elements** in any array field (a `children` list or a
  `rich_text` list).
- Whole request max **1000 block elements / 500 KB** — another reason to
  batch rather than send everything at once.
- URLs (like the `url` property) max **2000 characters**.
- Rate limit: about **3 requests per second**, on average, per connection
  (short bursts are fine) — see the 429-handling note above for what to do
  if you go over.

(Small detail, only relevant if you ever delete or restore pages through
this API: version `2026-03-11` renamed the `archived` field to `in_trash`.)

### 4. How to test it

Don't hit the real Notion API in automated tests. Mock the HTTP calls (fake
the `requests`/`httpx` responses in your tests, using something like
Python's `unittest.mock` or the `responses` library) so tests run offline
and don't depend on a real token or database. Separately, add a
`--dry-run` flag to the subcommand/script that builds the request bodies
and prints them instead of sending anything — useful for a human to
sanity-check what would be written before it actually reaches Notion.

### 5. Docs for the agent

Point the coding agent at these instead of relying on anyone's memory of
the Notion API (API details change over time):

- [developers.notion.com/llms.txt](https://developers.notion.com/llms.txt)
  — a machine-readable index of every page in Notion's developer docs; each
  page listed there also exists as plain `.md`, which makes it easy for an
  agent to fetch and read the current docs directly rather than relying on
  training data.
- [notion.so/profile/integrations](https://www.notion.so/profile/integrations)
  — where the internal connection is created and its token copied (step 1
  above).

For Python, plain `requests` or `httpx` with the headers above is enough.
If you'd rather use an SDK (software development kit — a pre-built helper
library), there is a community-maintained `notion-client` package on PyPI
(latest version 3.1.0 as of this spec; Notion's own official SDK is the
JavaScript one) — just set its Notion version explicitly to `2026-03-11`
rather than trusting its own default.

### Prompt you can paste to a coding agent

```
Implement a `notion-push` feature for AlwaysWhisper (see README.md's "Optional:
saving results to Notion" section for the full spec) that takes the output of
`alwayswhisper transcribe` (a working directory containing transcript_raw.srt
and transcript_words.json, or the dict returned by the Python transcribe_file()
API) and creates one Notion page per video.

Requirements:
- Read NOTION_TOKEN and NOTION_DATABASE_ID from the environment.
- Look up the database's data source via GET /v1/databases/{database_id}
  before creating any page.
- Create the page via POST /v1/pages under that data_source_id, with
  properties for title (video file name), date recorded, duration in
  seconds (number), language (select), and an optional video URL.
- Append the transcript as paragraph blocks via PATCH
  /v1/blocks/{page_id}/children, batched at no more than 100 blocks per
  request and no more than 2000 characters per text block.
- On HTTP 429, wait for the Retry-After header's value and retry.
- Send Notion-Version: 2026-03-11 on every request.
- Add a --dry-run flag that builds and prints the request bodies without
  sending them.
- Mock all HTTP calls in tests -- do not hit the real Notion API.

Fetch https://developers.notion.com/llms.txt yourself and read the current
docs before implementing, rather than relying on training data -- the Notion
API changes over time.
```

## License

AlwaysWhisper is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) — see [LICENSE](LICENSE).

You may use, modify and share it for **noncommercial purposes** (personal use, education, research, and use by noncommercial organizations). **Commercial use is not permitted** under this license. For a commercial license, contact <support@pmdao.org>.

Required Notice: Copyright (c) 2026 kzkhykw (support@pmdao.org)

In plain terms: a hobbyist, student, or researcher can use and change
AlwaysWhisper for free; a company using it to make money — including as
part of a paid product or service — needs to buy a commercial license
first.

Required Notice: Copyright (c) 2026 kzkhykw (support@pmdao.org)
