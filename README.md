# video-transcriber — Local Transcription with Speaker Diarization

![Python](https://img.shields.io/badge/python-3.12-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![uv](https://img.shields.io/badge/uv-0.11-4E5C66?style=for-the-badge&logo=astral&logoColor=white)
![WhisperX](https://img.shields.io/badge/WhisperX-3.8.6-2a6df4?style=for-the-badge)
![faster-whisper](https://img.shields.io/badge/faster--whisper-1.2-2a6df4?style=for-the-badge)
![pyannote-audio](https://img.shields.io/badge/pyannote--audio-4.0-FF6600?style=for-the-badge)
![ffmpeg](https://img.shields.io/badge/ffmpeg-9.0-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![GUI](https://img.shields.io/badge/GUI-Qt+%2B+tkinter-7c4dff?style=for-the-badge)

> **Author:** Juan Pablo Gómez Veira
>
> **Year:** August 2026

---

## Project Overview

Transcribes videos and audio (meetings, classes, lectures, podcasts, interviews — any recording) into timestamped Markdown transcripts and SRT/VTT subtitles, entirely on your own machine. No uploads, no subscriptions, no usage limits.

Built on [WhisperX](https://github.com/m-bain/whisperX) — faster-whisper for speech recognition, wav2vec2 forced alignment for word-accurate timestamps, and pyannote speaker diarization to separate the voices in the room.

## Features

- Local and offline after one-time model download (models cached in `models/`)
- GPU-accelerated with CUDA (falls back to CPU), quantized int8 to fit low-VRAM cards
- Speaker diarization: each voice becomes `SPEAKER_00`, `SPEAKER_01`, ...
- Configurable main speaker: assign a label to the most-spoken speaker automatically (the other voices get grouped as `Hablante 1..N`), or mark it manually by listening to a short audio sample of each voice
- Main-speaker extract: `*.solo-principal.md` with only the main speaker's lines, written whenever the main speaker is identified (`--auto-speaker`)
- Name every voice and merge duplicated ones: two speakers given the same label become one person
- Hallucination guard: drops low-confidence segments using Whisper's own `avg_logprob` signal
- Speech-only diarization: only the regions the transcript marks as speech are diarized (silence skipped, timestamps remapped — subtitles unaffected; CUDA only, skipped automatically on very dense audio)
- Fast diarization: the segmentation sliding-window step runs at 2s (~2-4x faster than pyannote's 1s default; main-speaker labels hold, the rest just get coarser grouping — `--seg-stride` to tune)
- The ASR model is cached across the batch and unloaded before diarization so it stays fast on low-VRAM cards; the alignment model reloads per file (cheap, ~360MB) and the diarization pipeline stays loaded
- Graceful GPU-out-of-memory recovery (halves the batch size and retries)
- Spanish by default, with a domain prompt tuned for classes and meetings
- One output folder per video, saved next to the original recording
- Drag-and-drop GUI (`gui_qt.py`, Qt) with per-file table, dark/light themes, live progress, phase feedback, speaker review with audio playback, finish toast + chime, stop control, and persistent settings; classic tkinter GUI (`gui.py`) as fallback; plus a full CLI (`transcribe.py`)

## Requirements

- Windows with an NVIDIA GPU (4GB VRAM is enough) — CPU works but is much slower
- Python 3.12 (managed by [uv](https://docs.astral.sh/uv/))
- ffmpeg on PATH (`winget install Gyan.FFmpeg`)
- A free Hugging Face token to unlock the diarization model (see Setup)

## Setup

```powershell
# 1. Install dependencies
uv sync

# 2. Install ffmpeg (one time, system-wide)
winget install Gyan.FFmpeg

# 3. Speaker diarization: create a free read token at
#    https://huggingface.co/settings/tokens and accept the model license at
#    https://huggingface.co/pyannote/speaker-diarization-community-1
setx HF_TOKEN "hf_xxxx"
```

Without a token the tool still works — transcripts are produced without speaker labels.

## Usage

### GUI (recommended: Qt)

```powershell
uv run gui_qt.py
```

Two interfaces, same pipeline and results. The Qt GUI (`gui_qt.py`) is the modern one: per-file table with durations and live states, dark/light themes, native file dialogs and drag & drop, speaker review with one-click audio samples, completion toast + chime, and persistent settings. The classic tkinter GUI (`gui.py`) remains as a lightweight fallback.

Drag and drop media files, pick formats/language, and press Transcribir. A progress bar plus a live log show file duration/size, per-phase timing (transcription, alignment, diarization), and detected speakers.

For double-clickable launchers without a console window, build once with:

```powershell
powershell -ExecutionPolicy Bypass -File build-exe.ps1
```

(`Transcribir.cmd` / `Transcribir-qt.cmd` are the console versions — handy for debugging.)

### CLI

```powershell
uv run transcribe.py "C:\media\video1.mp4"        # markdown + srt (defaults)
uv run transcribe.py "C:\media\video1.mp4" --formats markdown,srt,vtt,json
uv run transcribe.py "C:\media\videos"            # whole folder
uv run transcribe.py video.mp4 --lang es --auto-speaker --speaker-name Profesor
```

### Common options

| Option | Default | Purpose |
|---|---|---|
| `--lang` | `es` | Language code (`es`, `en`, `auto`, ...) |
| `--model` | `large-v3-turbo` | Whisper model; `medium` if low on VRAM |
| `--formats` | `markdown,srt` | Comma list: `markdown,srt,vtt,txt,json` |
| `--batch-size` | `8` | Lower if the GPU runs out of memory (auto-retries with half) |
| `--out` | next to video | Base output directory (one subfolder per video) |
| `--no-diarize` | off | Skip speaker labels |
| `--no-resume` | off | Ignore saved checkpoints; transcribe from scratch |
| `--benchmark` | off | Measure this machine's speed (bundled sample), save to `perf.json`, exit |
| `--auto-speaker` | off | Assign `--speaker-name` to the most-spoken speaker |
| `--speaker-name` | `Profesor` | Label for the main speaker |
| `--speaker-clips DIR` | — | Save up to 3 short WAVs per speaker to listen to the voices (one subfolder per file) |
| `--no-vad-crop` | off | Diarize the full audio instead of only the speech regions |
| `--seg-stride` | `2.0` | Segmentation window step (2.0 = ~2-4x faster diarization with negligible main-speaker-label impact; 1.0 = upstream default) |
| `--no-solo-principal` | off | Skip the main-speaker-only `*.solo-principal.md` extract (written by default when `--auto-speaker` identifies the main speaker) |

## Output layout

Each transcription goes into a folder named after the source file, next to the original video (defaults: `.md` + `.srt`; request more with `--formats`):

```text
media\
├── recording-2026-07-28.mp4
└── recording-2026-07-28\
    ├── recording-2026-07-28.md
    ├── recording-2026-07-28.srt
    ├── recording-2026-07-28.solo-principal.md   # only with --auto-speaker
    └── recording-2026-07-28.aligned.json        # checkpoint, enables resume
```

Subtitles are built as broadcast-style cues: at most two lines of ~44 characters, ~5 seconds each — long sentences are split automatically. Load the `.srt` in VLC alongside the recording to verify the transcript against the audio.

## Speaker identification

Diarization clusters voices automatically; it never needs to know how many speakers there are. Two ways to label the main speaker (e.g. the teacher in a class, the host in a meeting):

- **Automatic:** `--auto-speaker` assigns `--speaker-name` (default `Profesor`) to the speaker with the most speaking time.
- **Manual:** with auto off, after each run the GUI lets you review **each file of a batch individually** — every detected voice shows its speaking time and **several audio samples**, so you can name them by ear.

You can also name every voice, not just the main one — and two voices given the same name are merged into one person, which fixes the common case where diarization splits a single speaker in two. The GUI keeps these settings between sessions.

## Performance

Measured on a GTX 1650 SUPER (4GB) with `large-v3-turbo` + int8: roughly **10 minutes of processing per hour of audio** — ASR ~7 s/min of audio, alignment ~1 s/min, diarization ~1-2 s/min with the 2s segmentation stride (less on newer GPUs, more on dense multi-speaker recordings).

First run downloads the ASR model (~1.6GB) and the diarization pipeline (~100MB) into `models/`; afterwards everything is cached.

Transcriptions are checkpointed: after aligning, each file saves a `<name>.aligned.json` next to its outputs. If a run is interrupted or you re-run the same files, transcription is skipped and only diarization runs — a crashed batch resumes in minutes, not hours. Delete the `.aligned.json` files (or pass `--no-resume`) to force a fresh transcription.

Speed estimates self-calibrate: every run records its measured rates, and --benchmark (or the GUI's Probar velocidad button) measures your machine with a bundled sample. The pre-flight dialog shows a time range and warns when the GPU is shared with heavy apps. Calibration lives in %LOCALAPPDATA%\video-transcriber\perf.json (machine-local, not committed).

## Project structure

```text
video-transcriber/
├── transcribe.py        # core pipeline + CLI
├── gui_qt.py            # Qt interface (recommended)
├── gui.py               # classic tkinter interface (fallback)
├── launcher-qt.cs       # source for the Qt exe launcher
├── launcher.cs          # source for the classic exe launcher
├── build-exe.ps1        # compiles the launchers (csc)
├── Transcribir.cmd      # console launcher (classic)
├── Transcribir-qt.cmd   # console launcher (Qt)
└── pyproject.toml       # uv project (Python 3.12)
```

## Troubleshooting

- **`ffmpeg not found`** — install it and open a new terminal.
- **`torchcodec ... not installed correctly` warning** — harmless; audio is loaded via ffmpeg, torchcodec is never used.
- **CUDA out of memory** — lower `--batch-size` (e.g. `4`) or use `--model medium`.
- **No speaker labels** — the HF token is missing, or the model license at https://huggingface.co/pyannote/speaker-diarization-community-1 was not accepted.
- **Diarization merges several similar voices into one speaker** — expected; pyannote clusters similar-sounding voices, and forcing more speakers splits one person into two instead of fixing it.
