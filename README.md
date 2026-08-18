# video-transcriber — Local Transcription with Speaker Diarization

![Python](https://img.shields.io/badge/python-3.12-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![uv](https://img.shields.io/badge/uv-0.11-4E5C66?style=for-the-badge&logo=astral&logoColor=white)
![WhisperX](https://img.shields.io/badge/WhisperX-3.8.6-2a6df4?style=for-the-badge)
![faster-whisper](https://img.shields.io/badge/faster--whisper-1.2-2a6df4?style=for-the-badge)
![pyannote-audio](https://img.shields.io/badge/pyannote--audio-4.0-FF6600?style=for-the-badge)
![ffmpeg](https://img.shields.io/badge/ffmpeg-9.0-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![GUI](https://img.shields.io/badge/GUI-tkinter-7c4dff?style=for-the-badge)

> **Author:** Juan Pablo Gómez Veira
> **Year:** August 2026

---

## Project Overview

Transcribes videos and audio (meetings, classes, lectures, podcasts, interviews — any recording) into timestamped Markdown transcripts and SRT/VTT subtitles, entirely on your own machine. No uploads, no subscriptions, no usage limits.

Built on [WhisperX](https://github.com/m-bain/whisperX) — faster-whisper for speech recognition, wav2vec2 forced alignment for word-accurate timestamps, and pyannote speaker diarization to separate the voices in the room.

## Features

- Local and offline after one-time model download (models cached in `models/`)
- GPU-accelerated with CUDA (falls back to CPU), quantized int8 to fit low-VRAM cards
- Speaker diarization: each voice becomes `SPEAKER_00`, `SPEAKER_01`, ...
- Configurable main speaker: assign a label to the most-spoken speaker automatically, or mark it manually by listening to a short audio sample of each voice
- Bilingual-friendly: Spanish by default, with a prompt that keeps English technical terms intact
- One output folder per video, saved next to the original recording
- Drag-and-drop GUI (`gui.py`) and a full CLI (`transcribe.py`)

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

### GUI (recommended)

```powershell
uv run gui.py
```

Drag and drop media files, pick formats/language, and press Transcribir. A progress bar plus a live log show file duration/size, per-phase timing (transcription, alignment, diarization), and detected speakers.

For a double-clickable launcher without a console window, build one once with:

```powershell
powershell -ExecutionPolicy Bypass -File build-exe.ps1
```

(`Transcribir.cmd` is the console version — handy for debugging.)

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
| `--formats` | `markdown,srt` | Comma list: `markdown,srt,vtt,json` |
| `--batch-size` | `8` | Lower if the GPU runs out of memory |
| `--out` | next to video | Base output directory (one subfolder per video) |
| `--no-diarize` | off | Skip speaker labels |
| `--auto-speaker` | off | Assign `--speaker-name` to the most-spoken speaker |
| `--speaker-name` | `Profesor` | Label for the main speaker |
| `--speaker-clips DIR` | — | Save a short WAV per speaker to listen to the voices |

## Output layout

Each transcription goes into a folder named after the source file, next to the original video:

```text
media\
├── recording-2026-07-28.mp4
└── recording-2026-07-28\
    ├── recording-2026-07-28.md
    ├── recording-2026-07-28.srt
    └── recording-2026-07-28.vtt
```

Subtitles are merged into sentence-sized lines. Load the `.srt` in VLC alongside the recording to verify the transcript against the audio.

## Speaker identification

Diarization clusters voices automatically; it never needs to know how many speakers there are. Two ways to label the main speaker (e.g. the teacher in a class, the host in a meeting):

- **Automatic:** `--auto-speaker` assigns `--speaker-name` (default `Profesor`) to the speaker with the most speaking time.
- **Manual:** with auto off, the GUI plays a short audio sample of each voice so you can mark the right one by ear.

## Performance

Measured on a GTX 1650 SUPER (4GB) with `large-v3-turbo` + int8: roughly **3 minutes of ASR per hour of audio**, plus a few minutes for alignment and diarization.

First run downloads the ASR model (~1.6GB) and the diarization pipeline (~100MB) into `models/`; afterwards everything is cached.

## Project structure

```text
video-transcriber/
├── transcribe.py        # core pipeline + CLI
├── gui.py               # drag-and-drop interface
├── launcher.cs          # source for the optional exe launcher
├── build-exe.ps1        # compiles the launcher (csc)
├── Transcribir.cmd      # console launcher
└── pyproject.toml       # uv project (Python 3.12)
```

## Troubleshooting

- **`ffmpeg not found`** — install it and open a new terminal.
- **`torchcodec ... not installed correctly` warning** — harmless; audio is loaded via ffmpeg, torchcodec is never used.
- **CUDA out of memory** — lower `--batch-size` (e.g. `4`) or use `--model medium`.
- **No speaker labels** — the HF token is missing, or the model license at https://huggingface.co/pyannote/speaker-diarization-community-1 was not accepted.
- **Diarization merges several similar voices into one speaker** — expected; pyannote clusters similar-sounding voices, and forcing more speakers splits one person into two instead of fixing it.
