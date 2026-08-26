#!/usr/bin/env python3
"""transcribe.py — free, local video/audio transcription with speaker diarization.

Powered by WhisperX (faster-whisper ASR + forced alignment + pyannote diarization).
Runs entirely on your machine; no uploads, no limits.

Default output: timestamped Markdown transcript + SRT subtitles. Optionally also
VTT subtitles (open the original video in VLC and load the .srt to verify the
transcript against the recording), plain-text transcript, and a JSON dump of all
segments.

Speaker diarization (who said what) uses a free Hugging Face token — one-time setup:
  1. Create a (free) account + read token:  https://huggingface.co/settings/tokens
  2. Accept the model license:              https://huggingface.co/pyannote/speaker-diarization-community-1
Then set the HF_TOKEN environment variable, or pass --hf-token. Without a token the
transcript is produced but WITHOUT speaker labels (and a warning is printed).

Examples:
  uv run transcribe.py "C:\\media\\video1.mp4"
  uv run transcribe.py "C:\\media\\video1.mp4" --formats markdown,srt,vtt --lang es
  uv run transcribe.py "C:\\media\\videos" --model large-v3-turbo --batch-size 8
  uv run transcribe.py video.mp4 --auto-speaker --speaker-name Profesor
  uv run transcribe.py video.mp4 --no-diarize           # skip speaker labels
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import datetime
import shutil
import traceback
import warnings
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

MEDIA_EXTS = {
    ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".wmv", ".mpg", ".mpeg",
    ".m4a", ".mp3", ".wav", ".wma", ".flac", ".ogg", ".aac", ".opus",
}

MODELS_DIR = Path(__file__).resolve().parent / "models"

# Known whisper model name -> HF repo. Downloaded into MODELS_DIR as plain
# local directories (avoids Windows symlink issues in the HF cache).
ASR_MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

DIARIZE_MODEL_REPO = "pyannote/speaker-diarization-community-1"

APP_DATA_DIR = Path(os.environ.get(
    "LOCALAPPDATA", str(Path.home() / "AppData" / "Local")
)) / "video-transcriber"
LOG_DIR = APP_DATA_DIR / "logs"
LOG_KEEP = 30


def new_log_path() -> Path:
    """Path for a fresh per-run log file; prunes old logs beyond LOG_KEEP."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = LOG_DIR / f"run-{stamp}.log"
    try:
        logs = sorted(LOG_DIR.glob("run-*.log"))
        for old in logs[:-LOG_KEEP]:
            old.unlink(missing_ok=True)
    except Exception:
        pass
    return path


class Tee:
    """Duplicates writes to a log file while keeping the original stream."""

    def __init__(self, stream, log):
        self._stream = stream
        self._log = log

    def write(self, text):
        if not text:
            return
        try:
            self._stream.write(text)
        except Exception:
            pass
        try:
            self._log.write(text)
            self._log.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

# Subtitle layout targets (broadcast-ish): a cue is at most 2 lines of ~44
# characters and at most ~5 seconds long.
CUE_MAX_CHARS = 88
CUE_MAX_LINE = 44
CUE_MAX_LINES = 2
CUE_MAX_SECONDS = 5.0
MIN_CUE_SECONDS = 0.6


def ensure_model(repo_id: str, local_name: str, marker: str = "config.json") -> Path:
    """Download a HF model into MODELS_DIR as a plain folder (no symlinks).

    `marker` is a file that must be present to consider the download complete
    (guards against a partial download from a previous failed run)."""
    from huggingface_hub import snapshot_download

    dest = MODELS_DIR / local_name
    if not (dest / marker).exists():
        print(f"[info] Downloading model {repo_id} -> {dest} ...")
        snapshot_download(repo_id, local_dir=str(dest))
        print("[info] Model ready.")
    return dest


def asr_model_path(model: str) -> Path:
    if model in ASR_MODEL_REPOS:
        return ensure_model(ASR_MODEL_REPOS[model], f"faster-whisper-{model}")
    p = Path(model)
    if p.exists():
        return p
    raise SystemExit(
        f"[error] Unknown ASR model '{model}'. Known: {', '.join(ASR_MODEL_REPOS)}"
    )


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

def hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def srt_ts(seconds: float) -> str:
    seconds = max(0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        seconds += 1
        ms -= 1000
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_ts(seconds: float) -> str:
    return srt_ts(seconds).replace(",", ".")


def speaker_of(seg: dict) -> str | None:
    sp = seg.get("speaker")
    return sp if sp else None


def merge_segments(segments: list, max_chars: int, max_seconds: float, min_break_chars: int) -> list:
    """Merge short consecutive segments into readable blocks.

    Fixes the "one word per subtitle" problem: whisperx can emit very short
    segments (especially after per-speaker splitting), which would otherwise
    produce illegible output. Blocks are kept within one speaker — a speaker
    change always starts a new block (never an inline "[SPEAKER_X]" marker)."""
    merged: list[dict] = []
    cur: dict | None = None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        sp = speaker_of(seg)
        if cur is None:
            cur = {"start": seg["start"], "end": seg["end"], "speaker": sp, "text": text}
            continue

        new_len = len(cur["text"]) + 1 + len(text)
        new_span = seg["end"] - cur["start"]
        break_line = (
            sp != cur["speaker"]           # never merge across speakers
            or new_len > max_chars
            or new_span > max_seconds
        )
        if break_line:
            merged.append(cur)
            cur = {"start": seg["start"], "end": seg["end"], "speaker": sp, "text": text}
        else:
            cur["text"] += " " + text
            cur["end"] = seg["end"]

    if cur:
        merged.append(cur)
    return merged


def display_speaker(sp: str | None, names: dict) -> str | None:
    if not sp:
        return None
    return names.get(sp, sp)


def drop_repeated_segments(segments: list, min_logprob: float = -1.0) -> list:
    """Drop segments Whisper produced while failing.

    Uses Whisper's own `log_prob_threshold` (-1.0): the model-native signal
    that a segment was guessed, not heard. This is exactly what OpenAI's
    pipeline applies at transcription time but WhisperX's batched path skips;
    it is NOT a text-pattern hack. Real speech — even real speech that repeats
    words — carries healthy confidence (measured on real classes: -0.04 to
    -0.36), so it is never touched."""
    out: list[dict] = []
    dropped = 0
    for seg in segments:
        lp = seg.get("avg_logprob")
        if lp is not None and lp < min_logprob:
            dropped += 1
            continue
        out.append(seg)
    if dropped:
        print(f"[info] Hallucination guard: dropped {dropped} failed segment(s).")
    return out


# --------------------------------------------------------------------------- #
# Subtitle cue building (merge + wrap + split long sentences)
# --------------------------------------------------------------------------- #

def _wrap_lines(text: str, max_line: int, max_lines: int) -> list[str]:
    """Greedy word wrap into at most `max_lines` lines. Text that still does not
    fit joins the last line (slight overflow beats a third line)."""
    words = text.split()
    if not words:
        return [text]
    lines = [""]
    for word in words:
        cand = f"{lines[-1]} {word}".strip()
        if len(cand) <= max_line or len(lines) == max_lines:
            lines[-1] = cand
        else:
            lines.append(word)
    return lines


def build_cues(segments: list, first_line_reserve: int = 0) -> list[dict]:
    """Turn raw segments into properly sized subtitle cues.

    Words are packed into lines of at most CUE_MAX_LINE characters, lines are
    grouped into cues of at most CUE_MAX_LINES lines, and cue timings are
    interpolated proportionally to text length so subtitles stay roughly in
    sync with the speech. `first_line_reserve` shrinks each cue's first line to
    leave room for a speaker tag."""
    blocks = merge_segments(segments, max_chars=240,
                            max_seconds=12.0, min_break_chars=25)

    def line_budget(idx_in_cue: int) -> int:
        return CUE_MAX_LINE - (first_line_reserve if idx_in_cue == 0 else 0)

    cues: list[dict] = []
    for block in blocks:
        text = block["text"].strip()
        if not text:
            continue

        # 1. Pack words into cues: each cue holds up to CUE_MAX_LINES lines,
        #    the first one shortened by the speaker-tag reserve.
        cue_lines: list[str] = []
        all_cues: list[list[str]] = []

        def flush_cue() -> None:
            nonlocal cue_lines
            if cue_lines:
                all_cues.append(cue_lines)
                cue_lines = []

        for word in text.split():
            if not cue_lines:
                cue_lines.append(word)
                continue
            is_first_line = len(cue_lines) == 1
            budget = line_budget(0) if is_first_line else CUE_MAX_LINE
            cand = f"{cue_lines[-1]} {word}"
            if len(cand) <= budget:
                cue_lines[-1] = cand
            elif len(cue_lines) < CUE_MAX_LINES:
                cue_lines.append(word)
            else:
                flush_cue()
                cue_lines.append(word)
        flush_cue()

        # 2. Interpolate timings across the block, proportional to length.
        span = max(MIN_CUE_SECONDS * len(all_cues), block["end"] - block["start"])
        total_chars = sum(len(" ".join(g)) for g in all_cues) or 1
        consumed = 0.0
        prev_end = block["start"]
        for group in all_cues:
            c_start = prev_end
            consumed += len(" ".join(group))
            c_end = block["start"] + span * (consumed / total_chars)
            c_end = max(c_end, c_start + MIN_CUE_SECONDS)
            prev_end = c_end
            cues.append({
                "start": c_start,
                "end": c_end,
                "speaker": block["speaker"],
                "lines": group,
            })

    # Enforce monotonic, non-zero-length cues.
    for prev, cur in zip(cues, cues[1:]):
        if cur["start"] < prev["end"]:
            cur["start"] = prev["end"]
        if cur["end"] <= cur["start"]:
            cur["end"] = cur["start"] + MIN_CUE_SECONDS
    return cues


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def _write_text(path: Path, content: str) -> None:
    """Atomic write: a crash mid-write can't leave a truncated output file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def write_markdown(path: Path, title: str, source: str, meta: dict, segments: list, names: dict) -> None:
    merged = merge_segments(segments, max_chars=300, max_seconds=90, min_break_chars=25)
    lines = [f"# {title}", ""]
    lines.append(f"> **Fuente:** {source}")
    lines.append(f"> **Idioma:** {meta.get('language', '?')} · **Modelo:** {meta.get('model')}"
                 f" · **Diarización:** {'Sí' if meta.get('diarized') else 'No'}")
    lines.append(f"> **Transcrito:** {datetime.date.today().isoformat()}")
    lines.append("")
    lines.append("---")
    lines.append("")
    for seg in merged:
        sp = display_speaker(speaker_of(seg), names)
        ts = f"**[{hms(seg['start'])}]**"
        text = seg["text"].strip()
        if not text:
            continue
        if sp:
            lines.append(f"{ts} **{sp}:** {text}")
        else:
            lines.append(f"{ts} {text}")
        lines.append("")
    _write_text(path, "\n".join(lines))


def _speaker_tag_reserve(segments: list, names: dict) -> int:
    """Characters to reserve on a cue's first line for the longest speaker tag."""
    tags = {display_speaker(speaker_of(s), names) or "" for s in segments}
    longest = max((len(t) for t in tags if t), default=0)
    return longest + 3 if longest else 0


def write_srt(path: Path, segments: list, names: dict) -> None:
    reserve = _speaker_tag_reserve(segments, names)
    cues = build_cues(segments, first_line_reserve=reserve)
    blocks = []
    for i, cue in enumerate(cues, start=1):
        body = "\n".join(cue["lines"])
        sp = display_speaker(speaker_of(cue), names)
        if sp:
            body = f"[{sp}] {body}"
        blocks.append(f"{i}\n{srt_ts(cue['start'])} --> {srt_ts(cue['end'])}\n{body}\n")
    _write_text(path, "\n".join(blocks))


def write_vtt(path: Path, segments: list, names: dict) -> None:
    reserve = _speaker_tag_reserve(segments, names)
    cues = build_cues(segments, first_line_reserve=reserve)
    blocks = ["WEBVTT", ""]
    for cue in cues:
        body = "\n".join(cue["lines"])
        sp = display_speaker(speaker_of(cue), names)
        if sp:
            body = f"<v {sp}>{body}</v>"
        blocks.append(f"{vtt_ts(cue['start'])} --> {vtt_ts(cue['end'])}\n{body}\n")
    _write_text(path, "\n".join(blocks))


def write_txt(path: Path, title: str, meta: dict, segments: list, names: dict) -> None:
    merged = merge_segments(segments, max_chars=300, max_seconds=90, min_break_chars=25)
    lines = [
        title,
        f"{meta.get('language', '?')} · {meta.get('model')} · "
        f"{'diarized' if meta.get('diarized') else 'no diarization'}",
        "",
    ]
    for seg in merged:
        sp = display_speaker(speaker_of(seg), names)
        text = seg["text"].strip()
        if not text:
            continue
        prefix = f"[{hms(seg['start'])}] "
        if sp:
            lines.append(f"{prefix}{sp}: {text}")
        else:
            lines.append(f"{prefix}{text}")
    _write_text(path, "\n".join(lines) + "\n")


def write_json(path: Path, result: dict) -> None:
    _write_text(path, json.dumps(result, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- #
# Speaker helpers (main-speaker detection + voice clips)
# --------------------------------------------------------------------------- #

def speaker_totals(segments: list) -> dict[str, float]:
    """Total speaking time per speaker label."""
    totals: dict[str, float] = {}
    for seg in segments:
        sp = speaker_of(seg)
        if sp:
            totals[sp] = totals.get(sp, 0.0) + (seg["end"] - seg["start"])
    return totals


def dominant_speaker(segments: list) -> tuple[str | None, float]:
    """Return (speaker_label, total_seconds) of the speaker with the most
    speaking time."""
    totals = speaker_totals(segments)
    if not totals:
        return None, 0.0
    best = max(totals, key=totals.get)
    return best, totals[best]


def extract_speaker_clips(audio, segments: list, out_dir, clip_seconds: float = 6.0,
                          samples: int = 3) -> dict:
    """Save up to `samples` short WAVs per speaker (their longest turns,
    center-cropped) so a human can hear each voice several times.
    Returns {speaker_label: [wav_path, ...]}."""
    import wave

    import numpy as np

    turns: dict[str, list[tuple[float, float, float]]] = {}
    for seg in segments:
        sp = speaker_of(seg)
        if not sp:
            continue
        dur = seg["end"] - seg["start"]
        if dur < 2.0:
            continue
        turns.setdefault(sp, []).append((dur, seg["start"], seg["end"]))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: dict[str, list[str]] = {}
    for sp, sp_turns in turns.items():
        sp_turns.sort(reverse=True)
        chosen = sp_turns[:samples]
        paths: list[str] = []
        for idx, (dur, start, end) in enumerate(chosen, start=1):
            mid = (start + end) / 2.0
            cstart = max(start, mid - clip_seconds / 2)
            cend = min(end, cstart + clip_seconds)
            if cend - cstart < 1.5:
                cstart = max(start, cend - 1.5)
            s = int(cstart * 16000)
            e = int(cend * 16000)
            pcm = (np.clip(audio[s:e], -1.0, 1.0) * 32767).astype(np.int16)
            path = out_dir / f"{sp}_{idx}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm.tobytes())
            paths.append(str(path))
        if paths:
            clips[sp] = paths
    return clips


# --------------------------------------------------------------------------- #
# NLTK punkt (needed by WhisperX sentence segmentation)
# --------------------------------------------------------------------------- #

def ensure_punkt() -> None:
    try:
        import nltk

        nltk.data.find("tokenizers/punkt_tab")
    except Exception:
        try:
            nltk.download("punkt_tab", quiet=True)
            nltk.download("punkt", quiet=True)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] Could not download nltk punkt data: {exc}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Main transcription
# --------------------------------------------------------------------------- #

def resolve_token(args) -> str | None:
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    cache_file = Path.home() / ".cache" / "huggingface" / "token"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8").strip() or None
    return None


def pick_device(args) -> str:
    if args.device:
        return args.device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def free_gpu() -> None:
    if torch is not None and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


# Baseline rates measured on a GTX 1650 SUPER: minutes of wall-clock per
# minute of audio. Real machines self-calibrate via perf.json (recorded from
# runs / benchmark); these are the fallback when nothing is calibrated yet.
RATE_ASR = 0.060
RATE_ALIGN = 0.022
RATE_DIARIZE = 0.065
PER_FILE_OVERHEAD = 0.35  # model loads, VAD, I/O
RATE_MARGIN_CALIBRATED = 0.15
RATE_MARGIN_ESTIMATED = 0.40

PERF_FILE = APP_DATA_DIR / "perf.json"
BENCH_SAMPLE = "sample.wav"
BENCH_SAMPLE_PATH = Path(__file__).resolve().parent / "assets" / BENCH_SAMPLE


# --------------------------------------------------------------------------- #
# Speed calibration (per-machine): perf.json + heuristic fallback
# --------------------------------------------------------------------------- #

def _perf_key(args, device: str) -> str:
    return f"{device}|{getattr(args, 'model', 'large-v3-turbo')}|{getattr(args, 'compute_type', 'int8')}"


def load_perf() -> dict:
    try:
        return json.loads(PERF_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_perf(perf: dict) -> None:
    try:
        APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PERF_FILE.write_text(json.dumps(perf, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    except Exception:
        pass


def record_rates(args, device: str, asr_rate=None, align_rate=None,
                 diarize_rate=None, benchmark: bool = False) -> None:
    """Store measured rates for this machine. Real runs use an EWMA so a
    single contended run can't poison the estimate; benchmarks overwrite."""
    perf = load_perf()
    key = _perf_key(args, device)
    entry = perf.setdefault(key, {"samples": 0, "asr_rate": None,
                                  "align_rate": None, "diarize_rate": None})
    if benchmark:
        for field, val in (("asr_rate", asr_rate), ("align_rate", align_rate),
                           ("diarize_rate", diarize_rate)):
            if val is not None and val > 0:
                entry[field] = val
        entry["samples"] = 1
    else:
        got = False
        for field, val in (("asr_rate", asr_rate), ("align_rate", align_rate),
                           ("diarize_rate", diarize_rate)):
            if val is not None and val > 0:
                prev = entry.get(field)
                entry[field] = prev * 0.7 + val * 0.3 if prev else val
                got = True
        if got:
            entry["samples"] = entry.get("samples", 0) + 1
    entry["measured_at"] = datetime.date.today().isoformat()
    save_perf(perf)


def backfill_perf_from_logs() -> None:
    """Stub (intentionally unused): old logs describe runs slowed by GPU
    contention, so they cannot calibrate a clean estimate. The benchmark and
    EWMA from live runs are the honest source."""
    return


def _gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        return out.strip().splitlines()[0] if out.strip() else ""
    except Exception:
        return ""


def heuristic_rates(device: str) -> dict:
    """Coarse rates when this machine has no calibration yet."""
    base = {"asr_rate": RATE_ASR, "align_rate": RATE_ALIGN,
            "diarize_rate": RATE_DIARIZE}
    if device != "cuda":
        return {k: v * 8 for k, v in base.items()}
    name = _gpu_name().lower()
    scale = 1.0
    if any(t in name for t in ("rtx 3", "rtx 40", "rtx 5", "rtx 50")):
        scale = 2.5
    elif "rtx 20" in name:
        scale = 1.3
    return {k: v / scale for k, v in base.items()}


def estimate_runtime(files, args, device: str, diarize: bool = True):
    """(low_min, high_min, calibrated) estimate for the given files."""
    perf = load_perf()
    entry = perf.get(_perf_key(args, device))
    calibrated = bool(entry and entry.get("asr_rate") and entry.get("align_rate")
                      and entry.get("diarize_rate"))
    rates = {k: entry[k] for k in ("asr_rate", "align_rate", "diarize_rate")} \
        if calibrated else heuristic_rates(device)

    total = 0.0
    for f in files:
        duration = media_duration(Path(f)) or 0.0
        mins = duration / 60.0
        total += mins * rates["asr_rate"] + mins * rates["align_rate"] + PER_FILE_OVERHEAD
        if diarize:
            total += mins * rates["diarize_rate"]
    margin = RATE_MARGIN_CALIBRATED if calibrated else RATE_MARGIN_ESTIMATED
    return total * (1 - margin), total * (1 + margin), calibrated


def gpu_status() -> dict | None:
    """Quick snapshot of CUDA GPU load (util %, VRAM used, other GPU apps).

    Returns None when nvidia-smi is unavailable (e.g. AMD/Intel hardware)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        if not out:
            return None
        util, mem_used, mem_total = (float(x.strip()) for x in out[0].split(","))
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return {"util": util, "mem_used": mem_used, "mem_total": mem_total,
                "apps": list(apps)}
    except Exception:
        return None


def gpu_is_busy(status: dict | None, util_threshold: float = 25.0,
                mem_threshold: float = 1500.0) -> bool:
    """True when other apps are likely contending for the GPU."""
    if not status:
        return False
    return status["util"] >= util_threshold or status["mem_used"] >= mem_threshold


def _is_cuda_oom(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return "OutOfMemoryError" in type(exc).__name__ or "out of memory" in text.lower()


def _is_cuda_fatal(exc: Exception) -> bool:
    """True when the GPU device/context was lost (unrecoverable in-process)."""
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    return any(k in low for k in (
        "invalid device ordinal",
        "no cuda-capable device",
        "device-side assert",
        "no kernel image",
        "driver shutting down",
        "cuda error: out of range",
        "not used",
    ))


def quiet_third_party_noise() -> None:
    """Silence known-benign warnings from third-party libraries.

    These are noise, not errors: torchcodec is never used (audio is loaded via
    ffmpeg), the Lightning/TF32/pyannote messages are informational. Hiding them
    keeps the logs readable without hiding real problems (those still raise).
    """
    import logging

    for message in (
        r"\s*torchcodec is not installed correctly",
        r"Lightning automatically upgraded your loaded checkpoint",
        r"TensorFloat-32 \(TF32\) has been disabled",
        r"std\(\): degrees of freedom is <= 0",
        r"Xet Storage is enabled",
        r"cache-system uses symlinks by default",
    ):
        warnings.filterwarnings("ignore", message=message)

    for name in (
        "pytorch_lightning",
        "lightning",
        "lightning.pytorch",
        "lightning.fabric",
        "lightning.pytorch.utilities.migration",
        "lightning.pytorch.utilities.migration.utils",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def initial_prompt_for(lang: str | None) -> str:
    """A short domain prompt that nudges Whisper toward the recording's language.
    Empty = no bias."""
    if lang in (None, "", "auto"):
        return ""
    if lang == "es":
        return (
            "Bienvenidos. Esta es una conferencia, presentación o reunión. "
            "Se hablará de tecnología, programación, bases de datos, análisis "
            "de datos y herramientas informáticas."
        )
    return (
        "Welcome. This is a conference, presentation or meeting. "
        "We will talk about technology, programming, databases, data "
        "analytics and software tools."
    )


def media_duration(path: Path) -> float | None:
    """Best-effort media duration in seconds, via PyAV (bundled with faster-whisper)."""
    try:
        import av

        with av.open(str(path)) as container:
            return float(container.duration / av.time_base)
    except Exception:
        return None


def progress_printer():
    """Return a whisperx progress_callback that prints % done + elapsed + ETA.

    WhisperX otherwise runs silently, which makes long recordings look frozen.
    """
    start = time.monotonic()
    last = {"pct": -1}

    def cb(pct: float) -> None:
        if pct >= 100 or pct - last["pct"] >= 5:
            last["pct"] = pct
            elapsed = time.monotonic() - start
            eta = elapsed / pct * 100 if pct > 0 else 0
            print(
                f"[progress] {pct:5.1f}%  elapsed {elapsed/60:5.1f} min"
                f"  ~total {eta/60:5.1f} min"
            )

    return cb


class ModelCache:
    """Keeps the small alignment and diarization models loaded between files.

    The big ASR model is NOT kept cached across the diarization step: its
    CTranslate2 weights occupy VRAM that pyannote needs, making diarization
    ~5x slower. transcribe_one evicts it after transcription and reloads it
    per file instead."""

    def __init__(self) -> None:
        self._items: dict = {}

    def get(self, key):
        return self._items.get(key)

    def put(self, key, value):
        self._items[key] = value

    def evict(self, key) -> bool:
        obj = self._items.pop(key, None)
        return obj is not None


def _transcribe_align(path: Path, args, cache: ModelCache) -> dict:
    """Stage 1 of a file: load ASR (cached across the batch), transcribe,
    evict the ASR model, align, run the hallucination guard.

    Returns a "prep" dict for stage 2. Audio is intentionally not kept in
    memory (stage 2 reloads it)."""
    quiet_third_party_noise()
    import whisperx

    def phase(name: str) -> None:
        cb = getattr(args, "phase_callback", None)
        if cb is not None:
            try:
                cb(name)
            except Exception:
                pass

    t_start = time.monotonic()
    device = pick_device(args)
    lang = None if args.lang in (None, "", "auto") else args.lang
    prompt = initial_prompt_for(lang)

    print(f"\n=== {path.name} ===")
    size_mb = path.stat().st_size / (1024 * 1024)
    duration = media_duration(path)
    duration_str = hms(duration) if duration else "?"
    print(f"[info] duration: {duration_str} · size: {size_mb:.0f} MB")
    print(f"[info] device={device}  model={args.model}  compute_type={args.compute_type}"
          f"  batch_size={args.batch_size}"
          + (f"  language={lang}" if lang else "  language=auto"))

    phase("load")
    # The ASR model is loaded once and reused across every file in the batch
    # (only evicted after transcription so diarization keeps full VRAM).
    model_key = ("asr", args.model, device, args.compute_type, bool(lang))
    model = cache.get(model_key)
    if model is None:
        model = whisperx.load_model(
            str(asr_model_path(args.model)),
            device=device,
            compute_type=args.compute_type,
            language=lang,
            asr_options={"initial_prompt": prompt},
        )
        cache.put(model_key, model)
    else:
        try:  # keep the language-bias prompt in sync with the current file
            model.options.initial_prompt = prompt
        except Exception:
            pass

    audio = whisperx.load_audio(str(path))
    print("[info] Transcribing...")
    phase("asr")

    external_cb = getattr(args, "progress_callback", None)
    printer = progress_printer()

    def on_progress(pct: float) -> None:
        printer(pct)
        if external_cb is not None:
            external_cb(pct)

    t0 = time.monotonic()
    batch = args.batch_size
    result = None
    for attempt in range(3):
        try:
            result = model.transcribe(
                audio,
                batch_size=batch,
                language=lang,
                progress_callback=on_progress,
            )
            break
        except Exception as exc:
            if _is_cuda_fatal(exc):
                raise
            if _is_cuda_oom(exc) and batch > 1 and attempt < 2:
                batch = max(1, batch // 2)
                # Drop the possibly-corrupt model instance: reload fresh, it
                # may have leaked GPU memory or left the device in a bad state.
                try:
                    cache.evict(model_key)
                except Exception:
                    pass
                del model
                free_gpu()
                model = whisperx.load_model(
                    str(asr_model_path(args.model)),
                    device=device,
                    compute_type=args.compute_type,
                    language=lang,
                    asr_options={"initial_prompt": prompt},
                )
                cache.put(model_key, model)
                print(f"[warn] GPU out of memory; retrying with batch_size={batch}"
                      f" (fresh model load)...")
            else:
                raise
    print(f"[info] Transcription done in {(time.monotonic() - t0) / 60:.1f} min"
          f" ({result['language']}, {len(result['segments'])} segments)")

    # Evict the ASR model BEFORE alignment/diarization. Its weights live in
    # CTranslate2-managed VRAM (invisible to torch), so leaving it resident
    # starves pyannote of memory and makes diarization ~5x slower. The small
    # alignment and diarization pipelines stay cached across files instead.
    cache.evict(model_key)
    del model
    free_gpu()

    language = result.get("language", lang or args.lang)

    # 2. Forced alignment (accurate word timestamps)
    phase("align")
    t_align = time.monotonic()
    try:
        align_key = ("align", language, device)
        aligned = cache.get(align_key)
        if aligned is None:
            aligned = whisperx.load_align_model(language_code=language, device=device)
            cache.put(align_key, aligned)
        model_a, metadata = aligned
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False,
        )
        print(f"[info] Alignment done in {(time.monotonic() - t_align) / 60:.1f} min")
    except Exception as exc:
        print(f"[warn] Alignment failed ({exc}); keeping Whisper timestamps.")

    segments = drop_repeated_segments(result["segments"])
    # Importantly, write the cleaned segments back into `result` so that
    # diarization's assign_word_speakers works on the cleaned list (otherwise
    # it silently resurrects every hallucinated segment that was removed).
    result["segments"] = segments

    audio_minutes = len(audio) / 16000 / 60
    measure = {}
    if audio_minutes > 0:
        measure["asr_rate"] = (time.monotonic() - t0) / 60 / audio_minutes
        try:
            measure["align_rate"] = (time.monotonic() - t_align) / 60 / audio_minutes
        except (UnboundLocalError, NameError):
            measure["align_rate"] = None
    return {
        "path": path,
        "result": result,
        "segments": segments,
        "language": language,
        "t_start": t_start,
        "device": device,
        "lang": lang,
        "measure": measure,
    }


def _diarize_write(prep: dict, args, cache: ModelCache) -> dict | None:
    """Stage 2 of a file: reload audio, diarize (ASR already evicted so it
    stays fast), name speakers, write all outputs.

    Returns {"clips", "totals", "names"} when diarization ran, else None."""
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    path = prep["path"]
    result = prep["result"]
    segments = prep["segments"]
    language = prep["language"]
    device = prep["device"]
    lang = prep["lang"]

    def phase(name: str) -> None:
        cb = getattr(args, "phase_callback", None)
        if cb is not None:
            try:
                cb(name)
            except Exception:
                pass

    audio = whisperx.load_audio(str(path))

    def phase(name: str) -> None:
        cb = getattr(args, "phase_callback", None)
        if cb is not None:
            try:
                cb(name)
            except Exception:
                pass

    diar_progress = getattr(args, "diarize_progress_callback", None)

    # 3. Speaker diarization (optional)
    phase("diarize")
    diarized = False
    if not args.no_diarize:
        token = resolve_token(args)
        if token:
            t0 = time.monotonic()
            print("[info] Running speaker diarization...")
            for attempt in range(2):
                try:
                    dia = cache.get("diarize")
                    if dia is None:
                        diarize_model_path = ensure_model(
                            DIARIZE_MODEL_REPO, "speaker-diarization-community-1",
                            marker="config.yaml",
                        )
                        dia = DiarizationPipeline(
                            model_name=str(diarize_model_path), token=token, device=device
                        )
                        cache.put("diarize", dia)
                    diarize_segments = dia(audio, progress_callback=diar_progress)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    segments = result["segments"]
                    diarized = True
                    _audio_min = len(audio) / 16000 / 60
                    if _audio_min > 0:
                        prep["diarize_rate"] = (time.monotonic() - t0) / 60 / _audio_min
                    speakers = sorted({s.get("speaker") for s in segments if s.get("speaker")})
                    print(f"[info] Diarization done in {(time.monotonic() - t0) / 60:.1f} min"
                          f" — {len(speakers)} speaker(s): {', '.join(speakers)}")
                    break
                except Exception as exc:
                    asr_key = ("asr", args.model, device, args.compute_type, bool(lang))
                    if _is_cuda_oom(exc) and cache.evict(asr_key) and attempt == 0:
                        free_gpu()
                        print("[warn] GPU out of memory during diarization;"
                              " freed the ASR model and retrying...")
                    else:
                        print(f"[warn] Diarization failed ({exc});"
                              " transcript is without speaker labels.")
                        break
        else:
            print(
                "[warn] No Hugging Face token found -> skipping speaker labels.\n"
                "  To enable diarization (free):\n"
                "    1) https://huggingface.co/settings/tokens   (create a read token)\n"
                "    2) https://huggingface.co/pyannote/speaker-diarization-community-1  (accept license)\n"
                "  Then run with:  set HF_TOKEN=hf_xxxx  (or pass --hf-token)"
            )

    # 3.5 Speaker naming: auto main speaker + voice clips
    names: dict = {}
    payload = None
    if diarized:
        totals = speaker_totals(segments)
        if getattr(args, "auto_speaker", False):
            dom, dom_secs = dominant_speaker(segments)
            label = getattr(args, "speaker_name", None) or "Profesor"
            if dom:
                names[dom] = label
                print(f"[info] Main speaker: {dom} ({dom_secs / 60:.1f} min spoken) -> '{label}'")
        clip_dir = getattr(args, "speaker_clips_dir", None)
        clips = {}
        if clip_dir:
            clips = extract_speaker_clips(audio, segments, clip_dir)
        payload = {"clips": clips, "totals": totals, "names": dict(names), "outputs": []}

    # 4. Write outputs — one folder per transcription, named after the source
    #    file, saved next to the original video (or inside --out if given)
    phase("save")
    stem = path.stem
    base = Path(args.out) if args.out else path.parent
    out_dir = base / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    title = args.title or stem.replace("_", " ").replace("-", " ").title()
    meta = {
        "language": language,
        "model": args.model,
        "diarized": diarized,
        "device": device,
    }
    wanted = [f.strip() for f in args.formats.split(",") if f.strip()]
    written: list[str] = []

    if "markdown" in wanted:
        md_path = out_dir / f"{stem}.md"
        write_markdown(md_path, title, str(path), meta, segments, names)
        written.append(str(md_path))
        print(f"[ok] {md_path}")
    if "srt" in wanted:
        srt_path = out_dir / f"{stem}.srt"
        write_srt(srt_path, segments, names)
        written.append(str(srt_path))
        print(f"[ok] {srt_path}")
    if "vtt" in wanted:
        vtt_path = out_dir / f"{stem}.vtt"
        write_vtt(vtt_path, segments, names)
        written.append(str(vtt_path))
        print(f"[ok] {vtt_path}")
    if "txt" in wanted:
        txt_path = out_dir / f"{stem}.txt"
        write_txt(txt_path, title, meta, segments, names)
        written.append(str(txt_path))
        print(f"[ok] {txt_path}")
    if "json" in wanted:
        json_path = out_dir / f"{stem}.json"
        write_json(json_path, result)
        written.append(str(json_path))
        print(f"[ok] {json_path}")

    print(f"[info] Total: {(time.monotonic() - prep['t_start']) / 60:.1f} min")

    if payload:
        payload["outputs"] = written

    return payload


def transcribe_one(path: Path, args, cache: ModelCache | None = None) -> dict | None:
    """Transcribe a single file end-to-end (stage 1 + stage 2)."""
    cache = cache if cache is not None else ModelCache()
    prep = _transcribe_align(path, args, cache)
    return _diarize_write(prep, args, cache)


def run_benchmark(args) -> dict | None:
    """Measure ASR/alignment/diarization rates on the bundled sample and store
    them in perf.json. One-time ~1 min, including model loads."""
    quiet_third_party_noise()
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    device = pick_device(args)
    lang = None if args.lang in (None, "", "auto") else args.lang

    if not BENCH_SAMPLE_PATH.exists():
        print(f"[error] Benchmark sample not found: {BENCH_SAMPLE_PATH}")
        return None

    audio = whisperx.load_audio(str(BENCH_SAMPLE_PATH))
    audio_minutes = len(audio) / 16000 / 60
    print(f"[info] Benchmarking on {audio_minutes:.1f} min of speech "
          f"(model={args.model}, {args.compute_type})...")

    # ASR
    model = whisperx.load_model(
        str(asr_model_path(args.model)), device=device,
        compute_type=args.compute_type, language=lang,
        asr_options={"initial_prompt": initial_prompt_for(lang)},
    )
    t0 = time.monotonic()
    model.transcribe(audio, batch_size=args.batch_size, language=lang)
    asr_rate = (time.monotonic() - t0) / 60 / audio_minutes
    print(f"[info] ASR rate: {asr_rate * 60:.1f} s per minute of audio")
    del model
    free_gpu()

    # Alignment
    align_rate = RATE_ALIGN
    t0 = time.monotonic()
    try:
        model_a, metadata = whisperx.load_align_model(language_code="es", device=device)
        aligned = whisperx.align(  # noqa: F841
            [], model_a, metadata, audio, device, return_char_alignments=False,
        )
        align_rate = (time.monotonic() - t0) / 60 / audio_minutes
    except Exception:
        pass
    try:
        del model_a
    except Exception:
        pass
    free_gpu()
    print(f"[info] Alignment rate: {align_rate * 60:.1f} s per minute of audio")

    # Diarization
    diar_rate = None
    if not args.no_diarize:
        token = resolve_token(args)
        if token:
            dia = DiarizationPipeline(
                model_name=str(ensure_model(
                    DIARIZE_MODEL_REPO, "speaker-diarization-community-1",
                    marker="config.yaml")),
                token=token, device=device,
            )
            t0 = time.monotonic()
            dia(audio)
            diar_rate = (time.monotonic() - t0) / 60 / audio_minutes
            print(f"[info] Diarization rate: {diar_rate * 60:.1f} s per minute of audio")
            del dia
            free_gpu()
        else:
            print("[warn] No HF token; diarization rate not measured")
    else:
        print("[info] Diarization disabled; rate not measured")

    record_rates(args, device, asr_rate=asr_rate, align_rate=align_rate,
                 diarize_rate=diar_rate, benchmark=True)
    return {"asr_rate": asr_rate, "align_rate": align_rate, "diarize_rate": diar_rate}


# --------------------------------------------------------------------------- #
# Checkpoint / resume: stage 1 results are saved so an interrupted batch only
# redoes diarization, never transcription.
# --------------------------------------------------------------------------- #

CHECK_SUFFIX = ".aligned.json"


def _checkpoint_path(path: Path, args) -> Path:
    stem = path.stem
    base = Path(args.out) if args.out else path.parent
    return base / stem / f"{stem}{CHECK_SUFFIX}"


def _save_checkpoint(prep: dict, args) -> None:
    try:
        path = _checkpoint_path(prep["path"], args)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"language": prep["language"], "result": prep["result"],
                "segments": prep["segments"]}
        _write_text(path, json.dumps(data, ensure_ascii=False))
    except Exception as exc:
        print(f"[warn] Could not save checkpoint: {exc}")


def _load_checkpoint(path: Path, args) -> dict | None:
    cp = _checkpoint_path(path, args)
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
        segments = data["segments"]
        result = data["result"]
        result["segments"] = segments
        print(f"[info] Resumed aligned transcription from checkpoint: {cp}")
        return {
            "path": Path(path),
            "result": result,
            "segments": segments,
            "language": data.get("language", "?"),
            "t_start": time.monotonic(),
            "device": pick_device(args),
            "lang": None if args.lang in (None, "", "auto") else args.lang,
        }
    except Exception as exc:
        print(f"[warn] Could not load checkpoint ({exc}); transcribing from scratch.")
        return None


def get_prep(path: Path, args, cache: ModelCache, no_resume: bool = False) -> dict:
    """Stage 1 with resume: reuse the checkpoint if present, otherwise
    transcribe+align and save a checkpoint."""
    prep = None if no_resume else _load_checkpoint(path, args)
    if prep is None:
        prep = _transcribe_align(path, args, cache)
        _save_checkpoint(prep, args)
    return prep


def transcribe_batch(files: list[Path], args, cache: ModelCache | None = None) -> None:
    """Transcribe many files in two passes: transcribe+align everything first
    (the ASR model loads once for the whole batch), then diarize+write
    everything (the ASR model is gone, so diarization keeps full VRAM)."""
    cache = cache if cache is not None else ModelCache()
    no_resume = bool(getattr(args, "no_resume", False))
    prepared: list[dict] = []
    for f in files:
        try:
            prepared.append(get_prep(f, args, cache, no_resume))
        except Exception as exc:
            print(f"[error] Failed on {f}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            if _is_cuda_fatal(exc):
                print("[error] The GPU was lost (out of memory or another GPU app)."
                      " The remaining files were skipped — close other GPU-heavy"
                      " apps and re-run.", file=sys.stderr)
                break
    for prep in prepared:
        try:
            _diarize_write(prep, args, cache)
        except Exception as exc:
            print(f"[error] Failed on {prep['path']}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # Record measured rates so the next estimate is calibrated to this machine.
    try:
        asr = [p["measure"]["asr_rate"] for p in prepared
               if p.get("measure") and p["measure"].get("asr_rate") is not None]
        align = [p["measure"]["align_rate"] for p in prepared
                 if p.get("measure") and p["measure"].get("align_rate") is not None]
        diar = [p["diarize_rate"] for p in prepared if p.get("diarize_rate")]
        if asr or align or diar:
            record_rates(
                args, pick_device(args),
                asr_rate=statistics.fmean(asr) if asr else None,
                align_rate=statistics.fmean(align) if align else None,
                diarize_rate=statistics.fmean(diar) if diar else None,
            )
    except Exception:
        pass


# Thin note: heavy imports (whisperx, pyannote) live inside transcribe_one so
# they run after the warning filters are registered.


def collect_inputs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(f for f in sorted(pp.rglob("*")) if f.suffix.lower() in MEDIA_EXTS)
        elif pp.is_file():
            files.append(pp)
        else:
            print(f"[warn] Not found: {p}", file=sys.stderr)
    return files


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="transcribe.py",
        description="Transcribe video/audio locally with WhisperX (free, no uploads).",
    )
    p.add_argument("inputs", nargs="*", help="Video/audio file(s), or folder(s) of them")
    p.add_argument("--benchmark", action="store_true",
                   help="Measure speeds on this machine with the bundled sample, "
                        "save to perf.json and exit")
    p.add_argument("--lang", default="es", help="Language code (default: es)")
    p.add_argument("--model", default="large-v3-turbo",
                   help="Whisper model (default: large-v3-turbo; try medium if low VRAM)")
    p.add_argument("--compute-type", default="int8",
                   help="Quantization for ASR (default: int8; float16 if you have VRAM)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="ASR batch size (default: 8; auto-retries with half on OOM)")
    p.add_argument("--formats", default="markdown,srt",
                   help="Comma list of outputs: markdown,srt,vtt,txt,json (default: markdown,srt)")
    p.add_argument("--out", default=None,
                   help="Base output directory (default: next to each input video; "
                        "a subfolder named after the video is always created)")
    p.add_argument("--title", default=None, help="Optional title used in the Markdown header")
    p.add_argument("--no-diarize", action="store_true",
                   help="Skip speaker diarization entirely")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore saved checkpoints; transcribe from scratch")
    p.add_argument("--auto-speaker", action="store_true",
                   help="Assign --speaker-name to the speaker with the most speaking time")
    p.add_argument("--speaker-name", default="Profesor",
                   help="Label for the main speaker (default: Profesor)")
    p.add_argument("--speaker-clips", default=None, metavar="DIR",
                   help="Save one short WAV per speaker (their longest turn) into DIR")
    p.add_argument("--hf-token", default=None,
                   help="Hugging Face token for diarization (or set HF_TOKEN)")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None,
                   help="Force device (default: auto -> cuda if available)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    quiet_third_party_noise()

    # Windows consoles often default to cp1252/cp437; ensure we can print accents.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if shutil.which("ffmpeg") is None:
        print(
            "[error] ffmpeg not found on PATH. Install it (winget install Gyan.FFmpeg), "
            "then open a new terminal.",
            file=sys.stderr,
        )
        return 2

    if args.benchmark:
        result = run_benchmark(args)
        if result is None:
            return 2
        return 0

    files = collect_inputs(args.inputs)
    if not files:
        print("[error] No media files found in the given inputs.", file=sys.stderr)
        return 2

    log_path = new_log_path()
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)

        ensure_punkt()

        device = pick_device(args)
        low, high, calibrated = estimate_runtime(
            files, args, device, diarize=not args.no_diarize)
        gpu = gpu_status()
        tag = " (calibrated to this machine)" if calibrated else " (estimate, based on GPU type)"
        print(f"[info] Estimated runtime: ~{low:.0f}-{high:.0f} min for {len(files)} file(s)"
              f" (diarization {'on' if not args.no_diarize else 'off'}){tag}")
        if gpu_is_busy(gpu):
            seen = sorted({a.strip() for a in gpu["apps"]})
            print(f"[warn] GPU is busy: {gpu['util']:.0f}% util, "
                  f"{gpu['mem_used']:.0f}MB/{gpu['mem_total']:.0f}MB VRAM used "
                  f"({len(seen)} GPU process(es): {', '.join(seen[:6])}). "
                  "Close GPU-heavy apps (browsers, games) for much better speed.")

        t_run = time.monotonic()
        print(f"=== Run started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        transcribe_batch(files, args)
        print(f"=== Run completed: {len(files)} file(s) in "
              f"{(time.monotonic() - t_run) / 60:.1f} min ===")
        print(f"=== Log saved to: {log_path} ===")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
