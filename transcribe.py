#!/usr/bin/env python3
"""transcribe.py — free, local video/audio transcription with speaker diarization.

Powered by WhisperX (faster-whisper ASR + forced alignment + pyannote diarization).
Runs entirely on your machine; no uploads, no limits.

Default output: timestamped Markdown transcript + SRT subtitles. Optionally also
VTT subtitles (open the original video in VLC and load the .srt to verify the
transcript against the recording) and a JSON dump of all segments.

Speaker diarization (who said what) uses a free Hugging Face token — one-time setup:
  1. Create a (free) account + read token:  https://huggingface.co/settings/tokens
  2. Accept the model license:              https://huggingface.co/pyannote/speaker-diarization-community-1
Then set the HF_TOKEN environment variable, or pass --hf-token. Without a token the
transcript is produced but WITHOUT speaker labels (and a warning is printed).

Examples:
  uv run transcribe.py "C:\\Downloads\\video1.mp4"
  uv run transcribe.py "C:\\Downloads\\video1.mp4" --formats markdown,srt,vtt --lang es
  uv run transcribe.py "C:\\Downloads\\videos" --model large-v3-turbo --batch-size 8
  uv run transcribe.py video.mp4 --auto-speaker --speaker-name Profesor
  uv run transcribe.py video.mp4 --no-diarize      # skip speaker labels
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import datetime
import shutil
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
    """Merge short consecutive segments into readable lines.

    Fixes the "one word per subtitle" problem: whisperx can emit very short
    segments (especially after per-speaker splitting), which would otherwise
    produce illegible, oversized subtitle files. We merge until a line reaches
    a sensible size, breaking on speaker changes only once a line is long
    enough (so a noisy per-word speaker alternation can't fragment everything).
    """
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
        speaker_changed = sp != cur["speaker"]
        break_line = (
            (speaker_changed and len(cur["text"]) >= min_break_chars)
            or new_len > max_chars
            or new_span > max_seconds
        )
        if break_line:
            merged.append(cur)
            cur = {"start": seg["start"], "end": seg["end"], "speaker": sp, "text": text}
        else:
            if speaker_changed:
                cur["text"] += f" [{sp}] {text}"
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


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

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
    path.write_text("\n".join(lines), encoding="utf-8")


def write_srt(path: Path, segments: list, names: dict) -> None:
    merged = merge_segments(segments, max_chars=60, max_seconds=5.0, min_break_chars=20)
    blocks = []
    for i, seg in enumerate(merged, start=1):
        text = seg["text"].strip()
        if not text:
            continue
        sp = display_speaker(speaker_of(seg), names)
        body = f"[{sp}] {text}" if sp else text
        blocks.append(
            f"{i}\n{srt_ts(seg['start'])} --> {srt_ts(seg['end'])}\n{body}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def write_vtt(path: Path, segments: list, names: dict) -> None:
    merged = merge_segments(segments, max_chars=60, max_seconds=5.0, min_break_chars=20)
    blocks = ["WEBVTT", ""]
    for seg in merged:
        text = seg["text"].strip()
        if not text:
            continue
        sp = display_speaker(speaker_of(seg), names)
        if sp:
            body = f"<v {sp}>{text}</v>"
        else:
            body = text
        blocks.append(f"{vtt_ts(seg['start'])} --> {vtt_ts(seg['end'])}\n{body}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def write_json(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Speaker helpers (auto-teacher + voice clips)
# --------------------------------------------------------------------------- #

def dominant_speaker(segments: list) -> tuple[str | None, float]:
    """Return (speaker_label, total_seconds) of the speaker with the most
    speaking time — the heuristic for "the teacher is who talks the most"."""
    totals: dict[str, float] = {}
    for seg in segments:
        sp = speaker_of(seg)
        if sp:
            totals[sp] = totals.get(sp, 0.0) + (seg["end"] - seg["start"])
    if not totals:
        return None, 0.0
    best = max(totals, key=totals.get)
    return best, totals[best]


def extract_speaker_clips(audio, segments: list, out_dir, clip_seconds: float = 6.0) -> dict:
    """Save one short WAV per speaker (their longest turn, center-cropped) so a
    human can hear the voices. Returns {speaker_label: wav_path}."""
    import wave

    import numpy as np

    best: dict[str, tuple[float, float, float]] = {}
    for seg in segments:
        sp = speaker_of(seg)
        if not sp:
            continue
        dur = seg["end"] - seg["start"]
        if sp not in best or dur > best[sp][0]:
            best[sp] = (dur, seg["start"], seg["end"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: dict[str, str] = {}
    for sp, (dur, start, end) in best.items():
        if dur < 2.0:
            continue
        mid = (start + end) / 2.0
        cstart = max(start, mid - clip_seconds / 2)
        cend = min(end, cstart + clip_seconds)
        if cend - cstart < 1.5:
            cstart = max(start, cend - 1.5)
        s = int(cstart * 16000)
        e = int(cend * 16000)
        samples = np.clip(audio[s:e], -1.0, 1.0)
        pcm = (samples * 32767).astype(np.int16)
        path = out_dir / f"{sp}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.tobytes())
        clips[sp] = str(path)
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


def transcribe_one(path: Path, args) -> dict | None:
    quiet_third_party_noise()
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    t_start = time.monotonic()
    device = pick_device(args)
    lang = None if args.lang in (None, "", "auto") else args.lang
    print(f"\n=== {path.name} ===")
    print(f"[info] device={device}  model={args.model}  compute_type={args.compute_type}"
          f"  batch_size={args.batch_size}" + (f"  language={lang}" if lang else "  language=auto"))

    size_mb = path.stat().st_size / (1024 * 1024)
    duration = media_duration(path)
    duration_str = hms(duration) if duration else "?"
    print(f"[info] duration: {duration_str} · size: {size_mb:.0f} MB")

    # 1. Transcribe (batched faster-whisper)
    model_path = asr_model_path(args.model)
    model = whisperx.load_model(
        str(model_path),
        device=device,
        compute_type=args.compute_type,
        language=lang,
        asr_options={
            "initial_prompt": initial_prompt_for(lang),
        },
    )
    audio = whisperx.load_audio(str(path))
    print("[info] Transcribing...")
    external_cb = getattr(args, "progress_callback", None)
    printer = progress_printer()

    def on_progress(pct: float) -> None:
        printer(pct)
        if external_cb is not None:
            external_cb(pct)

    t0 = time.monotonic()
    result = model.transcribe(
        audio,
        batch_size=args.batch_size,
        language=lang,
        progress_callback=on_progress,
    )
    print(f"[info] Transcription done in {(time.monotonic() - t0) / 60:.1f} min")
    del model
    free_gpu()

    language = result.get("language", args.lang)

    # 2. Forced alignment (accurate word timestamps)
    t0 = time.monotonic()
    try:
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False,
        )
        del model_a
        free_gpu()
        print(f"[info] Alignment done in {(time.monotonic() - t0) / 60:.1f} min")
    except Exception as exc:
        print(f"[warn] Alignment failed ({exc}); keeping Whisper timestamps.")

    segments = result["segments"]

    # 3. Speaker diarization (optional)
    diarized = False
    if not args.no_diarize:
        token = resolve_token(args)
        if token:
            try:
                print("[info] Running speaker diarization...")
                t0 = time.monotonic()
                diarize_model_path = ensure_model(DIARIZE_MODEL_REPO, "speaker-diarization-community-1", marker="config.yaml")
                dia = DiarizationPipeline(model_name=str(diarize_model_path), token=token, device=device)
                diarize_segments = dia(audio)
                del dia
                free_gpu()
                result = whisperx.assign_word_speakers(diarize_segments, result)
                segments = result["segments"]
                diarized = True
                speakers = sorted({sp for sp in (s.get("speaker") for s in segments) if sp})
                print(f"[info] Diarization done in {(time.monotonic() - t0) / 60:.1f} min"
                      f" — {len(speakers)} speaker(s): {', '.join(speakers)}")
            except Exception as exc:
                print(f"[warn] Diarization failed ({exc}); transcript is without speaker labels.")
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
    clips = None
    if diarized:
        if getattr(args, "auto_speaker", False):
            dom, dom_secs = dominant_speaker(segments)
            label = args.speaker_name or "Speaker 1"
            if dom:
                names[dom] = label
                print(f"[info] Main speaker: {dom} ({dom_secs / 60:.1f} min hablados) -> '{label}'")
        clip_dir = getattr(args, "speaker_clips_dir", None)
        if clip_dir:
            clips = extract_speaker_clips(audio, segments, clip_dir)

    # 4. Write outputs — one folder per transcription, named after the source
    #    file, saved next to the original video (or inside --out if given)
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

    if "markdown" in wanted:
        md_path = out_dir / f"{stem}.md"
        write_markdown(md_path, title, str(path), meta, segments, names)
        print(f"[ok] {md_path}")
    if "srt" in wanted:
        srt_path = out_dir / f"{stem}.srt"
        write_srt(srt_path, segments, names)
        print(f"[ok] {srt_path}")
    if "vtt" in wanted:
        vtt_path = out_dir / f"{stem}.vtt"
        write_vtt(vtt_path, segments, names)
        print(f"[ok] {vtt_path}")
    if "json" in wanted:
        json_path = out_dir / f"{stem}.json"
        write_json(json_path, result)
        print(f"[ok] {json_path}")

    print(f"[info] Total: {(time.monotonic() - t_start) / 60:.1f} min")

    return clips


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
    p.add_argument("inputs", nargs="+", help="Video/audio file(s), or folder(s) of them")
    p.add_argument("--lang", default="es", help="Language code (default: es)")
    p.add_argument("--model", default="large-v3-turbo",
                   help="Whisper model (default: large-v3-turbo; try medium if low VRAM)")
    p.add_argument("--compute-type", default="int8",
                   help="Quantization for ASR (default: int8; float16 if you have VRAM)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="ASR batch size (default: 8; lower if GPU out of memory)")
    p.add_argument("--formats", default="markdown,srt",
                   help="Comma list of outputs: markdown,srt,vtt,json (default: markdown,srt)")
    p.add_argument("--out", default=None,
                   help="Base output directory (default: next to each input video; "
                        "a subfolder named after the video is always created)")
    p.add_argument("--title", default=None, help="Optional title used in the Markdown header")
    p.add_argument("--no-diarize", action="store_true",
                   help="Skip speaker diarization entirely")
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

    files = collect_inputs(args.inputs)
    if not files:
        print("[error] No media files found in the given inputs.", file=sys.stderr)
        return 2

    ensure_punkt()

    for f in files:
        try:
            transcribe_one(f, args)
        except Exception as exc:
            print(f"[error] Failed on {f}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
