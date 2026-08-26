#!/usr/bin/env python3
"""gui.py — drag-and-drop GUI for video-transcriber.

Launch:  uv run gui.py   (or double-click Transcribir.cmd)
"""

import argparse
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from tkinterdnd2 import DND_FILES, TkinterDnD

import transcribe

MEDIA_EXTS = transcribe.MEDIA_EXTS
LANGUAGES = [("Español", "es"), ("English", "en"), ("Auto-detección", "auto")]
MODELS = ["large-v3-turbo", "medium", "small", "large-v3"]
FORMATS = [
    ("Markdown", "markdown", True),
    ("Subtítulos .srt", "srt", True),
    ("Subtítulos .vtt", "vtt", False),
    ("Texto .txt", "txt", False),
    ("JSON", "json", False),
]

SETTINGS_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "video-transcriber"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"

PHASE_STATUS = {
    "load": "Cargando modelos...",
    "align": "Alineando palabras...",
    "diarize": "Identificando hablantes...",
    "save": "Escribiendo archivos...",
}


def build_theme():
    """Custom "darkpurple" theme: clean dark base (from darkly) with purple
    accents for the primary/secondary/info roles."""
    from ttkbootstrap.constants import DARK
    from ttkbootstrap.style.theme import ThemeDefinition
    from ttkbootstrap.themes.standard import STANDARD_THEMES

    colors = dict(STANDARD_THEMES["darkly"]["colors"])
    colors["primary"] = "#8b66cd"
    colors["secondary"] = "#6f42c1"
    colors["info"] = "#a98fe0"
    return ThemeDefinition(name="darkpurple", colors=colors, mode=DARK)


def enable_dark_titlebar(widget):
    """Paint the OS title bar (minimize/maximize/close area) dark instead of
    white, using Windows' DWM immersive dark mode."""
    try:
        import ctypes

        hwnd = widget.winfo_id()
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (Win11 / Win10)
            value = ctypes.c_int(1)
            ok = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)
            )
            if ok == 0:
                break
    except Exception:
        pass


class StreamRedirect:
    def __init__(self, sink, log=None):
        self.sink = sink
        self.log = log

    def write(self, text):
        if text:
            self.sink.put(text)
            if self.log is not None:
                try:
                    self.log.write(text)
                    self.log.flush()
                except Exception:
                    pass

    def flush(self):
        pass


class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transcripción de videos")
        self.geometry("820x700")
        self.minsize(720, 600)
        self.style = ttk.Style()
        self.style.register_theme(build_theme())
        self.style.theme_use("darkpurple")
        self.colors = self.style.colors
        enable_dark_titlebar(self)
        self.queue = queue.Queue()
        self.files: list[str] = []
        self.running = False
        self.stop_requested = False
        self.out_folder: str | None = None
        self.last_outputs: list[Path] = []
        self.last_output_dir: Path | None = None
        self.reviews: list[dict] = []
        self.last_log_path: Path | None = None
        self.run_start: float = 0.0
        self.clip_dir = tempfile.mkdtemp(prefix="transcribe-clips-")
        self.settings: dict = self._load_settings()

        self._build()
        self._apply_saved_settings()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self.on_drop)

    # ------------------------------------------------------------------ UI
    def _build(self):
        pad = {"padx": 10, "pady": 6}

        frm_files = ttk.LabelFrame(self, text=" 1. Archivos — arrastra y suelta aquí ")
        frm_files.pack(fill="x", **pad)
        row = ttk.Frame(frm_files)
        row.pack(fill="x", padx=6, pady=6)
        self.listbox = tk.Listbox(row, height=6, selectmode="extended",
                                  background=self.colors.bg, foreground=self.colors.fg,
                                  selectbackground=self.colors.primary,
                                  selectforeground=self.colors.fg,
                                  relief="flat", highlightthickness=0)
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.drop_target_register(DND_FILES)
        self.listbox.dnd_bind("<<Drop>>", self.on_drop)
        sb = ttk.Scrollbar(row, orient="vertical", command=self.listbox.yview,
                           bootstyle="round")
        sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sb.set)
        btns = ttk.Frame(frm_files)
        btns.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(btns, text="Añadir archivos…", command=self.add_files,
                   bootstyle="secondary-outline").pack(side="left")
        ttk.Button(btns, text="Quitar seleccionados", command=self.remove_selected,
                   bootstyle="secondary-outline").pack(side="left", padx=6)
        ttk.Button(btns, text="Limpiar lista", command=self.clear_files,
                   bootstyle="secondary-outline").pack(side="left")
        self.file_count = ttk.Label(btns, text="0 archivos")
        self.file_count.pack(side="right")

        frm_opts = ttk.LabelFrame(self, text=" 2. Opciones ")
        frm_opts.pack(fill="x", **pad)
        grid = ttk.Frame(frm_opts)
        grid.pack(fill="x", padx=6, pady=6)
        grid.columnconfigure(1, weight=1)

        ttk.Label(grid, text="Formatos:").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=3)
        self.var_fmts: dict[str, tk.BooleanVar] = {}
        for col, (label, key, default) in enumerate(FORMATS, start=1):
            var = tk.BooleanVar(value=default)
            self.var_fmts[key] = var
            ttk.Checkbutton(grid, text=label, variable=var).grid(
                row=0, column=col, sticky="w", padx=(0, 10), pady=3
            )

        ttk.Label(grid, text="Idioma:").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=3)
        self.var_lang = tk.StringVar(value="es")
        ttk.Combobox(
            grid, textvariable=self.var_lang,
            values=[label for label, _ in LANGUAGES], state="readonly", width=16,
        ).grid(row=1, column=1, sticky="w", pady=3)

        ttk.Label(grid, text="Modelo:").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=3)
        self.var_model = tk.StringVar(value=MODELS[0])
        ttk.Combobox(
            grid, textvariable=self.var_model, values=MODELS, state="readonly", width=16,
        ).grid(row=2, column=1, sticky="w", pady=3)
        ttk.Label(grid, text="(large-v3-turbo = mejor precisión/velocidad)",
                  foreground="gray").grid(row=2, column=2, sticky="w", padx=(10, 0))

        self.var_diar = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            grid, text="Identificar hablantes",
            variable=self.var_diar,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=3)

        self.var_auto_speaker = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            grid, text="Marcar al hablante principal automáticamente (el que más habla)",
            variable=self.var_auto_speaker,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=3)

        ttk.Label(grid, text="Nombre del hablante principal:").grid(
            row=5, column=0, sticky="w", padx=(0, 10), pady=3
        )
        self.var_speaker_name = tk.StringVar(value="Profesor")
        ttk.Entry(grid, textvariable=self.var_speaker_name).grid(
            row=5, column=1, columnspan=2, sticky="ew", pady=3
        )

        ttk.Label(grid, text="Guardar en:").grid(row=6, column=0, sticky="w", padx=(0, 10), pady=3)
        self.var_out = tk.StringVar(value="video")
        ttk.Radiobutton(
            grid, text="Junto al video (carpeta por video)",
            variable=self.var_out, value="video",
        ).grid(row=6, column=1, columnspan=2, sticky="w", pady=3)
        row7 = ttk.Frame(grid)
        row7.grid(row=7, column=1, columnspan=2, sticky="w", pady=3)
        ttk.Radiobutton(
            row7, text="Carpeta elegida:", variable=self.var_out, value="folder",
            command=self._on_out_mode,
        ).pack(side="left")
        ttk.Button(row7, text="Elegir…", command=self.choose_folder,
                   bootstyle="secondary-outline").pack(side="left", padx=6)
        self.lbl_out_path = ttk.Label(row7, text="(ninguna)", foreground="gray")
        self.lbl_out_path.pack(side="left")

        frm_run = ttk.LabelFrame(self, text=" 3. Transcribir ")
        frm_run.pack(fill="both", expand=True, **pad)
        self.btn_start = ttk.Button(frm_run, text="▶  Transcribir", command=self.start,
                                    bootstyle="success")
        self.btn_start.pack(pady=8)
        self.progress = ttk.Progressbar(frm_run, maximum=100, value=0,
                                        bootstyle="primary-striped")
        self.progress.pack(fill="x", padx=6, pady=(0, 4))
        btn_row = ttk.Frame(frm_run)
        btn_row.pack(fill="x", padx=6, pady=(0, 4))
        self.btn_stop = ttk.Button(btn_row, text="Detener", command=self.request_stop,
                                   bootstyle="danger", state="disabled")
        self.btn_stop.pack(side="left")
        self.btn_review = ttk.Button(btn_row, text="Revisar hablantes",
                                     command=self._review_speakers,
                                     bootstyle="info-outline", state="disabled")
        self.btn_review.pack(side="left", padx=6)
        self.btn_open = ttk.Button(btn_row, text="Abrir carpeta", command=self.open_output,
                                   bootstyle="secondary-outline", state="disabled")
        self.btn_open.pack(side="left")
        self.btn_log = ttk.Button(btn_row, text="Abrir registro", command=self.open_log,
                                  bootstyle="secondary-outline", state="disabled")
        self.btn_log.pack(side="left", padx=6)
        self.btn_bench = ttk.Button(btn_row, text="Probar velocidad",
                                    command=self.run_benchmark,
                                    bootstyle="secondary-outline")
        self.btn_bench.pack(side="left")
        self.status = ttk.Label(
            frm_run,
            text="Listo. Arrastra tus grabaciones, elige opciones y pulsa Transcribir.\n"
                 "La primera vez descarga los modelos (unos 2 GB).",
            foreground="gray",
            justify="center",
        )
        self.status.pack(fill="x", padx=6)
        c = self.colors
        self.log = tk.Text(frm_run, height=12, state="disabled", wrap="word",
                           background=c.dark, foreground=c.fg,
                           insertbackground=c.fg, relief="flat",
                           borderwidth=0, highlightthickness=0,
                           padx=8, pady=6)
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

    def _on_out_mode(self):
        if self.var_out.get() == "folder" and not self.out_folder:
            self.choose_folder()

    # ------------------------------------------------------------- helpers
    def _log(self, text: str):
        self.log.config(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.config(state="disabled")

    def _refresh(self):
        self.listbox.delete(0, "end")
        for f in self.files:
            self.listbox.insert("end", f)
        self.file_count.config(text=f"{len(self.files)} archivos")

    def _add_paths(self, paths: list[str]):
        added = 0
        for raw in paths:
            p = Path(raw)
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.suffix.lower() in MEDIA_EXTS and str(f) not in self.files:
                        self.files.append(str(f))
                        added += 1
            elif p.suffix.lower() in MEDIA_EXTS and str(p) not in self.files:
                self.files.append(str(p))
                added += 1
        self._refresh()
        return added

    # -------------------------------------------------------------- events
    def on_drop(self, event):
        paths = self.tk.splitlist(event.data)
        added = self._add_paths(paths)
        self._log(f"→ {added} archivo(s) añadido(s)\n")

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Elige videos o audios",
            filetypes=[("Video/Audio", "*.mp4 *.mov *.mkv *.webm *.m4v *.avi *.wmv "
                                     "*.m4a *.mp3 *.wav *.flac *.ogg *.aac *.opus"),
                       ("Todos los archivos", "*.*")],
        )
        if paths:
            self._add_paths(list(paths))

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        for idx in reversed(sel):
            del self.files[idx]
        self._refresh()

    def clear_files(self):
        self.files = []
        self._refresh()

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Carpeta de destino")
        if folder:
            self.out_folder = folder
            self.var_out.set("folder")
            self.lbl_out_path.config(text=folder)

    def on_close(self):
        if self.running and not messagebox.askyesno(
            "Salir", "La transcripción está en curso. ¿Salir y cancelarla?"
        ):
            return
        self._save_settings()
        shutil.rmtree(self.clip_dir, ignore_errors=True)
        self.destroy()

    # ----------------------------------------------------------- settings
    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self) -> None:
        try:
            data = {
                "formats": [k for k, v in self.var_fmts.items() if v.get()],
                "lang": self.var_lang.get(),
                "model": self.var_model.get(),
                "diarize": self.var_diar.get(),
                "auto_speaker": self.var_auto_speaker.get(),
                "speaker_name": self.var_speaker_name.get(),
                "out_mode": self.var_out.get(),
                "out_folder": self.out_folder or "",
            }
            SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        except Exception:
            pass

    def _apply_saved_settings(self) -> None:
        s = self.settings
        if not s:
            return
        for key in FORMATS:
            label_key = key[1]
            if label_key in s.get("formats", []):
                self.var_fmts[label_key].set(True)
        lang_labels = {code: label for label, code in LANGUAGES}
        if s.get("lang") in lang_labels:
            self.var_lang.set(lang_labels[s["lang"]])
        if s.get("model") in MODELS:
            self.var_model.set(s["model"])
        self.var_diar.set(bool(s.get("diarize", True)))
        self.var_auto_speaker.set(bool(s.get("auto_speaker", True)))
        if s.get("speaker_name"):
            self.var_speaker_name.set(s["speaker_name"])
        if s.get("out_mode") == "folder":
            self.var_out.set("folder")
        if s.get("out_folder"):
            self.out_folder = s["out_folder"]
            self.lbl_out_path.config(text=self.out_folder)

    # ------------------------------------------------------------ pipeline
    def _build_args(self) -> argparse.Namespace:
        formats = [key for key, var in self.var_fmts.items() if var.get()]
        if not formats:
            formats = ["markdown"]
        lang_code = dict(LANGUAGES).get(self.var_lang.get(), "es")
        return argparse.Namespace(
            lang=lang_code,
            model=self.var_model.get(),
            compute_type="int8",
            batch_size=8,
            formats=",".join(formats),
            out=self.out_folder if self.var_out.get() == "folder" else None,
            title=None,
            no_diarize=not self.var_diar.get(),
            hf_token=None,
            device=None,
            auto_speaker=self.var_auto_speaker.get(),
            speaker_name=self.var_speaker_name.get().strip() or "Profesor",
            speaker_clips_dir=self.clip_dir,
            progress_callback=self._on_progress,
            phase_callback=self._on_phase,
            diarize_progress_callback=self._on_diarize_progress,
        )

    def _on_progress(self, pct: float):
        self.queue.put(("progress", pct))

    def _on_diarize_progress(self, pct: float):
        self.queue.put(("diar_progress", pct))

    def _on_phase(self, name: str):
        self.queue.put(("phase", name))

    def request_stop(self):
        if self.running:
            self.stop_requested = True
            self.btn_stop.config(state="disabled")
            self.status.config(text="Deteniendo tras el archivo en curso...")

    def open_output(self):
        folder = self.last_output_dir or (
            Path(self.out_folder) if self.var_out.get() == "folder" and self.out_folder else None
        )
        if folder and Path(folder).exists() and hasattr(os, "startfile"):
            os.startfile(str(folder))

    def open_log(self):
        if self.last_log_path and Path(self.last_log_path).exists() and hasattr(os, "startfile"):
            os.startfile(str(self.last_log_path))

    def run_benchmark(self):
        if self.running:
            messagebox.showinfo("En proceso",
                                "Espera a que termine la transcripción actual.")
            return
        if shutil.which("ffmpeg") is None:
            messagebox.showerror(
                "ffmpeg no encontrado",
                "Instala ffmpeg (winget install Gyan.FFmpeg) y abre una terminal nueva.")
            return
        messagebox.showinfo(
            "Prueba de velocidad",
            "Mido tu velocidad real con la muestra incluida (~1 min, carga el modelo).\n"
            "Cierra apps con GPU para un resultado realista.")
        self.btn_bench.config(state="disabled")
        threading.Thread(target=self._bench_worker, daemon=True).start()

    def _bench_worker(self):
        q = self.queue
        redirect = StreamRedirect(q)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = redirect
        try:
            q.put(("status", "Midiendo velocidad..."))
            result = transcribe.run_benchmark(self._build_args())
            q.put(("bench_result", result or {}))
        except Exception:
            q.put("\n" + traceback.format_exc() + "\n")
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def start(self):
        if self.running:
            return
        files = list(self.files)
        if not files:
            messagebox.showinfo("Sin archivos", "Arrastra al menos un video o audio primero.")
            return
        if shutil.which("ffmpeg") is None:
            messagebox.showerror(
                "ffmpeg no encontrado",
                "Instala ffmpeg (winget install Gyan.FFmpeg) y abre una terminal nueva.",
            )
            return
        args = self._build_args()
        device = transcribe.pick_device(args)
        low, high, calibrated = transcribe.estimate_runtime(
            files, args, device, diarize=not args.no_diarize)
        mark = "calibrado a esta máquina" if calibrated else "estimación por tipo de GPU"
        gpu = transcribe.gpu_status()
        lines = [f"Tiempo estimado: ~{low:.0f}-{high:.0f} min  ({mark})"]
        if transcribe.gpu_is_busy(gpu):
            lines.append(
                f"La GPU está ocupada: {gpu['util']:.0f}% de uso, "
                f"{gpu['mem_used']:.0f}MB/{gpu['mem_total']:.0f}MB de VRAM "
                f"({len(gpu['apps'])} proceso(s) de GPU)."
            )
            lines.append("Cierra navegadores, juegos u otras apps con aceleración gráfica "
                         "para que vaya mucho más rápido.")
        else:
            lines.append("GPU libre, todo listo.")
        lines.append("")
        lines.append("¿Continuar?")
        if not messagebox.askyesno("Transcripción", "\n".join(lines)):
            return
        self.running = True
        self.stop_requested = False
        self.last_outputs = []
        self.last_output_dir = None
        self.reviews = []
        self.last_log_path = None
        self.run_start = time.monotonic()
        self.progress.config(mode="determinate", value=0)
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_review.config(state="disabled")
        self.btn_open.config(state="disabled")
        self.btn_log.config(state="disabled")
        self.status.config(text=f"Transcribiendo {len(files)} archivo(s)… no cierres la ventana")
        self._log(f"\n========== Iniciando ({len(files)} archivo(s)) ==========\n")
        self.worker = threading.Thread(target=self._worker, args=(files, args), daemon=True)
        self.worker.start()
        self.after(100, self._poll)

    def _worker(self, files: list[str], args: argparse.Namespace):
        q = self.queue
        log_path = transcribe.new_log_path()
        log_file = open(log_path, "w", encoding="utf-8")
        q.put(("logfile", str(log_path)))
        redirect = StreamRedirect(q, log_file)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = redirect
        cache = transcribe.ModelCache()
        try:
            transcribe.ensure_punkt()
            # Stage 1: transcribe + align every file (ASR loads once for the batch)
            prepared: list[dict] = []
            fatal = False
            for i, f in enumerate(files, start=1):
                if self.stop_requested:
                    q.put("[warn] Detenido por el usuario; archivos restantes omitidos.\n")
                    break
                q.put(("status", f"Transcribiendo {i}/{len(files)}: {Path(f).name}..."))
                q.put(f"\n>>> {Path(f).name}\n")
                try:
                    prepared.append(transcribe.get_prep(Path(f), args, cache))
                except Exception as exc:
                    q.put("\n" + traceback.format_exc() + "\n")
                    if transcribe._is_cuda_fatal(exc):
                        fatal = True
                        q.put("\n[error] La GPU se perdió (memoria insuficiente u otra"
                              " app usando la GPU). Se omitieron los archivos"
                              " restantes — cierra otras apps que usen GPU y"
                              " vuelve a ejecutar.\n")
                        break
            # Stage 2: diarize + write every file (ASR evicted, so it stays fast).
            # A user stop only halts *starting new work*; files already
            # transcribed are still diarized and written. Only a fatal GPU
            # error skips them entirely.
            for i, prep in enumerate(prepared, start=1):
                if fatal:
                    q.put("\n[error] La GPU se perdió; se omitió el resto del lote.\n")
                    break
                q.put(("status", f"Identificando hablantes {i}/{len(prepared)}: "
                                 f"{Path(prep['path']).name}..."))
                try:
                    payload = transcribe._diarize_write(prep, args, cache)
                    if payload:
                        q.put(("review", {
                            "key": Path(prep["path"]).name,
                            "path": str(prep["path"]),
                            "outputs": payload.get("outputs", []),
                            "clips": payload["clips"],
                            "totals": payload["totals"],
                            "names": payload["names"],
                        }))
                    q.put(f"\n>>> Completado: {Path(prep['path']).name}\n")
                except Exception:
                    q.put("\n" + traceback.format_exc() + "\n")
            # Record measured rates for better estimates on this machine.
            try:
                asr = [p["measure"]["asr_rate"] for p in prepared
                       if p.get("measure") and p["measure"].get("asr_rate") is not None]
                align = [p["measure"]["align_rate"] for p in prepared
                         if p.get("measure") and p["measure"].get("align_rate") is not None]
                diar = [p["diarize_rate"] for p in prepared if p.get("diarize_rate")]
                if asr or align or diar:
                    transcribe.record_rates(
                        args, transcribe.pick_device(args),
                        asr_rate=sum(asr) / len(asr) if asr else None,
                        align_rate=sum(align) / len(align) if align else None,
                        diarize_rate=sum(diar) / len(diar) if diar else None,
                    )
            except Exception:
                pass
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            try:
                log_file.close()
            except Exception:
                pass
        q.put("__DONE__")

    def _poll(self):
        try:
            while True:
                line = self.queue.get_nowait()
                if isinstance(line, tuple):
                    kind, *rest = line
                    if kind == "review":
                        self.reviews.append(rest[0])
                    elif kind == "logfile":
                        self.last_log_path = Path(rest[0])
                    elif kind == "progress":
                        self.progress.config(value=rest[0])
                    elif kind == "diar_progress":
                        self.progress.config(mode="determinate", value=rest[0])
                        self.status.config(text=f"Identificando hablantes… {rest[0]:.0f}%")
                    elif kind == "bench_result":
                        r = rest[0]
                        self.btn_bench.config(state="normal")
                        self.status.config(text="Listo.")
                        lines = ["Velocidades medidas (segundos por minuto de audio):"]
                        lines.append(f"  ASR:        {r.get('asr_rate', 0) * 60:.1f} s/min")
                        lines.append(f"  Alineación: {r.get('align_rate', 0) * 60:.1f} s/min")
                        if r.get("diarize_rate"):
                            lines.append(f"  Diarización:{r['diarize_rate'] * 60:.1f} s/min")
                        else:
                            lines.append("  Diarización: (sin token o deshabilitada)")
                        lines.append("")
                        lines.append("Guardado. Las próximas estimaciones usarán estos valores.")
                        messagebox.showinfo("Prueba de velocidad", "\n".join(lines))
                    elif kind == "status":
                        self.status.config(text=rest[0])
                    elif kind == "phase":
                        name = rest[0]
                        if name == "asr":
                            self.progress.stop()
                            self.progress.config(mode="determinate", value=0)
                        else:
                            self.progress.config(mode="indeterminate")
                            self.progress.start(14)
                        text = PHASE_STATUS.get(name)
                        if text:
                            self.status.config(text=text)
                    continue
                if line == "__DONE__":
                    self._finish()
                    return
                if isinstance(line, str) and line.startswith("[ok] "):
                    out = Path(line.split("] ", 1)[1].strip())
                    self.last_outputs.append(out)
                    self.last_output_dir = out.parent
                self._log(line)
        except queue.Empty:
            pass
        if self.running:
            self.after(100, self._poll)

    def _finish(self):
        self.running = False
        self.progress.stop()
        self.progress.config(mode="determinate", value=100)
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        if self.reviews:
            self.btn_review.config(state="normal")
        if self.last_output_dir:
            self.btn_open.config(state="normal")
        if self.last_log_path:
            self.btn_log.config(state="normal")
        stopped = " (detenido por el usuario)" if self.stop_requested else ""
        self.status.config(text="Listo. Puedes añadir más archivos y transcribir de nuevo.")
        self._log(f"\n========== Terminado{stopped} ==========\n")
        if self.last_log_path:
            self._log(f"Registro de esta ejecución: {self.last_log_path}\n")
        self._save_settings()
        if not self.var_auto_speaker.get():
            self._review_speakers()

    def _review_speakers(self):
        """Open the review flow. With several files, ask which one first, so
        every video of a batch can be reviewed individually."""
        if not self.reviews:
            messagebox.showinfo("Sin transcripciones",
                                "Primero transcribe algún video para revisar sus hablantes.")
            return
        if len(self.reviews) == 1:
            self._review_file(self.reviews[0])
            return
        dlg = tk.Toplevel(self)
        dlg.title("Revisar hablantes")
        dlg.resizable(False, False)
        dlg.configure(background=self.colors.bg)
        enable_dark_titlebar(dlg)
        ttk.Label(dlg, text="¿Qué archivo quieres revisar?", justify="center"
                  ).pack(padx=24, pady=(16, 8))
        keys = [r["key"] for r in self.reviews]
        var = tk.StringVar(value=keys[0])
        ttk.Combobox(dlg, textvariable=var, values=keys, state="readonly",
                     width=40).pack(padx=24, pady=4)

        def ok():
            chosen = next((r for r in self.reviews if r["key"] == var.get()), self.reviews[0])
            dlg.destroy()
            self._review_file(chosen)

        ttk.Button(dlg, text="Revisar", command=ok).pack(pady=(10, 16))
        dlg.update_idletasks()
        dlg.geometry(f"+{self.winfo_rootx() + 120}+{self.winfo_rooty() + 120}")
        dlg.grab_set()
        dlg.wait_window()

    def _review_file(self, entry: dict):
        """Name the voices of one file; two voices given the same name merge.
        Each voice offers several audio samples to make identification easy."""
        speakers = sorted(entry["totals"].items(), key=lambda kv: -kv[1])
        if not speakers:
            messagebox.showinfo("Sin hablantes",
                                "Esta transcripción no tiene hablantes identificados.")
            return
        dlg = tk.Toplevel(self)
        dlg.title(f"Revisar hablantes — {entry['key']}")
        dlg.resizable(False, False)
        dlg.configure(background=self.colors.bg)
        enable_dark_titlebar(dlg)
        ttk.Label(
            dlg,
            text="Escucha cada voz y ponle nombre. Deja un nombre para mantenerlo;\n"
                 "dos voces con el mismo nombre se fusionan en una sola persona.",
            justify="center",
        ).pack(padx=20, pady=(16, 8))
        entries: dict[str, ttk.Entry] = {}
        for sp, secs in speakers:
            row = ttk.Frame(dlg)
            row.pack(fill="x", padx=20, pady=2)
            current = entry["names"].get(sp, sp)
            mins = secs / 60
            ttk.Label(row, text=f"{mins:.1f} min").pack(side="left")
            for clip in entry["clips"].get(sp, []):
                ttk.Button(
                    row, text="▶", width=3,
                    command=lambda p=clip: self._play_clip(p),
                ).pack(side="left", padx=(4, 0))
            ttk.Label(row, text="·").pack(side="left", padx=4)
            entry_widget = ttk.Entry(row, width=20)
            entry_widget.insert(0, current)
            entry_widget.pack(side="left", padx=(4, 0))
            entries[sp] = entry_widget

        def ok():
            renames: dict[str, str] = {}
            for sp, entry_widget in entries.items():
                new = entry_widget.get().strip()
                cur = entry["names"].get(sp, sp)
                if new and new != cur:
                    renames[cur] = new
                    entry["names"][sp] = new
            self._apply_renames(renames, entry["outputs"])
            dlg.destroy()

        ttk.Button(dlg, text="Guardar", command=ok).pack(pady=(12, 16))
        dlg.update_idletasks()
        dlg.geometry(f"+{self.winfo_rootx() + 120}+{self.winfo_rooty() + 120}")
        dlg.grab_set()
        dlg.wait_window()

    def _play_clip(self, path: str):
        try:
            import winsound

            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:
            self._log(f"[warn] No se pudo reproducir la muestra: {exc}\n")

    def _apply_renames(self, renames: dict[str, str], outputs: list[str]):
        if not renames:
            return
        renamed = 0
        for p_str in outputs:
            p = Path(p_str)
            if not p.exists():
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            new = text
            for old_label, new_label in renames.items():
                new = new.replace(old_label, new_label)
            if new != text:
                p.write_text(new, encoding="utf-8")
                renamed += 1
        pairs = ", ".join(f"'{o}' → '{n}'" for o, n in renames.items())
        self._log(f"→ Renombrado(s): {pairs} ({renamed} archivo(s)).\n")


def main():
    # Crisp text on high-DPI (scaled) Windows displays.
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
