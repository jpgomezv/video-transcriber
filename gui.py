#!/usr/bin/env python3
"""gui.py — drag-and-drop GUI for video-transcriber.

Launch:  uv run gui.py   (or double-click Transcribir.cmd)
"""

import argparse
import queue
import re
import shutil
import sys
import tempfile
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD

import transcribe

MEDIA_EXTS = transcribe.MEDIA_EXTS
LANGUAGES = [("Español", "es"), ("English", "en"), ("Auto-detección", "auto")]
MODELS = ["large-v3-turbo", "medium", "small", "large-v3"]
FORMATS = [
    ("Markdown (transcripción)", "markdown", True),
    ("Subtítulos .srt", "srt", True),
    ("Subtítulos .vtt", "vtt", False),
    ("JSON (datos crudos)", "json", False),
]


class StreamRedirect:
    def __init__(self, sink):
        self.sink = sink

    def write(self, text):
        if text:
            self.sink.put(text)

    def flush(self):
        pass


class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transcripción de videos")
        self.geometry("760x700")
        self.minsize(660, 600)
        self.queue = queue.Queue()
        self.files: list[str] = []
        self.running = False
        self.out_folder: str | None = None
        self.last_outputs: list[Path] = []
        self.speaker_clips: dict[str, str] = {}
        self.clip_dir = tempfile.mkdtemp(prefix="transcribe-clips-")

        self._build()
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
        self.listbox = tk.Listbox(row, height=6, selectmode="extended")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.drop_target_register(DND_FILES)
        self.listbox.dnd_bind("<<Drop>>", self.on_drop)
        sb = ttk.Scrollbar(row, orient="vertical", command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sb.set)
        btns = ttk.Frame(frm_files)
        btns.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(btns, text="Añadir archivos…", command=self.add_files).pack(side="left")
        ttk.Button(btns, text="Quitar seleccionados", command=self.remove_selected).pack(side="left", padx=6)
        ttk.Button(btns, text="Limpiar lista", command=self.clear_files).pack(side="left")
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
        ttk.Button(row7, text="Elegir…", command=self.choose_folder).pack(side="left", padx=6)
        self.lbl_out_path = ttk.Label(row7, text="(ninguna)", foreground="gray")
        self.lbl_out_path.pack(side="left")

        frm_run = ttk.LabelFrame(self, text=" 3. Transcribir ")
        frm_run.pack(fill="both", expand=True, **pad)
        self.btn_start = ttk.Button(frm_run, text="▶  Transcribir", command=self.start)
        self.btn_start.pack(pady=8)
        self.progress = ttk.Progressbar(frm_run, maximum=100, value=0)
        self.progress.pack(fill="x", padx=6, pady=(0, 4))
        self.status = ttk.Label(
            frm_run,
            text="Listo. Arrastra tus grabaciones, elige opciones y pulsa Transcribir.\n"
                 "La primera vez descarga los modelos (unos 2 GB).",
            foreground="gray",
            justify="center",
        )
        self.status.pack(fill="x", padx=6)
        self.log = tk.Text(frm_run, height=12, state="disabled", wrap="word")
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
        shutil.rmtree(self.clip_dir, ignore_errors=True)
        self.destroy()

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
        )

    def _on_progress(self, pct: float):
        self.queue.put(("progress", pct))

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
        self.running = True
        self.last_outputs = []
        self.progress.config(value=0)
        self.btn_start.config(state="disabled")
        self.status.config(text=f"Transcribiendo {len(files)} archivo(s)… no cierres la ventana")
        self._log(f"\n========== Iniciando ({len(files)} archivo(s)) ==========\n")
        self.worker = threading.Thread(target=self._worker, args=(files, args), daemon=True)
        self.worker.start()
        self.after(100, self._poll)

    def _worker(self, files: list[str], args: argparse.Namespace):
        q = self.queue
        redirect = StreamRedirect(q)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = redirect
        try:
            transcribe.ensure_punkt()
            for i, f in enumerate(files, start=1):
                q.put(("status", f"Transcribiendo {i}/{len(files)}: {Path(f).name}..."))
                q.put(f"\n>>> {Path(f).name}\n")
                try:
                    clips = transcribe.transcribe_one(Path(f), args)
                    if clips:
                        q.put(("clips", clips))
                    q.put(f"\n>>> Completado: {Path(f).name}\n")
                except Exception:
                    q.put("\n" + traceback.format_exc() + "\n")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        q.put("__DONE__")

    def _poll(self):
        try:
            while True:
                line = self.queue.get_nowait()
                if isinstance(line, tuple):
                    kind, *rest = line
                    if kind == "clips":
                        self.speaker_clips.update(rest[0])
                    elif kind == "progress":
                        self.progress.config(value=rest[0])
                    elif kind == "status":
                        self.status.config(text=rest[0])
                    continue
                if line == "__DONE__":
                    self._finish()
                    return
                if isinstance(line, str) and line.startswith("[ok] "):
                    self.last_outputs.append(Path(line.split("] ", 1)[1].strip()))
                self._log(line)
        except queue.Empty:
            pass
        if self.running:
            self.after(100, self._poll)

    def _finish(self):
        self.running = False
        self.btn_start.config(state="normal")
        self.progress.config(value=100)
        self.status.config(text="Listo. Puedes añadir más archivos y transcribir de nuevo.")
        self._log("\n========== Terminado ==========\n")
        self._ask_teacher()

    def _ask_teacher(self):
        speakers = sorted({
            m
            for p in self.last_outputs
            if p.suffix == ".md" and p.exists()
            for m in re.findall(r"\*\*(SPEAKER_\d+):\*\*", p.read_text(encoding="utf-8"))
        })
        if not speakers:
            messagebox.showinfo("Terminado", "Transcripción completada.")
            return
        if self.var_auto_speaker.get():
            label = self.var_speaker_name.get().strip() or "Profesor"
            messagebox.showinfo(
                "Terminado",
                f"Transcripción completada.\nHablante principal ('{label}') "
                "detectado automáticamente: el que más tiempo habla.",
            )
            return
        dlg = tk.Toplevel(self)
        dlg.title("¿Quién es el hablante principal?")
        dlg.resizable(False, False)
        ttk.Label(
            dlg,
            text="Escucha cada voz y marca al hablante principal:",
            justify="center",
        ).pack(padx=20, pady=(16, 8))
        var = tk.StringVar(value="Ninguno")
        for sp in speakers:
            row = ttk.Frame(dlg)
            row.pack(fill="x", padx=20, pady=2)
            ttk.Radiobutton(row, text=sp, variable=var, value=sp).pack(side="left")
            clip = self.speaker_clips.get(sp)
            if clip:
                ttk.Button(
                    row, text="▶ Escuchar", width=10,
                    command=lambda p=clip: self._play_clip(p),
                ).pack(side="left", padx=(12, 0))
        row_n = ttk.Frame(dlg)
        row_n.pack(fill="x", padx=20, pady=2)
        ttk.Radiobutton(
            row_n, text="Ninguno (dejar como está)", variable=var, value="Ninguno",
        ).pack(side="left")

        def ok():
            chosen = var.get()
            if chosen != "Ninguno":
                self._apply_speaker(chosen)
            dlg.destroy()
            messagebox.showinfo("Terminado", "Transcripción completada.")

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

    def _apply_speaker(self, speaker_label: str):
        new_name = self.var_speaker_name.get().strip() or "Profesor"
        renamed = 0
        for p in self.last_outputs:
            if not p.exists():
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            new = text.replace(speaker_label, new_name)
            if new != text:
                p.write_text(new, encoding="utf-8")
                renamed += 1
        self._log(f"→ '{speaker_label}' renombrado a '{new_name}' en {renamed} archivo(s).\n")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
