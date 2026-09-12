#!/usr/bin/env python3
"""gui_qt.py — PROTOTYPE native Qt UI (PySide6) for video-transcriber.

Side-by-side experiment; gui.py (tkinter) is untouched. Drives the SAME
transcribe.py pipeline in-process (worker thread + polled queue, mirroring
the proven tkinter flow), so results are identical — only the shell differs.

Launch:  uv run gui_qt.py   (requires the PySide6 dependency)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import sys
import tempfile
import threading
import traceback
import winsound
from pathlib import Path

import transcribe

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QRadioButton,
    QSystemTrayIcon, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QAbstractItemView,
)

SETTINGS_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "video-transcriber"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"
APP_TITLE = "Transcripción de videos"
ICON_PATH = Path(__file__).resolve().parent / "assets" / "app-icon.ico"

LANGUAGES = [("Español", "es"), ("English", "en"), ("Auto-detección", "auto")]
MODELS = ["large-v3-turbo", "medium", "small", "large-v3"]
FORMATS = [("Markdown", "markdown", True), ("SRT", "srt", True),
           ("VTT", "vtt", False), ("TXT", "txt", False), ("JSON", "json", False)]

DARK_QSS = """
QMainWindow, QWidget#central { background: #0b0b10; }
QLabel { color: #e4e4e7; }
QLabel#muted { color: #a1a1aa; }
QLabel#title { font-size: 20px; font-weight: bold; color: #ffffff; }
QLabel#section { font-size: 13px; font-weight: bold; color: #ffffff; }
QGroupBox {
    background: #14141c; border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px; margin-top: 14px; padding: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 6px; color: #ffffff;
}
QPushButton {
    background: rgba(255, 255, 255, 0.05); color: #e4e4e7;
    border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 8px; padding: 7px 14px;
}
QPushButton:hover { border-color: #8b66cd; }
QPushButton:disabled { color: #63636b; border-color: rgba(255, 255, 255, 0.06); }
QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b66cd, stop:1 #6f42c1);
    color: white; border: none; border-radius: 10px; padding: 12px;
    font-size: 14px; font-weight: bold;
}
QPushButton#primary:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9a76dd, stop:1 #7d4fd1); }
QPushButton#primary:disabled { background: rgba(255, 255, 255, 0.08); color: #63636b; }
QPushButton#chip { border-radius: 14px; padding: 5px 14px; }
QPushButton#chip:checked {
    background: rgba(139, 102, 205, 0.28); border-color: #8b66cd; color: white;
}
QCheckBox { color: #e4e4e7; spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 5px;
    border: 1px solid rgba(255, 255, 255, 0.25); background: transparent; }
QCheckBox::indicator:checked { background: #8b66cd; border-color: #8b66cd; }
QComboBox, QLineEdit {
    background: #1c1c26; color: #f4f4f5; border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px; padding: 6px 10px; selection-background-color: #8b66cd;
}
QComboBox:focus, QLineEdit:focus { border-color: #8b66cd; }
QComboBox QAbstractItemView { background: #1c1c26; color: #f4f4f5; selection-background-color: #8b66cd; }
QRadioButton { color: #e4e4e7; spacing: 6px; }
QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px;
    border: 1px solid rgba(255, 255, 255, 0.3); background: transparent; }
QRadioButton::indicator:checked { background: #8b66cd; border-color: #8b66cd; }
QProgressBar {
    background: rgba(255, 255, 255, 0.08); border: none; border-radius: 4px;
    height: 8px; text-align: center; color: #a1a1aa; font-size: 10px;
}
QProgressBar::chunk { border-radius: 4px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b66cd, stop:1 #6f42c1); }
QPlainTextEdit {
    background: #08080d; color: #e4e4e7; border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px; padding: 8px;
}
QTableWidget {
    background: #14141c; color: #e4e4e7; gridline-color: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px;
    alternate-background-color: rgba(255, 255, 255, 0.025);
    selection-background-color: rgba(139, 102, 205, 0.35);
}
QHeaderView { background: #14141c; }
QHeaderView::section {
    background: #14141c; color: #a1a1aa; border: none;
    padding: 6px; font-weight: bold;
}
QTableWidget QTableCornerButton::section { background: #14141c; border: none; }
QScrollBar:vertical { background: transparent; width: 10px; }
QScrollBar::handle:vertical { background: rgba(255, 255, 255, 0.16);
    border-radius: 5px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QDialog { background: #14141c; }
QLabel#reviewhead { font-size: 13px; font-weight: bold; color: #ffffff; }
QPushButton#ghost {
    border: 1px solid rgba(139, 102, 205, 0.45); color: #b79df0;
    border-radius: 8px; padding: 7px 14px; background: transparent;
}
QPushButton#ghost:hover { background: rgba(139, 102, 205, 0.14); }
QPushButton#ghost:disabled { color: #63636b; border-color: rgba(255, 255, 255, 0.06); }
"""

PHASE_STATUS_QT = {
    "load": "Cargando modelos...",
    "align": "Alineando palabras...",
    "diarize": "Identificando hablantes...",
    "save": "Escribiendo archivos...",
}

LIGHT_QSS = DARK_QSS.replace("#ffffff", "#09090b") \
    .replace("#d9c8ff", "#6f42c1") \
    .replace("#b79df0", "#6f42c1") \
    .replace("#0b0b10", "#f4f4f6").replace("#14141c", "#ffffff") \
    .replace("#e4e4e7", "#27272a").replace("#f4f4f5", "#27272a") \
    .replace("#a1a1aa", "#52525b").replace("#71717a", "#a1a1aa") \
    .replace("#63636b", "#52525b") \
    .replace("#08080d", "#fafafa").replace("#1c1c26", "#f4f4f6") \
    .replace("rgba(255, 255, 255, 0.08)", "rgba(0, 0, 0, 0.1)") \
    .replace("rgba(255, 255, 255, 0.12)", "rgba(0, 0, 0, 0.15)") \
    .replace("rgba(255, 255, 255, 0.06)", "rgba(0, 0, 0, 0.06)") \
    .replace("rgba(255, 255, 255, 0.05)", "rgba(0, 0, 0, 0.04)") \
    .replace("rgba(255, 255, 255, 0.16)", "rgba(0, 0, 0, 0.22)") \
    .replace("rgba(255, 255, 255, 0.025)", "rgba(0, 0, 0, 0.025)") \
    .replace("rgba(255, 255, 255, 0.25)", "rgba(0, 0, 0, 0.2)") \
    .replace("rgba(255, 255, 255, 0.3)", "rgba(0, 0, 0, 0.25)") \
    .replace("rgba(255, 255, 255, 0.07)", "rgba(0, 0, 0, 0.06)") \
    .replace("rgba(255, 255, 255, 0.03)", "rgba(0, 0, 0, 0.03)") \
    .replace("rgba(255, 255, 255, 0.02)", "rgba(0, 0, 0, 0.02)")

THEMES = {"Oscuro": DARK_QSS, "Claro": LIGHT_QSS}


class ThemeSwitch(QWidget):
    """Moon/sun slider: the knob sits on the CURRENT theme's icon."""

    toggled = Signal(bool)

    def __init__(self, parent=None, dark: bool = True):
        super().__init__(parent)
        self._dark = dark
        self.setFixedSize(66, 30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Cambiar tema claro/oscuro")

    def set_dark(self, dark: bool) -> None:
        if dark != self._dark:
            self._dark = dark
            self.update()

    def mouseReleaseEvent(self, event):
        self._dark = not self._dark
        self.update()
        self.toggled.emit(self._dark)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        if self._dark:
            track, glyph = QColor("#3a3050"), QColor("#cfc6e8")
        else:
            track, glyph = QColor("#e4e4ea"), QColor("#6f42c1")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(2, 2, w - 4, h - 4, 13, 13)
        cy = h / 2
        self._moon(p, 13, cy, glyph)
        self._sun(p, w - 13, cy, glyph)
        kx = 17 if self._dark else w - 17
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(QPointF(kx, cy), 11, 11)
        # the CURRENT theme's glyph rides inside the knob
        if self._dark:
            self._moon(p, kx, cy, QColor("#3a3050"))
        else:
            self._sun(p, kx, cy, QColor("#6f42c1"))

    def _moon(self, p, cx, cy, color):
        path = QPainterPath()
        path.addEllipse(cx - 5, cy - 5, 10, 10)
        cut = QPainterPath()
        cut.addEllipse(cx - 1, cy - 6.5, 10, 13)
        try:
            path = path.subtracted(cut)
        except Exception:
            pass
        p.setBrush(color)
        p.drawPath(path)

    def _sun(self, p, cx, cy, color):
        pen = QPen(color, 1.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(color)
        p.drawEllipse(QPointF(cx, cy), 3, 3)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(8):
            a = math.pi / 4 * i
            p.drawLine(QPointF(cx + 5 * math.cos(a), cy + 5 * math.sin(a)),
                       QPointF(cx + 7.5 * math.cos(a), cy + 7.5 * math.sin(a)))


class StatusRing(QWidget):
    """Idle = dim ring (off); running = spinning arc; done = full green ring."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(26, 26)
        self._state = "idle"
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)

    def set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.update()
        if state == "running":
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()

    def _tick(self):
        self._angle = (self._angle + 9) % 360
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(4.5, 4.5, 17, 17)
        pen = QPen()
        pen.setWidth(3.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        if self._state == "idle":
            pen.setColor(QColor(128, 128, 128, 70))
            p.setPen(pen)
            p.drawArc(rect, 0, 360 * 16)
        elif self._state == "running":
            pen.setColor(QColor(128, 128, 128, 55))
            p.setPen(pen)
            p.drawArc(rect, 0, 360 * 16)
            pen.setColor(QColor("#8b66cd"))
            p.setPen(pen)
            p.drawArc(rect, -self._angle * 16, 110 * 16)
        else:
            pen.setColor(QColor("#22c55e"))
            p.setPen(pen)
            p.drawArc(rect, 0, 360 * 16)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1000, 860)
        try:
            if ICON_PATH.exists():
                self.setWindowIcon(QIcon(str(ICON_PATH)))
        except Exception:
            pass
        self._q: queue.Queue = queue.Queue()
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._poll)

        self.files: list[str] = []
        self.file_state: dict[str, str] = {}
        self.file_dur: dict[str, str] = {}
        self.running = False
        self.bench_running = False
        self.stop_requested = False
        self.out_folder: str | None = None
        self.last_outputs: list[str] = []
        self.last_output_dir: str | None = None
        self.reviews: list[dict] = []
        self.last_log_path: str | None = None
        self.clip_dir = tempfile.mkdtemp(prefix="transcribe-qt-clips-")
        self.tray: QSystemTrayIcon | None = None
        self.settings: dict = self._load_settings()

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 12, 18, 12)
        root.setSpacing(12)

        header = QHBoxLayout()
        titlebox = QVBoxLayout()
        t = QLabel("Transcripción <span style='color:#a98fe0'>de videos</span>")
        t.setObjectName("title")
        t.setTextFormat(Qt.TextFormat.RichText)
        titlebox.addWidget(t)
        self.title_label = t
        sub = QLabel("Local · sin subidas · sin límites")
        sub.setObjectName("muted")
        titlebox.addWidget(sub)
        header.addLayout(titlebox, 1)
        self.ring = StatusRing()
        self.ring.setToolTip("Estado del procesamiento")
        header.addWidget(self.ring)
        self.theme_switch = ThemeSwitch()
        self.theme_switch.toggled.connect(self._on_theme_toggled)
        header.addWidget(self.theme_switch)
        root.addLayout(header)

        # ---- 1. files
        box_files = QGroupBox("①  Archivos — arrastra y suelta aquí")
        lf = QVBoxLayout(box_files)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Archivo", "Duración", "Estado"])
        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 560)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 190)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(150)
        self.table.setAcceptDrops(True)
        lf.addWidget(self.table)
        row = QHBoxLayout()
        b_add = QPushButton("Añadir archivos…")
        b_add.setObjectName("ghost")
        b_add.clicked.connect(self.add_files)
        row.addWidget(b_add)
        b_rm = QPushButton("Quitar seleccionados")
        b_rm.setObjectName("ghost")
        b_rm.clicked.connect(self.remove_selected)
        row.addWidget(b_rm)
        b_clr = QPushButton("Limpiar lista")
        b_clr.setObjectName("ghost")
        b_clr.clicked.connect(self.clear_files)
        row.addWidget(b_clr)
        row.addStretch(1)
        self.file_count = QLabel("0 archivos")
        self.file_count.setObjectName("muted")
        row.addWidget(self.file_count)
        lf.addLayout(row)
        root.addWidget(box_files)
        self.setAcceptDrops(True)

        # ---- 2. options
        box_opts = QGroupBox("②  Opciones")
        grid = QGridLayout(box_opts)
        grid.addWidget(QLabel("Formatos:"), 0, 0)
        self.fmt_btns: dict[str, QPushButton] = {}
        frow = QHBoxLayout()
        for label, key, default in FORMATS:
            b = QPushButton(label)
            b.setObjectName("chip")
            b.setCheckable(True)
            b.setChecked(default)
            frow.addWidget(b)
            self.fmt_btns[key] = b
        frow.addStretch(1)
        grid.addLayout(frow, 0, 1, 1, 2)
        grid.addWidget(QLabel("Idioma:"), 1, 0)
        self.cmb_lang = QComboBox()
        for label, _code in LANGUAGES:
            self.cmb_lang.addItem(label)
        grid.addWidget(self.cmb_lang, 1, 1)
        grid.addWidget(QLabel("Modelo:"), 2, 0)
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(MODELS)
        grid.addWidget(self.cmb_model, 2, 1, 1, 2)
        self.chk_diar = QCheckBox("Identificar hablantes")
        self.chk_diar.setChecked(True)
        grid.addWidget(self.chk_diar, 3, 0, 1, 3)
        self.chk_auto = QCheckBox("Marcar al hablante principal automáticamente (el que más habla)")
        self.chk_auto.setChecked(True)
        grid.addWidget(self.chk_auto, 4, 0, 1, 3)
        self.chk_solo = QCheckBox("Extracto solo del hablante principal (*.solo-principal.md)")
        self.chk_solo.setChecked(True)
        grid.addWidget(self.chk_solo, 5, 0, 1, 3)
        grid.addWidget(QLabel("Nombre del hablante principal:"), 6, 0)
        self.edit_name = QLineEdit("Profesor")
        self.edit_name.setMaxLength(40)
        grid.addWidget(self.edit_name, 6, 1, 1, 2)
        grid.addWidget(QLabel("Guardar en:"), 7, 0)
        saverow = QHBoxLayout()
        self.radio_video = QRadioButton("Junto al video")
        self.radio_video.setChecked(True)
        self.radio_folder = QRadioButton("Carpeta:")
        self.radio_folder.toggled.connect(self._on_folder_toggled)
        saverow.addWidget(self.radio_video)
        saverow.addWidget(self.radio_folder)
        b_out = QPushButton("Elegir…")
        b_out.setObjectName("ghost")
        b_out.clicked.connect(self.choose_folder)
        saverow.addWidget(b_out)
        self.lbl_out = QLabel("(carpeta por video)")
        self.lbl_out.setObjectName("muted")
        saverow.addWidget(self.lbl_out, 1)
        grid.addLayout(saverow, 7, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        root.addWidget(box_opts)

        # ---- 3. run
        box_run = QGroupBox("③  Ejecutar")
        lr = QVBoxLayout(box_run)
        self.btn_start = QPushButton("▶  Transcribir")
        self.btn_start.setObjectName("primary")
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self.start)
        lr.addWidget(self.btn_start)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        lr.addWidget(self.progress)
        self.file_status = QLabel("")
        self.file_status.setObjectName("muted")
        lr.addWidget(self.file_status)
        brow = QHBoxLayout()
        self.btn_stop = QPushButton("Detener")
        self.btn_stop.setObjectName("ghost")
        self.btn_stop.clicked.connect(self.request_stop)
        self.btn_stop.setEnabled(False)
        brow.addWidget(self.btn_stop)
        self.btn_review = QPushButton("Revisar hablantes")
        self.btn_review.setObjectName("ghost")
        self.btn_review.clicked.connect(self._review_speakers)
        self.btn_review.setEnabled(False)
        brow.addWidget(self.btn_review)
        self.btn_open = QPushButton("Abrir carpeta")
        self.btn_open.setObjectName("ghost")
        self.btn_open.clicked.connect(self.open_output)
        self.btn_open.setEnabled(False)
        brow.addWidget(self.btn_open)
        self.btn_log = QPushButton("Abrir registro")
        self.btn_log.setObjectName("ghost")
        self.btn_log.clicked.connect(self.open_log)
        self.btn_log.setEnabled(False)
        brow.addWidget(self.btn_log)
        self.btn_bench = QPushButton("Probar velocidad")
        self.btn_bench.setObjectName("ghost")
        self.btn_bench.clicked.connect(self.run_benchmark)
        brow.addWidget(self.btn_bench)
        brow.addStretch(1)
        lr.addLayout(brow)
        self.status = QLabel("Arrastra tus grabaciones, elige opciones y pulsa Transcribir.")
        self.status.setObjectName("muted")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setWordWrap(True)
        lr.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(130)
        try:
            self.log.setFont(QFont("Consolas", 9))
        except Exception:
            pass
        lr.addWidget(self.log, 1)
        root.addWidget(box_run, 1)

        self._apply_saved_settings()
        self._apply_theme(save=False)
        self._apply_theme(save=False)

    # ---------------------------------------------------------------- theme
    def _apply_theme(self, save: bool = True) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(THEMES.get(self._theme_name(), DARK_QSS))
        accent = "#a98fe0" if self._theme_name() == "Oscuro" else "#6f42c1"
        self.title_label.setText(f"Transcripción <span style='color:{accent}'>de videos</span>")
        self.theme_switch.set_dark(self._theme_name() == "Oscuro")
        if save:
            self._save_settings()

    def _theme_name(self) -> str:
        return getattr(self, "_theme", "Oscuro")

    def _on_theme_toggled(self, dark: bool) -> None:
        self._theme = "Oscuro" if dark else "Claro"
        self._apply_theme()

    # --------------------------------------------------------------- helpers
    def _log(self, text: str):
        self.log.appendPlainText(text.rstrip("\n") or " ")
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _file_dur(self, f: str) -> str:
        if f not in self.file_dur:
            try:
                secs = transcribe.media_duration(Path(f))
                self.file_dur[f] = transcribe.hms(secs) if secs else "?"
            except Exception:
                self.file_dur[f] = "?"
        return self.file_dur[f]

    def _refresh(self):
        self.table.setRowCount(0)
        for f in self.files:
            r = self.table.rowCount()
            self.table.insertRow(r)
            name = QTableWidgetItem(Path(f).name)
            name.setData(Qt.ItemDataRole.UserRole, f)
            self.table.setItem(r, 0, name)
            dur = QTableWidgetItem(self._file_dur(f))
            dur.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, 1, dur)
            self.table.setItem(r, 2, QTableWidgetItem(self.file_state.get(f, "En cola")))
        self.file_count.setText(f"{len(self.files)} archivo" + ("" if len(self.files) == 1 else "s"))

    def _selected_paths(self) -> list[str]:
        out = []
        for idx in self.table.selectionModel().selectedRows():
            item = self.table.item(idx.row(), 0)
            if item is not None:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def _add_paths(self, paths) -> int:
        added = 0
        for raw in paths or []:
            p = Path(raw)
            if not p.exists():
                continue
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.suffix.lower() in transcribe.MEDIA_EXTS and str(f) not in self.files:
                        self.files.append(str(f))
                        added += 1
            elif p.suffix.lower() in transcribe.MEDIA_EXTS and str(p) not in self.files:
                self.files.append(str(p))
                added += 1
        self._refresh()
        return added

    # ---------------------------------------------------------------- events
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        n = self._add_paths(paths)
        if n:
            self._log(f"→ {n} archivo(s) añadido(s)\n")

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Elige videos o audios", "",
            "Video/Audio (*.mp4 *.mov *.mkv *.webm *.m4v *.avi *.wmv *.mpg *.mpeg "
            "*.m4a *.mp3 *.wav *.wma *.flac *.ogg *.aac *.opus);;Todos (*.*)")
        if paths:
            n = self._add_paths(paths)
            if n:
                self._log(f"→ {n} archivo(s) añadido(s)\n")
            else:
                self.status.setText("No se añadió nada.")

    def remove_selected(self):
        sel = set(self._selected_paths())
        if not sel:
            return
        n = len(sel)
        self.files = [f for f in self.files if f not in sel]
        for f in sel:
            self.file_state.pop(f, None)
            self.file_dur.pop(f, None)
        self._refresh()
        self._log(f"→ {n} archivo(s) quitado(s)\n")

    def clear_files(self):
        n = len(self.files)
        self.files = []
        self.file_state = {}
        self.file_dur = {}
        self._refresh()
        self._log(f"→ Lista limpiada ({n} archivo(s))\n")

    def _on_folder_toggled(self, checked: bool) -> None:
        if checked and not self.out_folder:
            self.choose_folder()
            if not self.out_folder:
                self.radio_video.setChecked(True)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de destino")
        if folder:
            self.out_folder = folder
            self.radio_folder.setChecked(True)
            self.lbl_out.setText(folder)

    def closeEvent(self, event):
        if self.running:
            ans = QMessageBox.question(
                self, "Salir", "La transcripción está en curso. ¿Salir y cancelarla?")
            if ans != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._save_settings()
        shutil.rmtree(self.clip_dir, ignore_errors=True)
        event.accept()

    # -------------------------------------------------------------- settings
    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self) -> None:
        try:
            data = {
                "formats": [k for _label, k, _d in FORMATS if self.fmt_btns[k].isChecked()],
                "lang": dict(LANGUAGES).get(self.cmb_lang.currentText(), "es"),
                "model": self.cmb_model.currentText(),
                "diarize": self.chk_diar.isChecked(),
                "auto_speaker": self.chk_auto.isChecked(),
                "solo_principal": self.chk_solo.isChecked(),
                "theme": self._theme_name(),
                "speaker_name": self.edit_name.text(),
                "out_mode": "folder" if self.radio_folder.isChecked() else "video",
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
        for label, key, _d in FORMATS:
            if key in s.get("formats", []):
                self.fmt_btns[key].setChecked(True)
            elif "formats" in s:
                self.fmt_btns[key].setChecked(False)
        codes = {code: label for label, code in LANGUAGES}
        labels = {label: code for label, code in LANGUAGES}
        lang_val = s.get("lang")
        if lang_val in codes:
            self.cmb_lang.setCurrentText(codes[lang_val])
        elif lang_val in labels:
            self.cmb_lang.setCurrentText(lang_val)
        if s.get("model") in MODELS:
            self.cmb_model.setCurrentText(s["model"])
        self.chk_diar.setChecked(bool(s.get("diarize", True)))
        self.chk_auto.setChecked(bool(s.get("auto_speaker", True)))
        self.chk_solo.setChecked(bool(s.get("solo_principal", True)))
        if s.get("speaker_name"):
            self.edit_name.setText(s["speaker_name"])
        if s.get("out_mode") == "folder":
            self.radio_folder.setChecked(True)
        if s.get("out_folder"):
            self.out_folder = s["out_folder"]
            self.lbl_out.setText(self.out_folder)
        if s.get("theme") in ("Oscuro", "Claro"):
            self._theme = s["theme"]

    # -------------------------------------------------------------- pipeline
    def _build_args(self) -> argparse.Namespace:
        formats = [k for _label, k, _d in FORMATS if self.fmt_btns[k].isChecked()] or ["markdown"]
        out = self.out_folder if self.radio_folder.isChecked() else None
        return argparse.Namespace(
            lang=dict(LANGUAGES).get(self.cmb_lang.currentText(), "es"),
            model=self.cmb_model.currentText(),
            compute_type="int8", batch_size=8,
            formats=",".join(formats), out=out, title=None,
            no_diarize=not self.chk_diar.isChecked(),
            hf_token=None, device=None,
            auto_speaker=self.chk_auto.isChecked(),
            speaker_name=self.edit_name.text().strip() or "Profesor",
            speaker_clips_dir=self.clip_dir,
            no_vad_crop=False, seg_stride=2.0,
            solo_principal=self.chk_solo.isChecked(),
            progress_callback=lambda pct: self._q.put(("progress", pct)),
            phase_callback=lambda name: self._q.put(("phase", name)),
            diarize_progress_callback=lambda pct: self._q.put(("diar_progress", pct)),
        )

    def _set_pill(self, mode: str):
        """Drive the status ring: 'work' = spinning, 'done' = full, else idle."""
        self.ring.set_state("running" if mode == "work" else
                            "done" if mode == "done" else "idle")

    def start(self):
        if self.running or self.bench_running:
            return
        files = list(self.files)
        if not files:
            QMessageBox.information(self, "Sin archivos",
                                    "Arrastra al menos un video o audio primero.")
            return
        if shutil.which("ffmpeg") is None:
            QMessageBox.critical(
                self, "ffmpeg no encontrado",
                "Instala ffmpeg (winget install Gyan.FFmpeg) y abre una terminal nueva.")
            return
        args = self._build_args()
        device = transcribe.pick_device(args)
        low, high, calibrated = transcribe.estimate_runtime(
            files, args, device, diarize=not args.no_diarize)
        mark = "calibrado a esta máquina" if calibrated else "estimación por tipo de GPU"
        gpu = transcribe.gpu_status()
        lines = [f"Tiempo estimado: ~{low:.0f}-{high:.0f} min ({mark})"]
        if transcribe.gpu_is_busy(gpu):
            lines.append(
                f"La GPU está ocupada: {gpu['util']:.0f}% de uso, "
                f"{gpu['mem_used']:.0f}MB/{gpu['mem_total']:.0f}MB de VRAM "
                f"({len(gpu['apps'])} proceso(s) de GPU).")
            lines.append("Cierra navegadores, juegos u otras apps con aceleración "
                         "gráfica para que vaya mucho más rápido.")
        else:
            lines.append("GPU libre, todo listo.")
        if QMessageBox.question(self, "Transcripción", "\n".join(lines) + "\n\n¿Continuar?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.Yes) != QMessageBox.StandardButton.Yes:
            return
        self.running = True
        self.stop_requested = False
        self.last_outputs = []
        self.last_output_dir = None
        self.reviews = []
        self.last_log_path = None
        self.file_state = {f: "En cola" for f in files}
        self._refresh()
        self._set_pill("work")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_review.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.btn_log.setEnabled(False)
        self.status.setText(f"Transcribiendo {len(files)} archivo(s)… no cierres la ventana")
        self.file_status.setText("")
        self.setWindowTitle(f"Transcribiendo {len(files)} archivo(s) — {APP_TITLE}")
        self._log(f"\n========== Iniciando ({len(files)} archivo(s)) ==========\n")
        threading.Thread(target=self._worker, args=(files, args), daemon=True).start()
        self.timer.start()

    def request_stop(self):
        if self.running:
            self.stop_requested = True
            self.btn_stop.setEnabled(False)
            self.status.setText("Deteniendo tras el archivo en curso...")

    def _worker(self, files: list[str], args: argparse.Namespace):
        q = self._q
        log_path = transcribe.new_log_path()
        try:
            log_file = open(log_path, "w", encoding="utf-8")
        except Exception:
            log_file = None
        q.put(("logfile", str(log_path)))
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _Sink(q, log_file)
        cache = transcribe.ModelCache()
        n = len(files)
        try:
            transcribe.ensure_punkt()
            prepared: list[dict] = []
            fatal = False
            for i, f in enumerate(files, start=1):
                if self.stop_requested:
                    q.put("[warn] Detenido por el usuario; archivos restantes omitidos.\n")
                    break
                q.put(("status", f"Transcribiendo {i}/{n}: {Path(f).name}..."))
                q.put(("file_status", f, "Transcribiendo…", i, n))
                q.put(f"\n>>> {Path(f).name}\n")
                try:
                    prepared.append(transcribe.get_prep(Path(f), args, cache))
                except Exception as exc:
                    q.put(("file_status", f, "Error", i, n))
                    q.put("\n" + traceback.format_exc() + "\n")
                    if transcribe._is_cuda_fatal(exc):
                        fatal = True
                        q.put("\n[error] La GPU se perdió (memoria insuficiente u otra"
                              " app usando la GPU). Se omitieron los archivos"
                              " restantes — cierra otras apps que usen GPU y"
                              " vuelve a ejecutar.\n")
                        break
            m = len(prepared)
            for i, prep in enumerate(prepared, start=1):
                if fatal:
                    q.put("\n[error] La GPU se perdió; se omitió el resto del lote.\n")
                    break
                pname = str(prep["path"])
                q.put(("status", f"Identificando hablantes {i}/{m}: {Path(pname).name}..."))
                q.put(("file_status", pname, "Identificando hablantes…", i, m))
                try:
                    payload = transcribe._diarize_write(prep, args, cache)
                    if payload:
                        q.put(("review", {
                            "key": Path(pname).name, "path": pname,
                            "outputs": payload.get("outputs", []),
                            "clips": payload["clips"],
                            "totals": payload["totals"],
                            "names": payload["names"],
                        }))
                    q.put(("file_status", pname, "Completado ✓", i, m))
                    q.put(f"\n>>> Completado: {Path(pname).name}\n")
                except Exception:
                    q.put(("file_status", pname, "Error", i, m))
                    q.put("\n" + traceback.format_exc() + "\n")
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
                        diarize_rate=sum(diar) / len(diar) if diar else None)
            except Exception:
                pass
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            try:
                if log_file is not None:
                    log_file.close()
            except Exception:
                pass
        q.put("__DONE__")

    def _poll(self):
        try:
            while True:
                line = self._q.get_nowait()
                if isinstance(line, tuple):
                    kind, *rest = line
                    if kind == "review":
                        self.reviews.append(rest[0])
                    elif kind == "logfile":
                        self.last_log_path = Path(rest[0])
                    elif kind == "progress":
                        self.progress.setRange(0, 100)
                        self.progress.setValue(int(rest[0]))
                    elif kind == "diar_progress":
                        self.progress.setRange(0, 100)
                        self.progress.setValue(int(rest[0]))
                        self.status.setText(f"Identificando hablantes… {rest[0]:.0f}%")
                    elif kind == "bench_result":
                        self.bench_running = False
                        if not self.running:
                            self.timer.stop()
                        self.btn_bench.setEnabled(True)
                        self.status.setText("Listo.")
                        self._set_pill("done")
                        r = rest[0] or {}
                        if r.get("asr_rate") or r.get("align_rate") or r.get("diarize_rate"):
                            lines = ["Velocidades medidas (segundos por minuto de audio):",
                                     f"  ASR:        {r.get('asr_rate', 0) * 60:.1f} s/min",
                                     f"  Alineación: {r.get('align_rate', 0) * 60:.1f} s/min"]
                            if r.get("diarize_rate"):
                                lines.append(f"  Diarización:{r['diarize_rate'] * 60:.1f} s/min")
                            else:
                                lines.append("  Diarización: (sin token o deshabilitada)")
                            lines += ["", "Guardado. Las próximas estimaciones usarán estos valores."]
                        else:
                            lines = ["No se pudo medir la velocidad.",
                                     "Revisa el registro de arriba para ver el detalle."]
                        QMessageBox.information(self, "Prueba de velocidad", "\n".join(lines))
                    elif kind == "status":
                        self.status.setText(rest[0])
                    elif kind == "phase":
                        if rest[0] == "asr":
                            self.progress.setRange(0, 100)
                            self.progress.setValue(0)
                        else:
                            self.progress.setRange(0, 0)
                        text = PHASE_STATUS_QT.get(rest[0])
                        if text:
                            self.status.setText(text)
                    elif kind == "file_status":
                        fpath, text, i, n = rest[0], rest[1], rest[2], rest[3]
                        self.file_state[fpath] = text
                        self._refresh_row(fpath)
                        base = Path(fpath).name
                        self.file_status.setText(f"Actual: {base} — {text}")
                        self.setWindowTitle(f"[{i}/{n}] {base} — {APP_TITLE}")
                    continue
                if line == "__DONE__":
                    self._finish()
                    return
                if isinstance(line, str) and line.startswith("[ok] "):
                    out = Path(line.split("] ", 1)[1].strip())
                    self.last_outputs.append(str(out))
                    self.last_output_dir = str(out.parent)
                self._log(line)
        except queue.Empty:
            pass
        except Exception:
            self._log("\n" + traceback.format_exc() + "\n")
        finally:
            if self.running or self.bench_running:
                self.timer.start()

    def _refresh_row(self, fpath: str):
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == fpath:
                state = QTableWidgetItem(self.file_state.get(fpath, ""))
                self.table.setItem(r, 2, state)
                break


    def _finish(self):
        self.running = False
        self.timer.stop()
        self.timer.stop()
        self.setWindowTitle(APP_TITLE)
        self._set_pill("done")
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.reviews:
            self.btn_review.setEnabled(True)
        if self.last_output_dir:
            self.btn_open.setEnabled(True)
        if self.last_log_path:
            self.btn_log.setEnabled(True)
        stopped = " (detenido por el usuario)" if self.stop_requested else ""
        self.status.setText("Listo. Puedes añadir más archivos y transcribir de nuevo.")
        self._log(f"\n========== Terminado{stopped} ==========\n")
        if self.last_log_path:
            self._log(f"Registro de esta ejecución: {self.last_log_path}\n")
        self._save_settings()
        # Notice me: chime + taskbar flash + native Windows toast (only when
        # the window itself is not focused, i.e. you were doing something else).
        try:
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass
        try:
            QApplication.alert(self)
        except Exception:
            pass
        try:
            if self.tray is None:
                self.tray = QSystemTrayIcon(QIcon(str(ICON_PATH)), self)
            n = len(self.last_outputs)
            title = "Transcripción detenida" if self.stop_requested else "Transcripción completa"
            body = f"{n} archivo(s) listo(s)"
            if self.last_output_dir:
                body += f" — {self.last_output_dir}"
            self.tray.show()
            self.tray.showMessage(title, body,
                                  QSystemTrayIcon.MessageIcon.Information, 6000)
            QTimer.singleShot(9000, self._hide_tray)
        except Exception:
            pass
        if not self.chk_auto.isChecked():
            self._review_speakers()

    def _hide_tray(self):
        try:
            if self.tray is not None:
                self.tray.hide()
        except Exception:
            pass

    # --------------------------------------------------------------- review
    def _review_speakers(self):
        if not self.reviews:
            QMessageBox.information(self, "Sin transcripciones",
                                    "Primero transcribe algún video para revisar sus hablantes.")
            return
        if len(self.reviews) == 1:
            self._review_file(self.reviews[0])
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Revisar hablantes")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("¿Qué archivo quieres revisar?"))
        combo = QComboBox()
        keys = [r["key"] for r in self.reviews]
        labels = []
        for i, r in enumerate(self.reviews):
            label = r["key"]
            if keys.count(r["key"]) > 1:
                label = f"{r['key']} — {Path(r['path']).parent.name}"
            labels.append(label)
        combo.addItems(labels)
        lay.addWidget(combo)
        ok = QPushButton("Revisar")
        ok.setObjectName("primary")
        ok.setDefault(True)
        lay.addWidget(ok)
        chosen = {"r": self.reviews[0]}

        def go():
            idx = combo.currentIndex()
            if 0 <= idx < len(self.reviews):
                chosen["r"] = self.reviews[idx]
            dlg.accept()

        ok.clicked.connect(go)
        if dlg.exec():
            self._review_file(chosen["r"])

    def _review_file(self, entry: dict):
        speakers = sorted(entry["totals"].items(), key=lambda kv: -kv[1])
        if not speakers:
            QMessageBox.information(self, "Sin hablantes",
                                    "Esta transcripción no tiene hablantes identificados.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Revisar hablantes — {entry['key']}")
        dlg.setMinimumWidth(600)
        lay = QVBoxLayout(dlg)
        head = QLabel(f"{entry['key']} · {len(speakers)} voces")
        head.setObjectName("section")
        lay.addWidget(head)
        info = QLabel("Escucha cada voz y ponle nombre. Dos voces con el mismo "
                      "nombre se fusionan en una sola persona.")
        info.setObjectName("muted")
        info.setWordWrap(True)
        lay.addWidget(info)
        player = QMediaPlayer(dlg)
        player.setAudioOutput(QAudioOutput(dlg))
        edits: dict[str, QLineEdit] = {}
        for sp, secs in speakers:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{secs / 60:.1f} min"))
            for clip in entry["clips"].get(sp, []):
                btn = QPushButton("▶")
                btn.setObjectName("ghost")
                btn.setFixedWidth(46)
                btn.clicked.connect(lambda _=False, p=clip: self._play_clip(player, p))
                row.addWidget(btn)
            edit = QLineEdit(entry["names"].get(sp, sp))
            edit.setMaxLength(40)
            row.addWidget(edit, 1)
            edits[sp] = edit
            lay.addLayout(row)
        save = QPushButton("Guardar")
        save.setObjectName("primary")
        save.setDefault(True)
        lay.addWidget(save)

        def ok():
            renames: dict[str, str] = {}
            for sp, widget in edits.items():
                new = widget.text().strip()
                cur = entry["names"].get(sp, sp)
                if new and new != cur:
                    renames[cur] = new
                    entry["names"][sp] = new
            self._apply_renames(renames, entry["outputs"])
            dlg.accept()

        save.clicked.connect(ok)
        dlg.exec()

    def _play_clip(self, player: QMediaPlayer, path: str):
        try:
            player.stop()
            player.setSource(QUrl.fromLocalFile(path))
            player.play()
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
            new = transcribe.rename_labels(text, renames)
            if new != text:
                p.write_text(new, encoding="utf-8")
                renamed += 1
        pairs = ", ".join(f"'{o}' → '{n}'" for o, n in renames.items())
        self._log(f"→ Renombrado(s): {pairs} ({renamed} archivo(s)).\n")

    # ---------------------------------------------------------------- bench
    def open_output(self):
        folder = self.last_output_dir or (self.out_folder or None)
        if folder and Path(folder).exists() and hasattr(os, "startfile"):
            os.startfile(str(folder))

    def open_log(self):
        if self.last_log_path and Path(self.last_log_path).exists() and hasattr(os, "startfile"):
            os.startfile(str(self.last_log_path))

    def run_benchmark(self):
        if self.running:
            QMessageBox.information(self, "En proceso",
                                    "Espera a que termine la transcripción actual.")
            return
        if shutil.which("ffmpeg") is None:
            QMessageBox.critical(
                self, "ffmpeg no encontrado",
                "Instala ffmpeg (winget install Gyan.FFmpeg) y abre una terminal nueva.")
            return
        ans = QMessageBox.information(
            self, "Prueba de velocidad",
            "Mido tu velocidad real con la muestra incluida (~1 min, carga el modelo).\n"
            "Cierra apps con GPU para un resultado realista.",
            QMessageBox.StandardButton.Ok)
        if ans != QMessageBox.StandardButton.Ok:
            return
        self.btn_bench.setEnabled(False)
        self.bench_running = True
        self._set_pill("work")
        self.status.setText("Midiendo velocidad...")
        threading.Thread(target=self._bench_worker, daemon=True).start()
        self.timer.start()

    def _bench_worker(self):
        q = self._q
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _Sink(q, None)
        result = None
        try:
            result = transcribe.run_benchmark(self._bench_args())
        except Exception:
            q.put("\n" + traceback.format_exc() + "\n")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            q.put(("bench_result", result or {}))

    def _bench_args(self) -> argparse.Namespace:
        s = self._load_settings()
        return argparse.Namespace(
            lang=dict(LANGUAGES).get(self.cmb_lang.currentText(), "es"),
            model=self.cmb_model.currentText(),
            compute_type="int8", batch_size=8, formats="markdown,srt",
            out=None, title=None, no_diarize=not self.chk_diar.isChecked(),
            hf_token=None, device=None, auto_speaker=True,
            speaker_name="Profesor", speaker_clips_dir=None,
            no_vad_crop=False, seg_stride=2.0, solo_principal=True)


class _Sink:
    """File-like object routing prints into the poll queue (+ optional file)."""

    def __init__(self, q: queue.Queue, log_file):
        self.q = q
        self.log_file = log_file

    def write(self, text):
        if text:
            self.q.put(text)
            if self.log_file is not None:
                try:
                    self.log_file.write(text)
                    self.log_file.flush()
                except Exception:
                    pass

    def flush(self):
        pass


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        app.setFont(QFont("Segoe UI", 10))
    except Exception:
        pass
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
