@echo off
title Transcripcion de videos (Qt)
cd /d "%~dp0"
echo Iniciando Transcripcion de videos (Qt)...
uv run gui_qt.py
if errorlevel 1 (
    echo.
    echo Ocurrio un error. Revisa el mensaje arriba.
    pause
)
