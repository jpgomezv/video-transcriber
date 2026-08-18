@echo off
title Transcripcion de videos
cd /d "%~dp0"
echo Iniciando Transcripcion de videos...
uv run gui.py
if errorlevel 1 (
    echo.
    echo Ocurrio un error. Revisa el mensaje arriba.
    pause
)
