@echo off
cd /d "%~dp0"
title Build Meeting Transcriber EXE
echo ========================================================
echo   Building MeetingTranscriber.exe (PyInstaller, onedir)
echo ========================================================

python -m pip install --upgrade pyinstaller

python -m PyInstaller --noconfirm --clean --windowed ^
  --name MeetingTranscriber ^
  --collect-all customtkinter ^
  --collect-all faster_whisper ^
  --collect-all ctranslate2 ^
  --collect-binaries av ^
  --hidden-import av ^
  --hidden-import tokenizers ^
  --hidden-import huggingface_hub ^
  app.py

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo.
echo Build complete: dist\MeetingTranscriber\MeetingTranscriber.exe
pause
