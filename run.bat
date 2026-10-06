@echo off
cd /d "%~dp0"
title Meeting Transcriber AI
echo ========================================================
echo   Starting Meeting Transcriber AI...
echo ========================================================
python app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%
    echo If libraries are not installed, please run install.bat first.
    echo.
    pause
)
