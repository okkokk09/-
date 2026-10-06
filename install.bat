@echo off
cd /d "%~dp0"
title Installing Meeting Transcriber Dependencies
echo ========================================================
echo   Installing Meeting Transcriber AI Dependencies...
echo ========================================================
echo.

echo [1/3] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/3] Installing core libraries (customtkinter, faster-whisper, av)...
pip install customtkinter faster-whisper av

echo.
echo.
echo [3/3] Installing NVIDIA CUDA and cuDNN wheels for Windows...
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

echo.
echo Configuring CUDA DLLs for Windows...
python -c "from utils import setup_cuda_dll_paths; setup_cuda_dll_paths(); print('CUDA DLL setup verified!')"

echo.
echo ========================================================
echo   Installation completed successfully!
echo   You can now launch the app by running run.bat
echo ========================================================
echo.
pause
