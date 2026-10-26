@echo off
title AI Image Caption Generator - BLIP
echo ============================================================
echo   AI Image Caption Generator Using BLIP
echo   Starting local server...
echo ============================================================

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

if not exist ".setup_done" (
    echo [INFO] First-time setup: checking and installing dependencies...
    python -m pip install -r requirements.txt --quiet
    echo 1 > .setup_done
)

if not exist "sample_images\park_bench.jpg" (
    echo [INFO] Generating demo sample images...
    python create_samples.py
)

echo.
echo ============================================================
echo   Application is starting!
echo   Opening browser at: http://127.0.0.1:7860
echo   (Keep this window open while using the application)
echo ============================================================
echo.

:: Automatically open browser after 2 seconds
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:7860"

python app.py
pause
