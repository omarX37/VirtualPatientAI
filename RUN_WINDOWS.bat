@echo off
setlocal
cd /d "%~dp0"
title Virtual Patient AI - Setup and Run

echo ============================================
echo       Virtual Patient AI - Local App
echo ============================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON=py"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PYTHON=python"
    ) else (
        echo Python was not found.
        echo Install Python 3.11 from https://www.python.org/downloads/windows/
        echo During setup, tick "Add python.exe to PATH".
        echo Then run this file again.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating local Python environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto error
)

echo [2/3] Installing/checking required libraries...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error

echo [3/3] Starting the local website...
echo Keep this window open while using the app.
echo Your browser should open automatically.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto error
exit /b 0

:error
echo.
echo Something went wrong. Take a screenshot of the error and send it for troubleshooting.
pause
exit /b 1
