@echo off
rem ===========================================================================
rem  Hindi AI Dubbing Platform - Windows setup and launcher (no Docker)
rem
rem  1. finds Python 3.10-3.12        4. installs dependencies (skipped when unchanged)
rem  2. checks FFmpeg / FFprobe       5. creates .env from .env.example on first run
rem  3. creates .venv                 6. builds the web UI if frontend\dist is missing (needs Node.js)
rem                                   7. starts the FastAPI server (API + web UI on one port)
rem
rem  Optional overrides (set before running):
rem    DUB_PYTHON       full path to python.exe to use
rem    DUB_TORCH_INDEX  cpu | cu126 | cu128   (default: auto-detected from nvidia-smi)
rem    DUB_SKIP_RUN=1   install only, do not start the server
rem ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "TORCH_VERSION=2.8.0"
set "TORCHAUDIO_VERSION=2.8.0"

rem ---------------------------------------------------------------- 1. Python
set "PY_CMD="
if defined DUB_PYTHON (
    "%DUB_PYTHON%" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PY_CMD="%DUB_PYTHON%""
    if not defined PY_CMD (
        echo [ERROR] DUB_PYTHON=%DUB_PYTHON% is not Python 3.10-3.12.
        exit /b 1
    )
)
if not defined PY_CMD (
    for %%V in (3.11 3.12 3.10) do (
        if not defined PY_CMD (
            py -%%V -c "import sys" >nul 2>&1 && set "PY_CMD=py -%%V"
        )
    )
)
if not defined PY_CMD (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PY_CMD=python"
)
if not defined PY_CMD (
    echo [ERROR] Python 3.10, 3.11 or 3.12 ^(64-bit^) is required. 3.11 is recommended:
    echo         winget install --id Python.Python.3.11 -e
    exit /b 1
)
echo [OK] Python: !PY_CMD!

rem ---------------------------------------------------------------- 2. FFmpeg
where ffmpeg >nul 2>&1
if errorlevel 1 goto :no_ffmpeg
where ffprobe >nul 2>&1
if errorlevel 1 goto :no_ffmpeg
echo [OK] FFmpeg found on PATH

rem ---------------------------------------------------------------- 3. venv
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [..] Creating virtual environment in %VENV_DIR%
    !PY_CMD! -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        exit /b 1
    )
)
set "VPY=%CD%\%VENV_DIR%\Scripts\python.exe"

rem ---------------------------------------------------------------- 4. dependencies
set "TORCH_INDEX=cpu"
set "CC_MAJOR="
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set "TORCH_INDEX=cu126"
    for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do set "CC_MAJOR=%%a"
)
rem RTX 50-series (Blackwell, compute capability 12.x) needs the CUDA 12.8 build.
if defined CC_MAJOR if !CC_MAJOR! GEQ 12 set "TORCH_INDEX=cu128"
if defined DUB_TORCH_INDEX set "TORCH_INDEX=%DUB_TORCH_INDEX%"
echo [OK] PyTorch build: !TORCH_INDEX!

set "REQ_HASH="
for /f "delims=" %%h in ('certutil -hashfile requirements.txt SHA256 ^| findstr /v ":"') do (
    if not defined REQ_HASH set "REQ_HASH=%%h"
)
if defined REQ_HASH set "REQ_HASH=!REQ_HASH: =!"
set "WANT=torch-%TORCH_VERSION%-!TORCH_INDEX!-req-!REQ_HASH!"
set "STAMP=%VENV_DIR%\.install-stamp"
set "HAVE="
if exist "%STAMP%" set /p HAVE=<"%STAMP%"

if /i "!HAVE!"=="!WANT!" (
    echo [OK] Dependencies are up to date
) else (
    echo [..] Installing dependencies - the first run downloads several GB
    "%VPY%" -m pip install --upgrade pip setuptools wheel
    if errorlevel 1 goto :pip_failed
    "%VPY%" -m pip install torch==%TORCH_VERSION% torchaudio==%TORCHAUDIO_VERSION% --index-url https://download.pytorch.org/whl/!TORCH_INDEX!
    if errorlevel 1 goto :pip_failed
    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :pip_failed
    > "%STAMP%" echo !WANT!
)

"%VPY%" -c "import torch; print('[OK] torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
if errorlevel 1 (
    echo [ERROR] PyTorch failed to import. Delete %VENV_DIR% and run this script again.
    exit /b 1
)

rem ---------------------------------------------------------------- 5. .env
if not exist ".env" (
    copy /y ".env.example" ".env" >nul
    echo [NOTE] Created .env from .env.example.
    echo      Set HF_TOKEN and OPENROUTER_API_KEY ^(or TRANSLATION_ENGINE=ollama^),
    echo      then run this script again.
    exit /b 0
)

rem ---------------------------------------------------------------- 6. web UI
if not exist "frontend\dist\index.html" (
    where npm >nul 2>&1
    if errorlevel 1 (
        echo [WARN] frontend\dist is missing and Node.js is not installed; only the API will be available.
        echo        Install Node.js 20+ ^(winget install --id OpenJS.NodeJS.LTS -e^) and run this script again.
    ) else (
        echo [..] Building the web UI
        pushd frontend
        call npm ci --no-audit --no-fund
        if not errorlevel 1 call npm run build
        popd
        if not exist "frontend\dist\index.html" echo [WARN] Web UI build failed; only the API will be available.
    )
)

rem ---------------------------------------------------------------- 7. run
if "%DUB_SKIP_RUN%"=="1" (
    echo [OK] Setup complete. DUB_SKIP_RUN=1, not starting the server.
    exit /b 0
)
echo [..] Starting server. Web UI: http://127.0.0.1:8000/   API docs: http://127.0.0.1:8000/docs  ^(Ctrl+C to stop^)
"%VPY%" -m backend.app.main
exit /b %errorlevel%

:no_ffmpeg
echo [ERROR] ffmpeg / ffprobe not found on PATH. Install a full build, then open a NEW terminal:
echo         winget install --id Gyan.FFmpeg -e
exit /b 1

:pip_failed
echo [ERROR] pip install failed. Fix the error above and run this script again.
exit /b 1
