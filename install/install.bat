@echo off
REM HPVsim installer for Windows.
REM Installs Miniforge (if needed) and pip-installs hpvsim + jupyter.
REM
REM Usage: double-click, or run from Command Prompt:
REM    install.bat

setlocal enabledelayedexpansion

echo === HPVsim installer ===

set "MINIFORGE_DIR=%USERPROFILE%\miniforge3"
set "CONDA_EXE="

REM Prefer conda already on PATH, otherwise look in the default Miniforge location.
where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    for /f "delims=" %%i in ('where conda') do (
        if not defined CONDA_EXE set "CONDA_EXE=%%i"
    )
    echo ^>^>^> Found existing conda at: !CONDA_EXE!
) else if exist "%MINIFORGE_DIR%\Scripts\conda.exe" (
    set "CONDA_EXE=%MINIFORGE_DIR%\Scripts\conda.exe"
    echo ^>^>^> Found existing Miniforge at: %MINIFORGE_DIR%
) else (
    call :install_miniforge
    if errorlevel 1 exit /b 1
    set "CONDA_EXE=%MINIFORGE_DIR%\Scripts\conda.exe"
)

echo ^>^>^> Installing hpvsim and jupyter into the base conda env
"!CONDA_EXE!" run --no-capture-output -n base pip install --upgrade pip
if errorlevel 1 exit /b 1
"!CONDA_EXE!" run --no-capture-output -n base pip install hpvsim jupyter
if errorlevel 1 exit /b 1

echo.
echo === Done ===
echo Open a new Command Prompt and verify with:
echo     python -c "import hpvsim; print(hpvsim.__version__)"
echo.
echo Launch Jupyter with:
echo     jupyter notebook
echo.
pause
exit /b 0

:install_miniforge
set "INSTALLER=%TEMP%\Miniforge3-Windows-x86_64.exe"
set "URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
echo ^>^>^> Downloading %URL%
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%URL%' -OutFile '%INSTALLER%'"
if errorlevel 1 exit /b 1
echo ^>^>^> Installing Miniforge to %MINIFORGE_DIR%
start /wait "" "%INSTALLER%" /InstallationType=JustMe /RegisterPython=1 /AddToPath=1 /S /D=%MINIFORGE_DIR%
if errorlevel 1 exit /b 1
del "%INSTALLER%" >nul 2>nul
exit /b 0
