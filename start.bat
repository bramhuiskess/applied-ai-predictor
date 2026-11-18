@echo off
rem One-click launcher for Windows: finds (or installs) Python, builds .venv, starts the setup.
rem
rem   start.bat                    guided: asks before installing anything, then offers to open the dashboard
rem   start.bat --yes --no-start   no questions, no dashboard (for scripts and CI)
rem   start.bat --dry-run          print what would happen, change nothing
rem
rem Flags: --yes  --no-start  --minimal (no training libraries)  --extras LIST  --dry-run  --help
rem Exit codes: 0 ok, 1 no usable Python and none installed, 2 an install step failed.
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "MIN_MINOR=11"
set "PREFERRED=3.12"
if exist .python-version set /p PREFERRED=<.python-version
set "UV_URL=https://astral.sh/uv/install.ps1"
set "YES=0"
set "NO_START=0"
set "DRY=0"
set "EXTRAS=train,datasets"
set "RC=0"

:args
if "%~1"=="" goto args_done
if /i "%~1"=="--yes" set "YES=1"
if /i "%~1"=="-y" set "YES=1"
if /i "%~1"=="--no-start" set "NO_START=1"
if /i "%~1"=="--minimal" set "EXTRAS="
if /i "%~1"=="--dry-run" set "DRY=1"
if /i "%~1"=="--extras" (
  set "EXTRAS=%~2"
  shift
)
if /i "%~1"=="--help" goto help
if /i "%~1"=="-h" goto help
shift
goto args
:args_done

rem ---------------------------------------------------------------- 1. find a Python
rem A candidate only counts when it really runs and is new enough (this also skips the Microsoft Store stub).
set "PY="
call :try "py -%PREFERRED%"
call :try "py -3.13"
call :try "py -3.12"
call :try "py -3.11"
call :try "python3"
call :try "python"

set "UV="
where uv >nul 2>nul && set "UV=uv"
if not defined UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"

set "USE_UV=0"
if defined PY (
  for /f "delims=" %%V in ('%PY% --version 2^>^&1') do echo Python: %%V ^(%PY%^)
  goto venv
)

echo No Python 3.%MIN_MINOR% or newer found.
call :ask "Install Python %PREFERRED% for this project with uv - no admin rights, no system-wide changes?"
if errorlevel 1 goto manual_help
set "USE_UV=1"
if defined UV goto venv
echo uv is a small tool by Astral that can download Python. It will be installed in %USERPROFILE%\.local\bin
echo from the official installer: %UV_URL%
if "%DRY%"=="1" (
  echo [dry-run] would: download and run %UV_URL%, then uv venv --python %PREFERRED% .venv
  set "UV=%USERPROFILE%\.local\bin\uv.exe"
  goto venv
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_NO_MODIFY_PATH='1'; $env:UV_INSTALL_DIR='%USERPROFILE%\.local\bin'; irm %UV_URL% | iex"
if errorlevel 1 goto uv_failed
rem This window's PATH does not know uv yet, so call it by its full path.
set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not exist "%UV%" goto uv_failed

rem ---------------------------------------------------------------- 2. the virtual environment
:venv
set "NEW_VENV=0"
if not exist .venv set "NEW_VENV=1"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys; sys.exit(0 if sys.version_info >= (3, %MIN_MINOR%) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo .venv was built with an older Python.
    call :ask "Recreate .venv?"
    if errorlevel 1 (
      set "RC=1"
      goto finish
    )
    set "NEW_VENV=1"
    if "%DRY%"=="1" (echo [dry-run] would: remove .venv and create it again) else rmdir /s /q .venv
  )
)

set "SPEC=.[%EXTRAS%]"
if "%EXTRAS%"=="" set "SPEC=."
set "UP_TO_DATE=0"
if exist .venv\Scripts\predictor.exe if exist .venv\.pyproject.stamp if exist .venv\.extras.stamp (
  fc /b pyproject.toml .venv\.pyproject.stamp >nul 2>nul
  if not errorlevel 1 (
    set "OLD_EXTRAS="
    set /p OLD_EXTRAS=<.venv\.extras.stamp
    if "!OLD_EXTRAS!"=="%EXTRAS%" set "UP_TO_DATE=1"
  )
)
if "%UP_TO_DATE%"=="1" (
  echo Environment: .venv is already installed and up to date.
  goto start
)

if "%NEW_VENV%"=="1" (
  if "%USE_UV%"=="1" (
    echo [create] .venv with Python %PREFERRED% ^(uv^)
    if not "%DRY%"=="1" "%UV%" venv --python %PREFERRED% .venv || goto install_failed
  ) else (
    echo [create] .venv with %PY%
    if not "%DRY%"=="1" %PY% -m venv .venv || goto install_failed
  )
)
echo Installing the project (%SPEC%) ...
if "%DRY%"=="1" (
  echo [dry-run] would: pip install -e "%SPEC%" into .venv
  goto start
)
.venv\Scripts\python.exe -m pip --version >nul 2>nul
if not errorlevel 1 (
  .venv\Scripts\python.exe -m pip install --quiet --upgrade pip || goto install_failed
  .venv\Scripts\python.exe -m pip install -e "%SPEC%" || goto install_failed
) else if defined UV (
  "%UV%" pip install --python .venv\Scripts\python.exe -e "%SPEC%" || goto install_failed
) else (
  echo This .venv has no pip and uv is not installed.
  goto install_failed
)
copy /y pyproject.toml .venv\.pyproject.stamp >nul
> .venv\.extras.stamp echo(%EXTRAS%

rem ---------------------------------------------------------------- 3. start
:start
if "%NO_START%"=="1" (
  echo Done. Next: .venv\Scripts\activate ^&^& predictor setup
  goto finish
)
call :ask "Open the dashboard now - this starts the guided setup in your browser?"
if errorlevel 1 (
  echo.
  echo Whenever you want it:
  echo   .venv\Scripts\activate
  echo   predictor setup        ^(guided, in the browser; add --cli for the terminal^)
  echo   predictor run --open   ^(straight to the dashboard once setup is done^)
  goto finish
)
if "%DRY%"=="1" (
  echo [dry-run] would: run .venv\Scripts\predictor.exe setup
  goto finish
)
.venv\Scripts\predictor.exe setup
goto finish

rem ---------------------------------------------------------------- helpers and exits
:try
if defined PY exit /b 0
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3, %MIN_MINOR%) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=%~1"
exit /b 0

:ask
if "%DRY%"=="1" (
  echo [dry-run] would ask: %~1
  exit /b 0
)
if "%YES%"=="1" exit /b 0
set "ANS="
set /p "ANS=%~1 [Y/n] "
if /i "!ANS!"=="n" exit /b 1
if /i "!ANS!"=="no" exit /b 1
exit /b 0

:manual_help
echo.
echo Install Python %PREFERRED% yourself, then run this file again:
echo   Windows:  winget install Python.Python.%PREFERRED%   ^(or https://www.python.org/downloads/^)
set "RC=1"
goto finish

:uv_failed
echo Could not install uv ^(offline, or blocked?^).
echo.
echo Install Python %PREFERRED% yourself, then run this file again:
echo   Windows:  winget install Python.Python.%PREFERRED%   ^(or https://www.python.org/downloads/^)
set "RC=2"
goto finish

:install_failed
echo.
echo An install step failed; see the messages above.
set "RC=2"
goto finish

:help
echo One-click launcher: finds or installs Python, builds .venv, starts the setup.
echo Flags: --yes  --no-start  --minimal  --extras LIST  --dry-run  --help
echo Exit codes: 0 ok, 1 no usable Python and none installed, 2 an install step failed.
goto finish

:finish
rem Keep the window open when it was started by a double-click, except in scripted use.
if not "%YES%"=="1" pause
endlocal & exit /b %RC%
