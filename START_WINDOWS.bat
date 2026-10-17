@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 app.py
) else (
    python app.py
)
if errorlevel 1 (
    echo.
    echo Install Python 3.10 or newer from python.org with Tcl/Tk and Add Python to PATH enabled.
    echo See README.md for setup and troubleshooting.
    pause
)
endlocal
