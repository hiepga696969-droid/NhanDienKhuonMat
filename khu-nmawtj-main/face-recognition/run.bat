@echo off
rem ============================================================
rem  Employee Attendance App - Quick launcher (double-click to run)
rem  ASCII-only file: cmd.exe cannot parse UTF-8 characters.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

rem Fix Qt plugin path (project path contains spaces)
set "QT_PLUGIN_PATH=%~dp0.venv\Lib\site-packages\PySide6\plugins"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Cannot find .venv\Scripts\python.exe
    echo Please check that the venv folder exists.
    pause
    exit /b 1
)

rem pyodbc is required for the SQL Server backend - install once if missing
.venv\Scripts\python.exe -c "import pyodbc" >nul 2>&1
if errorlevel 1 (
    echo pyodbc not found - installing ^(one time, needs internet^)...
    .venv\Scripts\python.exe -m pip install pyodbc
    if errorlevel 1 (
        echo [ERROR] Failed to install pyodbc. Install manually with:
        echo   .venv\Scripts\python.exe -m pip install pyodbc
        pause
        exit /b 1
    )
)

echo Starting Employee Attendance App...
.venv\Scripts\python.exe -m app.main
echo.
echo App closed. Press any key to exit.
pause
