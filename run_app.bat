@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating local Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -c "import streamlit, pandas, plotly" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install required packages.
        pause
        exit /b 1
    )
)

echo Stopping any previous instances of this app...
powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*health_status_analyzer*' } | Stop-Process -Force -ErrorAction SilentlyContinue"

echo Starting Samsung health record analyzer...
echo.
echo Open this URL in your browser:
echo http://localhost:8501
echo.
echo (If the page looks stale, hard-refresh with Ctrl+Shift+R)
echo.

".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

endlocal
