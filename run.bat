@echo off
cd /d "%~dp0"
echo Starting 3D Asset Generator (hot-reload enabled)...
echo.
echo File changes will auto-restart the server — just refresh your browser.
echo Press Ctrl+C to stop.
echo.
gradio app.py
pause
