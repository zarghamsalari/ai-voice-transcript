@echo off
REM Double-click to launch the AI Voice Transcript GUI.
REM Uses pythonw.exe so no console window appears beside the app.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" "%~dp0app.py"
