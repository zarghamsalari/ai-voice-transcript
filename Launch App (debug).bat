@echo off
REM Same as "Launch App" but keeps a console window open so you can see errors.
REM Use this one if the regular launcher does nothing when double-clicked.
cd /d "%~dp0"
".venv\Scripts\python.exe" "%~dp0app.py"
pause
