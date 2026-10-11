@echo off
rem Double-click to start Inside the Transformer (see start.py for what it does).
title Inside the Transformer
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" start.py %*
) else (
    where py >nul 2>nul && (py -3 start.py %*) || (python start.py %*)
)
if errorlevel 1 pause
