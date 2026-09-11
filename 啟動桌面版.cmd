@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_desktop.ps1"
if errorlevel 1 pause
