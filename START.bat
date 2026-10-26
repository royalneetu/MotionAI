
@echo off
cd /d "%~dp0"

start "HAR Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul

cd /d "%~dp0frontend"
start "HAR Frontend" cmd /k "npm.cmd run dev"

timeout /t 5 /nobreak >nul
start "" "http://localhost:5175"

exit
