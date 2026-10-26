@echo off
cd /d "%~dp0"
if not exist .venv (py -3.11 -m venv .venv)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r backend\requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
