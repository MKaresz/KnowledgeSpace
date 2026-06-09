@echo off
cd /d %~dp0

call venv\Scripts\activate.bat

start "" python knowledge_space.py

timeout /t 5 >nul

start http://127.0.0.1:7860

