@echo off
cd /d "%~dp0"
title Event Timeline Extractor
echo.
echo  Starting http://127.0.0.1:8766  (keep this window open)
echo  Close this window or press Ctrl+C to stop the server.
echo.
python -m uvicorn event_timeline_extractor.web.app:app --host 127.0.0.1 --port 8766
if errorlevel 1 pause
