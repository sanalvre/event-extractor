# Stable local web server (no --reload) so long /api/timeline requests are not cut off mid-job.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m uvicorn event_timeline_extractor.web.app:app --host 127.0.0.1 --port 8766
