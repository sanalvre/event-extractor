# Event Timeline Extractor

Turn long YouTube videos into grounded event timelines for review.

This project is a focused pipeline for:

- downloading a YouTube video
- extracting audio
- transcribing it
- chunking the transcript into reviewable windows
- generating a structured timeline with timestamped evidence

It ships both a CLI and a localhost web UI.

## Project focus

The current product boundary is intentionally narrow:

- primary input: YouTube URLs
- primary output: structured event timelines
- primary use cases: QA, coaching, research review, and incident-style playback

The goal is not generic summarization. The goal is a timeline a human can scan and trace back to source transcript segments.

## Features

- YouTube-first ingestion with `yt-dlp`
- local transcription by default with `faster-whisper`
- optional Groq and memories.ai transcription adapters
- grounded events with source segment IDs and source timestamps
- artifact persistence for long runs
- batch-based extraction for larger videos
- resumable reuse of transcript, window, and extraction artifacts
- local FastAPI UI with job status polling
- reviewer-friendly timeline view plus raw JSON output

## Pipeline

```text
YouTube URL
  -> yt-dlp download
  -> ffmpeg audio extraction
  -> transcription
  -> transcript windows
  -> OpenRouter timeline extraction
  -> validation + post-processing
  -> JSON timeline
```

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
- project dependencies installed
- `OPENROUTER_API_KEY` for non-dry timeline extraction

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in only the credentials you need.

## Quick start

### CLI

```bash
ete run --url "https://www.youtube.com/watch?v=..." -o timeline.json
```

Short test run:

```bash
ete run --url "https://www.youtube.com/watch?v=..." --max-seconds 60 -o timeline.json
```

Dry run without calling OpenRouter:

```powershell
$env:ETE_USE_STUB = "1"
$env:ETE_TRANSCRIBER = "stub"
ete run --url "https://www.youtube.com/watch?v=..." --dry-run
```

### Web UI

```bash
python serve.py
```

Then open:

```text
http://127.0.0.1:8765/
```

The web app runs on localhost by default and exposes:

- `POST /api/jobs`
- `POST /api/jobs/dry-run`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/result`
- `GET /api/health`

## Configuration

Common settings:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`
- `ETE_TRANSCRIBER`
- `ETE_VALIDATE_EVIDENCE`
- `ETE_EXTRACTION_BATCH_SIZE`
- `ETE_EXTRACTION_MAX_BATCHES`
- `ETE_EXTRACTION_MAX_BATCH_SIZE`

Optional transcription backends:

- `GROQ_API_KEY`
- `GROQ_MODEL`
- `MEMORIES_API_KEY`
- `MEMORIES_TRANSCRIPTION_SPEAKER`

Optional extras:

- diarization: `pip install -e ".[diarize]"`
- vision: `pip install -e ".[vision]"`

## Output shape

Each event can include:

- `time`
- `event`
- `event_type`
- `confidence`
- `evidence`
- `source_segment_ids`
- `source_start`
- `source_end`

Result metadata can include:

- input provenance
- ASR backend/model
- batch plan
- validation stats
- post-processing stats
- warnings
- artifacts directory

## Reliability notes

- long runs persist artifacts under the work directory
- transcript and window stages can be reused on rerun
- extraction batches are saved individually and can be reused
- the web UI tracks stage progress and failure details in job status files

## Development

Run checks:

```bash
ruff check src tests
.venv\Scripts\python.exe -m pytest
```

CI runs on GitHub Actions with `ruff` and `pytest`.

## Open source hygiene

- local secrets belong in `.env`
- local test artifacts belong in ignored folders such as `.ete_pilot/`
- do not commit API keys, generated media, or local work directories

See:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [docs/youtube-timeline-roadmap.md](docs/youtube-timeline-roadmap.md)

## License

MIT
