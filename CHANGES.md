# Changelog

All notable changes to the Event Timeline Extractor are documented here.

---

## [Unreleased]

### Added — memories.ai Visual Frame Analysis (memories-s0)
Local, open-source Vision-Language Model (Apache 2.0) from memories.ai.
Runs fully on-device via HuggingFace — **no API key required**.

- **`src/event_timeline_extractor/vision/memories_s0.py`** — new module
  - `MemoriesS0Analyzer` class with lazy model loading and CUDA auto-detection
  - `FrameDescription` dataclass (`timestamp`, `description`)
  - CPU fallback with runtime warning when no GPU available
  - Clear `ImportError` with install hint when `[vision]` extras absent
- **`pyproject.toml`** — added `[vision]` optional dependency group (`torch>=2.0`, `transformers>=4.37.0`, `Pillow>=10.0`)
- **`src/event_timeline_extractor/config.py`** — added `ETE_VISION_ENABLED` (default `false`) and `ETE_VISION_FRAME_INTERVAL` (default `10`)
- **`src/event_timeline_extractor/chunking.py`** — added `vision_context: str = ""` field to `TimeWindow`; `vision_map` param and `_vision_context_in_range()` helper to `chunk_segments()`
- **`src/event_timeline_extractor/pipeline.py`** — added `_run_vision_analysis()` helper; conditionally extracts frames and runs memories-s0
- **`src/event_timeline_extractor/llm/openrouter.py`** — injects `[VISUAL CONTEXT]` blocks into the LLM prompt when `vision_context` is non-empty

**Usage:**
```bash
pip install -e ".[vision]"
# .env
ETE_VISION_ENABLED=true
ETE_VISION_FRAME_INTERVAL=10
```

---

### Added — memories.ai Cloud Transcription + Built-in Diarization
Replaces local Whisper + pyannote with a single memories.ai API call.
Requires `MEMORIES_API_KEY` (free tier at api-platform.memories.ai).

- **`src/event_timeline_extractor/transcription/memories_backend.py`** — new module
  - `MemoriesTranscriber` implements the `Transcriber` protocol
  - Two-step flow: `POST /upload` → `POST /transcriptions/sync-generate-audio`
  - Returns segments with speaker labels (`SPEAKER_00`, `SPEAKER_01`, …) built-in — no pyannote needed
- **`src/event_timeline_extractor/config.py`** — added `MEMORIES_API_KEY`, `MEMORIES_TRANSCRIPTION_SPEAKER` (default `true`), `memories_key_plain()`
- **`src/event_timeline_extractor/transcription/factory.py`** — added `"memories"` case

**Usage:**
```bash
# .env
ETE_TRANSCRIBER=memories
MEMORIES_API_KEY=sk-mai-...
```

---

### Added — Speaker-Aware (Scene-Aware) Chunking
When the memories.ai transcriber provides speaker labels, the pipeline automatically
breaks LLM windows at speaker turns instead of fixed 20-second boundaries.

- **`src/event_timeline_extractor/chunking.py`** — added `speaker_aware: bool = False` to `chunk_segments()`. Breaks at speaker changes; long monologues still split at `window_sec`.
- **`src/event_timeline_extractor/pipeline.py`** — auto-enables `speaker_aware=True` when `ETE_TRANSCRIBER=memories` + speaker labels are present. Exposed as `speaker_aware_chunking` in result metadata.

---

### Added — Groq Cloud Transcription Backend
Whisper Large v3 on Groq hardware at ~189× real-time speed. Free tier. No new Python packages needed.

- **`src/event_timeline_extractor/transcription/groq_backend.py`** — new module
  - `GroqTranscriber` implements the `Transcriber` protocol
  - File size guard: raises a clear error with `--max-seconds` hint for audio > 25 MB
- **`src/event_timeline_extractor/config.py`** — added `GROQ_API_KEY`, `GROQ_MODEL` (default `whisper-large-v3`), `GROQ_BASE_URL`
- **`src/event_timeline_extractor/transcription/factory.py`** — added `"groq"` case

**Usage:**
```bash
# .env
ETE_TRANSCRIBER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=whisper-large-v3          # or whisper-large-v3-turbo
```

---

### Fixed

#### English language forced on Whisper (`ETE_WHISPER_LANGUAGE`)
The `small` Whisper model can mis-detect language on short clips (outputting e.g. Thai for English speech).

- **`src/event_timeline_extractor/config.py`** — added `ETE_WHISPER_LANGUAGE: str | None = "en"`. Set to empty for auto-detection.
- **`src/event_timeline_extractor/transcription/faster_whisper_backend.py`** — passes `language` kwarg to `model.transcribe()`

#### OpenMP runtime crash on Windows (`KMP_DUPLICATE_LIB_OK`)
Server crashed silently (`OMP Error #15`) when CTranslate2 and another OpenMP runtime (PyTorch/MKL) were both loaded.

- **`src/event_timeline_extractor/transcription/faster_whisper_backend.py`** — sets `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` before the first faster-whisper import
- **`serve.py`** — convenience launcher that sets the flag before uvicorn starts: `python serve.py`

---

### Changed

- **`pipeline.py`** result `meta` now includes: `transcriber`, `speaker_aware_chunking`, `vision`
- **`transcription/factory.py`** docstring updated to list all supported `ETE_TRANSCRIBER` values

---

### Tests added (106 total, all passing)

| File | What it covers |
|---|---|
| `tests/unit/test_vision_memories_s0.py` | `MemoriesS0Analyzer` device resolution, empty inputs, mocked inference, import error path |
| `tests/unit/test_chunking_vision.py` | `vision_context` field, `_vision_context_in_range`, `chunk_segments` with `vision_map` |
| `tests/unit/test_config_vision.py` | Vision settings defaults and env-file loading |
| `tests/unit/test_openrouter_vision.py` | `[VISUAL CONTEXT]` injection/omission in LLM prompt |
| `tests/integration/test_pipeline_vision.py` | Vision disabled/enabled; meta field; context flows into windows |
| `tests/unit/test_groq_backend.py` | Construction, file size guard, segment parsing, HTTP mocks |
| `tests/unit/test_memories_backend.py` | Construction, `_to_segments`, error handling, upload+transcribe flow |
| `tests/unit/test_chunking_speaker_aware.py` | Speaker splits, same-speaker merging, time-limit override, None-speaker fallback |
| `tests/unit/test_config_vision.py` | Vision config defaults, env-file loading |

---

### Environment variables reference

| Variable | Default | Used by |
|---|---|---|
| `ETE_TRANSCRIBER` | `faster_whisper` | All backends |
| `ETE_WHISPER_LANGUAGE` | `en` | faster_whisper |
| `ETE_WHISPER_MODEL_SIZE` | `small` | faster_whisper |
| `MEMORIES_API_KEY` | — | memories transcription + (future) vision API |
| `MEMORIES_TRANSCRIPTION_SPEAKER` | `true` | memories |
| `GROQ_API_KEY` | — | groq |
| `GROQ_MODEL` | `whisper-large-v3` | groq |
| `ETE_VISION_ENABLED` | `false` | memories-s0 vision |
| `ETE_VISION_FRAME_INTERVAL` | `10` | memories-s0 vision |
| `OPENROUTER_API_KEY` | — | LLM synthesis |
| `OPENROUTER_MODEL` | `deepseek/deepseek-chat` | LLM synthesis |
