# Changelog

All notable changes to this project are documented here.

## [0.3.0] - 2026-04-20

### Added

- persistent pipeline artifacts for input, media, transcript, windows, extraction batches, and final timelines
- resumable reuse of transcript, window, and extraction batch artifacts
- filesystem-backed web jobs with status polling, result retrieval, and persisted failure details
- stage-level progress reporting across pipeline and web job execution
- source segment IDs and source timestamp grounding on extracted events
- conservative duplicate merging and low-signal event filtering
- adaptive extraction batch planning for longer YouTube videos
- reviewer-friendly web timeline rendering with a raw JSON toggle
- contributor and security docs

### Changed

- narrowed the project around a YouTube-first event timeline workflow
- improved extraction prompting to favor meaningful review events over transcript-like utterance lists
- normalized event-type drift from model output instead of failing on unknown labels
- aligned source-reference validation with line-anchored transcript timestamps
- refreshed README and roadmap documentation to match the current product direction

### Fixed

- metadata now reports the actual transcription backend and model
- long-running jobs now preserve clearer status and error context
- grounded events are no longer dropped because of transcript line/end-time mismatch

### Verification

- full local suite passing: `137 passed`

## [0.2.0] - 2026-04-13

### Added

- optional Groq transcription backend
- optional memories.ai transcription backend
- optional speaker-aware chunking
- optional visual frame analysis path

### Fixed

- Whisper language handling for short clips
- Windows OpenMP runtime startup issue
