# YouTube Timeline Roadmap

This document turns the current repo audit into an execution plan.

Target product direction:

> Extract a reliable, reviewable event timeline from long YouTube videos.

The plan below narrows scope, improves reliability for long-running jobs, and tightens timeline quality so outputs are easier to trust in QA, review, coaching, and incident analysis workflows.

## Product Boundary

Primary supported input:
- YouTube URLs

Primary supported output:
- Structured timeline JSON grounded in transcript evidence

Non-goals for the focused product path:
- Local-file-first positioning
- Equal emphasis on all transcription backends
- Experimental features presented as core functionality
- Single huge request flows for long videos

## Milestones

### Milestone 1: Narrow the Product

Goal:
- Make the codebase, docs, and UX tell one story: YouTube in, grounded timeline out.

Definition of done:
- README, UI copy, and CLI help align around YouTube timeline extraction.
- The main pipeline is organized into clear stages.
- Metadata accurately reflects what backend/model actually ran.
- Experimental features are clearly isolated from the core flow.

Backlog:

#### M1-01 Narrow README and repo positioning
Scope:
- Rewrite the README around YouTube timeline extraction.
- Move local files and experimental features into secondary sections.

Acceptance criteria:
- Opening paragraphs describe YouTube as the main supported input.
- Long-video processing and timeline grounding are central themes.
- Experimental paths are clearly labeled.

Likely files:
- `README.md`

#### M1-02 Update web UI copy and API framing
Scope:
- Make the web interface clearly about YouTube submission and timeline extraction.
- Remove mixed messaging that implies broader input support.

Acceptance criteria:
- UI text consistently says YouTube URL.
- API docs and route comments match the narrowed product.

Likely files:
- `src/event_timeline_extractor/web/static/index.html`
- `src/event_timeline_extractor/web/static/app.js`
- `src/event_timeline_extractor/web/app.py`

#### M1-03 Simplify CLI help and defaults
Scope:
- Keep the CLI usable, but make its help text reflect the YouTube-first product.
- Decide whether local file support stays as internal/developer mode or is deprecated.

Acceptance criteria:
- Help output emphasizes YouTube.
- If local files remain, they are clearly marked non-core.

Likely files:
- `src/event_timeline_extractor/cli.py`

#### M1-04 Refactor pipeline into explicit stages
Scope:
- Break `pipeline.py` into clearer stage-oriented units.
- Introduce typed handoff objects between stages.

Acceptance criteria:
- Stages are individually testable.
- Pipeline orchestration is easier to follow and extend.

Likely files:
- `src/event_timeline_extractor/pipeline.py`
- New modules such as:
  - `src/event_timeline_extractor/pipeline_models.py`
  - `src/event_timeline_extractor/stages/download.py`
  - `src/event_timeline_extractor/stages/media.py`
  - `src/event_timeline_extractor/stages/transcription.py`
  - `src/event_timeline_extractor/stages/extraction.py`

#### M1-05 Introduce typed domain models for pipeline state
Scope:
- Define stable models for media source, transcript, windows, and run metadata.

Acceptance criteria:
- Stage boundaries pass typed models instead of loose dict-like assumptions.
- Metadata and artifact naming are more consistent.

Likely files:
- `src/event_timeline_extractor/schema.py`
- New typed-model module(s)

#### M1-06 Normalize metadata and provenance
Scope:
- Make `meta` accurately describe the actual run.
- Avoid Whisper-only metadata on non-Whisper runs.

Acceptance criteria:
- `meta` reports actual transcriber/backend/model details.
- Optional features appear only when used.

Likely files:
- `src/event_timeline_extractor/pipeline.py`
- `src/event_timeline_extractor/transcription/factory.py`
- `src/event_timeline_extractor/transcription/faster_whisper_backend.py`
- `src/event_timeline_extractor/transcription/groq_backend.py`
- `src/event_timeline_extractor/transcription/memories_backend.py`

#### M1-07 Improve stage-specific error handling
Scope:
- Replace broad exception treatment with clearer error classes and messages.

Acceptance criteria:
- Failures are categorized by stage.
- Web/API messages remain safe but more useful.

Likely files:
- `src/event_timeline_extractor/web/app.py`
- `src/event_timeline_extractor/pipeline.py`
- Potential new `errors.py`

#### M1-08 Isolate experimental features
Scope:
- Mark vision and similar optional enrichments as experimental.
- Keep them out of the critical path and primary messaging.

Acceptance criteria:
- Vision does not appear as a core promise in the README/UI.
- Experimental code paths are isolated by docs and module structure.

Likely files:
- `README.md`
- `src/event_timeline_extractor/pipeline.py`
- `src/event_timeline_extractor/vision/*`

#### M1-09 Decide the fate of local file support
Scope:
- Choose one of:
  - remove from core product
  - keep as internal/developer mode
  - fully deprecate

Acceptance criteria:
- One documented decision.
- No ambiguous product messaging remains.

Likely files:
- `README.md`
- `src/event_timeline_extractor/cli.py`
- `src/event_timeline_extractor/pipeline.py`

### Milestone 2: Long-Video Reliability

Goal:
- Process long YouTube videos incrementally, persist progress, and resume failed jobs without starting over.

Definition of done:
- Runs have persistent job folders or artifact directories.
- Long runs checkpoint progress across stages.
- Re-running a failed job resumes from checkpoints.
- The web flow is job-based instead of one blocking request.

Backlog:

#### M2-01 Add persistent job/artifact directory model
Scope:
- Introduce a stable work directory keyed by job ID or video ID.
- Separate transient temp files from resumable artifacts.

Acceptance criteria:
- Each run has a predictable artifact directory.
- Artifacts are named and organized consistently.

Likely files:
- New `src/event_timeline_extractor/jobs.py`
- `src/event_timeline_extractor/pipeline.py`
- `src/event_timeline_extractor/web/app.py`

#### M2-02 Persist stage outputs
Scope:
- Save download metadata, probed duration, transcript, windows, and final results.

Acceptance criteria:
- A completed or partial run leaves enough state to inspect and resume.

Likely files:
- New artifact IO module(s)
- `src/event_timeline_extractor/pipeline.py`

#### M2-03 Add checkpoint after transcription
Scope:
- Save transcript and transcript metadata before LLM extraction.

Acceptance criteria:
- Failed extraction does not require re-downloading or re-transcribing.

Likely files:
- `src/event_timeline_extractor/pipeline.py`
- New artifact persistence module(s)

#### M2-04 Persist partial LLM extraction results
Scope:
- Save extraction output per batch or per group of windows.

Acceptance criteria:
- A failure partway through extraction preserves earlier completed work.

Likely files:
- `src/event_timeline_extractor/llm/openrouter.py`
- New extraction orchestration module(s)

#### M2-05 Add batched extraction for long transcripts
Scope:
- Prevent giant prompt construction for very long videos.
- Introduce batched window groups and later merge results.

Acceptance criteria:
- Long videos are processed in bounded LLM calls.
- Batch size is configurable and tested.

Likely files:
- `src/event_timeline_extractor/chunking.py`
- `src/event_timeline_extractor/llm/openrouter.py`
- `src/event_timeline_extractor/pipeline.py`

#### M2-06 Add resume logic
Scope:
- Detect existing artifacts and continue from the latest valid stage.

Acceptance criteria:
- Re-running a failed job skips already-complete stages when inputs match.

Likely files:
- `src/event_timeline_extractor/pipeline.py`
- New `jobs.py` or `artifacts.py`

#### M2-07 Introduce job status model
Scope:
- Track stages such as queued, downloading, transcribing, extracting, validating, complete, failed.

Acceptance criteria:
- CLI and web can display current status and progress.

Likely files:
- New `src/event_timeline_extractor/jobs.py`
- `src/event_timeline_extractor/schema.py`

#### M2-08 Replace blocking web processing with job endpoints
Scope:
- Move from blocking POST to submit/status/result endpoints.

Acceptance criteria:
- Web API can start a job and return quickly.
- Clients can poll status and fetch results when ready.

Likely files:
- `src/event_timeline_extractor/web/app.py`
- `src/event_timeline_extractor/web/static/app.js`
- `src/event_timeline_extractor/web/static/index.html`

#### M2-09 Add retry/timeout strategy per dependency
Scope:
- Handle yt-dlp, ffmpeg, ASR, and LLM failures more predictably.

Acceptance criteria:
- Retry policy is explicit.
- Permanent failures are distinguishable from retryable ones.

Likely files:
- `src/event_timeline_extractor/fetch.py`
- `src/event_timeline_extractor/ffmpeg_tools.py`
- `src/event_timeline_extractor/llm/openrouter.py`
- Backend modules under `transcription/`

#### M2-10 Add cleanup and retention rules
Scope:
- Define what artifacts are kept, for how long, and what can be deleted.

Acceptance criteria:
- Artifact growth is controlled.
- Result inspection remains possible after completion.

Likely files:
- New artifact management module(s)
- Docs

### Milestone 3: Timeline Quality and Grounding

Goal:
- Make outputs more specific, more reviewable, and more clearly anchored to transcript evidence.

Definition of done:
- Events are traceable to source segments.
- Duplicate or vague events are reduced.
- Timestamps and evidence are validated more strictly.
- Output quality metadata helps users understand confidence and limitations.

Backlog:

#### M3-01 Add stable segment IDs
Scope:
- Extend transcript segments with IDs that survive chunking and extraction.

Acceptance criteria:
- Each event can reference exact source segments.

Likely files:
- `src/event_timeline_extractor/transcription/base.py`
- `src/event_timeline_extractor/chunking.py`
- Transcriber backends

#### M3-02 Expand timeline schema with source references
Scope:
- Add fields such as `source_segment_ids`, `source_start`, `source_end`, or equivalent.

Acceptance criteria:
- Output supports traceability without guessing.

Likely files:
- `src/event_timeline_extractor/schema.py`
- `src/event_timeline_extractor/llm/openrouter.py`

#### M3-03 Tighten extraction prompts around timeline semantics
Scope:
- Ask for timeline events, not generic summaries.
- Emphasize chronology, distinct events, and evidence grounding.

Acceptance criteria:
- Prompt contract is specific to timeline extraction.
- Tests cover prompt content expectations.

Likely files:
- `src/event_timeline_extractor/llm/openrouter.py`
- `tests/unit/test_openrouter.py`

#### M3-04 Add event typing
Scope:
- Introduce useful categories such as speech, action, transition, incident, or decision.

Acceptance criteria:
- Event typing improves downstream filtering without overcomplicating the schema.

Likely files:
- `src/event_timeline_extractor/schema.py`
- `src/event_timeline_extractor/llm/openrouter.py`

#### M3-05 Add merge/dedup across windows
Scope:
- Consolidate repeated events produced in adjacent batches.

Acceptance criteria:
- Timeline duplication decreases on longer videos.

Likely files:
- New post-processing module
- `src/event_timeline_extractor/pipeline.py`

#### M3-06 Add stricter timestamp grounding checks
Scope:
- Validate that event timestamps map back to known transcript segment times.

Acceptance criteria:
- Unsupported timestamps are dropped or flagged.

Likely files:
- `src/event_timeline_extractor/validation.py`
- `src/event_timeline_extractor/schema.py`

#### M3-07 Strengthen evidence validation
Scope:
- Expand current substring validation into stronger grounding checks where practical.

Acceptance criteria:
- Unsupported evidence is flagged or removed consistently.

Likely files:
- `src/event_timeline_extractor/validation.py`

#### M3-08 Add quality/confidence metadata
Scope:
- Record warnings, grounding gaps, and extraction-quality indicators.

Acceptance criteria:
- `meta` gives reviewers more context about confidence and limitations.

Likely files:
- `src/event_timeline_extractor/schema.py`
- `src/event_timeline_extractor/pipeline.py`
- `src/event_timeline_extractor/validation.py`

#### M3-09 Improve speaker handling for review use cases
Scope:
- Preserve neutral labels consistently and avoid role guessing.
- Ensure chunking and prompt logic stay aligned with available speaker metadata.

Acceptance criteria:
- Speaker-aware outputs are stable when labels are available.

Likely files:
- `src/event_timeline_extractor/chunking.py`
- `src/event_timeline_extractor/llm/openrouter.py`
- `src/event_timeline_extractor/transcription/diarization.py`

#### M3-10 Add reviewer-friendly output modes
Scope:
- Keep JSON as the system output while preparing a cleaner human-readable timeline view later.

Acceptance criteria:
- Structured output remains primary.
- A reviewer-facing rendering path is documented or scaffolded.

Likely files:
- `src/event_timeline_extractor/schema.py`
- `src/event_timeline_extractor/web/static/app.js`
- Docs

## Target Architecture

The current code already has good building blocks. The next step is to make stage boundaries more explicit.

Recommended target structure:

```text
src/event_timeline_extractor/
  cli.py
  config.py
  errors.py
  jobs.py
  schema.py
  pipeline.py
  pipeline_models.py
  artifacts.py
  stages/
    download.py
    media.py
    transcription.py
    chunking.py
    extraction.py
    validation.py
  transcription/
  llm/
  web/
  vision/
```

Recommended orchestration shape:

1. Build job context from YouTube URL
2. Resolve/persist artifact directory
3. Download or reuse downloaded media
4. Probe media duration
5. Extract WAV
6. Transcribe or reuse transcript
7. Chunk transcript into bounded windows/batches
8. Extract events batch-by-batch
9. Merge/dedup/validate
10. Persist final timeline and metadata

## Testing Strategy

Testing should move toward the narrowed product promise.

Add or strengthen tests for:
- YouTube-only input validation
- persistent artifact layout
- resume behavior
- long-video batch extraction
- metadata accuracy by backend
- event grounding and source references
- duplicate suppression across batches
- job-status API behavior

De-emphasize or isolate tests for:
- peripheral or experimental flows presented outside the core product path

## Recommended First Sprint

Start here before building resume logic:

1. M1-01 Narrow README and repo positioning
2. M1-02 Update web UI copy and API framing
3. M1-04 Refactor pipeline into explicit stages
4. M1-06 Normalize metadata and provenance
5. M2-01 Add persistent job/artifact directory model
6. M2-02 Persist stage outputs

Reason:
- This gives the project a clear product shape.
- It improves the codebase before major feature work.
- It creates the base needed for resumability and better extraction quality.

## Suggested Delivery Sequence

### Phase A
- M1-01
- M1-02
- M1-03
- M1-04
- M1-05
- M1-06
- M1-07
- M1-08
- M1-09

### Phase B
- M2-01
- M2-02
- M2-03
- M2-04
- M2-05
- M2-06
- M2-07
- M2-08
- M2-09
- M2-10

### Phase C
- M3-01
- M3-02
- M3-03
- M3-04
- M3-05
- M3-06
- M3-07
- M3-08
- M3-09
- M3-10

## Notes

- "Any size" should be implemented as incremental processing with bounded LLM requests, persistent artifacts, and resume support.
- The focused product can still keep optional backends, but they should no longer distort the main architecture or messaging.
- The strongest long-term differentiator is not just transcription speed; it is trustworthy, reviewable timeline extraction grounded in transcript evidence.
