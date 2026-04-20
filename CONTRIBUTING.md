# Contributing

Thanks for taking a look at Event Timeline Extractor.

## Scope

This project is intentionally narrow:

- primary input: YouTube URLs
- primary output: grounded event timelines for review
- primary quality bar: traceable, timestamped, reviewable events

Changes that improve long-video reliability, grounding, test coverage, and reviewer-facing output are especially welcome.

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Install `ffmpeg` and `ffprobe`, and make sure both are available on `PATH`.

## Running checks

```bash
ruff check src tests
.venv\Scripts\python.exe -m pytest
```

## Before opening a PR

- keep the YouTube-first product boundary intact unless you are intentionally changing project scope
- add or update tests for behavior changes
- do not commit `.env`, API keys, or local run artifacts
- prefer small, focused PRs with a short explanation of user-facing impact

## Notes on secrets and artifacts

- use `.env` for local credentials
- keep local work artifacts under ignored folders such as `.ete_pilot/`
- if you run live extraction tests, make sure generated artifacts are not staged

## Good areas for contribution

- extraction quality and grounding
- better batch planning and resumability
- clearer reviewer-facing output
- docs, examples, and troubleshooting
