"""CLI entrypoint."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

from event_timeline_extractor import __version__
from event_timeline_extractor.config import load_settings
from event_timeline_extractor.pipeline import PipelineInput, run_pipeline

app = typer.Typer(
    name="ete",
    help="Event Timeline Extractor - grounded timelines from YouTube videos.",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    _setup_logging(verbose)


@app.command("run")
def run_cmd(
    url: str | None = typer.Option(None, "--url", help="YouTube URL."),
    file: Path | None = typer.Option(
        None,
        "--file",
        help="Local video/audio file for developer workflows.",
    ),
    work_dir: Path = typer.Option(
        Path(".ete_work"),
        "--work-dir",
        help="Working directory for downloads and extracted audio.",
    ),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write JSON to this file."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Skip OpenRouter and emit placeholder events from transcript windows.",
    ),
    window_sec: float = typer.Option(
        20.0,
        "--window-sec",
        help="Transcript chunk size for the LLM in seconds.",
    ),
    max_minutes: float | None = typer.Option(
        None,
        "--max-minutes",
        help="Reject media longer than this many minutes.",
    ),
    max_seconds: float | None = typer.Option(
        None,
        "--max-seconds",
        help="Process only the first N seconds of the source.",
    ),
) -> None:
    """Run the full timeline extraction pipeline."""
    if bool(url) == bool(file):
        typer.echo("Specify exactly one of --url or --file.", err=True)
        raise typer.Exit(code=2)

    settings = load_settings()
    inp = PipelineInput(
        youtube_url=url,
        file_path=file,
    )
    try:
        result = run_pipeline(
            inp,
            work_dir=work_dir,
            settings=settings,
            window_sec=window_sec,
            max_minutes=max_minutes,
            max_seconds=max_seconds,
            dry_run=dry_run,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    text = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
    if out:
        out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")


@app.command("version")
def version_cmd() -> None:
    typer.echo(__version__)


def cli_entry() -> None:
    app()


if __name__ == "__main__":
    app()
