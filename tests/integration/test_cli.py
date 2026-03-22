from unittest.mock import patch

from typer.testing import CliRunner

from event_timeline_extractor.cli import app

runner = CliRunner()


def test_version() -> None:
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0
    assert r.output.strip()


def test_run_requires_single_input() -> None:
    r = runner.invoke(app, ["run"])
    assert r.exit_code == 2


@patch("event_timeline_extractor.cli.run_pipeline")
def test_run_file_mocked(mock_run, tmp_path) -> None:
    from event_timeline_extractor.schema import TimelineEvent, TimelineResult

    mock_run.return_value = TimelineResult(events=[TimelineEvent(time="00:00", event="x")])

    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"fake")

    r = runner.invoke(
        app,
        ["run", "--file", str(vid), "--dry-run", "--work-dir", str(tmp_path / "w")],
    )
    assert r.exit_code == 0
    assert "x" in r.output
