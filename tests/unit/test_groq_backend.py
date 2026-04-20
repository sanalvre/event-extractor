"""Unit tests for GroqTranscriber."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from event_timeline_extractor.transcription.groq_backend import GroqTranscriber

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_raises_if_api_key_empty() -> None:
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqTranscriber(api_key="")


def test_raises_if_api_key_missing_via_factory() -> None:
    from event_timeline_extractor.config import Settings
    from event_timeline_extractor.transcription.factory import get_transcriber

    s = Settings(ete_transcriber="groq", groq_api_key=None)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_transcriber(s)


def test_factory_returns_groq_transcriber_when_key_set() -> None:
    from event_timeline_extractor.config import Settings
    from event_timeline_extractor.transcription.factory import get_transcriber

    s = Settings(ete_transcriber="groq", groq_api_key="gsk_test")
    t = get_transcriber(s)
    assert isinstance(t, GroqTranscriber)


# ---------------------------------------------------------------------------
# File size guard
# ---------------------------------------------------------------------------


def test_check_file_size_raises_for_oversized_file(tmp_path: Path) -> None:
    big = tmp_path / "big.wav"
    big.write_bytes(b"\x00" * (26 * 1024 * 1024))  # 26 MB > 25 MB limit
    t = GroqTranscriber(api_key="gsk_test")
    with pytest.raises(ValueError, match="25 MB"):
        t._check_file_size(big)


def test_check_file_size_passes_for_small_file(tmp_path: Path) -> None:
    small = tmp_path / "small.wav"
    small.write_bytes(b"\x00" * 1024)
    t = GroqTranscriber(api_key="gsk_test")
    t._check_file_size(small)  # should not raise


# ---------------------------------------------------------------------------
# _parse_segments
# ---------------------------------------------------------------------------


def test_parse_segments_maps_to_transcript_segments() -> None:
    data = {
        "segments": [
            {"start": 0.0, "end": 3.5, "text": " Hello officer."},
            {"start": 3.5, "end": 7.1, "text": " Good morning."},
        ]
    }
    result = GroqTranscriber._parse_segments(data)
    assert len(result) == 2
    assert result[0].start == 0.0
    assert result[0].end == 3.5
    assert result[0].text == "Hello officer."
    assert result[0].speaker is None
    assert result[1].text == "Good morning."


def test_parse_segments_skips_empty_text() -> None:
    data = {"segments": [{"start": 0.0, "end": 1.0, "text": "   "}]}
    assert GroqTranscriber._parse_segments(data) == []


def test_parse_segments_handles_missing_segments_key() -> None:
    assert GroqTranscriber._parse_segments({}) == []


# ---------------------------------------------------------------------------
# transcribe() — HTTP mocked
# ---------------------------------------------------------------------------


def _make_groq_response(segments: list[dict]) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"segments": segments, "text": "hello"}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = mock_resp
    return mock_client


@patch("event_timeline_extractor.transcription.groq_backend.httpx.Client")
def test_transcribe_returns_segments(mock_cls, tmp_path: Path) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 100)

    mock_cls.return_value = _make_groq_response(
        [{"start": 0.0, "end": 2.0, "text": " Test segment."}]
    )

    t = GroqTranscriber(api_key="gsk_test")
    result = t.transcribe(wav)

    assert len(result) == 1
    assert result[0].text == "Test segment."
    assert result[0].start == 0.0


@patch("event_timeline_extractor.transcription.groq_backend.httpx.Client")
def test_transcribe_raises_on_http_error(mock_cls, tmp_path: Path) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 100)

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = '{"error": {"message": "Invalid API key."}}'
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = mock_resp
    mock_cls.return_value = mock_client

    t = GroqTranscriber(api_key="gsk_bad")
    with pytest.raises(RuntimeError, match="HTTP 401"):
        t.transcribe(wav)


@patch("event_timeline_extractor.transcription.groq_backend.httpx.Client")
def test_transcribe_posts_to_correct_endpoint(mock_cls, tmp_path: Path) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 100)
    mock_cls.return_value = _make_groq_response([])

    t = GroqTranscriber(api_key="gsk_test", base_url="https://api.groq.com/openai/v1")
    t.transcribe(wav)

    call_url = mock_cls.return_value.post.call_args[0][0]
    assert "audio/transcriptions" in call_url
    assert "groq.com" in call_url


@patch("event_timeline_extractor.transcription.groq_backend.httpx.Client")
def test_transcribe_sends_correct_model(mock_cls, tmp_path: Path) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 100)
    mock_cls.return_value = _make_groq_response([])

    GroqTranscriber(api_key="gsk_test", model="whisper-large-v3-turbo").transcribe(wav)

    data_arg = mock_cls.return_value.post.call_args[1]["data"]
    assert data_arg["model"] == "whisper-large-v3-turbo"
    assert data_arg["response_format"] == "verbose_json"


@patch("event_timeline_extractor.transcription.groq_backend.httpx.Client")
def test_transcribe_retries_transient_network_error(mock_cls, tmp_path: Path) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 100)

    first_client = MagicMock()
    first_client.__enter__.return_value = first_client
    first_client.__exit__.return_value = None
    first_client.post.side_effect = httpx.ReadTimeout("slow")

    second_client = _make_groq_response([{"start": 0.0, "end": 1.0, "text": " Retry worked."}])
    mock_cls.side_effect = [first_client, second_client]

    result = GroqTranscriber(api_key="gsk_test").transcribe(wav)

    assert len(result) == 1
    assert result[0].text == "Retry worked."
    assert first_client.post.call_count == 1
