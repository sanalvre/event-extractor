"""Unit tests for MemoriesTranscriber (memories.ai cloud transcription)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from event_timeline_extractor.transcription.memories_backend import (
    MemoriesTranscriber,
    _raise_for_status,
)

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_raises_if_api_key_empty() -> None:
    with pytest.raises(ValueError, match="MEMORIES_API_KEY"):
        MemoriesTranscriber(api_key="")


def test_factory_raises_without_key() -> None:
    from event_timeline_extractor.config import Settings
    from event_timeline_extractor.transcription.factory import get_transcriber

    s = Settings(ete_transcriber="memories", memories_api_key=None)
    with pytest.raises(ValueError, match="MEMORIES_API_KEY"):
        get_transcriber(s)


def test_factory_returns_memories_transcriber_when_key_set() -> None:
    from event_timeline_extractor.config import Settings
    from event_timeline_extractor.transcription.factory import get_transcriber

    s = Settings(ete_transcriber="memories", memories_api_key="sk-mai-test")
    t = get_transcriber(s)
    assert isinstance(t, MemoriesTranscriber)


# ---------------------------------------------------------------------------
# _to_segments — static parsing
# ---------------------------------------------------------------------------


def test_to_segments_maps_fields_correctly() -> None:
    items = [
        {"text": " Hello there.", "start_time": 0.0, "end_time": 2.5, "speaker": "SPEAKER_00"},
        {"text": " Good morning.", "start_time": 2.6, "end_time": 5.0, "speaker": "SPEAKER_01"},
    ]
    segs = MemoriesTranscriber._to_segments(items)
    assert len(segs) == 2
    assert segs[0].text == "Hello there."
    assert segs[0].start == 0.0
    assert segs[0].end == 2.5
    assert segs[0].speaker == "SPEAKER_00"
    assert segs[1].speaker == "SPEAKER_01"


def test_to_segments_strips_whitespace_only_items() -> None:
    items = [{"text": "   ", "start_time": 0.0, "end_time": 1.0, "speaker": None}]
    assert MemoriesTranscriber._to_segments(items) == []


def test_to_segments_handles_missing_speaker() -> None:
    items = [{"text": "Hello", "start_time": 0.0, "end_time": 1.0}]
    segs = MemoriesTranscriber._to_segments(items)
    assert segs[0].speaker is None


def test_to_segments_empty_list() -> None:
    assert MemoriesTranscriber._to_segments([]) == []


# ---------------------------------------------------------------------------
# _raise_for_status helper
# ---------------------------------------------------------------------------


def test_raise_for_status_passes_on_2xx() -> None:
    mock_r = MagicMock()
    mock_r.status_code = 200
    _raise_for_status(mock_r, "upload")  # should not raise


def test_raise_for_status_raises_on_4xx() -> None:
    mock_r = MagicMock()
    mock_r.status_code = 401
    mock_r.text = '{"code":401,"msg":"Unauthorized"}'
    with pytest.raises(RuntimeError, match="HTTP 401"):
        _raise_for_status(mock_r, "upload")


# ---------------------------------------------------------------------------
# transcribe() — full flow with HTTP mocked
# ---------------------------------------------------------------------------


def _make_mock_client(upload_asset_id: str, transcribe_items: list[dict]) -> MagicMock:
    upload_resp = MagicMock()
    upload_resp.status_code = 200
    upload_resp.json.return_value = {"data": {"asset_id": upload_asset_id}}

    transcribe_resp = MagicMock()
    transcribe_resp.status_code = 200
    transcribe_resp.json.return_value = {"data": {"items": transcribe_items}}

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    # First call = upload, second = transcribe
    mock_client.post.side_effect = [upload_resp, transcribe_resp]
    return mock_client


@patch("event_timeline_extractor.transcription.memories_backend.httpx.Client")
def test_transcribe_full_flow(mock_cls, tmp_path: Path) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 100)

    items = [
        {
            "text": " Officer asks driver to exit.",
            "start_time": 0.5,
            "end_time": 3.0,
            "speaker": "SPEAKER_00",
        },
        {"text": " Driver complies.", "start_time": 3.1, "end_time": 5.0, "speaker": "SPEAKER_01"},
    ]
    mock_cls.return_value = _make_mock_client("re_test_asset", items)

    t = MemoriesTranscriber(api_key="sk-mai-test", speaker=True)
    segs = t.transcribe(wav)

    assert len(segs) == 2
    assert segs[0].text == "Officer asks driver to exit."
    assert segs[0].speaker == "SPEAKER_00"
    assert segs[1].speaker == "SPEAKER_01"


@patch("event_timeline_extractor.transcription.memories_backend.httpx.Client")
def test_transcribe_sends_speaker_flag(mock_cls, tmp_path: Path) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 100)
    mock_cls.return_value = _make_mock_client("re_xyz", [])

    MemoriesTranscriber(api_key="sk-mai-test", speaker=True).transcribe(wav)

    # Second POST call is the transcription; check speaker=True in payload
    transcription_call = mock_cls.return_value.post.call_args_list[1]
    body = transcription_call[1]["json"]
    assert body["speaker"] is True
    assert body["model"] == "whisper-1"


@patch("event_timeline_extractor.transcription.memories_backend.httpx.Client")
def test_transcribe_raises_on_upload_error(mock_cls, tmp_path: Path) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 100)

    bad_resp = MagicMock()
    bad_resp.status_code = 402
    bad_resp.text = '{"msg":"Insufficient balance"}'
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = bad_resp
    mock_cls.return_value = mock_client

    with pytest.raises(RuntimeError, match="HTTP 402"):
        MemoriesTranscriber(api_key="sk-mai-test").transcribe(wav)


@patch("event_timeline_extractor.transcription.memories_backend.httpx.Client")
def test_transcribe_retries_transient_upload_error(mock_cls, tmp_path: Path) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 100)

    first_client = MagicMock()
    first_client.__enter__.return_value = first_client
    first_client.__exit__.return_value = None
    first_client.post.side_effect = httpx.ConnectError("offline")

    upload_resp = MagicMock()
    upload_resp.status_code = 200
    upload_resp.json.return_value = {"data": {"asset_id": "re_test_asset"}}
    second_client = MagicMock()
    second_client.__enter__.return_value = second_client
    second_client.__exit__.return_value = None
    second_client.post.return_value = upload_resp

    transcribe_resp = MagicMock()
    transcribe_resp.status_code = 200
    transcribe_resp.json.return_value = {"data": {"items": []}}
    third_client = MagicMock()
    third_client.__enter__.return_value = third_client
    third_client.__exit__.return_value = None
    third_client.post.return_value = transcribe_resp
    mock_cls.side_effect = [first_client, second_client, third_client]

    MemoriesTranscriber(api_key="sk-mai-test", speaker=True).transcribe(wav)

    assert first_client.post.call_count == 1
