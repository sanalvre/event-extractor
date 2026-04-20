import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from event_timeline_extractor.chunking import TimeWindow
from event_timeline_extractor.config import Settings
from event_timeline_extractor.llm.openrouter import TimelineSynthesizer, _parse_events_json


def test_parse_events_json_object() -> None:
    raw = (
        '{"events":[{"time":"00:00","event":"Start","event_type":"speech","confidence":0.91,'
        '"speaker":null,"evidence":null,'
        '"source_segment_ids":["seg-000001"],"source_start":"00:00","source_end":"00:01"}]}'
    )
    ev = _parse_events_json(raw)
    assert len(ev) == 1
    assert ev[0].time == "00:00"
    assert ev[0].event_type == "speech"
    assert ev[0].confidence == pytest.approx(0.91)
    assert ev[0].source_segment_ids == ["seg-000001"]


def test_parse_events_json_strips_markdown_fence() -> None:
    raw = '```json\n{"events":[{"time":"00:01","event":"X"}]}\n```'
    ev = _parse_events_json(raw)
    assert ev[0].event == "X"


@patch("event_timeline_extractor.llm.openrouter.httpx.Client")
def test_synthesize_calls_openrouter(mock_client_cls) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "{}"
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "events": [
                                {
                                    "time": "00:00",
                                    "event": "Test",
                                    "event_type": "decision",
                                    "confidence": 0.82,
                                    "speaker": None,
                                    "evidence": None,
                                }
                            ]
                        }
                    )
                }
            }
        ]
    }
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    settings = Settings(openrouter_api_key="sk-test-key-for-mock")
    synth = TimelineSynthesizer(settings)
    out = synth.synthesize(
        [TimeWindow(0, 5, "hello", [])],
    )
    assert len(out.events) == 1
    assert out.events[0].event == "Test"
    assert out.events[0].event_type == "decision"
    assert out.events[0].confidence == pytest.approx(0.82)
    call_kw = mock_client.post.call_args
    payload = call_kw[1]["json"]
    assert payload["temperature"] == pytest.approx(0.05)
    assert "timeline extraction task" in payload["messages"][0]["content"]
    assert "2 to 4 meaningful events per minute" in payload["messages"][0]["content"]
    assert "DO NOT emit low-signal backchannels" in payload["messages"][0]["content"]
    assert "clear requests, refusals, notable questions" in payload["messages"][0]["content"]
    assert "event_type" in payload["messages"][0]["content"]
    assert "confidence" in payload["messages"][0]["content"]
    assert "source_segment_ids" in payload["messages"][0]["content"]
    assert "seg-XXXXXX" in payload["messages"][1]["content"]
    assert "Extract only the strongest timeline events." in payload["messages"][1]["content"]
    headers = call_kw[1]["headers"]
    assert headers["Authorization"].startswith("Bearer ")
    assert headers["Authorization"] != "Bearer "


def test_synthesize_missing_key() -> None:
    settings = Settings(openrouter_api_key=None)
    synth = TimelineSynthesizer(settings)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        synth.synthesize([TimeWindow(0, 1, "x", [])])


def test_dry_run_includes_source_references() -> None:
    settings = Settings(openrouter_api_key="sk-test-key-for-mock")
    synth = TimelineSynthesizer(settings)
    out = synth.dry_run(
        [
            TimeWindow(
                0,
                5,
                "[00:00] seg-000001 hello",
                [],
                source_segment_ids=["seg-000001"],
            )
        ]
    )
    assert out.events[0].source_segment_ids == ["seg-000001"]
    assert out.events[0].event_type == "other"
    assert out.events[0].confidence == pytest.approx(0.25)
    assert out.events[0].source_start == "00:00"
    assert out.events[0].source_end == "00:05"


@patch("event_timeline_extractor.llm.openrouter.httpx.Client")
def test_synthesize_retries_transient_network_errors(mock_client_cls) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "{}"
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"events":[{"time":"00:00","event":"Recovered"}]}'}}]
    }

    first_client = MagicMock()
    first_client.__enter__.return_value = first_client
    first_client.__exit__.return_value = None
    first_client.post.side_effect = httpx.ConnectError("offline")

    second_client = MagicMock()
    second_client.__enter__.return_value = second_client
    second_client.__exit__.return_value = None
    second_client.post.return_value = mock_resp

    mock_client_cls.side_effect = [first_client, second_client]

    settings = Settings(openrouter_api_key="sk-test-key-for-mock")
    out = TimelineSynthesizer(settings).synthesize([TimeWindow(0, 5, "hello", [])])

    assert len(out.events) == 1
    assert out.events[0].event == "Recovered"
    assert first_client.post.call_count == 1
    assert second_client.post.call_count == 1
