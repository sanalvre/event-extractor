import json
from unittest.mock import MagicMock, patch

import pytest

from event_timeline_extractor.chunking import TimeWindow
from event_timeline_extractor.config import Settings
from event_timeline_extractor.llm.openrouter import TimelineSynthesizer, _parse_events_json


def test_parse_events_json_object() -> None:
    raw = '{"events":[{"time":"00:00","event":"Start","speaker":null,"evidence":null}]}'
    ev = _parse_events_json(raw)
    assert len(ev) == 1
    assert ev[0].time == "00:00"


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
    call_kw = mock_client.post.call_args
    payload = call_kw[1]["json"]
    assert payload["temperature"] == pytest.approx(0.05)
    headers = call_kw[1]["headers"]
    assert headers["Authorization"].startswith("Bearer ")
    assert headers["Authorization"] != "Bearer "


def test_synthesize_missing_key() -> None:
    settings = Settings(openrouter_api_key=None)
    synth = TimelineSynthesizer(settings)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        synth.synthesize([TimeWindow(0, 1, "x", [])])
