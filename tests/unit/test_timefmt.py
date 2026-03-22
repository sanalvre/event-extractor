import pytest

from event_timeline_extractor.timefmt import (
    download_section_first_seconds,
    seconds_to_end_timestamp,
)


def test_seconds_to_end_timestamp() -> None:
    assert seconds_to_end_timestamp(20) == "0:20"
    assert seconds_to_end_timestamp(65) == "1:05"
    assert seconds_to_end_timestamp(3600) == "1:00:00"


def test_seconds_to_end_timestamp_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        seconds_to_end_timestamp(0)
    with pytest.raises(ValueError):
        seconds_to_end_timestamp(-1)


def test_download_section_first_seconds() -> None:
    assert download_section_first_seconds(20) == "*0:00-0:20"
    assert download_section_first_seconds(125) == "*0:00-2:05"
