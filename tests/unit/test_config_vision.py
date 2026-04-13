"""Tests for vision-related config settings added to Settings."""

from __future__ import annotations

from pathlib import Path

from event_timeline_extractor.config import Settings, load_settings


def test_vision_enabled_defaults_to_false() -> None:
    s = Settings()
    assert s.ete_vision_enabled is False


def test_vision_frame_interval_defaults_to_10() -> None:
    s = Settings()
    assert s.ete_vision_frame_interval == 10


def test_vision_settings_loaded_from_env_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("ETE_VISION_ENABLED=true\nETE_VISION_FRAME_INTERVAL=5\n")
    s = load_settings(env_file=env)
    assert s.ete_vision_enabled is True
    assert s.ete_vision_frame_interval == 5


def test_vision_enabled_can_be_set_directly() -> None:
    s = Settings(ete_vision_enabled=True, ete_vision_frame_interval=30)
    assert s.ete_vision_enabled is True
    assert s.ete_vision_frame_interval == 30
