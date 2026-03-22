from pathlib import Path

from event_timeline_extractor.config import Settings, load_settings


def test_settings_repr_redacts_secret() -> None:
    s = Settings(
        openrouter_api_key="sk-test-secret-key",
        ete_use_stub=True,
    )
    r = repr(s)
    assert "sk-test" not in r
    assert "<redacted>" in r or "SecretStr" not in r  # pydantic may show ***


def test_openrouter_key_plain_none() -> None:
    s = Settings(openrouter_api_key=None)
    assert s.openrouter_key_plain() is None


def test_openrouter_key_plain_value() -> None:
    s = Settings(openrouter_api_key="abc")
    assert s.openrouter_key_plain() == "abc"


def test_load_settings_tmp_env_file(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / ".env"
    p.write_text("OPENROUTER_API_KEY=fromfile\n", encoding="utf-8")
    s = load_settings(env_file=p)
    assert s.openrouter_key_plain() == "fromfile"
