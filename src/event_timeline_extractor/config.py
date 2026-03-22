"""Environment-backed settings. Values are never printed in __repr__."""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "deepseek/deepseek-chat"
    # Lower = less paraphrase drift for extraction (0.0–0.1 typical).
    ete_openrouter_temperature: float = 0.05
    ete_transcriber: str = "faster_whisper"
    ete_use_stub: bool = False
    # faster-whisper: tiny < base < small < medium < large-v3 (quality / cost tradeoff)
    ete_whisper_model_size: str = "small"
    ete_whisper_beam_size: int = 1
    ete_whisper_vad: bool = True
    # Finer alignment for time anchoring; slightly more compute.
    ete_whisper_word_timestamps: bool = False
    # none | pyannote — optional speaker diarization (requires extras + HF_TOKEN).
    ete_diarization: str = "none"
    # Hugging Face token for pyannote models (same as `huggingface-cli login`).
    hf_token: SecretStr | None = None
    # Drop events whose evidence is not a substring of the full transcript (after whitespace norm).
    ete_validate_evidence: bool = True

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(openrouter_api_key=<redacted>, "
            f"openrouter_base_url={self.openrouter_base_url!r}, "
            f"openrouter_model={self.openrouter_model!r}, "
            f"ete_transcriber={self.ete_transcriber!r}, ete_use_stub={self.ete_use_stub!r}, "
            f"ete_whisper_model_size={self.ete_whisper_model_size!r})"
        )

    def openrouter_key_plain(self) -> str | None:
        k = self.openrouter_api_key
        return k.get_secret_value() if k else None

    def hf_token_plain(self) -> str | None:
        t = self.hf_token
        return t.get_secret_value() if t else None


def load_settings(*, env_file: Path | None = None) -> Settings:
    """Load settings; optional explicit .env path for tests."""
    if env_file is not None:
        return Settings(_env_file=(env_file,))
    return Settings()
