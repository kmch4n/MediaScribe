from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_CONFIG_PATH = Path("mediascribe.toml")


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration values for application logging."""

    verbose: bool = False
    log_file: Path | None = None


@dataclass(frozen=True)
class TranscribeConfig:
    """Configuration values for transcription defaults."""

    model: str = "large-v3"
    language: str | None = None
    device: str = "auto"
    compute_type: str = "default"
    beam_size: int = 5
    chunk_length: int = 30
    temperature: float = 0.0
    min_silence_ms: int = 500
    vad: bool = True
    overwrite: bool = False
    output_format: str = "txt"


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object loaded from TOML."""

    transcribe: TranscribeConfig = TranscribeConfig()
    logging: LoggingConfig = LoggingConfig()


def _require_type(value: Any, expected_type: type[Any], key: str) -> Any:
    if not isinstance(value, expected_type):
        raise ValueError(f"Config key `{key}` must be of type {expected_type.__name__}.")
    return value


def _require_float(value: Any, key: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"Config key `{key}` must be a float.")


def _optional_path(value: Any, key: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Config key `{key}` must be a string path.")
    return Path(value)


def _optional_language(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Config key `{key}` must be a string.")
    normalized = value.strip()
    return normalized or None


def load_config(path: Path | None = None) -> AppConfig:
    """Load TOML configuration from disk if present."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AppConfig()

    with config_path.open("rb") as f:
        payload = tomllib.load(f)

    transcribe_section = payload.get("transcribe", {})
    logging_section = payload.get("logging", {})

    if not isinstance(transcribe_section, dict):
        raise ValueError("Config section `transcribe` must be a table.")
    if not isinstance(logging_section, dict):
        raise ValueError("Config section `logging` must be a table.")

    transcribe = TranscribeConfig(
        model=_require_type(transcribe_section.get("model", "large-v3"), str, "transcribe.model"),
        language=_optional_language(
            transcribe_section.get("language"),
            "transcribe.language",
        ),
        device=_require_type(transcribe_section.get("device", "auto"), str, "transcribe.device"),
        compute_type=_require_type(
            transcribe_section.get("compute_type", "default"),
            str,
            "transcribe.compute_type",
        ),
        beam_size=_require_type(transcribe_section.get("beam_size", 5), int, "transcribe.beam_size"),
        chunk_length=_require_type(
            transcribe_section.get("chunk_length", 30),
            int,
            "transcribe.chunk_length",
        ),
        temperature=_require_float(
            transcribe_section.get("temperature", 0.0),
            "transcribe.temperature",
        ),
        min_silence_ms=_require_type(
            transcribe_section.get("min_silence_ms", 500),
            int,
            "transcribe.min_silence_ms",
        ),
        vad=_require_type(transcribe_section.get("vad", True), bool, "transcribe.vad"),
        overwrite=_require_type(
            transcribe_section.get("overwrite", False),
            bool,
            "transcribe.overwrite",
        ),
        output_format=_require_type(
            transcribe_section.get("output_format", "txt"),
            str,
            "transcribe.output_format",
        ),
    )
    logging = LoggingConfig(
        verbose=_require_type(logging_section.get("verbose", False), bool, "logging.verbose"),
        log_file=_optional_path(logging_section.get("log_file"), "logging.log_file"),
    )
    return AppConfig(transcribe=transcribe, logging=logging)
