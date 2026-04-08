from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SUPPORTED_MEDIA_EXTENSIONS = {
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".wav",
}


@dataclass(frozen=True)
class SegmentRecord:
    """Serializable transcription segment."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionOptions:
    """Runtime options for transcription."""

    model_name: str = "large-v3"
    language: str | None = None
    device: str = "auto"
    compute_type: str = "default"
    beam_size: int = 5
    vad_filter: bool = True
    min_silence_duration_ms: int = 500
    chunk_length: int = 30
    temperature: float = 0.0
    overwrite: bool = False
    output_format: str = "txt"


def _should_fallback_to_cpu(exc: Exception) -> bool:
    """Return whether the exception suggests missing CUDA runtime libraries."""
    message = str(exc).lower()
    indicators = [
        "cublas",
        "cudnn",
        "cuda",
        "cufft",
        "curand",
        "cannot be loaded",
    ]
    return any(indicator in message for indicator in indicators)


def resolve_media_file(input_path: Path) -> Path:
    """Resolve and validate a single media file path."""
    path = input_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path must be a file: {path}")
    if path.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
        raise ValueError(f"Unsupported media file extension: {path.suffix}")
    return path


def _format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def _format_vtt_timestamp(seconds: float) -> str:
    return _format_srt_timestamp(seconds).replace(",", ".")


def _build_json_payload(source: Path, segments: Sequence[SegmentRecord]) -> str:
    payload = {
        "source": str(source),
        "segments": [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments
        ],
    }
    return json.dumps(payload, indent=4, ensure_ascii=False)


def write_transcript(
    output_path: Path,
    source: Path,
    segments: Sequence[SegmentRecord],
    output_format: str,
) -> None:
    """Write transcription output in the selected format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if output_format == "txt":
        content = "".join(segment.text for segment in segments).strip() + "\n"
    elif output_format == "srt":
        blocks = []
        for index, segment in enumerate(segments, start=1):
            blocks.append(
                "\n".join(
                    [
                        str(index),
                        (
                            f"{_format_srt_timestamp(segment.start)} --> "
                            f"{_format_srt_timestamp(segment.end)}"
                        ),
                        segment.text.strip(),
                    ]
                )
            )
        content = "\n\n".join(blocks).strip() + "\n"
    elif output_format == "vtt":
        blocks = ["WEBVTT"]
        for segment in segments:
            blocks.append(
                "\n".join(
                    [
                        (
                            f"{_format_vtt_timestamp(segment.start)} --> "
                            f"{_format_vtt_timestamp(segment.end)}"
                        ),
                        segment.text.strip(),
                    ]
                )
            )
        content = "\n\n".join(blocks).strip() + "\n"
    elif output_format == "json":
        content = _build_json_payload(source, segments) + "\n"
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(output_path)


def _load_model(model_name: str, device: str, compute_type: str):
    warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _transcribe_with_fallback(
    media_path: Path,
    options: TranscriptionOptions,
    logger: logging.Logger,
) -> list[SegmentRecord]:
    """Run transcription and fall back to CPU when CUDA runtime libraries are missing."""
    def _run_once(device: str, compute_type: str) -> list[SegmentRecord]:
        model = _load_model(
            model_name=options.model_name,
            device=device,
            compute_type=compute_type,
        )
        segments_iter, _ = model.transcribe(
            str(media_path),
            language=options.language,
            beam_size=options.beam_size,
            vad_filter=options.vad_filter,
            vad_parameters={"min_silence_duration_ms": options.min_silence_duration_ms},
            chunk_length=options.chunk_length,
            temperature=options.temperature,
        )
        return [
            SegmentRecord(start=segment.start, end=segment.end, text=segment.text)
            for segment in segments_iter
        ]

    try:
        return _run_once(options.device, options.compute_type)
    except Exception as exc:
        if options.device == "cpu" or not _should_fallback_to_cpu(exc):
            raise

        logger.warning(
            "CUDA runtime libraries were not available. Falling back to CPU inference."
        )
        return _run_once("cpu", "int8")


def build_output_path(
    source: Path,
    output_path: Path | None,
    output_format: str,
) -> Path:
    """Return the output path for a transcription artifact."""
    if output_path is not None:
        return output_path.expanduser().resolve()
    suffix = f".{output_format}"
    return source.with_suffix(suffix)


def transcribe_file(
    input_path: Path,
    output_path: Path | None,
    options: TranscriptionOptions,
    logger: logging.Logger,
) -> Path:
    """Transcribe a single media file and return the written output path."""
    media_path = resolve_media_file(input_path)
    resolved_output_path = build_output_path(
        source=media_path,
        output_path=output_path,
        output_format=options.output_format,
    )

    if resolved_output_path.exists() and not options.overwrite:
        raise FileExistsError(f"Output file already exists: {resolved_output_path}")

    logger.info("Transcribing: %s", media_path)
    segments = _transcribe_with_fallback(
        media_path=media_path,
        options=options,
        logger=logger,
    )
    write_transcript(
        output_path=resolved_output_path,
        source=media_path,
        segments=segments,
        output_format=options.output_format,
    )
    logger.info("Wrote transcript: %s", resolved_output_path)
    return resolved_output_path
