from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from mediascribe.config import DEFAULT_CONFIG_PATH, load_config
from mediascribe.logging_utils import configure_logger
from mediascribe.transcribe import TranscriptionOptions, transcribe_file


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="mediascribe")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional path to a log file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a TOML config file. Defaults to mediascribe.toml.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe a single audio or video file.",
    )
    transcribe_parser.add_argument(
        "input",
        type=Path,
        help="Input media file.",
    )
    transcribe_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output file path.",
    )
    transcribe_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcript files.",
    )
    transcribe_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["txt", "srt", "vtt", "json"],
        default=None,
        help="Transcript output format.",
    )
    transcribe_parser.add_argument(
        "--model",
        default=None,
        help="Whisper model name.",
    )
    transcribe_parser.add_argument(
        "--language",
        default=None,
        help="Optional ISO language code.",
    )
    transcribe_parser.add_argument(
        "--device",
        default=None,
        help="Inference device for faster-whisper.",
    )
    transcribe_parser.add_argument(
        "--compute-type",
        default=None,
        help="Compute type for faster-whisper.",
    )
    transcribe_parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size for decoding.",
    )
    transcribe_parser.add_argument(
        "--chunk-length",
        type=int,
        default=None,
        help="Chunk length in seconds.",
    )
    transcribe_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Decoding temperature.",
    )
    transcribe_parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=None,
        help="Minimum silence duration for VAD.",
    )
    transcribe_parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD filtering.",
    )

    return parser


def _run_transcribe(args: argparse.Namespace, logger: logging.Logger) -> int:
    config = load_config(args.config)
    options = TranscriptionOptions(
        model_name=args.model or config.transcribe.model,
        language=args.language if args.language is not None else config.transcribe.language,
        device=args.device or config.transcribe.device,
        compute_type=args.compute_type or config.transcribe.compute_type,
        beam_size=args.beam_size if args.beam_size is not None else config.transcribe.beam_size,
        vad_filter=(not args.no_vad) if args.no_vad else config.transcribe.vad,
        min_silence_duration_ms=(
            args.min_silence_ms
            if args.min_silence_ms is not None
            else config.transcribe.min_silence_ms
        ),
        chunk_length=(
            args.chunk_length if args.chunk_length is not None else config.transcribe.chunk_length
        ),
        temperature=(
            args.temperature if args.temperature is not None else config.transcribe.temperature
        ),
        overwrite=args.overwrite if args.overwrite else config.transcribe.overwrite,
        output_format=args.output_format or config.transcribe.output_format,
    )
    written_file = transcribe_file(
        input_path=args.input,
        output_path=args.output,
        options=options,
        logger=logger,
    )
    logger.info("Finished transcription. Wrote: %s", written_file)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        logger = configure_logger(
            verbose=args.verbose or config.logging.verbose,
            log_file=args.log_file or config.logging.log_file,
        )
        if args.command == "transcribe":
            return _run_transcribe(args, logger)
    except Exception as exc:
        fallback_logger = logging.getLogger("mediascribe-fallback")
        if not fallback_logger.handlers:
            fallback_logger.addHandler(logging.StreamHandler())
        fallback_logger.error("%s", exc)
        return 1

    parser.error(f"Unsupported command: {args.command}")
    return 2
