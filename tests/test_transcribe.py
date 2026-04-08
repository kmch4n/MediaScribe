from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mediascribe.config import load_config
from mediascribe.transcribe import (
    SegmentRecord,
    _should_fallback_to_cpu,
    build_output_path,
    resolve_media_file,
    write_transcript,
)


class TranscribeHelpersTest(unittest.TestCase):
    def test_resolve_media_file_accepts_supported_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            media_path = Path(temp_dir) / "lecture.mp3"
            media_path.write_text("x", encoding="utf-8")

            resolved = resolve_media_file(media_path)

            self.assertEqual(resolved, media_path.resolve())

    def test_build_output_path_uses_explicit_output_file(self) -> None:
        source = Path("/tmp/example.wav")
        output_file = Path("/tmp/out/example.txt")

        output_path = build_output_path(source, output_file, "json")

        self.assertEqual(output_path, output_file.resolve())

    def test_write_transcript_json_uses_utf8_json_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "sample.json"
            source = Path(temp_dir) / "sample.wav"
            segments = [SegmentRecord(start=0.0, end=1.5, text="hello")]

            write_transcript(output_path, source, segments, "json")

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["segments"][0]["text"], "hello")

    def test_load_config_reads_transcribe_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "mediascribe.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[transcribe]",
                        'model = "small"',
                        'language = "ja"',
                        "",
                        "[logging]",
                        "verbose = true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.transcribe.model, "small")
            self.assertEqual(config.transcribe.language, "ja")
            self.assertTrue(config.logging.verbose)

    def test_load_config_treats_empty_language_as_auto_detect(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "mediascribe.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[transcribe]",
                        'language = ""',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertIsNone(config.transcribe.language)

    def test_should_fallback_to_cpu_detects_missing_cuda_library(self) -> None:
        exc = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

        self.assertTrue(_should_fallback_to_cpu(exc))


if __name__ == "__main__":
    unittest.main()
