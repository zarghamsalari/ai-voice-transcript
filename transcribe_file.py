"""Transcribe an existing audio file (mp3, wav, m4a, etc.) into text."""

import argparse
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe(audio_path: Path, model_size: str, language: str | None) -> str:
    # int8 = quantized weights -> fast on CPU, small memory footprint.
    # device="cpu" works everywhere; switch to "cuda" if you have an NVIDIA GPU.
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        str(audio_path),
        language=language,           # None = auto-detect the spoken language
        beam_size=5,                 # higher = more accurate, slower (5 is standard)
        vad_filter=True,             # skip silent regions to save time + avoid hallucinations
    )

    print(f"Detected language: {info.language} (confidence {info.language_probability:.2f})")
    print(f"Audio duration: {info.duration:.1f}s\n")

    lines: list[str] = []
    for seg in segments:
        line = f"[{seg.start:6.2f}s -> {seg.end:6.2f}s] {seg.text.strip()}"
        print(line)
        lines.append(line)

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio file using Whisper.")
    parser.add_argument("audio", type=Path, help="Path to the audio file")
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--language", default=None,
                        help="Force a language code like 'en', 'ur', 'es'. Default: auto-detect.")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"ERROR: file not found: {args.audio}", file=sys.stderr)
        return 1

    transcript = transcribe(args.audio, args.model, args.language)

    out_dir = Path(__file__).parent / "transcripts"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / (args.audio.stem + ".txt")
    out_file.write_text(transcript, encoding="utf-8")

    print(f"\nTranscript saved to: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
