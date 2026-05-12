"""Record audio from the microphone, save it, then transcribe it."""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from faster_whisper import WhisperModel


SAMPLE_RATE = 16000  # Whisper was trained on 16 kHz audio; matching this avoids resampling.


def record(seconds: int, out_path: Path, device: int | None) -> None:
    dev_label = "default mic" if device is None else f"device #{device}"
    print(f"Recording for {seconds}s from {dev_label} — speak now...")
    # Mono (channels=1) is enough for speech. int16 = standard WAV PCM format.
    audio = sd.rec(int(seconds * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype=np.int16,
                   device=device)
    # sd.rec returns immediately and records in the background. We must wait.
    for remaining in range(seconds, 0, -1):
        print(f"  ...{remaining}s left", end="\r", flush=True)
        time.sleep(1)
    sd.wait()
    print("Recording done.            ")

    wavfile.write(str(out_path), SAMPLE_RATE, audio)
    print(f"Audio saved to: {out_path}")


def transcribe(audio_path: Path, model_size: str, language: str | None) -> str:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
    )

    print(f"\nDetected language: {info.language} (confidence {info.language_probability:.2f})\n")
    lines: list[str] = []
    for seg in segments:
        line = f"[{seg.start:6.2f}s -> {seg.end:6.2f}s] {seg.text.strip()}"
        print(line)
        lines.append(line)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record from mic and transcribe with Whisper.")
    parser.add_argument("--seconds", type=int, default=10, help="How many seconds to record (default: 10)")
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--language", default=None,
                        help="Force a language code like 'en', 'ur'. Default: auto-detect.")
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    recordings_dir = project_dir / "recordings"
    transcripts_dir = project_dir / "transcripts"
    recordings_dir.mkdir(exist_ok=True)
    transcripts_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = recordings_dir / f"rec_{stamp}.wav"
    txt_path = transcripts_dir / f"rec_{stamp}.txt"

    record(args.seconds, wav_path)
    transcript = transcribe(wav_path, args.model, args.language)
    txt_path.write_text(transcript, encoding="utf-8")

    print(f"\nTranscript saved to: {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
