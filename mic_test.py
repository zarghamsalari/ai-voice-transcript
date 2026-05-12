"""Diagnose microphone issues:
- Lists all audio input devices on the system.
- Records a short clip from a chosen device and reports the loudness.

This bypasses Whisper entirely so we can isolate audio-capture problems.
"""

import argparse
import sys

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
TEST_SECONDS = 3


def list_devices() -> None:
    print("=== Audio input devices ===\n")
    default_in = sd.default.device[0]
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = "  <-- DEFAULT" if idx == default_in else ""
            print(f"  [{idx:>2}] {dev['name']}  "
                  f"(channels={dev['max_input_channels']}, "
                  f"rate={int(dev['default_samplerate'])}){marker}")
    print()


def test_device(device: int | None) -> None:
    dev_label = "default device" if device is None else f"device {device}"
    print(f"Recording {TEST_SECONDS}s from {dev_label}. Speak now (loudly).")
    audio = sd.rec(int(TEST_SECONDS * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype=np.int16,
                   device=device)
    sd.wait()

    peak = int(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak_pct = peak / 32767 * 100
    rms_pct = rms / 32767 * 100

    print(f"\nPeak amplitude : {peak:>5} / 32767  ({peak_pct:6.2f}% of max)")
    print(f"RMS loudness   : {rms:7.1f}        ({rms_pct:6.3f}% of full scale)\n")

    if peak < 50:
        print("STATUS: SILENT. The mic is producing no signal at all.")
        print("        -> Check Windows mute, app permissions, or pick a different device.")
    elif peak_pct < 3:
        print("STATUS: very quiet. Whisper will probably miss the speech.")
        print("        -> Raise Windows mic boost, or speak closer to the mic.")
    elif peak_pct < 30:
        print("STATUS: OK. Decent level, Whisper should work.")
    elif peak_pct > 95:
        print("STATUS: CLIPPING. Mic level is too high -- audio is distorted.")
        print("        -> Lower Windows mic volume.")
    else:
        print("STATUS: EXCELLENT. Strong, clean signal.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Microphone diagnostic tool.")
    parser.add_argument("--list", action="store_true", help="List input devices and exit")
    parser.add_argument("--device", type=int, default=None,
                        help="Test a specific device index (from --list).")
    args = parser.parse_args()

    list_devices()
    if args.list:
        return 0
    test_device(args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
