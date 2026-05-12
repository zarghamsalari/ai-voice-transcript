---
title: AI Voice Transcript
sdk: streamlit
sdk_version: "1.40.0"
app_file: streamlit_app.py
pinned: false
license: mit
---

# AI Voice Transcript

Local voice-to-text powered by OpenAI's Whisper (via [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)). Runs fully offline on your machine. No API keys, no per-minute charges. Three ways to use it:

| Mode | File | Best for |
|---|---|---|
| **Desktop GUI** | `app.py` | Day-to-day use on your own machine |
| **Web app** | `streamlit_app.py` | Sharing via a URL, deploying to the cloud |
| **CLI** | `transcribe_file.py`, `record_and_transcribe.py` | Scripting, batch jobs |

The YAML block above is read by Hugging Face Spaces — it tells HF this is a Streamlit app and which file to run.

## Features

- Drag-and-drop audio/video file transcription (desktop)
- Microphone capture with device selector and silent-recording detection (desktop)
- Live-streaming transcript display as Whisper decodes
- 9 pre-configured languages plus auto-detect
- 5 model sizes — pick speed vs. accuracy
- Optional timestamps
- Copy / Save As / Open Folder shortcuts
- Same backend across all three modes

## Quick start (local)

```powershell
git clone https://github.com/<your-username>/ai-voice-transcript.git
cd ai-voice-transcript

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt           # core deps (web + transcription)
pip install -r requirements-desktop.txt   # add GUI + mic recording
```

On Linux/macOS use `bash` and `source .venv/bin/activate` instead.

### Run the desktop GUI

Windows: double-click `Launch App.bat`.

Any OS:
```bash
python app.py
```

### Run the web app locally

```bash
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually <http://localhost:8501>).

### Run the CLI

```bash
python transcribe_file.py path/to/audio.m4a --model base --language en
python record_and_transcribe.py --seconds 15
```

## Model sizes

| Model | Disk | Speed | Accuracy | Memory |
|-------|------|-------|----------|--------|
| `tiny` | ~75 MB | fastest | OK | ~400 MB |
| `base` | ~150 MB | fast | good (default) | ~500 MB |
| `small` | ~500 MB | medium | better | ~1 GB |
| `medium` | ~1.5 GB | slow | great | ~3 GB |
| `large-v3` | ~3 GB | slowest | best | ~6 GB |

Models auto-download to `~/.cache/huggingface/` on first use.

## Deploying online

See [DEPLOY.md](DEPLOY.md) for step-by-step guides:

- **Hugging Face Spaces** — recommended (16 GB RAM free tier, no cold starts)
- **Render.com** — generic web host (works but 512 MB free tier is tight for Whisper)
- **Streamlit Community Cloud** — native home for Streamlit apps

## How it works

1. **Whisper** is a neural network trained on ~680,000 hours of multilingual audio.
2. **faster-whisper** converts Whisper to the CTranslate2 runtime with int8 quantization, giving ~4x speedup on CPU and lower memory use.
3. **VAD (Voice Activity Detection)** filters silence before decoding — speeds things up and prevents Whisper from "hallucinating" text on empty audio.
4. **16 kHz mono** is Whisper's native input; the desktop recorder uses this directly so no resampling is needed.
5. **Background threading** in the GUI keeps the UI responsive during long transcriptions — workers post events to a queue, the main thread polls and renders.

## Project layout

```
ai-voice-transcript/
├── app.py                       # Desktop GUI (Tkinter + tkinterdnd2)
├── streamlit_app.py             # Web app (Streamlit)
├── transcribe_file.py           # CLI: transcribe a file
├── record_and_transcribe.py     # CLI: record from mic + transcribe
├── mic_test.py                  # CLI: diagnose mic issues
├── Launch App.bat               # Windows: double-click to launch GUI
├── Launch App (debug).bat       # Windows: same but with visible console
├── requirements.txt             # Cloud / web deps
├── requirements-desktop.txt     # Adds GUI + mic recording deps
├── DEPLOY.md                    # Step-by-step deployment guide
├── recordings/                  # Captured audio (gitignored)
└── transcripts/                 # Generated text output (gitignored)
```

## License

MIT
