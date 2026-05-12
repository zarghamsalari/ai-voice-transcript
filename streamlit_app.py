"""Browser-based UI for AI voice transcription.

Same Whisper backend as app.py, but served as a web page via Streamlit so it
can run on Hugging Face Spaces, Render, Streamlit Community Cloud, or any
other Python web host. File upload only -- a server cannot capture audio
from the visitor's microphone.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel


LANGUAGES: list[tuple[str, str | None]] = [
    ("Auto-detect", None),
    ("English", "en"),
    ("Urdu", "ur"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Chinese", "zh"),
]
# On low-memory hosts (e.g. Render free tier with 512 MB RAM), only the tiny
# model fits. Set CLOUD_TIER=free as an env var on the host to restrict choices.
if os.environ.get("CLOUD_TIER") == "free":
    MODELS = ["tiny"]
else:
    MODELS = ["tiny", "base", "small", "medium", "large-v3"]


@st.cache_resource(show_spinner=False)
def load_model(size: str) -> WhisperModel:
    # Cached across reruns so we don't redownload/reload on every click.
    return WhisperModel(size, device="cpu", compute_type="int8")


def main() -> None:
    st.set_page_config(page_title="AI Voice Transcript", layout="wide")
    st.title("AI Voice Transcript")
    st.caption(
        "Upload an audio or video file. Whisper runs in the browser server, "
        "fully open-source, no API keys."
    )

    with st.sidebar:
        st.header("Options")
        model_size = st.selectbox(
            "Model", MODELS, index=1,
            help="Larger models are more accurate but slower and need more memory. "
                 "tiny is best on free hosting tiers; base is the balanced default.",
        )
        lang_label = st.selectbox("Language", [name for name, _ in LANGUAGES], index=0)
        language = next((code for name, code in LANGUAGES if name == lang_label), None)
        include_ts = st.checkbox("Include timestamps in output", value=False)
        st.markdown("---")
        st.caption("Tip: deploy your own copy on Hugging Face Spaces or Render. See DEPLOY.md.")

    uploaded = st.file_uploader(
        "Choose an audio or video file",
        type=["wav", "mp3", "m4a", "mp4", "aac", "ogg", "flac", "opus", "webm"],
    )

    if uploaded is None:
        st.info("Waiting for a file...")
        return

    size_mb = uploaded.size / 1024 / 1024
    st.write(f"**File:** `{uploaded.name}` -- {size_mb:.1f} MB")

    if not st.button("Transcribe", type="primary"):
        return

    # Whisper requires a real file path; persist the upload to a temp file.
    suffix = Path(uploaded.name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner(f"Loading model '{model_size}'... (first run downloads weights)"):
        model = load_model(model_size)

    segments, info = model.transcribe(
        tmp_path, language=language, beam_size=5, vad_filter=True
    )
    st.info(
        f"Detected language: **{info.language}** "
        f"({info.language_probability:.0%}) -- Duration: {info.duration:.0f}s"
    )

    duration = max(info.duration, 1.0)
    progress = st.progress(0.0)
    live_box = st.empty()
    lines: list[str] = []

    # Iterating segments triggers the actual decode work; we stream results live.
    for seg in segments:
        text = seg.text.strip()
        if include_ts:
            lines.append(f"[{seg.start:6.2f}s -> {seg.end:6.2f}s] {text}")
        else:
            lines.append(text)
        display = "\n".join(lines) if include_ts else " ".join(lines)
        live_box.text_area("Transcript", value=display, height=380)
        progress.progress(min(seg.end / duration, 1.0))

    progress.progress(1.0)
    final_text = "\n".join(lines) if include_ts else " ".join(lines)
    st.success("Done.")
    st.download_button(
        "Download transcript (.txt)",
        data=final_text,
        file_name=f"{Path(uploaded.name).stem}.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()
