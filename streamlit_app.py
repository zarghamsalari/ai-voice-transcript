"""Browser-based UI for AI voice transcription.

Same Whisper backend as app.py, but served as a web page via Streamlit so it
can run on Hugging Face Spaces, Render, Streamlit Community Cloud, or any
other Python web host. File upload only -- a server cannot capture audio
from the visitor's microphone.

Transcript history is kept in st.session_state and survives Streamlit reruns
within the same browser tab. It is cleared when the tab closes or the app
restarts -- Streamlit Cloud's filesystem is ephemeral, so true persistence
would require an external database or object store.
"""

import io
import os
import tempfile
import zipfile
from datetime import datetime
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

MAX_HISTORY = 50  # cap to protect container memory; oldest entries drop off


@st.cache_resource(show_spinner=False)
def load_model(size: str) -> WhisperModel:
    # Cached across reruns so we don't redownload/reload on every click.
    return WhisperModel(size, device="cpu", compute_type="int8")


def render_history_sidebar() -> None:
    history = st.session_state.get("history", [])
    st.subheader(f"History ({len(history)})")

    if not history:
        st.caption(
            "Transcripts you create in this tab will appear here. "
            "History resets when the tab closes."
        )
        return

    if st.button("Clear all", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    if len(history) > 1:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for entry in history:
                safe_stamp = entry["timestamp"].replace(":", "-").replace(" ", "_")
                fname = f"{Path(entry['filename']).stem}_{safe_stamp}.txt"
                zf.writestr(fname, entry["text"])
        st.download_button(
            "Download all (.zip)",
            data=buf.getvalue(),
            file_name="transcripts.zip",
            mime="application/zip",
            use_container_width=True,
        )

    for i, entry in enumerate(reversed(history)):
        # Truncate long filenames so the expander label stays tidy.
        label = entry["filename"]
        if len(label) > 28:
            label = label[:25] + "..."
        with st.expander(label):
            st.caption(entry["timestamp"])
            st.caption(
                f"{entry['model']} | {entry['language']} | "
                f"{int(entry['duration'])}s | {len(entry['text'])} chars"
            )
            st.download_button(
                "Download .txt",
                data=entry["text"],
                file_name=f"{Path(entry['filename']).stem}.txt",
                mime="text/plain",
                key=f"dl_hist_{i}",
                use_container_width=True,
            )


def main() -> None:
    st.set_page_config(page_title="AI Voice Transcript", layout="wide")
    st.title("AI Voice Transcript")
    st.caption(
        "Upload an audio or video file. Whisper runs on the server, "
        "fully open-source, no API keys."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("Options")
        model_size = st.selectbox(
            "Model", MODELS, index=min(1, len(MODELS) - 1),
            help="Larger models are more accurate but slower and need more memory. "
                 "tiny is best on free hosting tiers; base is the balanced default.",
        )
        lang_label = st.selectbox("Language", [name for name, _ in LANGUAGES], index=0)
        language = next((code for name, code in LANGUAGES if name == lang_label), None)
        include_ts = st.checkbox("Include timestamps in output", value=False)
        st.markdown("---")
        render_history_sidebar()

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

    st.session_state.history.append({
        "filename": uploaded.name,
        "text": final_text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_size,
        "language": info.language,
        "duration": info.duration,
    })
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[-MAX_HISTORY:]

    st.success("Done. Saved to history (see sidebar — it refreshes on your next click).")
    st.download_button(
        "Download transcript (.txt)",
        data=final_text,
        file_name=f"{Path(uploaded.name).stem}.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()
