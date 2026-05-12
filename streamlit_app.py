"""Browser-based UI for AI voice transcription.

Same Whisper backend as app.py. Two input modes (file upload + browser
microphone recording). Mobile-friendly layout. Session history.

Transcript history is kept in st.session_state and survives Streamlit reruns
within the same browser tab. It is cleared when the tab closes or the app
restarts.
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
if os.environ.get("CLOUD_TIER") == "free":
    MODELS = ["tiny"]
else:
    MODELS = ["tiny", "base", "small", "medium", "large-v3"]

MAX_HISTORY = 50


@st.cache_resource(show_spinner=False)
def load_model(size: str) -> WhisperModel:
    return WhisperModel(size, device="cpu", compute_type="int8")


def render_history_sidebar() -> None:
    history = st.session_state.get("history", [])
    st.subheader(f"History ({len(history)})")

    if not history:
        st.caption("Transcripts you create in this tab will appear here.")
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
            mime="application/octet-stream",
            use_container_width=True,
        )

    for i, entry in enumerate(reversed(history)):
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
                mime="application/octet-stream",
                key=f"dl_hist_{i}",
                use_container_width=True,
            )


def main() -> None:
    # No layout="wide" -- centred layout reads much better on phones.
    st.set_page_config(page_title="AI Voice Transcript")
    st.title("AI Voice Transcript")
    st.caption(
        "Upload a file or record your voice. Whisper runs on the server, "
        "fully open-source, no API keys."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("Options")
        model_size = st.selectbox(
            "Model", MODELS, index=min(1, len(MODELS) - 1),
            help="Larger models are more accurate but slower and need more memory.",
        )
        lang_label = st.selectbox("Language", [name for name, _ in LANGUAGES], index=0)
        language = next((code for name, code in LANGUAGES if name == lang_label), None)
        include_ts = st.checkbox("Include timestamps in output", value=False)
        st.markdown("---")
        render_history_sidebar()

    # Two input modes side by side -- works on mobile (tabs stack nicely).
    tab_file, tab_record = st.tabs(["Upload file", "Record voice"])

    audio_bytes: bytes | None = None
    audio_name: str | None = None
    audio_suffix: str = ".wav"

    with tab_file:
        uploaded = st.file_uploader(
            "Choose an audio or video file",
            type=["wav", "mp3", "m4a", "mp4", "aac", "ogg", "flac", "opus", "webm"],
        )
        if uploaded is not None:
            size_mb = uploaded.size / 1024 / 1024
            st.write(f"**File:** `{uploaded.name}` -- {size_mb:.1f} MB")
            audio_bytes = uploaded.getvalue()
            audio_name = uploaded.name
            audio_suffix = Path(uploaded.name).suffix or ".bin"

    with tab_record:
        st.caption(
            "Tap the microphone below, allow browser access, "
            "speak, then tap stop. Works on iPhone Safari too."
        )
        recorded = st.audio_input("Record from your microphone")
        if recorded is not None:
            audio_bytes = recorded.getvalue()
            audio_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            audio_suffix = ".wav"
            kb = len(audio_bytes) / 1024
            st.write(f"**Recording captured** -- {kb:.1f} KB")

    if audio_bytes is None:
        st.info("Upload a file or record your voice to begin.")
        return

    if not st.button("Transcribe", type="primary", use_container_width=True):
        return

    suffix = audio_suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
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
        live_box.text_area("Transcribing...", value=display, height=240)
        progress.progress(min(seg.end / duration, 1.0))

    progress.progress(1.0)
    final_text = "\n".join(lines) if include_ts else " ".join(lines)

    st.session_state.history.append({
        "filename": audio_name,
        "text": final_text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_size,
        "language": info.language,
        "duration": info.duration,
    })
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[-MAX_HISTORY:]

    st.success("Done!")

    # st.code has a built-in copy-to-clipboard button that works on iOS Safari.
    # wrap_lines keeps long sentences readable on narrow mobile screens.
    st.subheader("Transcript")
    st.code(final_text, language=None, wrap_lines=True)

    # application/octet-stream forces iOS Safari to show the Share/Save sheet
    # instead of opening the text inline in a new tab.
    st.download_button(
        "Download as .txt",
        data=final_text,
        file_name=f"{Path(audio_name).stem}.txt",
        mime="application/octet-stream",
        use_container_width=True,
    )
    st.caption(
        "On iPhone: tap Download -> 'Save to Files'. "
        "Or tap the copy icon at the top-right of the transcript above."
    )


if __name__ == "__main__":
    main()
