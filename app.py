"""Desktop GUI for AI voice transcription.

Two modes:
  - File: drag & drop or browse for an audio/video file.
  - Record: capture from a chosen microphone for N seconds.

Whisper inference and audio recording run in background threads. The UI
thread polls a queue to apply updates, because Tkinter is not thread-safe.
"""

import os
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    print("Missing tkinterdnd2. Run: pip install tkinterdnd2")
    sys.exit(1)

from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
PROJECT_DIR = Path(__file__).parent
RECORDINGS_DIR = PROJECT_DIR / "recordings"
TRANSCRIPTS_DIR = PROJECT_DIR / "transcripts"
RECORDINGS_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

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
MODELS = ["tiny", "base", "small", "medium", "large-v3"]


class TranscriptApp:
    def __init__(self, root: TkinterDnD.Tk) -> None:
        self.root = root
        self.root.title("AI Voice Transcript")
        self.root.geometry("780x740")
        self.root.minsize(640, 540)

        self.events: queue.Queue = queue.Queue()
        self.audio_path: Path | None = None
        self.cached_model: WhisperModel | None = None
        self.cached_model_size: str = ""
        self.last_transcript_path: Path | None = None
        self.is_busy = False
        self.device_options: list[tuple[str, int]] = []

        self._build_ui()
        self._refresh_devices()
        self._poll_events()

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="x", padx=10, pady=(10, 0))

        notebook.add(self._build_file_tab(notebook), text="  File  ")
        notebook.add(self._build_record_tab(notebook), text="  Record  ")

        self._build_options_frame()
        self._build_status_and_progress()
        self._build_output_area()
        self._build_bottom_buttons()

    def _build_file_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(parent)

        self.drop_zone = tk.Label(
            tab,
            text="Drag an audio or video file here",
            relief="ridge",
            borderwidth=2,
            font=("Segoe UI", 11),
            height=4,
            bg="#f0f4f8",
            fg="#445",
        )
        self.drop_zone.pack(fill="x", padx=20, pady=(15, 5))
        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind("<<Drop>>", self._on_file_drop)

        ttk.Button(tab, text="Browse...", command=self._browse).pack(pady=(0, 5))

        self.file_label_var = tk.StringVar(value="No file selected")
        ttk.Label(tab, textvariable=self.file_label_var, foreground="#334").pack(pady=2)

        self.transcribe_btn = ttk.Button(
            tab, text="Transcribe", command=self._start_file_transcribe
        )
        self.transcribe_btn.pack(pady=(8, 10))
        return tab

    def _build_record_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(parent)

        mic_row = ttk.Frame(tab)
        mic_row.pack(fill="x", padx=20, pady=15)
        ttk.Label(mic_row, text="Microphone:").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            mic_row, textvariable=self.device_var, state="readonly", width=48
        )
        self.device_combo.pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(mic_row, text="Refresh", command=self._refresh_devices).pack(side="left")

        sec_row = ttk.Frame(tab)
        sec_row.pack(fill="x", padx=20)
        ttk.Label(sec_row, text="Duration (seconds):").pack(side="left")
        self.seconds_var = tk.IntVar(value=10)
        self.seconds_label = ttk.Label(sec_row, text="10", width=4)
        ttk.Scale(
            sec_row,
            from_=3,
            to=120,
            orient="horizontal",
            variable=self.seconds_var,
            command=lambda v: self.seconds_label.config(text=str(int(float(v)))),
        ).pack(side="left", fill="x", expand=True, padx=10)
        self.seconds_label.pack(side="left")

        self.record_btn = ttk.Button(
            tab, text="Record and Transcribe", command=self._start_record_transcribe
        )
        self.record_btn.pack(pady=15)
        return tab

    def _build_options_frame(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Options")
        frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame, text="Model:").grid(row=0, column=0, sticky="w", padx=5, pady=6)
        self.model_var = tk.StringVar(value="base")
        ttk.Combobox(
            frame, textvariable=self.model_var, values=MODELS,
            state="readonly", width=12,
        ).grid(row=0, column=1, padx=5)

        ttk.Label(frame, text="Language:").grid(row=0, column=2, sticky="w", padx=(15, 5))
        self.lang_var = tk.StringVar(value="Auto-detect")
        ttk.Combobox(
            frame, textvariable=self.lang_var,
            values=[label for label, _ in LANGUAGES],
            state="readonly", width=15,
        ).grid(row=0, column=3, padx=5)

        self.timestamps_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, text="Include timestamps", variable=self.timestamps_var
        ).grid(row=0, column=4, padx=15)

    def _build_status_and_progress(self) -> None:
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, foreground="#06a").pack(
            anchor="w", padx=15
        )
        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill="x", padx=15, pady=(2, 8))

    def _build_output_area(self) -> None:
        wrap = ttk.Frame(self.root)
        wrap.pack(fill="both", expand=True, padx=10, pady=5)
        self.text_widget = tk.Text(wrap, wrap="word", font=("Segoe UI", 10), undo=True)
        scroll = ttk.Scrollbar(wrap, command=self.text_widget.yview)
        self.text_widget.config(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text_widget.pack(side="left", fill="both", expand=True)

    def _build_bottom_buttons(self) -> None:
        row = ttk.Frame(self.root)
        row.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(row, text="Copy", command=self._copy).pack(side="left")
        ttk.Button(row, text="Save As...", command=self._save_as).pack(side="left", padx=5)
        ttk.Button(row, text="Open Folder", command=self._open_folder).pack(side="left")
        ttk.Button(row, text="Clear", command=self._clear).pack(side="right")

    # ---------- File mode ----------

    def _on_file_drop(self, event) -> None:
        raw = event.data.strip()
        # tkdnd wraps paths containing spaces in braces; multiple files are space-separated.
        if raw.startswith("{"):
            end = raw.find("}")
            path = raw[1:end] if end > 0 else raw[1:]
        else:
            path = raw.split()[0]
        self._set_audio_file(path)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an audio or video file",
            filetypes=[
                ("Audio/Video", "*.wav *.mp3 *.m4a *.mp4 *.aac *.ogg *.flac *.opus *.webm *.mkv *.mov"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._set_audio_file(path)

    def _set_audio_file(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return
        self.audio_path = p
        size_mb = p.stat().st_size / 1024 / 1024
        self.file_label_var.set(f"{p.name}    ({size_mb:.1f} MB)")

    def _start_file_transcribe(self) -> None:
        if self.is_busy:
            return
        if not self.audio_path:
            messagebox.showwarning("No file", "Drag in or browse for an audio file first.")
            return
        self._set_busy(True)
        threading.Thread(target=self._file_thread, daemon=True).start()

    def _file_thread(self) -> None:
        try:
            self._post("status", "Loading model...")
            model = self._load_model()
            language = self._language_code()

            assert self.audio_path is not None
            self._post("status", f"Transcribing {self.audio_path.name}...")
            self._post("clear", None)

            segments, info = model.transcribe(
                str(self.audio_path),
                language=language,
                beam_size=5,
                vad_filter=True,
            )

            self._post(
                "status",
                f"Language: {info.language} ({info.language_probability:.0%})   "
                f"Duration: {info.duration:.0f}s",
            )

            transcript = self._consume_segments(segments, info.duration)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = TRANSCRIPTS_DIR / f"{self.audio_path.stem}_{stamp}.txt"
            out.write_text(transcript, encoding="utf-8")
            self.last_transcript_path = out
            self._post("done", str(out))
        except Exception as e:
            self._post("error", str(e))

    # ---------- Record mode ----------

    def _refresh_devices(self) -> None:
        default_in = sd.default.device[0]
        self.device_options = []
        for idx, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                marker = "  (default)" if idx == default_in else ""
                self.device_options.append((f"[{idx}] {d['name']}{marker}", idx))
        labels = [o[0] for o in self.device_options]
        self.device_combo["values"] = labels
        if not labels:
            return
        for i, (_, idx) in enumerate(self.device_options):
            if idx == default_in:
                self.device_combo.current(i)
                return
        self.device_combo.current(0)

    def _selected_device_index(self) -> int | None:
        label = self.device_var.get()
        for opt_label, idx in self.device_options:
            if opt_label == label:
                return idx
        return None

    def _start_record_transcribe(self) -> None:
        if self.is_busy:
            return
        self._set_busy(True)
        threading.Thread(target=self._record_thread, daemon=True).start()

    def _record_thread(self) -> None:
        try:
            seconds = self.seconds_var.get()
            device = self._selected_device_index()
            self._post("clear", None)
            self._post("progress", 0)
            self._post("status", f"Recording for {seconds}s...")

            # sd.rec returns immediately; the buffer fills in the background.
            audio = sd.rec(
                int(seconds * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.int16,
                device=device,
            )

            t0 = time.time()
            while True:
                elapsed = time.time() - t0
                if elapsed >= seconds:
                    break
                self._post("progress", (elapsed / seconds) * 100)
                self._post("status", f"Recording... {int(seconds - elapsed)}s left")
                time.sleep(0.2)
            sd.wait()
            self._post("progress", 100)

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wav = RECORDINGS_DIR / f"rec_{stamp}.wav"
            wavfile.write(str(wav), SAMPLE_RATE, audio)
            self.audio_path = wav

            peak = int(np.max(np.abs(audio)))
            if peak < 100:
                self._post(
                    "error",
                    f"Recording was silent (peak {peak}/32767).\n\n"
                    "Try a different microphone in the dropdown, check Windows mic\n"
                    "permissions (Settings -> Privacy -> Microphone), or unmute the\n"
                    "selected device.",
                )
                return

            self._post("status", f"Audio level OK (peak {peak}/32767). Loading model...")
            model = self._load_model()
            language = self._language_code()
            self._post("status", "Transcribing...")

            segments, info = model.transcribe(
                str(wav), language=language, beam_size=5, vad_filter=True
            )
            transcript = self._consume_segments(segments, info.duration)
            out = TRANSCRIPTS_DIR / f"rec_{stamp}.txt"
            out.write_text(transcript, encoding="utf-8")
            self.last_transcript_path = out
            self._post("done", str(out))
        except Exception as e:
            self._post("error", str(e))

    # ---------- Shared transcription helpers ----------

    def _consume_segments(self, segments, duration: float) -> str:
        # Iterating segments triggers the actual decoding work in faster-whisper.
        duration = max(duration, 1.0)
        include_ts = self.timestamps_var.get()
        lines: list[str] = []
        for seg in segments:
            text = seg.text.strip()
            if include_ts:
                line = f"[{seg.start:6.2f}s -> {seg.end:6.2f}s] {text}"
                self._post("append", line + "\n")
            else:
                line = text
                self._post("append", text + " ")
            lines.append(line)
            self._post("progress", min((seg.end / duration) * 100, 100))
        return "\n".join(lines) if include_ts else " ".join(lines)

    def _load_model(self) -> WhisperModel:
        size = self.model_var.get()
        if self.cached_model is None or self.cached_model_size != size:
            self.cached_model = WhisperModel(size, device="cpu", compute_type="int8")
            self.cached_model_size = size
        return self.cached_model

    def _language_code(self) -> str | None:
        chosen = self.lang_var.get()
        for label, code in LANGUAGES:
            if label == chosen:
                return code
        return None

    # ---------- Thread <-> UI bridge ----------

    def _post(self, kind: str, data) -> None:
        self.events.put((kind, data))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "status":
                    self.status_var.set(data)
                elif kind == "clear":
                    self.text_widget.delete("1.0", tk.END)
                elif kind == "append":
                    self.text_widget.insert(tk.END, data)
                    self.text_widget.see(tk.END)
                elif kind == "progress":
                    self.progress["value"] = float(data)
                elif kind == "done":
                    self.status_var.set(f"Done. Saved to {data}")
                    self.progress["value"] = 100
                    self._set_busy(False)
                elif kind == "error":
                    self.status_var.set("Error")
                    self._set_busy(False)
                    messagebox.showerror("Error", data)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_events)

    def _set_busy(self, busy: bool) -> None:
        self.is_busy = busy
        state = "disabled" if busy else "normal"
        self.transcribe_btn.config(state=state)
        self.record_btn.config(state=state)

    # ---------- Bottom-row actions ----------

    def _copy(self) -> None:
        text = self.text_widget.get("1.0", tk.END).strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Copied to clipboard")

    def _save_as(self) -> None:
        text = self.text_widget.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Nothing to save", "Transcribe something first.")
            return
        default_name = (
            self.last_transcript_path.stem if self.last_transcript_path else "transcript"
        )
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text", "*.txt"), ("Markdown", "*.md"), ("All files", "*.*")],
        )
        if path:
            Path(path).write_text(text, encoding="utf-8")
            self.status_var.set(f"Saved to {path}")

    def _open_folder(self) -> None:
        os.startfile(TRANSCRIPTS_DIR)

    def _clear(self) -> None:
        self.text_widget.delete("1.0", tk.END)
        self.progress["value"] = 0
        self.status_var.set("Ready")


def main() -> int:
    root = TkinterDnD.Tk()
    TranscriptApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
