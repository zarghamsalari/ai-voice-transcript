# Deployment guide

This project has three runnable entry points:

| File | Mode | Where it runs |
|---|---|---|
| `app.py` | Desktop GUI (Tkinter) | Your local machine |
| `streamlit_app.py` | Web app | Any Python web host |
| `transcribe_file.py` / `record_and_transcribe.py` | CLI | Your local machine |

Only `streamlit_app.py` is intended for cloud hosting. The desktop versions need a screen and a microphone, which a server doesn't have.

---

## Option 1 — Hugging Face Spaces (recommended)

**Why HF Spaces is the right choice for this app:** it's purpose-built for ML demos, offers ~16 GB RAM on the free tier (vs. 512 MB on Render free), never sleeps, and auto-installs from `requirements.txt`.

### Steps

1. **Push this repo to GitHub** (see the GitHub section below) so you have a stable URL.
2. Sign in at <https://huggingface.co> (free account).
3. Click **New** -> **Space**.
4. Fill in:
   - **Space name:** `ai-voice-transcript` (or whatever you like)
   - **License:** MIT (or your choice)
   - **SDK:** Streamlit
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public or Private
5. After it creates the empty Space, clone it locally:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/ai-voice-transcript hf-space
   ```
6. Copy these files from this repo into the cloned `hf-space` folder:
   - `streamlit_app.py`
   - `requirements.txt`
   - `README.md` (the YAML frontmatter at the top is what configures the Space)
7. Push:
   ```bash
   cd hf-space
   git add .
   git commit -m "Initial deployment"
   git push
   ```
8. HF builds the image, installs packages, and starts the app. After ~3-5 min you'll have a public URL like `https://huggingface.co/spaces/<you>/ai-voice-transcript`.

### After deployment

- The first transcription is slow (Whisper model downloads to the Space's disk on first request).
- Stick to `base` or `small` models on free hardware; `medium` and `large-v3` need ~6-10 GB RAM and are sluggish on CPU.

---

## Option 2 — Render.com

Render hosts general web services. Free tier has constraints that matter for an ML app: **512 MB RAM** (Whisper `base` is borderline) and **services sleep after 15 minutes idle** (~30s cold start on next request). For a real audience, upgrade to the $7/mo paid plan.

### Steps

1. Push this repo to GitHub.
2. Sign in at <https://render.com>.
3. **New** -> **Web Service** -> connect your GitHub repo.
4. Fill in:
   - **Environment:** Python 3
   - **Build Command:**
     ```
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```
     streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
     ```
   - **Instance Type:** Free (or Starter for $7/mo if you want it to stay warm)
5. Click **Create Web Service**. First deploy takes ~5-10 minutes.

### Free-tier survival tips

- Force the model to `tiny` to fit in 512 MB. In `streamlit_app.py`, change `index=1` to `index=0` in the model dropdown so `tiny` is the default.
- Or lock the model: set the dropdown's `value="tiny", disabled=True` so users can't pick a larger one.

---

## Option 3 — Streamlit Community Cloud

Native home for Streamlit apps, free, easy.

1. Push to GitHub.
2. Go to <https://share.streamlit.io>.
3. Connect your repo, pick `streamlit_app.py` as the entry point, deploy.
4. Free tier: 1 GB RAM, may cold-start. `base` model fits.

---

## Keeping GitHub + your host in sync

Easiest workflow:

- **Single source of truth:** push code changes to GitHub.
- For HF Spaces: either (a) add the HF Space as a second remote and push to both, or (b) use the HF Spaces "Connect to GitHub" feature in the Space settings, which auto-syncs.
- For Render: enable auto-deploy on push (default when you connect a repo).

After that, every `git push` to your `main` branch re-deploys the hosted app.
