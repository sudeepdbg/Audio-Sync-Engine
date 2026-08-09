# Dub Sync & Audio QC Validator (v9)

**Author:** Sudeep Kumar | sudeepdbg@gmail.com | +91 95906 75753

A Flask-based Broadcast/OTT localization QC tool designed to validate dubbed audio against master references and perform advanced standalone audio quality checks. It bridges the gap between basic file comparison and deep signal analysis using `librosa`, `ffmpeg`, and optional ASR.

## 🎯 Core Modes

1. **Dub Sync + QC (`/upload`)**: Compares a dub against a master reference. Analyzes start offset, drift, clock-speed factor, onset DNA match, chroma DNA match, and runs the full advanced QC suite.
2. **Standalone Audio QC (`/qc`)**: No reference needed. Runs every check that doesn't require a comparison file against one or more independent audio files.

---

## 🚀 Quickstart

### Prerequisites
You need `ffmpeg` and `ffprobe` installed and accessible on your system `PATH`.
```bash
# Debian/Ubuntu
apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Installation & Execution
```bash
# Clone the repo
git clone https://github.com/sudeepdbg/Dub-Sync-Audio-QC-Validator.git
cd Dub-Sync-Audio-QC-Validator

# Install dependencies (use a virtualenv recommended)
pip install -r requirements.txt --break-system-packages

# Run the application
python3 audio_align.py
```
By default, the app binds to `http://127.0.0.1:5001`. For production, use Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5001 audio_align:app
```

### Optional: Enable ASR (Language ID + Profanity)
To enable Automatic Speech Recognition for language identification and profanity scanning:
```bash
pip install faster-whisper --break-system-packages
```
*Note: ASR is opt-in per request via the "Run ASR" checkbox. The first run downloads the model (~75MB).*

---

## 📁 Repository Contents

```
.
├── audio_align.py              # Flask app: routes, alignment analysis, orchestration
├── capability_extensions.py    # ASR, DME check, AD detection, spatial loudness
├── requirements.txt
└── templates/
    └── index.html              # Frontend (vanilla JS + ECharts)
```
*`audio_align.py` imports directly from `capability_extensions.py`—both must reside in the same folder.*

---

## 📡 API Endpoints

| Route | Method | Description |
| :--- | :--- | :--- |
| `/upload` | `POST` | **Sync Mode.** Upload `reference` and `comparison[]` files. Returns offset, drift, DNA match, and QC checks. |
| `/qc` | `POST` | **Standalone Mode.** Upload `files[]`. Returns levels, waveform, spectrum, and QC checks. |
| `/health` | `GET` | Liveness check. |
| `/metrics` | `GET` | Prometheus-format metrics. |
| `/wipe` | `POST` | Clears all session data. |

---

## 🛠 QC Capability Matrix

| Check | Sync Mode | Standalone | Notes |
| :--- | :---: | :---: | :--- |
| **Start offset / drift / speed** | ✅ | ❌ | Calculates clock-rate ratio and time-compression actions. |
| **Onset / Chroma DNA match** | ✅ | ❌ | Transient correlation vs. spectral/harmonic correlation. |
| **Loudness (LUFS) / True Peak** | ✅ | ✅ | Integrated loudness and sample peak measurement. |
| **Dropouts / Silence Gaps** | ✅ | ✅ | Uses `ffmpeg silencedetect`. |
| **Hum & Buzz / Rumble** | ✅ | ✅ | 50/60Hz hum and low-freq rumble (First 10s sample). |
| **Mono-in-Stereo (Dual-mono)** | ✅ | ✅ | Detects identical channels (First 5s sample). |
| **Spatial Loudness** | ✅ | ✅ | Targets -27 LUFS (Dolby immersive-mix guidance). |
| **DME Structural Check** | ✅ | ✅ | Heuristic band-energy/correlation (Requires M&E stem upload). |
| **Language ID / Profanity** | ✅ | ✅ | Opt-in via `run_asr=true`. Uses `faster-whisper`. |
| **Atmos Bed / AD Detection** | ✅ | ✅ | Container tags/disposition only. Object counts are *not* measurable. |

---

## ⚠️ Known Limitations & Design Choices

- **Sampling:** Hum, rumble, and dual-mono checks sample the first 5–10 seconds of the file. Intermittent mid-file issues may be missed.
- **Level Spikes:** Uses a coarse statistical peak-outlier heuristic (`ffmpeg astats`), not sample-accurate click/pop detection.
- **ASR Processing:** Whisper model calls are serialized behind a lock. Expect sequential processing for multi-file ASR requests.
- **Atmos & DME:** Dolby Renderer metadata (object count/position) cannot be extracted via `ffprobe`. Unmeasurable checks are explicitly reported as `null` rather than false passes.
- **AV Sync:** Lip-sync/AV sync is **not implemented** as this is strictly an audio-only tool.

---

## 💻 Frontend UI

The UI intentionally follows a utilitarian "pro-tool" aesthetic: dark background, monospace typography, flat borders, and no gradients or rounded corners. 

- Status is communicated via bracketed text (`[PASS]`, `[FAIL]`, `[WARN]`, `[SKIP]`) and color.
- Results feature tabbed views (Sync Analysis, Advanced QC, Spectrum).
- A `skip` state is rendered dim/gray, ensuring a check that wasn't performed never looks like a check that failed.




