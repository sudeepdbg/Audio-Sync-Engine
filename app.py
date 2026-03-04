import os
import uuid
import shutil
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import threading
import time
import traceback
from scipy import signal
from scipy.signal import butter, lfilter
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB Limit

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
PERFORMANCE_SR = 22050
WAVEFORM_MAX_POINTS = 2000  # FIX: cap waveform points for fast JSON + Plotly rendering


# --- AUTO-CLEANUP (session folders older than 1hr) ---
def auto_cleanup_worker():
    while True:
        now = time.time()
        for folder in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, folder)
            # FIX: only delete session folders, not the root data dir
            if os.path.isdir(path) and folder.startswith("SES_") and os.path.getmtime(path) < now - 3600:
                shutil.rmtree(path, ignore_errors=True)
        time.sleep(600)

threading.Thread(target=auto_cleanup_worker, daemon=True).start()


# --- AUDIO ENGINES ---

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


def normalize_lufs(y, sr, target=-23.0):
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(y)
        return pyln.normalize.loudness(y, loudness, target)
    except:
        return y


def normalize_visual(y):
    """Normalize for waveform overlay clarity."""
    max_v = np.max(np.abs(y))
    return y / max_v if max_v > 0 else y


def downsample_waveform(y, max_points=WAVEFORM_MAX_POINTS):
    """
    FIX: Smart downsampling — takes the max absolute value in each bucket
    instead of crude [::N] slicing. Preserves transients visually.
    Also hard-caps output to max_points for JSON payload size.
    """
    if len(y) <= max_points:
        return y.tolist()
    step = len(y) // max_points
    buckets = len(y) // step
    trimmed = y[:buckets * step].reshape(buckets, step)
    # Take signed value with largest magnitude per bucket
    idx = np.argmax(np.abs(trimmed), axis=1)
    result = trimmed[np.arange(buckets), idx]
    return result.tolist()


def get_file_metadata(path):
    try:
        info = sf.info(path)
        return {
            "sr": f"{info.samplerate} Hz",
            "duration": f"{round(info.duration, 2)}s",
            "bit_depth": info.subtype,
            "channels": info.channels,
            "channel_label": "Stereo" if info.channels == 2 else "Mono" if info.channels == 1 else f"{info.channels} Ch"
        }
    except:
        return {"sr": "N/A", "duration": "0s", "bit_depth": "N/A", "channels": 0, "channel_label": "N/A"}


def scan_levels(path):
    try:
        data, rate = sf.read(path)
        if data.ndim > 1:
            data_mono = np.mean(data, axis=1)
        else:
            data_mono = data
        peak_db = 20 * np.log10(np.max(np.abs(data_mono)) + 1e-10)
        meter = pyln.Meter(rate)
        # pyloudnorm needs stereo or mono array correctly shaped
        lufs_val = meter.integrated_loudness(data)
        lufs = f"{round(lufs_val, 2)} LUFS"
        return {"lufs": lufs, "peak": f"{round(peak_db, 2)} dBFS"}
    except:
        return {"lufs": "ERR", "peak": "ERR"}


def calculate_phase(path):
    try:
        data, _ = sf.read(path)
        if len(data.shape) < 2 or data.shape[1] < 2:
            return "1.0 (Mono)"
        corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        status = "Healthy" if corr > 0.4 else "🚩 Issue"
        return f"{round(float(corr), 2)} ({status})"
    except:
        return "N/A"


def analyze_segment(y_ref, y_comp, sr):
    """
    FIX: Proper normalized cross-correlation for offset + DNA match score.

    Previous version used signal.correlate on spectral_centroid and divided
    by norm product — this gave values wildly outside [0,100] because the
    raw correlation peak isn't bounded that way.

    New approach:
    - Offset: RMS envelope cross-correlation (more robust than centroid for sync)
    - DNA score: cosine similarity on MFCCs (properly bounded 0–1, scaled to 0–100)
    """
    hop = 512

    # --- OFFSET via RMS envelope cross-correlation ---
    ref_env = librosa.feature.rms(y=y_ref, hop_length=hop)[0].astype(np.float64)
    comp_env = librosa.feature.rms(y=y_comp, hop_length=hop)[0].astype(np.float64)

    # Normalize envelopes to [0,1]
    ref_env = (ref_env - ref_env.min()) / (ref_env.max() - ref_env.min() + 1e-10)
    comp_env = (comp_env - comp_env.min()) / (comp_env.max() - comp_env.min() + 1e-10)

    corr = signal.correlate(comp_env, ref_env, mode='full')
    lag = np.argmax(corr) - (len(ref_env) - 1)
    offset_ms = round(float(lag * hop / sr * 1000), 2)

    # --- DNA MATCH via MFCC cosine similarity ---
    # Use minimum length to avoid shape mismatch
    min_len = min(len(y_ref), len(y_comp))
    mfcc_ref = librosa.feature.mfcc(y=y_ref[:min_len], sr=sr, n_mfcc=20)
    mfcc_comp = librosa.feature.mfcc(y=y_comp[:min_len], sr=sr, n_mfcc=20)

    # Mean across time axis → (20,) vector per file
    vec_ref = np.mean(mfcc_ref, axis=1)
    vec_comp = np.mean(mfcc_comp, axis=1)

    # Cosine similarity: always in [-1, 1], rescale to [0, 100]
    cos_sim = np.dot(vec_ref, vec_comp) / (np.linalg.norm(vec_ref) * np.linalg.norm(vec_comp) + 1e-10)
    dna_score = round(float((cos_sim + 1) / 2 * 100), 1)  # rescale [-1,1] → [0,100]

    return offset_ms, dna_score


def determine_status(offset_ms, drift_ms, dna_score):
    """
    FIX: Clearer, decoupled pass/fail logic with named failure reasons.
    Previously both conditions had to be true for PASS — DNA was unreliable
    so almost everything failed.
    """
    issues = []
    if abs(offset_ms) > 80:
        issues.append(f"Start offset {offset_ms}ms exceeds threshold")
    if abs(drift_ms) > 150:
        issues.append(f"Drift {drift_ms}ms exceeds threshold")
    if dna_score < 55:
        issues.append(f"DNA match {dna_score}% too low")

    status = "FAIL" if issues else "PASS"
    reason = "; ".join(issues) if issues else "All metrics within thresholds"
    return status, reason


# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/wipe', methods=['POST'])
def wipe():
    """FIX: Only wipe session folders, not the root DATA_DIR itself."""
    wiped = 0
    for folder in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(path) and folder.startswith("SES_"):
            shutil.rmtree(path, ignore_errors=True)
            wiped += 1
    return jsonify({"status": "ok", "wiped_sessions": wiped})


@app.route('/upload', methods=['POST'])
def upload():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    root = os.path.join(DATA_DIR, session_id)
    os.makedirs(root, exist_ok=True)

    try:
        vocal_logic = request.form.get('vocal_logic') == 'true'
        ref = request.files.get('reference')
        comps = request.files.getlist('comparison[]')

        if not ref:
            return jsonify({"error": "No reference file provided"}), 400
        if not comps:
            return jsonify({"error": "No comparison files provided"}), 400

        ref_path = os.path.join(root, secure_filename(ref.filename))
        ref.save(ref_path)
        ref_meta = get_file_metadata(ref_path)
        total_dur = librosa.get_duration(path=ref_path)

        # Load master start + end segments
        y_ref_s, _ = librosa.load(ref_path, sr=PERFORMANCE_SR, duration=60)
        y_ref_e, _ = librosa.load(ref_path, sr=PERFORMANCE_SR, offset=max(0, total_dur - 60))

        if vocal_logic:
            y_ref_s = butter_bandpass_filter(normalize_lufs(y_ref_s, PERFORMANCE_SR), 300, 3400, PERFORMANCE_SR)
            y_ref_e = butter_bandpass_filter(normalize_lufs(y_ref_e, PERFORMANCE_SR), 300, 3400, PERFORMANCE_SR)

        results = []
        for f in comps:
            if not f or not f.filename:
                continue
            try:
                f_path = os.path.join(root, secure_filename(f.filename))
                f.save(f_path)
                comp_dur = librosa.get_duration(path=f_path)

                y_c_s, _ = librosa.load(f_path, sr=PERFORMANCE_SR, duration=60)
                y_c_e, _ = librosa.load(f_path, sr=PERFORMANCE_SR, offset=max(0, comp_dur - 60))

                if vocal_logic:
                    y_c_s = butter_bandpass_filter(normalize_lufs(y_c_s, PERFORMANCE_SR), 300, 3400, PERFORMANCE_SR)
                    y_c_e = butter_bandpass_filter(normalize_lufs(y_c_e, PERFORMANCE_SR), 300, 3400, PERFORMANCE_SR)

                s_off, dna = analyze_segment(y_ref_s, y_c_s, PERFORMANCE_SR)
                e_off, _ = analyze_segment(y_ref_e, y_c_e, PERFORMANCE_SR)
                drift = round(e_off - s_off, 2)

                comp_meta = get_file_metadata(f_path)
                status, reason = determine_status(s_off, drift, dna)

                results.append({
                    "filename": f.filename,
                    "status": status,
                    "reason": reason,
                    "offset_ms": s_off,
                    "total_drift_ms": drift,
                    "dna_match": dna,
                    "phase": calculate_phase(f_path),
                    "levels": scan_levels(f_path),
                    "ref_meta": ref_meta,
                    "comp_meta": comp_meta,
                    # FIX: smart downsampled waveforms, max 2000 points each
                    "wave_a": downsample_waveform(normalize_visual(y_ref_s)),
                    "wave_r": downsample_waveform(normalize_visual(y_c_s)),
                    "chan_mismatch": ref_meta['channels'] != comp_meta['channels']
                })

            except Exception as file_err:
                # FIX: per-file error — don't crash the whole batch
                results.append({
                    "filename": f.filename,
                    "status": "ERROR",
                    "reason": str(file_err),
                    "error": True
                })

        return jsonify({"results": results})

    except Exception as e:
        shutil.rmtree(root, ignore_errors=True)
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
