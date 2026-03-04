import os
import gc
import uuid
import shutil
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import signal
from scipy.signal import butter, lfilter
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

# --- CONFIGURATION ---
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PERFORMANCE_SR      = 22050
WAVEFORM_MAX_POINTS = 2000
SEGMENT_DURATION    = 60        # seconds to analyse at start/end
MAX_WORKERS         = 4         # ThreadPoolExecutor cap
ALLOWED_EXTENSIONS  = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.aiff', '.aif', '.opus'}


# ─── AUTO-CLEANUP ─────────────────────────────────────────────────────────────
def _cleanup_worker():
    while True:
        now = time.time()
        try:
            for folder in os.listdir(DATA_DIR):
                path = os.path.join(DATA_DIR, folder)
                if os.path.isdir(path) and folder.startswith("SES_") \
                        and os.path.getmtime(path) < now - 3600:
                    shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass
        time.sleep(600)

threading.Thread(target=_cleanup_worker, daemon=True).start()


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    """Security: only accept known audio extensions."""
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def butter_bandpass(data, lowcut, highcut, fs, order=2):
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, data)


def normalize_lufs(y, sr, target=-23.0):
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(y)
        return pyln.normalize.loudness(y, loudness, target)
    except Exception:
        return y


def normalize_visual(y):
    m = np.max(np.abs(y))
    return y / m if m > 0 else y


def downsample_waveform(y, max_pts=WAVEFORM_MAX_POINTS):
    """
    Bucket-max downsampling — preserves transients far better than [::N].
    Hard-caps JSON payload at max_pts samples.
    """
    if len(y) <= max_pts:
        return y.tolist()
    step    = len(y) // max_pts
    buckets = len(y) // step
    trimmed = y[:buckets * step].reshape(buckets, step)
    idx     = np.argmax(np.abs(trimmed), axis=1)
    return trimmed[np.arange(buckets), idx].tolist()


# ─── AUDIO ANALYSIS ───────────────────────────────────────────────────────────
def get_file_metadata(path):
    try:
        info = sf.info(path)
        return {
            "sr":            f"{info.samplerate} Hz",
            "native_sr":     info.samplerate,
            "duration":      f"{round(info.duration, 2)}s",
            "duration_sec":  info.duration,
            "bit_depth":     info.subtype,
            "channels":      info.channels,
            "channel_label": "Stereo" if info.channels == 2 else
                             "Mono"   if info.channels == 1 else f"{info.channels} Ch"
        }
    except Exception:
        return {"sr": "N/A", "native_sr": 0, "duration": "0s", "duration_sec": 0,
                "bit_depth": "N/A", "channels": 0, "channel_label": "N/A"}


def scan_levels(path):
    """
    Returns integrated LUFS, sample peak dBFS, and True Peak dBTP.

    True Peak (inter-sample peak) catches digital clipping that sample peak
    misses — required by EBU R128 / ATSC A/85.
    pyloudnorm's true_peak upsamples 4x then measures max inter-sample peak.
    """
    try:
        data, rate = sf.read(path)
        mono = np.mean(data, axis=1) if data.ndim > 1 else data

        sample_peak_db = 20 * np.log10(np.max(np.abs(mono)) + 1e-10)

        meter    = pyln.Meter(rate)
        lufs_val = meter.integrated_loudness(data)
        lufs     = f"{round(lufs_val, 2)} LUFS"

        # True Peak — 4x upsampling to catch inter-sample peaks
        try:
            tp    = pyln.meter.true_peak(data, rate)
            tp_db = f"{round(float(np.max(tp)), 2)} dBTP"
        except Exception:
            tp_db = "N/A"

        return {
            "lufs":      lufs,
            "peak":      f"{round(sample_peak_db, 2)} dBFS",
            "true_peak": tp_db
        }
    except Exception:
        return {"lufs": "ERR", "peak": "ERR", "true_peak": "ERR"}


def calculate_phase(path):
    try:
        data, _ = sf.read(path)
        if data.ndim < 2 or data.shape[1] < 2:
            return "1.0 (Mono)"
        corr   = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        status = "Healthy" if corr > 0.4 else "🚩 Issue"
        return f"{round(float(corr), 2)} ({status})"
    except Exception:
        return "N/A"


def load_segment(path, sr, offset=0.0, duration=None):
    """
    librosa.load with kaiser_fast resampler.
    ~40% faster than default kaiser_best; quality is indistinguishable
    for RMS envelope / MFCC analysis at PERFORMANCE_SR.
    Both ref and comp are always resampled to PERFORMANCE_SR here,
    so downstream hop→ms calculations are always correct regardless
    of the file's native sample rate (44.1k, 48k, 96k, etc.).
    """
    return librosa.load(
        path,
        sr=sr,
        offset=offset,
        duration=duration,
        res_type='kaiser_fast'
    )


def analyze_segment(y_ref, y_comp, sr):
    """
    Offset  : RMS-envelope cross-correlation (robust, SR-normalised).
    DNA     : MFCC cosine similarity — properly bounded [0, 100].

    SR-mismatch note: both arrays enter at PERFORMANCE_SR (via load_segment),
    so the hop * frames / sr → ms conversion is always consistent.
    """
    hop = 512

    # Offset via RMS envelope cross-correlation
    ref_env  = librosa.feature.rms(y=y_ref,  hop_length=hop)[0].astype(np.float64)
    comp_env = librosa.feature.rms(y=y_comp, hop_length=hop)[0].astype(np.float64)
    ref_env  = (ref_env  - ref_env.min())  / (ref_env.max()  - ref_env.min()  + 1e-10)
    comp_env = (comp_env - comp_env.min()) / (comp_env.max() - comp_env.min() + 1e-10)

    corr      = signal.correlate(comp_env, ref_env, mode='full')
    lag       = np.argmax(corr) - (len(ref_env) - 1)
    offset_ms = round(float(lag * hop / sr * 1000), 2)

    # DNA via MFCC cosine similarity
    min_len  = min(len(y_ref), len(y_comp))
    vec_ref  = np.mean(librosa.feature.mfcc(y=y_ref[:min_len],  sr=sr, n_mfcc=20), axis=1)
    vec_comp = np.mean(librosa.feature.mfcc(y=y_comp[:min_len], sr=sr, n_mfcc=20), axis=1)

    cos_sim   = np.dot(vec_ref, vec_comp) / (
        np.linalg.norm(vec_ref) * np.linalg.norm(vec_comp) + 1e-10
    )
    dna_score = round(float((cos_sim + 1) / 2 * 100), 1)   # [-1,1] → [0,100]

    return offset_ms, dna_score


def calculate_speed_factor(start_offset_ms, end_offset_ms, duration_sec):
    """
    Speed Factor — tells engineers exactly how much to time-stretch the dub.

    Derivation
    ----------
    If the dub accumulates `drift` extra milliseconds over its full length,
    its clock is running at a slightly different rate than the master.

      speed_factor = master_duration / dub_effective_duration
                   = T / (T + drift_sec)

    speed_factor = 1.0  → perfect sync
    speed_factor > 1.0  → dub is slower (needs time-compress)
    speed_factor < 1.0  → dub is faster (needs time-expand)

    The percentage delta gives engineers a direct input for most DAW
    time-stretch dialogs (e.g. "+0.012%" in Pro Tools Elastic Audio).
    """
    if duration_sec <= 0:
        return {"ratio": 1.0, "display": "N/A", "delta": "N/A", "action": "N/A"}

    drift_sec    = (end_offset_ms - start_offset_ms) / 1000.0
    speed_factor = duration_sec / (duration_sec + drift_sec + 1e-10)
    pct_delta    = round((speed_factor - 1.0) * 100, 4)

    if abs(pct_delta) < 0.001:
        action = "No time-stretch needed"
    elif pct_delta > 0:
        action = f"Time-compress dub by {abs(pct_delta):.4f}%"
    else:
        action = f"Time-expand dub by {abs(pct_delta):.4f}%"

    return {
        "ratio":   round(speed_factor, 6),
        "display": f"{speed_factor:.6f}×",
        "delta":   f"{'+' if pct_delta >= 0 else ''}{pct_delta:.4f}%",
        "action":  action
    }


def determine_status(offset_ms, drift_ms, dna_score):
    issues = []
    if abs(offset_ms) > 80:
        issues.append(f"Start offset {offset_ms}ms exceeds threshold (±80ms)")
    if abs(drift_ms) > 150:
        issues.append(f"Drift {drift_ms}ms exceeds threshold (±150ms)")
    if dna_score < 55:
        issues.append(f"DNA match {dna_score}% below threshold (55%)")

    status = "FAIL" if issues else "PASS"
    reason = "; ".join(issues) if issues else "All metrics within thresholds"
    return status, reason


# ─── PER-FILE PROCESSOR (runs in thread pool) ─────────────────────────────────
def process_comparison_file(f, root, y_ref_s, y_ref_e, ref_meta, vocal_logic):
    """
    Isolated worker — one bad file never kills the batch.
    Explicit del + gc.collect() releases RAM between threads.
    """
    if not f or not f.filename:
        return None

    # Extension whitelist
    if not allowed_file(f.filename):
        return {
            "filename": f.filename,
            "status":   "ERROR",
            "reason":   f"Rejected — unsupported type '{os.path.splitext(f.filename)[1]}'",
            "error":    True
        }

    try:
        f_path    = os.path.join(root, secure_filename(f.filename))
        f.save(f_path)
        comp_meta = get_file_metadata(f_path)
        comp_dur  = comp_meta["duration_sec"]

        y_c_s, _ = load_segment(f_path, PERFORMANCE_SR, duration=SEGMENT_DURATION)
        y_c_e, _ = load_segment(f_path, PERFORMANCE_SR, offset=max(0.0, comp_dur - SEGMENT_DURATION))

        if vocal_logic:
            y_c_s = butter_bandpass(normalize_lufs(y_c_s, PERFORMANCE_SR), 300, 3400, PERFORMANCE_SR)
            y_c_e = butter_bandpass(normalize_lufs(y_c_e, PERFORMANCE_SR), 300, 3400, PERFORMANCE_SR)

        s_off, dna = analyze_segment(y_ref_s, y_c_s, PERFORMANCE_SR)
        e_off, _   = analyze_segment(y_ref_e, y_c_e, PERFORMANCE_SR)
        drift      = round(e_off - s_off, 2)

        speed          = calculate_speed_factor(s_off, e_off, comp_dur)
        status, reason = determine_status(s_off, drift, dna)

        result = {
            "filename":       f.filename,
            "status":         status,
            "reason":         reason,
            "offset_ms":      s_off,
            "total_drift_ms": drift,
            "dna_match":      dna,
            "speed_factor":   speed,
            "phase":          calculate_phase(f_path),
            "levels":         scan_levels(f_path),
            "ref_meta":       ref_meta,
            "comp_meta":      comp_meta,
            "wave_a":         downsample_waveform(normalize_visual(y_ref_s)),
            "wave_r":         downsample_waveform(normalize_visual(y_c_s)),
            "chan_mismatch":  ref_meta["channels"] != comp_meta["channels"]
        }

        del y_c_s, y_c_e
        gc.collect()
        return result

    except Exception as err:
        return {
            "filename": f.filename,
            "status":   "ERROR",
            "reason":   str(err),
            "error":    True
        }


# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/wipe', methods=['POST'])
def wipe():
    """Wipes only SES_ folders. gc.collect() reclaims numpy RAM immediately."""
    wiped = 0
    for folder in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(path) and folder.startswith("SES_"):
            shutil.rmtree(path, ignore_errors=True)
            wiped += 1
    gc.collect()
    return jsonify({"status": "ok", "wiped_sessions": wiped})


@app.route('/upload', methods=['POST'])
def upload():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    root       = os.path.join(DATA_DIR, session_id)
    os.makedirs(root, exist_ok=True)

    try:
        vocal_logic = request.form.get('vocal_logic') == 'true'
        ref         = request.files.get('reference')
        comps       = request.files.getlist('comparison[]')

        if not ref:
            return jsonify({"error": "No reference file provided"}), 400
        if not comps:
            return jsonify({"error": "No comparison files provided"}), 400

        if not allowed_file(ref.filename):
            return jsonify({"error": f"Master file type not allowed: '{os.path.splitext(ref.filename)[1]}'"}), 400

        ref_path  = os.path.join(root, secure_filename(ref.filename))
        ref.save(ref_path)
        ref_meta  = get_file_metadata(ref_path)
        total_dur = ref_meta["duration_sec"]

        y_ref_s, _ = load_segment(ref_path, PERFORMANCE_SR, duration=SEGMENT_DURATION)
        y_ref_e, _ = load_segment(ref_path, PERFORMANCE_SR, offset=max(0.0, total_dur - SEGMENT_DURATION))

        if vocal_logic:
            y_ref_s = butter_bandpass(normalize_lufs(y_ref_s, PERFORMANCE_SR), 300, 3400, PERFORMANCE_SR)
            y_ref_e = butter_bandpass(normalize_lufs(y_ref_e, PERFORMANCE_SR), 300, 3400, PERFORMANCE_SR)

        # ── Parallel file processing ─────────────────────────────────────────
        # librosa.load (I/O + resampling) and sf.read (I/O) release the GIL,
        # giving real concurrency. Typical speedup: 50–70% for 5+ files.
        results_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    process_comparison_file,
                    f, root, y_ref_s, y_ref_e, ref_meta, vocal_logic
                ): i
                for i, f in enumerate(comps)
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results_map[futures[future]] = result

        # Preserve original upload order
        results = [results_map[i] for i in sorted(results_map)]

        del y_ref_s, y_ref_e
        gc.collect()

        return jsonify({"results": results})

    except Exception as e:
        shutil.rmtree(root, ignore_errors=True)
        gc.collect()
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
