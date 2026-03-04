import os
import uuid
import shutil
import numpy as np
import librosa
import librosa.display
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
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 # 1GB Limit

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
PERFORMANCE_SR = 22050 

# --- AUTO-CLEANUP ---
def auto_cleanup_worker():
    while True:
        now = time.time()
        for folder in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, folder)
            if os.path.isdir(path) and os.path.getmtime(path) < now - 3600:
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

def get_file_metadata(path):
    try:
        info = sf.info(path)
        return {
            "sr": f"{info.samplerate} Hz",
            "duration": f"{round(info.duration, 2)}s",
            "bit_depth": info.subtype,
            "channels": info.channels,
            "channel_label": "5.1 Surround" if info.channels == 6 else "Stereo" if info.channels == 2 else f"{info.channels} Ch"
        }
    except:
        return {"sr": "N/A", "duration": "0s", "bit_depth": "N/A", "channels": 0, "channel_label": "N/A"}

def calculate_phase(path):
    try:
        data, _ = sf.read(path)
        if len(data.shape) < 2 or data.shape[1] < 2:
            return "1.0 (Mono)"
        corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        status = "Healthy" if corr > 0.4 else "Wide" if corr > 0 else "🚩 Phase Issue"
        return f"{round(float(corr), 2)} ({status})"
    except:
        return "N/A"

def scan_levels(path):
    try:
        data, rate = sf.read(path)
        peak_db = 20 * np.log10(np.max(np.abs(data)) + 1e-10)
        meter = pyln.Meter(rate)
        lufs = f"{round(meter.integrated_loudness(data), 2)} LUFS"
        return {"lufs": lufs, "peak": f"{round(peak_db, 2)} dBFS"}
    except:
        return {"lufs": "ERR", "peak": "ERR"}

def analyze_sync(y_a, y_r, sr):
    # Feature Extraction (Spectral Centroid)
    feat_a = librosa.feature.spectral_centroid(y=y_a, sr=sr)[0]
    feat_r = librosa.feature.spectral_centroid(y=y_r, sr=sr)[0]
    
    # Feature Normalization
    feat_a = (feat_a - np.mean(feat_a)) / (np.std(feat_a) + 1e-10)
    feat_r = (feat_r - np.mean(feat_r)) / (np.std(feat_r) + 1e-10)

    # Correlation
    corr = signal.correlate(feat_r, feat_a, mode='same')
    lag = np.argmax(corr) - len(feat_a) // 2
    offset_ms = round(float(lag * 512 / sr * 1000), 2)
    
    # DNA Match Score
    score = round(float(np.max(corr) / (np.linalg.norm(feat_a) * np.linalg.norm(feat_r))), 2) * 100
    return offset_ms, min(score, 100.0)

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/wipe', methods=['POST'])
def wipe():
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    return jsonify({"status": "Cache Wiped"})

@app.route('/upload', methods=['POST'])
def upload():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    root = os.path.join(DATA_DIR, session_id)
    os.makedirs(root, exist_ok=True)
    
    vocal_logic = request.form.get('vocal_logic') == 'true'
    ref = request.files['reference']
    comps = request.files.getlist('comparison[]')

    ref_path = os.path.join(root, secure_filename(ref.filename))
    ref.save(ref_path)
    ref_meta = get_file_metadata(ref_path)
    
    # Load Master for Analysis
    y_ref, _ = librosa.load(ref_path, sr=PERFORMANCE_SR, duration=60)
    if vocal_logic:
        y_ref = normalize_lufs(y_ref, PERFORMANCE_SR)
        y_ref_proc = butter_bandpass_filter(y_ref, 300, 3400, PERFORMANCE_SR)
    else:
        y_ref_proc = y_ref

    results = []
    for f in comps:
        f_path = os.path.join(root, secure_filename(f.filename))
        f.save(f_path)
        
        y_comp, _ = librosa.load(f_path, sr=PERFORMANCE_SR, duration=60)
        if vocal_logic:
            y_comp = normalize_lufs(y_comp, PERFORMANCE_SR)
            y_comp_proc = butter_bandpass_filter(y_comp, 300, 3400, PERFORMANCE_SR)
        else:
            y_comp_proc = y_comp

        offset, dna = analyze_sync(y_ref_proc, y_comp_proc, PERFORMANCE_SR)
        levels = scan_levels(f_path)
        phase = calculate_phase(f_path)
        comp_meta = get_file_metadata(f_path)

        # Downsample for Plotly performance (1 point per 50)
        step = 50
        results.append({
            "filename": f.filename,
            "offset_ms": offset,
            "dna_match": dna,
            "phase": phase,
            "levels": levels,
            "ref_meta": ref_meta,
            "comp_meta": comp_meta,
            "wave_a": y_ref_proc[::step].tolist(),
            "wave_r": y_comp_proc[::step].tolist(),
            "chan_mismatch": ref_meta['channels'] != comp_meta['channels']
        })

    return jsonify({"reference": ref.filename, "results": results})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
