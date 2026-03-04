import os
import uuid
import shutil
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import threading
import time
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
    except: return y

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
    except: return {"sr": "N/A", "duration": "0s", "bit_depth": "N/A", "channels": 0, "channel_label": "N/A"}

def calculate_phase(path):
    try:
        data, _ = sf.read(path)
        if len(data.shape) < 2 or data.shape[1] < 2: return "1.0 (Mono)"
        corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        status = "Healthy" if corr > 0.4 else "Wide" if corr > 0 else "🚩 Phase Issue"
        return f"{round(float(corr), 2)} ({status})"
    except: return "N/A"

def scan_levels(path):
    try:
        data, rate = sf.read(path)
        peak_db = 20 * np.log10(np.max(np.abs(data)) + 1e-10)
        meter = pyln.Meter(rate)
        lufs = f"{round(meter.integrated_loudness(data), 2)} LUFS"
        return {"lufs": lufs, "peak": f"{round(peak_db, 2)} dBFS"}
    except: return {"lufs": "ERR", "peak": "ERR"}

def analyze_segment(y_ref, y_comp, sr):
    feat_a = librosa.feature.spectral_centroid(y=y_ref, sr=sr)[0]
    feat_r = librosa.feature.spectral_centroid(y=y_comp, sr=sr)[0]
    feat_a = (feat_a - np.mean(feat_a)) / (np.std(feat_a) + 1e-10)
    feat_r = (feat_r - np.mean(feat_r)) / (np.std(feat_r) + 1e-10)
    corr = signal.correlate(feat_r, feat_a, mode='same')
    lag = np.argmax(corr) - len(feat_a) // 2
    offset_ms = round(float(lag * 512 / sr * 1000), 2)
    score = round(float(np.max(corr) / (np.linalg.norm(feat_a) * np.linalg.norm(feat_r))), 2) * 100
    return offset_ms, min(score, 100.0)

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

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
    total_dur = librosa.get_duration(path=ref_path)
    
    # Process Master segments (Start and End)
    y_ref_s, _ = librosa.load(ref_path, sr=PERFORMANCE_SR, duration=60)
    y_ref_e, _ = librosa.load(ref_path, sr=PERFORMANCE_SR, offset=max(0, total_dur-60))
    
    if vocal_logic:
        y_ref_s = butter_bandpass_filter(normalize_lufs(y_ref_s, PERFORMANCE_SR), 300, 3400, PERFORMANCE_SR)
        y_ref_e = butter_bandpass_filter(normalize_lufs(y_ref_e, PERFORMANCE_SR), 300, 3400, PERFORMANCE_SR)

    results = []
    for f in comps:
        f_path = os.path.join(root, secure_filename(f.filename))
        f.save(f_path)
        comp_dur = librosa.get_duration(path=f_path)
        
        y_c_s, _ = librosa.load(f_path, sr=PERFORMANCE_SR, duration=60)
        y_c_e, _ = librosa.load(f_path, sr=PERFORMANCE_SR, offset=max(0, comp_dur-60))

        if vocal_logic:
            y_c_s = butter_bandpass_filter(normalize_lufs(y_c_s, PERFORMANCE_SR), 300, 3400, PERFORMANCE_SR)
            y_c_e = butter_bandpass_filter(normalize_lufs(y_c_e, PERFORMANCE_SR), 300, 3400, PERFORMANCE_SR)

        s_off, dna = analyze_segment(y_ref_s, y_c_s, PERFORMANCE_SR)
        e_off, _ = analyze_segment(y_ref_e, y_c_e, PERFORMANCE_SR)
        drift = round(e_off - s_off, 2)
        
        levels = scan_levels(f_path)
        phase = calculate_phase(f_path)
        comp_meta = get_file_metadata(f_path)

        results.append({
            "filename": f.filename,
            "status": "PASS" if (dna > 70 and abs(drift) < 100) else "FAIL",
            "offset_ms": s_off,
            "end_offset_ms": e_off,
            "total_drift_ms": drift,
            "dna_match": dna,
            "phase": phase,
            "levels": levels,
            "ref_meta": ref_meta,
            "comp_meta": comp_meta,
            "wave_a": y_ref_s[::50].tolist(),
            "wave_r": y_c_s[::50].tolist(),
            "chan_mismatch": ref_meta['channels'] != comp_meta['channels']
        })

    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
