import os
import uuid
import shutil
import hashlib
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import traceback
import subprocess
import soundfile as sf
import time
import threading
import gc
from scipy import signal
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

# --- OPTIONAL ADVANCED LIBRARIES ---
try:
    import demucs.separate
    HAS_DEMUCS = True
except ImportError:
    HAS_DEMUCS = False

try:
    import pyloudnorm as fln
    HAS_LOUDNORM = True
except ImportError:
    HAS_LOUDNORM = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 # 1GB Limit

# --- BROADCAST QUALITY THRESHOLDS ---
MAX_DRIFT_MS = 30
MAX_START_OFFSET_MS = 50
DUB_MATCH_THRESHOLD = 15
PERFORMANCE_SR = 22050  # "Goldilocks" rate for speed/accuracy balance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_VOLATILE_PATH = os.path.join(BASE_DIR, "data")
os.makedirs(MEDIA_VOLATILE_PATH, exist_ok=True)

FINGERPRINT_CACHE = {}
_cache_lock = threading.Lock()

# --- ANALYTICS ENGINES ---

def get_file_metadata(path):
    """Extracts technical specs including Bit-Depth for delivery verification."""
    ext = path.rsplit('.', 1)[1].upper() if '.' in path else "UNK"
    try:
        info = sf.info(path)
        return {
            "format": f"{ext} ({info.format})",
            "sr": f"{info.samplerate} Hz",
            "duration": info.duration,
            "duration_str": f"{round(info.duration, 2)}s",
            "bit_depth": info.subtype,
            "channels": info.channels
        }
    except:
        return {"format": ext, "sr": "N/A", "duration": 0, "duration_str": "0s", "bit_depth": "N/A", "channels": 0}

def classify_quality(path):
    """Scans for loudness (LUFS) and peak clipping."""
    try:
        data, rate = sf.read(path)
        if len(data.shape) > 1: data = np.mean(data, axis=1)
        max_val = np.max(np.abs(data))
        lufs = "N/A"
        if HAS_LOUDNORM:
            meter = fln.Meter(rate)
            lufs = f"{round(meter.integrated_loudness(data), 2)} LUFS"
        return {
            "lufs": lufs,
            "peak": "Clipping" if max_val > 0.99 else "Clean",
            "status": "Silent" if max_val < 0.001 else "Active"
        }
    except:
        return {"lufs": "N/A", "peak": "N/A", "status": "Error"}

def isolate_vocals(file_path, output_root):
    """Isolates dialogue using Demucs for better DNA matching."""
    if not HAS_DEMUCS: return file_path
    try:
        demucs.separate.main(["--mp3", "-n", "htdemucs", "-o", output_root, file_path])
        base = os.path.basename(file_path).rsplit('.', 1)[0]
        vocal_path = os.path.join(output_root, "htdemucs", base, "vocals.mp3")
        return vocal_path if os.path.exists(vocal_path) else file_path
    except: return file_path

def calculate_phase_correlation(path):
    """Detects if Left/Right channels are out of phase."""
    try:
        y, _ = librosa.load(path, sr=PERFORMANCE_SR, mono=False, duration=30)
        if y.ndim < 2 or y.shape[0] < 2: return "N/A (Mono)"
        correlation = np.corrcoef(y[0], y[1])[0, 1]
        return f"{round(correlation, 2)} ({'Healthy' if correlation > 0 else 'Phase Issue'})"
    except: return "Error"

def get_efficient_fingerprint(file_path):
    """Uses Chromaprint for Content DNA matching."""
    fpcalc_path = shutil.which("fpcalc") or "/usr/local/bin/fpcalc"
    try:
        cmd = [fpcalc_path, "-plain", file_path]
        return subprocess.check_output(cmd, timeout=30).decode().strip()
    except: return None

def compare_dna(fp_a, fp_b):
    if not fp_a or not fp_b: return 0.0
    from difflib import SequenceMatcher
    return round(SequenceMatcher(None, fp_a, fp_b).ratio() * 100, 2)

def get_offset(y_ref, y_comp, sr):
    hop = 512
    ref_env = librosa.feature.rms(y=y_ref, hop_length=hop)[0]
    comp_env = librosa.feature.rms(y=y_comp, hop_length=hop)[0]
    corr = signal.correlate(comp_env, ref_env, mode='full')
    lag = np.argmax(corr) - (len(ref_env) - 1)
    return round(float(lag * hop / sr * 1000), 2)

def analyze_sync(ref_path, comp_path, ref_meta):
    comp_meta = get_file_metadata(comp_path)
    
    # PERFORMANCE: Downsample to Goldilocks SR
    y_r, _ = librosa.load(ref_path, sr=PERFORMANCE_SR, duration=30)
    y_c, _ = librosa.load(comp_path, sr=PERFORMANCE_SR, duration=30)
    
    start_offset = get_offset(y_r, y_c, PERFORMANCE_SR)
    phase = calculate_phase_correlation(comp_path)
    
    # DNA Comparison
    dna_score = compare_dna(get_efficient_fingerprint(ref_path), get_efficient_fingerprint(comp_path))
    
    # Drift calculation
    duration = min(ref_meta['duration'], comp_meta['duration'])
    drift = 0.0
    if duration > 60:
        y_r_e, _ = librosa.load(ref_path, sr=PERFORMANCE_SR, offset=duration-30, duration=30)
        y_c_e, _ = librosa.load(comp_path, sr=PERFORMANCE_SR, offset=duration-30, duration=30)
        drift = round(abs(get_offset(y_r_e, y_c_e, PERFORMANCE_SR) - start_offset), 2)

    # Visualization
    plt.figure(figsize=(10, 2))
    plt.plot(y_r[:PERFORMANCE_SR*5], color='blue', alpha=0.3)
    plt.plot(y_c[:PERFORMANCE_SR*5], color='orange', alpha=0.7)
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

    return {
        "offset": start_offset, "drift": drift, "phase": phase, "dna": dna_score,
        "visual": base64.b64encode(buf.getvalue()).decode('utf-8'),
        "ref_meta": ref_meta, "comp_meta": comp_meta,
        "needs_review": abs(start_offset) > MAX_START_OFFSET_MS or drift > MAX_DRIFT_MS or dna_score < DUB_MATCH_THRESHOLD
    }

# --- ROUTES ---

@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    root = os.path.join(MEDIA_VOLATILE_PATH, session_id)
    os.makedirs(root, exist_ok=True)
    deep = request.form.get('deepAnalysis') == 'true'
    
    try:
        ref_file = request.files['reference']
        ref_path = os.path.join(root, secure_filename(ref_file.filename))
        ref_file.save(ref_path)
        ref_meta = get_file_metadata(ref_path)
        ref_proc = isolate_vocals(ref_path, root) if deep else ref_path
        
        results = []
        for f in request.files.getlist('comparison[]'):
            if not f.filename: continue
            f_path = os.path.join(root, secure_filename(f.filename))
            f.save(f_path)
            
            quality = classify_quality(f_path)
            f_proc = isolate_vocals(f_path, root) if deep else f_path
            sync = analyze_sync(ref_proc, f_proc, ref_meta)
            
            results.append({
                'filename': f.filename,
                'offset_ms': sync['offset'],
                'drift_ms': sync['drift'],
                'phase': sync['phase'],
                'dna': sync['dna'],
                'visual': sync['visual'],
                'ref_meta': sync['ref_meta'],
                'comp_meta': sync['comp_meta'],
                'quality': quality,
                'needs_review': sync['needs_review']
            })
            
        return jsonify({'reference': ref_file.filename, 'results': results})
    except Exception as e:
        return jsonify({'error': str(traceback.format_exc())}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
