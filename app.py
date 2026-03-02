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
    import pyloudnorm as fct
    HAS_LOUDNORM = True
except ImportError:
    HAS_LOUDNORM = False

app = Flask(__name__)

# --- SYSTEM CONFIGURATION ---
app.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024 
SUPPORTED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'aac', 'mp4'}

# --- QUALITY THRESHOLDS ---
DUB_MATCH_THRESHOLD = 15
MAX_DRIFT_MS = 30
MAX_START_OFFSET_MS = 50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_VOLATILE_PATH = os.path.join(BASE_DIR, "data")
if not os.path.exists(MEDIA_VOLATILE_PATH):
    os.makedirs(MEDIA_VOLATILE_PATH)

FINGERPRINT_CACHE = {}
_cache_lock = threading.Lock()

def get_file_metadata(path):
    try:
        info = sf.info(path)
        return {
            "sr": f"{info.samplerate} Hz",
            "duration": info.duration,
            "duration_str": f"{round(info.duration, 2)}s",
            "bit_depth": str(info.subtype),
            "channels": int(info.channels)
        }
    except Exception:
        duration = librosa.get_duration(path=path)
        y, sr = librosa.load(path, sr=None, duration=1)
        return {"sr": f"{sr} Hz", "duration": duration, "duration_str": f"{round(duration, 2)}s", "bit_depth": "N/A", "channels": "N/A"}

def classify_audio_quality(file_path):
    try:
        data, rate = sf.read(file_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        max_val = np.max(np.abs(data))
        clipping = "Possible Clipping" if max_val > 0.99 else "Clean Peaks"
        loudness_val = "N/A"
        if HAS_LOUDNORM:
            meter = fct.Meter(rate) 
            loudness_val = f"{round(meter.integrated_loudness(data), 2)} LUFS"
        return {"lufs": loudness_val, "peak": clipping, "label": "Verified" if max_val > 0.001 else "Silent"}
    except:
        return {"lufs": "N/A", "peak": "N/A", "label": "Error"}

def get_offset_at_time(y_ref, y_comp, sr, hop_length):
    ref_env = librosa.feature.rms(y=y_ref, hop_length=hop_length)[0]
    comp_env = librosa.feature.rms(y=y_comp, hop_length=hop_length)[0]
    ref_env = (ref_env - ref_env.min()) / (ref_env.max() - ref_env.min() + 1e-10)
    comp_env = (comp_env - comp_env.min()) / (comp_env.max() - comp_env.min() + 1e-10)
    correlation = signal.correlate(comp_env, ref_env, mode='full')
    lag_frame = np.argmax(correlation) - (len(ref_env) - 1)
    return round(float(lag_frame * hop_length / sr * 1000), 2)

def analyze_sync(anchor_path, rendition_path, ref_meta, sr=22050, hop_length=512):
    # Fingerprinting
    fpcalc_path = shutil.which("fpcalc") or "/usr/local/bin/fpcalc"
    def get_fp(p):
        try: return subprocess.check_output([fpcalc_path, "-plain", p]).decode().strip()
        except: return ""
    
    fp_a, fp_b = get_fp(anchor_path), get_fp(rendition_path)
    from difflib import SequenceMatcher
    match_score = round(SequenceMatcher(None, fp_a, fp_b).ratio() * 100, 2)
    
    comp_meta = get_file_metadata(rendition_path)
    
    # Timing Analysis (Start and End for Drift)
    y_ref_s, _ = librosa.load(anchor_path, sr=sr, duration=20)
    y_comp_s, _ = librosa.load(rendition_path, sr=sr, duration=20)
    start_offset = get_offset_at_time(y_ref_s, y_comp_s, sr, hop_length)
    
    try:
        y_ref_e, _ = librosa.load(anchor_path, sr=sr, offset=max(0, ref_meta['duration']-20))
        y_comp_e, _ = librosa.load(rendition_path, sr=sr, offset=max(0, comp_meta['duration']-20))
        end_offset = get_offset_at_time(y_ref_e, y_comp_e, sr, hop_length)
        drift_ms = round(abs(end_offset - start_offset), 2)
    except:
        drift_ms = 0.0

    # Plot
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y_ref_s[:sr*10], sr=sr, color='blue', alpha=0.5, label="Master")
    librosa.display.waveshow(y_comp_s[:sr*10], sr=sr, color='orange', alpha=0.5, label="Dub")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    issues = []
    if abs(start_offset) > MAX_START_OFFSET_MS: issues.append(f"Offset Issue: {start_offset}ms")
    if drift_ms > MAX_DRIFT_MS: issues.append(f"Drift Issue: {drift_ms}ms")
    if match_score < DUB_MATCH_THRESHOLD: issues.append(f"Content DNA Mismatch ({match_score}%)")

    return {
        "start_offset": start_offset, "drift_ms": drift_ms, "match_score": match_score,
        "issues": issues, "visual": base64.b64encode(buf.getvalue()).decode('utf-8'),
        "ref_meta": ref_meta, "comp_meta": comp_meta
    }

@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    analysis_root = os.path.join(MEDIA_VOLATILE_PATH, session_id)
    os.makedirs(analysis_root, exist_ok=True)
    deep_analysis = request.form.get('deepAnalysis') == 'true'
    
    try:
        ref_file = request.files['reference']
        ref_path = os.path.join(analysis_root, secure_filename(ref_file.filename))
        ref_file.save(ref_path)
        ref_meta = get_file_metadata(ref_path)
        
        results = []
        for f in request.files.getlist('comparison[]'):
            if not f.filename: continue
            f_path = os.path.join(analysis_root, secure_filename(f.filename))
            f.save(f_path)
            
            quality = classify_audio_quality(f_path)
            analysis = analyze_sync(ref_path, f_path, ref_meta)
            
            summary = "Detection complete. Review metrics for drift." if analysis['issues'] else "Detection complete. Perfectly aligned."
            
            results.append({
                'filename': f.filename,
                'offset_ms': float(analysis['start_offset']),
                'drift_ms': float(analysis['drift_ms']),
                'match_confidence': float(analysis['match_score']),
                'issues': analysis['issues'],
                'visual': analysis['visual'],
                'ref_meta': analysis['ref_meta'],
                'comp_meta': analysis['comp_meta'],
                'quality': quality,
                'status_summary': summary,
                'needs_review': bool(analysis['issues'])
            })
        return jsonify({'reference': ref_file.filename, 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
