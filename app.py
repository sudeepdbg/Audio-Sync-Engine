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

# Advanced Signal Processing
import demucs.separate
import pyloudnorm as fct

app = Flask(__name__)

# --- SYSTEM CONFIGURATION ---
app.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024 
SUPPORTED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'aac', 'mp4'}

# --- QUALITY THRESHOLDS ---
EXACT_MATCH_THRESHOLD = 95
DUB_MATCH_THRESHOLD = 15
MAX_DRIFT_MS = 30
MAX_START_OFFSET_MS = 50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_VOLATILE_PATH = os.path.join(BASE_DIR, "data")
if not os.path.exists(MEDIA_VOLATILE_PATH):
    os.makedirs(MEDIA_VOLATILE_PATH)

FINGERPRINT_CACHE = {}
_cache_lock = threading.Lock()

# --- UTILITIES ---

def classify_audio_quality(file_path):
    """Lighter weight quality scan using pyloudnorm and librosa (macOS Compatible)."""
    try:
        data, rate = sf.read(file_path)
        
        # 1. Check for Digital Clipping
        max_val = np.max(np.abs(data))
        clipping = "Possible Clipping" if max_val > 0.99 else "Clean Peaks"
        
        # 2. Measure Integrated Loudness (Broadcast standard)
        meter = fct.Meter(rate) 
        loudness = meter.integrated_loudness(data)
        
        # 3. Detect Silence
        is_silent = "Silent" if max_val < 0.001 else "Contains Signal"

        return {
            "dynamic_range": f"{round(loudness, 2)} LUFS",
            "peak_status": clipping,
            "quality_label": is_silent if is_silent == "Silent" else "Verified"
        }
    except Exception as e:
        print(f"Quality Check Failed: {e}")
        return {"quality_label": "Scan Error", "dynamic_range": "N/A", "peak_status": "N/A"}

def isolate_vocals(file_path, output_root):
    """Uses Neural separation to strip background music/effects."""
    try:
        # Using htdemucs for the best speed/quality balance
        demucs.separate.main(["--mp3", "-n", "htdemucs", "-o", output_root, file_path])
        
        base_name = os.path.basename(file_path).rsplit('.', 1)[0]
        # Search for the isolated vocal stem
        vocal_path = os.path.join(output_root, "htdemucs", base_name, "vocals.mp3")
        
        return vocal_path if os.path.exists(vocal_path) else file_path
    except Exception as e:
        print(f"Vocal Isolation Error: {e}")
        return file_path

def get_file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def get_file_metadata(path):
    try:
        info = sf.info(path)
        return {
            "sr": f"{info.samplerate} Hz",
            "duration": info.duration,
            "duration_str": f"{round(info.duration, 2)}s",
            "bit_depth": info.subtype,
            "channels": info.channels
        }
    except Exception:
        duration = librosa.get_duration(path=path)
        y, sr = librosa.load(path, sr=None, duration=1)
        return {"sr": f"{sr} Hz", "duration": duration, "duration_str": f"{round(duration, 2)}s", "bit_depth": "N/A", "channels": "N/A"}

def get_efficient_fingerprint(file_path):
    file_hash = get_file_hash(file_path)
    with _cache_lock:
        if file_hash in FINGERPRINT_CACHE: return FINGERPRINT_CACHE[file_hash]
    
    fpcalc_path = shutil.which("fpcalc") or "/usr/local/bin/fpcalc"
    try:
        cmd = [fpcalc_path, "-plain", file_path]
        fp = subprocess.check_output(cmd, timeout=30).decode().strip()
        with _cache_lock: FINGERPRINT_CACHE[file_hash] = fp
        return fp
    except Exception: return None

def compare_fingerprints(fp_a, fp_b):
    try:
        if not fp_a or not fp_b: return 0.0
        if fp_a == fp_b: return 100.0
        
        if ',' in fp_a:
            list_a = [int(x) for x in fp_a.split(',') if x.strip()]
            list_b = [int(x) for x in fp_b.split(',') if x.strip()]
            min_len = min(len(list_a), len(list_b))
            if min_len == 0: return 0.0
            matches = sum(1 for a, b in zip(list_a[:min_len], list_b[:min_len]) if a == b)
            return round((matches / min_len) * 100, 2)
            
        from difflib import SequenceMatcher
        return round(SequenceMatcher(None, fp_a, fp_b).ratio() * 100, 2)
    except Exception: return 0.0

def get_offset_at_time(y_ref, y_comp, sr, hop_length):
    ref_env = librosa.feature.rms(y=y_ref, hop_length=hop_length)[0]
    comp_env = librosa.feature.rms(y=y_comp, hop_length=hop_length)[0]
    
    ref_env = (ref_env - ref_env.min()) / (ref_env.max() - ref_env.min() + 1e-10)
    comp_env = (comp_env - comp_env.min()) / (comp_env.max() - comp_env.min() + 1e-10)
    
    correlation = signal.correlate(comp_env, ref_env, mode='full')
    lag_frame = np.argmax(correlation) - (len(ref_env) - 1)
    return round(float(lag_frame * hop_length / sr * 1000), 2)

def analyze_sync(anchor_path, rendition_path, ref_meta, sr=22050, hop_length=512):
    fp_a = get_efficient_fingerprint(anchor_path)
    fp_b = get_efficient_fingerprint(rendition_path)
    match_score = compare_fingerprints(fp_a, fp_b)
    
    comp_meta = get_file_metadata(rendition_path)
    duration = min(ref_meta['duration'], comp_meta['duration'])
    
    # Load analysis windows (Start and potentially End for drift)
    y_ref_start, _ = librosa.load(anchor_path, sr=sr, duration=60)
    y_comp_start, _ = librosa.load(rendition_path, sr=sr, duration=60)
    
    start_offset = get_offset_at_time(y_ref_start, y_comp_start, sr, hop_length)
    
    end_offset = start_offset
    if duration > 120:
        y_ref_end, _ = librosa.load(anchor_path, sr=sr, offset=duration-60, duration=60)
        y_comp_end, _ = librosa.load(rendition_path, sr=sr, offset=duration-60, duration=60)
        end_offset = get_offset_at_time(y_ref_end, y_comp_end, sr, hop_length)
    
    drift = round(abs(start_offset - end_offset), 2)
    
    # Generate visualization
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y_ref_start[:sr*10], sr=sr, color='blue', alpha=0.7)
    plt.title("Reference Signal (First 10s)")
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y_comp_start[:sr*10], sr=sr, color='orange', alpha=0.7)
    plt.title(f"Comparison Signal (Detected Offset: {start_offset}ms)")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    
    # Issues flagging
    issues = []
    if abs(start_offset) > MAX_START_OFFSET_MS: issues.append(f"Start Offset: {start_offset}ms")
    if drift > MAX_DRIFT_MS: issues.append(f"Drift Detected: {drift}ms")
    if match_score < DUB_MATCH_THRESHOLD: issues.append(f"Low DNA Score ({match_score}%)")
    
    return {
        "start_offset": start_offset, "end_offset": end_offset, "drift": drift,
        "match_score": match_score, "issues": issues,
        "visual": base64.b64encode(buf.getvalue()).decode('utf-8'),
        "ref_meta": ref_meta, "comp_meta": comp_meta
    }

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    try:
        gc.collect() # Force free memory/file handles
        time.sleep(0.5) # Allow OS to release locks
        with _cache_lock:
            FINGERPRINT_CACHE.clear()
        
        if os.path.exists(MEDIA_VOLATILE_PATH):
            for item in os.listdir(MEDIA_VOLATILE_PATH):
                item_path = os.path.join(MEDIA_VOLATILE_PATH, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    else:
                        os.remove(item_path)
                except: continue
        return jsonify({'status': 'Cache and storage cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        
        ref_to_analyze = isolate_vocals(ref_path, analysis_root) if deep_analysis else ref_path
        
        comp_files = request.files.getlist('comparison[]')
        results = []
        for f in comp_files:
            if not f.filename: continue
            f_path = os.path.join(analysis_root, secure_filename(f.filename))
            f.save(f_path)
            
            quality_report = classify_audio_quality(f_path)
            comp_to_analyze = isolate_vocals(f_path, analysis_root) if deep_analysis else f_path
            
            analysis = analyze_sync(ref_to_analyze, comp_to_analyze, ref_meta)
            
            results.append({
                'filename': f.filename,
                'offset_ms': analysis['start_offset'],
                'drift_ms': analysis['drift'],
                'match_confidence': analysis['match_score'],
                'issues': analysis['issues'],
                'visual': analysis['visual'],
                'ref_meta': analysis['ref_meta'],
                'comp_meta': analysis['comp_meta'],
                'quality': quality_report,
                'deep_mode_active': deep_analysis,
                'needs_review': len(analysis['issues']) > 0
            })
            
        return jsonify({'reference': ref_file.filename, 'results': results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
