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
from scipy import signal
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# CONFIGURATION
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 
app.config['MAX_FORM_MEMORY_SIZE'] = 500 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_VOLATILE_PATH = os.path.join(BASE_DIR, "data")
if not os.path.exists(MEDIA_VOLATILE_PATH):
    os.makedirs(MEDIA_VOLATILE_PATH)

SUPPORTED_CONTAINERS = {'wav', 'mp3', 'm4a', 'flac', 'aac', 'mp4'}
FINGERPRINT_CACHE = {}
_cache_lock = threading.Lock()

# --- UTILITIES ---

def get_file_hash(path):
    """Memory-efficient chunked MD5 hashing for 500MB+ files."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def cleanup_session(path, delay=600):
    """Deletes session files after a delay."""
    def _delete():
        time.sleep(delay)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    threading.Thread(target=_delete, daemon=True).start()

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
        y, sr = librosa.load(path, sr=None, duration=1)
        duration = librosa.get_duration(path=path)
        return {
            "sr": f"{sr} Hz",
            "duration": duration,
            "duration_str": f"{round(duration, 2)}s",
            "bit_depth": "Compressed",
            "channels": "Unknown"
        }

# --- CORE LOGIC: FINGERPRINTING & SYNC ---

def compare_fingerprints(fp_a, fp_b):
    """Robust comparison of Chromaprint fingerprints."""
    try:
        # 1. Clean the strings
        fp_a = fp_a.strip()
        fp_b = fp_b.strip()

        # 2. Check for exact string match first (covers identical files)
        if fp_a == fp_b:
            return 100.0

        # 3. Handle comma-separated integers
        if ',' in fp_a and ',' in fp_b:
            list_a = [int(x) for x in fp_a.split(',')]
            list_b = [int(x) for x in fp_b.split(',')]
            
            min_len = min(len(list_a), len(list_b))
            if min_len == 0: return 0.0
            
            matches = sum(1 for a, b in zip(list_a[:min_len], list_b[:min_len]) if a == b)
            return round((matches / min_len) * 100, 2)
        
        # 4. Fallback: If it's a giant single integer string, use SequenceMatcher
        # as a last resort or check substring similarity
        from difflib import SequenceMatcher
        return round(SequenceMatcher(None, fp_a, fp_b).ratio() * 100, 2)

    except Exception as e:
        print(f"Comparison Error: {e}")
        return 0.0

def get_efficient_fingerprint(file_path):
    file_hash = get_file_hash(file_path)
    with _cache_lock:
        if file_hash in FINGERPRINT_CACHE:
            return FINGERPRINT_CACHE[file_hash]
    
    fpcalc_path = shutil.which("fpcalc") or "/opt/homebrew/bin/fpcalc"
    try:
        cmd = [fpcalc_path, "-plain", file_path]
        fp = subprocess.check_output(cmd, timeout=30).decode().strip()
        with _cache_lock:
            FINGERPRINT_CACHE[file_hash] = fp
        return fp
    except: return None

def get_offset_at_time(y_ref, y_comp, sr, hop_length):
    """Calculates the ms lag between two audio buffers."""
    ref_env = librosa.feature.rms(y=y_ref, hop_length=hop_length)[0]
    comp_env = librosa.feature.rms(y=y_comp, hop_length=hop_length)[0]
    
    # Normalize envelopes
    ref_env = (ref_env - ref_env.min()) / (ref_env.max() - ref_env.min() + 1e-10)
    comp_env = (comp_env - comp_env.min()) / (comp_env.max() - comp_env.min() + 1e-10)
    
    correlation = signal.correlate(comp_env, ref_env, mode='full')
    lag_frame = np.argmax(correlation) - (len(ref_env) - 1)
    return round(float(lag_frame * hop_length / sr * 1000), 2)

def analyze_sync(anchor_path, rendition_path, sr=22050, hop_length=512):
    try:
        # 1. Content Integrity (Acoustic DNA)
        fp_a = get_efficient_fingerprint(anchor_path)
        fp_b = get_efficient_fingerprint(rendition_path)
        match_score = compare_fingerprints(fp_a, fp_b)

        # 2. Metadata & Duration Check
        ref_meta = get_file_metadata(anchor_path)
        comp_meta = get_file_metadata(rendition_path)
        duration = min(ref_meta['duration'], comp_meta['duration'])

        # 3. Start Window Check (0-60s)
        y_ref_start, _ = librosa.load(anchor_path, sr=sr, duration=60)
        y_comp_start, _ = librosa.load(rendition_path, sr=sr, duration=60)
        start_offset = get_offset_at_time(y_ref_start, y_comp_start, sr, hop_length)

        # 4. End Window Check (Last 60s)
        end_offset = start_offset # Default fallback
        if duration > 120:
            offset_start_time = duration - 60
            y_ref_end, _ = librosa.load(anchor_path, sr=sr, offset=offset_start_time, duration=60)
            y_comp_end, _ = librosa.load(rendition_path, sr=sr, offset=offset_start_time, duration=60)
            end_offset = get_offset_at_time(y_ref_end, y_comp_end, sr, hop_length)

        # 5. Logic: Sync vs Drift
        drift_variance = abs(start_offset - end_offset)
        issues = []
        if abs(start_offset) > 50: issues.append(f"Start Offset: {start_offset}ms")
        if drift_variance > 30: issues.append(f"Drift Detected: {round(drift_variance, 2)}ms variance over duration")
        if match_score < 75: issues.append(f"Content DNA match low ({match_score}%)")

        # Visualization (Start window)
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y_ref_start[:sr*10], sr=sr, alpha=0.5, label='Reference')
        librosa.display.waveshow(y_comp_start[:sr*10], sr=sr, alpha=0.5, label='Comparison')
        plt.title(f"Sync Visual (First 10s) | Match: {match_score}%")
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()

        return {
            "start_offset": start_offset,
            "end_offset": end_offset,
            "drift": round(drift_variance, 2),
            "match_score": match_score,
            "issues": issues,
            "visual": base64.b64encode(buf.getvalue()).decode('utf-8'),
            "ref_meta": ref_meta,
            "comp_meta": comp_meta
        }
    except Exception as e:
        traceback.print_exc()
        raise e

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    try:
        for item in os.listdir(MEDIA_VOLATILE_PATH):
            path = os.path.join(MEDIA_VOLATILE_PATH, item)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        with _cache_lock: FINGERPRINT_CACHE.clear()
        return jsonify({'status': 'Volatile storage and cache cleared'})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    analysis_root = os.path.join(MEDIA_VOLATILE_PATH, session_id)
    os.makedirs(analysis_root, exist_ok=True)
    
    try:
        ref_file = request.files['reference']
        comp_files = request.files.getlist('comparison[]')

        ref_path = os.path.join(analysis_root, secure_filename(ref_file.filename))
        ref_file.save(ref_path)

        results = []
        for f in comp_files:
            if f.filename:
                f_path = os.path.join(analysis_root, secure_filename(f.filename))
                f.save(f_path)
                
                analysis = analyze_sync(ref_path, f_path)
                results.append({
                    'filename': f.filename,
                    'offset_ms': analysis['start_offset'],
                    'end_offset_ms': analysis['end_offset'],
                    'drift_ms': analysis['drift'],
                    'match_confidence': analysis['match_score'],
                    'issues': analysis['issues'],
                    'visual': analysis['visual'],
                    'ref_meta': analysis['ref_meta'],
                    'comp_meta': analysis['comp_meta'],
                    'needs_review': len(analysis['issues']) > 0
                })

        cleanup_session(analysis_root)
        return jsonify({'reference': ref_file.filename, 'results': results})
    except Exception as e: return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
