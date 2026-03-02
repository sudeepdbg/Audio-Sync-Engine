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

app = Flask(__name__)

# --- SYSTEM CONFIGURATION ---
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 
SUPPORTED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'aac', 'mp4'}

# --- QUALITY THRESHOLDS (Defined Constants) ---
EXACT_MATCH_THRESHOLD = 95    # High confidence for identical files
DUB_MATCH_THRESHOLD = 15      # Min similarity for localized dubs
MAX_DRIFT_MS = 30             # Perceptibility limit for sync drift
MAX_START_OFFSET_MS = 50      # Standard broadcast acceptable delay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_VOLATILE_PATH = os.path.join(BASE_DIR, "data")
if not os.path.exists(MEDIA_VOLATILE_PATH):
    os.makedirs(MEDIA_VOLATILE_PATH)

FINGERPRINT_CACHE = {}
_cache_lock = threading.Lock()

# --- UTILITIES ---

def allowed_file(filename):
    """Ensures only valid audio/video containers are processed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in SUPPORTED_EXTENSIONS

def get_file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def cleanup_session(path, delay=600):
    def _delete():
        time.sleep(delay)
        gc.collect() 
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    threading.Thread(target=_delete, daemon=True).start()

def get_file_metadata(path):
    """Retrieves technical specifications of the audio file."""
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
        return {
            "sr": f"{sr} Hz",
            "duration": duration,
            "duration_str": f"{round(duration, 2)}s",
            "bit_depth": "Compressed",
            "channels": "Unknown"
        }

# --- CORE LOGIC ---

def compare_fingerprints(fp_a, fp_b):
    try:
        if not fp_a or not fp_b: return 0.0
        fp_a, fp_b = fp_a.strip(), fp_b.strip()
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
    except (ValueError, TypeError, AttributeError): 
        return 0.0

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

def get_offset_at_time(y_ref, y_comp, sr, hop_length):
    ref_env = librosa.feature.rms(y=y_ref, hop_length=hop_length)[0]
    comp_env = librosa.feature.rms(y=y_comp, hop_length=hop_length)[0]
    ref_env = (ref_env - ref_env.min()) / (ref_env.max() - ref_env.min() + 1e-10)
    comp_env = (comp_env - comp_env.min()) / (comp_env.max() - comp_env.min() + 1e-10)
    correlation = signal.correlate(comp_env, ref_env, mode='full')
    lag_frame = np.argmax(correlation) - (len(ref_env) - 1)
    return round(float(lag_frame * hop_length / sr * 1000), 2)

def generate_summary(match_score, drift, start_offset):
    """Generates the text verdict using defined thresholds."""
    if match_score >= EXACT_MATCH_THRESHOLD:
        content_txt = "Exact match detected."
    elif match_score >= DUB_MATCH_THRESHOLD:
        content_txt = "Acoustically related (consistent with a localized dub)."
    else:
        content_txt = "Content DNA mismatch (unrelated audio)."
        
    if drift > MAX_DRIFT_MS:
        sync_txt = f"Significant drift detected ({drift}ms variation)."
    elif abs(start_offset) > MAX_START_OFFSET_MS:
        sync_txt = f"Constant delay of {start_offset}ms detected."
    else:
        sync_txt = "Sync is frame-accurate."
    return f"{content_txt} {sync_txt}"

def analyze_sync(anchor_path, rendition_path, ref_meta, sr=22050, hop_length=512):
    """
    Analyzes temporal sync between two files. 
    ref_meta is passed in to avoid redundant disk I/O.
    """
    fp_a = get_efficient_fingerprint(anchor_path)
    fp_b = get_efficient_fingerprint(rendition_path)
    match_score = compare_fingerprints(fp_a, fp_b)
    
    comp_meta = get_file_metadata(rendition_path)
    duration = min(ref_meta['duration'], comp_meta['duration'])
    
    y_ref_start, _ = librosa.load(anchor_path, sr=sr, duration=60)
    y_comp_start, _ = librosa.load(rendition_path, sr=sr, duration=60)
    
    start_offset = get_offset_at_time(y_ref_start, y_comp_start, sr, hop_length)
    
    end_offset = start_offset
    if duration > 120:
        y_ref_end, _ = librosa.load(anchor_path, sr=sr, offset=duration-60, duration=60)
        y_comp_end, _ = librosa.load(rendition_path, sr=sr, offset=duration-60, duration=60)
        end_offset = get_offset_at_time(y_ref_end, y_comp_end, sr, hop_length)
    
    drift = round(abs(start_offset - end_offset), 2)
    summary = generate_summary(match_score, drift, start_offset)
    
    issues = []
    if abs(start_offset) > MAX_START_OFFSET_MS: issues.append(f"Start Offset: {start_offset}ms")
    if drift > MAX_DRIFT_MS: issues.append(f"Drift Detected: {drift}ms variance")
    if match_score < DUB_MATCH_THRESHOLD: issues.append(f"DNA Match Low ({match_score}%)")
    
    # Visualization: Restored Subplots for better QA visibility
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y_ref_start[:sr*10], sr=sr, color='blue', alpha=0.7)
    plt.title("Reference (First 10s)")
    
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y_comp_start[:sr*10], sr=sr, color='orange', alpha=0.7)
    plt.title(f"Comparison (Offset: {start_offset}ms)")
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    return {
        "start_offset": start_offset, "end_offset": end_offset, "drift": drift,
        "match_score": match_score, "summary": summary, "issues": issues,
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
        gc.collect() 
        time.sleep(0.5)
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
                except Exception as e:
                    print(f"Skipping {item}: {e}")
                    continue

        return jsonify({'status': 'Volatile storage and cache cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    analysis_root = os.path.join(MEDIA_VOLATILE_PATH, session_id)
    os.makedirs(analysis_root, exist_ok=True)
    try:
        ref_file = request.files['reference']
        if not ref_file or not allowed_file(ref_file.filename):
            return jsonify({'error': 'Reference file missing or unsupported format'}), 400

        ref_path = os.path.join(analysis_root, secure_filename(ref_file.filename))
        ref_file.save(ref_path)
        
        # Optimization: Scan reference once
        ref_meta = get_file_metadata(ref_path)
        
        comp_files = request.files.getlist('comparison[]')
        results = []
        for f in comp_files:
            if not f.filename or not allowed_file(f.filename):
                continue
            
            f_path = os.path.join(analysis_root, secure_filename(f.filename))
            f.save(f_path)
            
            analysis = analyze_sync(ref_path, f_path, ref_meta)
            results.append({
                'filename': f.filename,
                'offset_ms': analysis['start_offset'],
                'end_offset_ms': analysis['end_offset'],
                'drift_ms': analysis['drift'],
                'match_confidence': analysis['match_score'],
                'summary': analysis['summary'],
                'issues': analysis['issues'],
                'visual': analysis['visual'],
                'ref_meta': analysis['ref_meta'],
                'comp_meta': analysis['comp_meta'],
                'needs_review': len(analysis['issues']) > 0
            })
        
        cleanup_session(analysis_root)
        return jsonify({'reference': ref_file.filename, 'results': results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
