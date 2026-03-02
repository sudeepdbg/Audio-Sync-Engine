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
from difflib import SequenceMatcher
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

# --- FIX #6: BACKGROUND CLEANUP ---
def cleanup_session(path, delay=600):
    """Deletes session files after a delay to prevent disk bloat."""
    def _delete():
        time.sleep(delay)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    threading.Thread(target=_delete, daemon=True).start()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in SUPPORTED_CONTAINERS

def get_file_metadata(path):
    try:
        info = sf.info(path)
        return {
            "sr": f"{info.samplerate} Hz",
            "duration": f"{round(info.duration, 2)}s",
            "bit_depth": info.subtype,
            "channels": info.channels
        }
    except Exception:
        y, sr = librosa.load(path, sr=None, duration=1)
        duration = librosa.get_duration(path=path)
        return {
            "sr": f"{sr} Hz",
            "duration": f"{round(duration, 2)}s",
            "bit_depth": "Compressed",
            "channels": "Unknown"
        }

# --- FIX #2: SECURE FINGERPRINTING ---
def get_efficient_fingerprint(file_path):
    # Use full file hash for 100% binary match accuracy
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    if file_hash in FINGERPRINT_CACHE:
        return FINGERPRINT_CACHE[file_hash]
    
    # Securely find fpcalc and execute without shell=True
    fpcalc_path = shutil.which("fpcalc") or "/opt/homebrew/bin/fpcalc"
    try:
        cmd = [fpcalc_path, "-plain", file_path]
        fp = subprocess.check_output(cmd, timeout=30).decode().strip()
        FINGERPRINT_CACHE[file_hash] = fp
        return fp
    except Exception as e:
        print(f"fpcalc error: {e}")
        return None

def generate_visual_comparison(anchor_y, rendition_y, drift_ms, match_score, sr):
    plt.figure(figsize=(10, 5), facecolor='#f8fafc')
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(anchor_y, sr=sr, alpha=0.6, color='#3b82f6')
    plt.title(f"Sync: {drift_ms}ms | Content Integrity: {match_score}%", fontsize=10)
    plt.ylabel("Reference")
    plt.xticks([]) 
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(rendition_y, sr=sr, alpha=0.6, color='#f59e0b')
    plt.ylabel("Comparison")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def analyze_temporal_drift(anchor_path, rendition_path, sr=22050, hop_length=512):
    try:
        abs_anchor = os.path.abspath(anchor_path)
        abs_rendition = os.path.abspath(rendition_path)
        
        # Binary Match Check
        match_score = 0.0
        with open(abs_anchor, 'rb') as f1, open(abs_rendition, 'rb') as f2:
            if hashlib.md5(f1.read()).digest() == hashlib.md5(f2.read()).digest():
                match_score = 100.0

        # --- FIX #7: OFFLINE FINGERPRINT COMPARISON ---
        if match_score < 100:
            fp_a = get_efficient_fingerprint(abs_anchor)
            fp_b = get_efficient_fingerprint(abs_rendition)
            if fp_a and fp_b:
                # Use SequenceMatcher to find similarity ratio between fingerprint strings
                match_score = round(SequenceMatcher(None, fp_a, fp_b).ratio() * 100, 2)

        # Load audio for sync analysis
        anchor_buffer, _ = librosa.load(abs_anchor, sr=sr, mono=True, duration=60)
        rendition_buffer, _ = librosa.load(abs_rendition, sr=sr, mono=True, duration=60)
        
        a_trimmed, _ = librosa.effects.trim(anchor_buffer)
        r_trimmed, _ = librosa.effects.trim(rendition_buffer)
        
        # Envelopes
        anchor_env = librosa.feature.rms(y=a_trimmed, hop_length=hop_length)[0]
        rendition_env = librosa.feature.rms(y=r_trimmed, hop_length=hop_length)[0]
        
        # Normalize
        anchor_env = (anchor_env - anchor_env.min()) / (anchor_env.max() - anchor_env.min() + 1e-10)
        rendition_env = (rendition_env - rendition_env.min()) / (rendition_env.max() - rendition_env.min() + 1e-10)
        
        # --- FIX #3: CORRELATION LOGIC ---
        # Use mode='full' for mathematically accurate lag discovery
        correlation = signal.correlate(rendition_env, anchor_env, mode='full')
        # Center of the 'full' correlation is at len(anchor_env) - 1
        lag_frame = np.argmax(correlation) - (len(anchor_env) - 1)
        drift_ms = round(float(lag_frame * hop_length / sr * 1000), 2)
        
        issues = []
        if abs(drift_ms) > 100: issues.append("Severe desync (>100ms)")
        elif abs(drift_ms) > 50: issues.append("Minor desync (50-100ms)")
        if match_score < 30: issues.append("Content mismatch - wrong dub?")
        elif match_score < 70: issues.append("Low confidence match")
            
        validation_flag = len(issues) > 0
        viz = generate_visual_comparison(anchor_buffer[:sr*15], rendition_buffer[:sr*15], drift_ms, match_score, sr)
        
        return drift_ms, validation_flag, viz, match_score, issues

    except Exception as e:
        traceback.print_exc()
        raise Exception(f"Analysis failed: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    analysis_root = os.path.join(MEDIA_VOLATILE_PATH, session_id)
    os.makedirs(analysis_root, exist_ok=True)
    
    try:
        anchor_track = request.files['reference']
        rendition_tracks = request.files.getlist('comparison[]')

        anchor_path = os.path.join(analysis_root, anchor_track.filename)
        anchor_track.save(anchor_path)
        ref_metadata = get_file_metadata(anchor_path)

        results = []
        for track in rendition_tracks:
            if track.filename and allowed_file(track.filename):
                r_path = os.path.join(analysis_root, track.filename)
                track.save(r_path)
                comp_metadata = get_file_metadata(r_path)
                
                drift, needs_val, viz, score, issues = analyze_temporal_drift(anchor_path, r_path)
                
                results.append({
                    'filename': track.filename, 
                    'offset_ms': drift,
                    'match_confidence': score, 
                    'needs_review': needs_val, 
                    'visual': viz,
                    'issues': issues,
                    'ref_meta': ref_metadata,
                    'comp_meta': comp_metadata
                })
        
        # Trigger cleanup for this session
        cleanup_session(analysis_root)
        
        return jsonify({'reference': anchor_track.filename, 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
