import os
import uuid
import shutil
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
import threading
import time
from scipy import signal
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

# --- ADVANCED AUDIO ENGINES ---
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

# --- BROADCAST STANDARDS ---
MAX_DRIFT_MS = 30
MAX_START_OFFSET_MS = 50
PERFORMANCE_SR = 22050 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- BACKGROUND AUTO-CLEANUP ---
def cleanup_worker():
    """Deletes temporary session folders older than 1 hour to save disk space."""
    while True:
        now = time.time()
        if os.path.exists(DATA_DIR):
            for folder in os.listdir(DATA_DIR):
                path = os.path.join(DATA_DIR, folder)
                if os.path.isdir(path) and os.path.getmtime(path) < (now - 3600):
                    shutil.rmtree(path, ignore_errors=True)
        time.sleep(600)

threading.Thread(target=cleanup_worker, daemon=True).start()

# --- CORE UTILITIES ---
def get_file_metadata(path):
    try:
        info = sf.info(path)
        return {
            "format": info.format,
            "sr": f"{info.samplerate} Hz",
            "duration": info.duration,
            "bit_depth": info.subtype,
            "channels": info.channels,
            "channel_label": "5.1 Surround" if info.channels == 6 else "Stereo" if info.channels == 2 else f"{info.channels} Ch"
        }
    except:
        return {"format": "ERR", "sr": "N/A", "duration": 0, "bit_depth": "N/A", "channels": 0, "channel_label": "N/A"}

def scan_levels(path):
    """Calculates Integrated LUFS and True Peak for broadcast compliance."""
    try:
        data, rate = sf.read(path)
        peak_db = 20 * np.log10(np.max(np.abs(data)) + 1e-10)
        lufs = "N/A"
        if HAS_LOUDNORM:
            if len(data.shape) == 1: data = data.reshape(-1, 1)
            meter = fln.Meter(rate)
            lufs = f"{round(meter.integrated_loudness(data), 2)} LUFS"
        return {"lufs": lufs, "peak": f"{round(peak_db, 2)} dBFS"}
    except:
        return {"lufs": "Scan Err", "peak": "Scan Err"}

def isolate_vocals(file_path, output_root):
    """Dialogue isolation to ensure background music doesn't skew sync results."""
    if not HAS_DEMUCS: return file_path
    try:
        demucs.separate.main(["--mp3", "-n", "htdemucs", "-o", output_root, file_path])
        base = os.path.basename(file_path).rsplit('.', 1)[0]
        v_path = os.path.join(output_root, "htdemucs", base, "vocals.mp3")
        return v_path if os.path.exists(v_path) else file_path
    except: return file_path

def get_offset(y_ref, y_comp, sr):
    hop = 512
    ref_env = librosa.feature.rms(y=y_ref, hop_length=hop)[0]
    comp_env = librosa.feature.rms(y=y_comp, hop_length=hop)[0]
    corr = signal.correlate(comp_env, ref_env, mode='full')
    lag = np.argmax(corr) - (len(ref_env) - 1)
    return round(float(lag * hop / sr * 1000), 2)

def calculate_dna_match(y_ref, y_comp):
    """Cross-correlation based confidence score (Content DNA)."""
    try:
        y_ref_norm = (y_ref - np.mean(y_ref)) / (np.std(y_ref) + 1e-6)
        y_comp_norm = (y_comp - np.mean(y_comp)) / (np.std(y_comp) + 1e-6)
        correlation = np.corrcoef(y_ref_norm[:5000], y_comp_norm[:5000])[0, 1]
        return round(max(0, correlation * 100), 2)
    except: return 0.0

def generate_visual(y_ref, y_comp):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 3), facecolor='#0f172a')
    ax = fig.add_subplot(111)
    r = y_ref / (np.max(np.abs(y_ref)) + 1e-6)
    c = y_comp / (np.max(np.abs(y_comp)) + 1e-6)
    ax.plot(r[:PERFORMANCE_SR*15], color='#2563eb', alpha=0.5, label="Master")
    ax.plot(c[:PERFORMANCE_SR*15], color='#f59e0b', alpha=0.8, label="Dub")
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='#0f172a')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/wipe', methods=['POST'])
def wipe():
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    return jsonify({"status": "All cache and session data cleared."})

@app.route('/upload', methods=['POST'])
def upload():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    root = os.path.join(DATA_DIR, session_id)
    os.makedirs(root, exist_ok=True)
    deep = request.form.get('deepAnalysis') == 'true'
    
    try:
        ref_file = request.files['reference']
        ref_path = os.path.join(root, secure_filename(ref_file.filename))
        ref_file.save(ref_path)
        ref_meta = get_file_metadata(ref_path)
        
        # Isolate Master if needed
        r_work = isolate_vocals(ref_path, root) if deep else ref_path
        y_r, _ = librosa.load(r_work, sr=PERFORMANCE_SR, duration=30)
        
        results = []
        for f in request.files.getlist('comparison[]'):
            if not f.filename: continue
            f_path = os.path.join(root, secure_filename(f.filename))
            f.save(f_path)
            
            c_work = isolate_vocals(f_path, root) if deep else f_path
            y_c, _ = librosa.load(c_work, sr=PERFORMANCE_SR, duration=30)
            
            comp_meta = get_file_metadata(f_path)
            levels = scan_levels(f_path)
            offset = get_offset(y_r, y_c, PERFORMANCE_SR)
            dna = calculate_dna_match(y_r, y_c)
            
            # Channel Mapping Check
            chan_mismatch = ref_meta['channels'] != comp_meta['channels']

            results.append({
                'filename': f.filename,
                'offset_ms': offset,
                'drift_ms': 0.0,
                'dna_match': dna,
                'visual': generate_visual(y_r, y_c),
                'ref_meta': ref_meta,
                'comp_meta': comp_meta,
                'levels': levels,
                'chan_mismatch': chan_mismatch,
                'needs_review': abs(offset) > MAX_START_OFFSET_MS or chan_mismatch or dna < 15
            })
            
        return jsonify({'reference': ref_file.filename, 'results': results})
    except Exception:
        return jsonify({'error': str(traceback.format_exc())}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
