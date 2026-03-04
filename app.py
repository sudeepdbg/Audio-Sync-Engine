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
from scipy import signal
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

# --- ADVANCED LIBRARIES ---
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

def get_file_metadata(path):
    """Deep technical scan of the audio container."""
    try:
        info = sf.info(path)
        return {
            "format": info.format,
            "sr": f"{info.samplerate} Hz",
            "duration": info.duration,
            "duration_str": f"{round(info.duration, 2)}s",
            "bit_depth": info.subtype,
            "channels": info.channels
        }
    except:
        return {"format": "ERR", "sr": "N/A", "duration": 0, "bit_depth": "N/A"}

def scan_levels(path):
    """Calculates Integrated LUFS and True Peak."""
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
    """Isolates dialogue to prevent music from confusing the sync logic."""
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

def generate_visual(y_ref, y_comp):
    """High-contrast professional waveform comparison."""
    plt.style.use('default')
    fig = plt.figure(figsize=(12, 3), facecolor='white')
    ax = fig.add_subplot(111)
    
    r = y_ref / (np.max(np.abs(y_ref)) + 1e-6)
    c = y_comp / (np.max(np.abs(y_comp)) + 1e-6)
    
    ax.plot(r[:PERFORMANCE_SR*10], color='#2563eb', alpha=0.4, label="Master")
    ax.plot(c[:PERFORMANCE_SR*10], color='#f59e0b', alpha=0.7, label="Dub")
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def analyze_sync(ref_path, comp_path, ref_meta, deep_mode, root):
    # Process files
    r_work = isolate_vocals(ref_path, root) if deep_mode else ref_path
    c_work = isolate_vocals(comp_path, root) if deep_mode else comp_path
    
    y_r, _ = librosa.load(r_work, sr=PERFORMANCE_SR, duration=30)
    y_c, _ = librosa.load(c_work, sr=PERFORMANCE_SR, duration=30)
    
    start_offset = get_offset(y_r, y_c, PERFORMANCE_SR)
    comp_meta = get_file_metadata(comp_path)
    levels = scan_levels(comp_path)
    
    duration = min(ref_meta['duration'], comp_meta['duration'])
    drift = 0.0
    if duration > 40:
        y_r_e, _ = librosa.load(r_work, sr=PERFORMANCE_SR, offset=duration-20, duration=20)
        y_c_e, _ = librosa.load(c_work, sr=PERFORMANCE_SR, offset=duration-20, duration=20)
        drift = round(abs(get_offset(y_r_e, y_c_e, PERFORMANCE_SR) - start_offset), 2)

    return {
        "offset_ms": start_offset,
        "drift_ms": drift,
        "visual": generate_visual(y_r, y_c),
        "ref_meta": ref_meta,
        "comp_meta": comp_meta,
        "quality": levels,
        "needs_review": abs(start_offset) > MAX_START_OFFSET_MS or drift > MAX_DRIFT_MS
    }

@app.route('/')
def index(): return render_template('index.html')

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
        
        results = []
        for f in request.files.getlist('comparison[]'):
            if not f.filename: continue
            f_path = os.path.join(root, secure_filename(f.filename))
            f.save(f_path)
            results.append({**{'filename': f.filename}, **analyze_sync(ref_path, f_path, ref_meta, deep, root)})
            
        return jsonify({'reference': ref_file.filename, 'results': results})
    except Exception:
        return jsonify({'error': str(traceback.format_exc())}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
