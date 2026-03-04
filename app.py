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
import soundfile as sf
import threading
import time
from scipy import signal
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

# --- ADVANCED ENGINES ---
try:
    import pyloudnorm as fln
    HAS_LOUDNORM = True
except ImportError:
    HAS_LOUDNORM = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 # 1GB Support
PERFORMANCE_SR = 22050 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- AUTO-CLEANUP (Disk Protection) ---
def auto_cleanup_worker():
    """Background thread to delete temporary audio data older than 1 hour."""
    while True:
        now = time.time()
        for folder in os.listdir(DATA_DIR):
            folder_path = os.path.join(DATA_DIR, folder)
            if os.path.isdir(folder_path):
                if os.path.getmtime(folder_path) < now - 3600:
                    shutil.rmtree(folder_path, ignore_errors=True)
        time.sleep(600)

threading.Thread(target=auto_cleanup_worker, daemon=True).start()

# --- BROADCAST UTILITIES ---

def get_file_metadata(path):
    try:
        info = sf.info(path)
        return {
            "sr": f"{info.samplerate} Hz",
            "duration": info.duration,
            "bit_depth": info.subtype,
            "channels": info.channels,
            "channel_label": "5.1 Surround" if info.channels == 6 else "Stereo" if info.channels == 2 else f"{info.channels} Ch"
        }
    except:
        return {"sr": "N/A", "duration": 0, "bit_depth": "N/A", "channels": 0, "channel_label": "N/A"}

def calculate_phase(path):
    """Calculates Pearson correlation between L/R channels for phase health."""
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
    """Broadcast scanning for Integrated Loudness and Peak dBFS."""
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
        return {"lufs": "ERR", "peak": "ERR"}

def generate_visual(y_ref, y_comp):
    """Professional high-contrast overlay waveform."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 4), facecolor='#0f172a')
    ax = fig.add_subplot(111)
    r = y_ref / (np.max(np.abs(y_ref)) + 1e-6)
    c = y_comp / (np.max(np.abs(y_comp)) + 1e-6)
    ax.fill_between(range(len(r[:PERFORMANCE_SR*15])), r[:PERFORMANCE_SR*15], color='#3b82f6', alpha=0.4, label="Master")
    ax.plot(c[:PERFORMANCE_SR*15], color='#f59e0b', linewidth=1.0, alpha=0.9, label="Dub")
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
    return jsonify({"status": "Cache Wiped"})

@app.route('/upload', methods=['POST'])
def upload():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    root = os.path.join(DATA_DIR, session_id)
    os.makedirs(root, exist_ok=True)
    try:
        ref_file = request.files['reference']
        ref_path = os.path.join(root, secure_filename(ref_file.filename))
        ref_file.save(ref_path)
        ref_meta = get_file_metadata(ref_path)
        y_r, _ = librosa.load(ref_path, sr=PERFORMANCE_SR, duration=30)
        
        results = []
        for f in request.files.getlist('comparison[]'):
            f_path = os.path.join(root, secure_filename(f.filename))
            f.save(f_path)
            y_c, _ = librosa.load(f_path, sr=PERFORMANCE_SR, duration=30)
            
            comp_meta = get_file_metadata(f_path)
            levels = scan_levels(f_path)
            phase = calculate_phase(f_path)
            
            # Temporal Sync Calculation
            corr = signal.correlate(y_c, y_r, mode='full')
            lag = np.argmax(corr) - (len(y_r) - 1)
            offset = round(float(lag / PERFORMANCE_SR * 1000), 2)
            
            # Content DNA
            dna = round(max(0, np.corrcoef(y_r[:5000], y_c[:5000])[0, 1] * 100), 2)
            
            results.append({
                'filename': f.filename,
                'offset_ms': offset,
                'dna_match': dna,
                'phase': phase,
                'levels': levels,
                'ref_meta': ref_meta,
                'comp_meta': comp_meta,
                'visual': generate_visual(y_r, y_c),
                'chan_mismatch': ref_meta['channels'] != comp_meta['channels']
            })
        return jsonify({'reference': ref_file.filename, 'results': results})
    except:
        return jsonify({'error': str(traceback.format_exc())}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
