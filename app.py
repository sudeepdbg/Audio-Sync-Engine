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
import soundfile as sf
import time
import threading
import gc
from scipy import signal
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

# Optional Loudness Scanning
try:
    import pyloudnorm as fln
    HAS_LOUDNORM = True
except ImportError:
    HAS_LOUDNORM = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_PATH = os.path.join(BASE_DIR, "data")
os.makedirs(MEDIA_PATH, exist_ok=True)

# --- IMPACT METRIC ENGINES ---

def get_loudness(y, sr):
    if not HAS_LOUDNORM: return "N/A"
    try:
        meter = fln.Meter(sr)
        return f"{round(meter.integrated_loudness(y), 2)} LUFS"
    except: return "Error"

def get_lip_flap_density(y, sr):
    """Measures syllable/speech density for Lip-Flap Sync."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    density = np.sum(onset_env > (np.max(onset_env) * 0.2)) / (duration + 1e-6)
    return float(density)

def get_tone_lock_score(y_ref, y_comp):
    """Compares spectral texture to verify emotional consistency."""
    ref_contrast = np.mean(librosa.feature.spectral_contrast(y=y_ref))
    comp_contrast = np.mean(librosa.feature.spectral_contrast(y=y_comp))
    score = 100 - (abs(ref_contrast - comp_contrast) * 12) 
    return float(max(0, min(100, score)))

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
    except:
        return {"sr": "Unknown", "duration": 0, "duration_str": "0s", "bit_depth": "N/A", "channels": "N/A"}

def get_offset(y_ref, y_comp, sr):
    hop = 512
    ref_env = librosa.feature.rms(y=y_ref, hop_length=hop)[0]
    comp_env = librosa.feature.rms(y=y_comp, hop_length=hop)[0]
    ref_env = (ref_env - ref_env.min()) / (ref_env.max() - ref_env.min() + 1e-10)
    comp_env = (comp_env - comp_env.min()) / (comp_env.max() - comp_env.min() + 1e-10)
    corr = signal.correlate(comp_env, ref_env, mode='full')
    lag = np.argmax(corr) - (len(ref_env) - 1)
    return round(float(lag * hop / sr * 1000), 2)

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    work_dir = os.path.join(MEDIA_PATH, session_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        ref_file = request.files['reference']
        ref_path = os.path.join(work_dir, secure_filename(ref_file.filename))
        ref_file.save(ref_path)
        
        y_ref, sr = librosa.load(ref_path, sr=22050)
        ref_meta = get_file_metadata(ref_path)
        ref_meta['loudness'] = get_loudness(y_ref, sr)
        ref_dens = get_lip_flap_density(y_ref, sr)

        results = []
        for f in request.files.getlist('comparison[]'):
            if not f.filename: continue
            f_path = os.path.join(work_dir, secure_filename(f.filename))
            f.save(f_path)
            
            y_comp, _ = librosa.load(f_path, sr=22050)
            comp_meta = get_file_metadata(f_path)
            comp_meta['loudness'] = get_loudness(y_comp, sr)
            
            # Impact Calculations
            offset = get_offset(y_ref[:30*sr], y_comp[:30*sr], sr)
            comp_dens = get_lip_flap_density(y_comp, sr)
            lip_flap = 100 - (abs(ref_dens - comp_dens) / (ref_dens + 1e-6) * 100)
            tone_lock = get_tone_lock_score(y_ref, y_comp)
            
            # Summary Logic
            summary = "Timing is frame-accurate."
            if abs(offset) > 50: summary = f"Significant delay of {offset}ms detected."
            if tone_lock < 70: summary += " Tone texture differs from master."

            # Waveform
            plt.figure(figsize=(10, 2))
            librosa.display.waveshow(y_ref[:10*sr], sr=sr, alpha=0.4, label="Master")
            librosa.display.waveshow(y_comp[:10*sr], sr=sr, alpha=0.4, label="Dub")
            plt.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            results.append({
                'filename': f.filename,
                'offset_ms': offset,
                'lip_flap_sync': round(max(0, lip_flap), 2),
                'tone_lock': round(tone_lock, 2),
                'summary': summary,
                'visual': base64.b64encode(buf.getvalue()).decode('utf-8'),
                'ref_meta': ref_meta,
                'comp_meta': comp_meta,
                'needs_review': abs(offset) > 50 or lip_flap < 80
            })

        return jsonify({'reference': ref_file.filename, 'results': results})
    except Exception as e:
        return jsonify({'error': str(traceback.format_exc())}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
