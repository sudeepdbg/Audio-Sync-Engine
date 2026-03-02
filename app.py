import os
import uuid
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import soundfile as sf
import traceback
from scipy import signal
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

try:
    import pyloudnorm as fln
    HAS_LOUDNORM = True
except ImportError:
    HAS_LOUDNORM = False

app = Flask(__name__)
# Increased to 2GB for massive files
app.config['MAX_CONTENT_LENGTH'] = 2048 * 1024 * 1024 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- BROADCASTER UTILITIES ---

def ms_to_timecode(ms, fps=24):
    """Converts milliseconds to HH:MM:SS:FF."""
    total_seconds = abs(ms) / 1000.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    frames = int((total_seconds % 1) * fps)
    sign = "-" if ms < 0 else ""
    return f"{sign}{hours:02}:{minutes:02}:{seconds:02}:{frames:02}"

def get_phase_correlation(y, channels):
    """Detects phase issues in stereo files (+1 is perfect, -1 is out of phase)."""
    if channels < 2: return 1.0
    # Split stereo into L/R
    left = y[0] if y.ndim > 1 else y
    right = y[1] if y.ndim > 1 else y
    correlation = np.corrcoef(left, right)[0, 1]
    return round(float(correlation), 3)

def get_compliance_data(path, target_sr=16000):
    """Memory-efficient analysis of large files."""
    info = sf.info(path)
    # Load in mono for sync logic, but keep stereo for phase/loudness
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    
    loudness = "N/A"
    phase = 1.0
    
    if HAS_LOUDNORM:
        try:
            # We load a small chunk for loudness to save RAM
            data, rate = sf.read(path, frames=min(int(rate*60), info.frames))
            meter = fln.Meter(rate)
            loudness = f"{round(meter.integrated_loudness(data), 2)} LUFS"
            phase = get_phase_correlation(data.T, info.channels)
        except: pass

    # Speech density for Lip-Flap
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    density = np.sum(onset_env > (np.max(onset_env) * 0.2)) / (info.duration + 1e-6)

    return {
        "y": y, "sr": sr, "density": density,
        "meta": {
            "sr": f"{info.samplerate}Hz", "ch": info.channels,
            "bit": info.subtype, "dur": info.duration,
            "lufs": loudness, "phase": phase
        }
    }

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    session_id = f"QC_{uuid.uuid4().hex[:6].upper()}"
    work_dir = os.path.join(DATA_DIR, session_id)
    os.makedirs(work_dir, exist_ok=True)
    
    fps = float(request.form.get('fps', 24))

    try:
        ref_file = request.files['reference']
        ref_path = os.path.join(work_dir, secure_filename(ref_file.filename))
        ref_file.save(ref_path)
        
        ref_data = get_compliance_data(ref_path)

        results = []
        for f in request.files.getlist('comparison[]'):
            if not f.filename: continue
            f_path = os.path.join(work_dir, secure_filename(f.filename))
            f.save(f_path)
            
            comp_data = get_compliance_data(f_path)
            
            # Sync Logic
            hop = 512
            ref_env = librosa.feature.rms(y=ref_data['y'][:20*ref_data['sr']], hop_length=hop)[0]
            comp_env = librosa.feature.rms(y=comp_data['y'][:20*comp_data['sr']], hop_length=hop)[0]
            corr = signal.correlate(comp_env, ref_env, mode='full')
            lag = np.argmax(corr) - (len(ref_env) - 1)
            offset_ms = round(float(lag * hop / ref_data['sr'] * 1000), 2)
            
            # Impact Scores
            lip_flap = 100 - (abs(ref_data['density'] - comp_data['density']) / (ref_data['density'] + 1e-6) * 100)
            
            # Visuals
            plt.figure(figsize=(10, 1.5))
            plt.plot(ref_env[:500], color='gray', alpha=0.3)
            plt.plot(comp_env[:500], color='#2563eb', alpha=0.7)
            plt.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            results.append({
                'filename': f.filename,
                'offset_ms': offset_ms,
                'timecode': ms_to_timecode(offset_ms, fps),
                'lip_flap': max(0, round(lip_flap, 2)),
                'phase': comp_data['meta']['phase'],
                'lufs': comp_data['meta']['lufs'],
                'meta': comp_data['meta'],
                'visual': base64.b64encode(buf.getvalue()).decode('utf-8'),
                'passed': abs(offset_ms) < 45 and comp_data['meta']['phase'] > 0
            })

        return jsonify({'results': results, 'ref_lufs': ref_data['meta']['lufs']})
    except Exception:
        return jsonify({'error': str(traceback.format_exc())}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
