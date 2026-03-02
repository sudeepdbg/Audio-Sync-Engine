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
app.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- ENGINE LOGIC ---

def get_audio_metrics(y, sr, path):
    """Gathers all technical and impact data in one pass."""
    # Technical Specs
    info = sf.info(path)
    
    # Loudness
    loudness = "N/A"
    if HAS_LOUDNORM:
        try:
            meter = fln.Meter(sr)
            loudness = f"{round(meter.integrated_loudness(y), 2)} LUFS"
        except: pass

    # Speech Density (Lip-Flap)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    density = np.sum(onset_env > (np.max(onset_env) * 0.2)) / (librosa.get_duration(y=y, sr=sr) + 1e-6)

    return {
        "meta": {
            "sr": f"{info.samplerate}Hz",
            "channels": "Stereo" if info.channels > 1 else "Mono",
            "bit_depth": info.subtype,
            "duration": f"{round(info.duration, 2)}s",
            "loudness": loudness
        },
        "density": density,
        "raw_y": y
    }

def get_comparison(ref_data, comp_data, sr):
    """Calculates relative impact scores."""
    # Offset
    hop = 512
    ref_env = librosa.feature.rms(y=ref_data['raw_y'][:20*sr], hop_length=hop)[0]
    comp_env = librosa.feature.rms(y=comp_data['raw_y'][:20*sr], hop_length=hop)[0]
    corr = signal.correlate(comp_env, ref_env, mode='full')
    lag = np.argmax(corr) - (len(ref_env) - 1)
    offset = round(float(lag * hop / sr * 1000), 2)

    # Lip Flap Score
    lip_flap = 100 - (abs(ref_data['density'] - comp_data['density']) / (ref_data['density'] + 1e-6) * 100)
    
    # Tone Lock (Spectral Contrast)
    ref_c = np.mean(librosa.feature.spectral_contrast(y=ref_data['raw_y']))
    comp_c = np.mean(librosa.feature.spectral_contrast(y=comp_data['raw_y']))
    tone_lock = 100 - (abs(ref_c - comp_c) * 12)

    return offset, max(0, round(lip_flap, 2)), max(0, round(tone_lock, 2))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    session_id = f"QC_{uuid.uuid4().hex[:6].upper()}"
    work_dir = os.path.join(DATA_DIR, session_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        ref_file = request.files['reference']
        ref_path = os.path.join(work_dir, secure_filename(ref_file.filename))
        ref_file.save(ref_path)
        
        y_r, sr = librosa.load(ref_path, sr=22050)
        ref_analysis = get_audio_metrics(y_r, sr, ref_path)

        results = []
        for f in request.files.getlist('comparison[]'):
            if not f.filename: continue
            f_path = os.path.join(work_dir, secure_filename(f.filename))
            f.save(f_path)
            
            y_c, _ = librosa.load(f_path, sr=22050)
            comp_analysis = get_audio_metrics(y_c, sr, f_path)
            
            offset, lip_flap, tone_lock = get_comparison(ref_analysis, comp_analysis, sr)

            # Generate Waveform
            plt.figure(figsize=(12, 2))
            librosa.display.waveshow(y_r[:10*sr], sr=sr, alpha=0.3, color='gray')
            librosa.display.waveshow(y_c[:10*sr], sr=sr, alpha=0.6, color='#2563eb')
            plt.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            results.append({
                'filename': f.filename,
                'offset_ms': offset,
                'lip_flap': lip_flap,
                'tone_lock': tone_lock,
                'visual': base64.b64encode(buf.getvalue()).decode('utf-8'),
                'ref_meta': ref_analysis['meta'],
                'comp_meta': comp_analysis['meta'],
                'passed': abs(offset) < 50 and lip_flap > 80
            })

        return jsonify({'session': session_id, 'reference': ref_file.filename, 'results': results})
    except Exception:
        return jsonify({'error': str(traceback.format_exc())}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
