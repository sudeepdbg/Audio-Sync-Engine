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
from scipy import signal
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- IMPACT ENGINE MODULES ---

def extract_vocal_characteristics(y, sr):
    """Filters audio to focus on human speech frequencies (300Hz-3400Hz)."""
    # Use a bandpass-like effect via short-time fourier transform
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    # Mask out non-vocal frequencies to simulate isolation
    vocal_mask = (freqs >= 300) & (freqs <= 3400)
    return S[vocal_mask, :]

def get_lip_flap_score(y, sr):
    """Calculates speech density (syllables per second) for sync impact."""
    if len(y) == 0: return 0
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    density = np.sum(onset_env > (np.max(onset_env) * 0.2)) / (duration + 1e-6)
    return float(density)

def get_tone_lock(y_ref, y_comp):
    """Verifies emotional texture match via spectral contrast."""
    ref_c = np.mean(librosa.feature.spectral_contrast(y=y_ref))
    comp_c = np.mean(librosa.feature.spectral_contrast(y=y_comp))
    score = 100 - (abs(ref_c - comp_c) * 15)
    return float(max(0, min(100, score)))

def get_offset(y_ref, y_comp, sr):
    """High-precision cross-correlation for temporal alignment."""
    # Use RMS envelopes for stable matching
    ref_env = librosa.feature.rms(y=y_ref, hop_length=512)[0]
    comp_env = librosa.feature.rms(y=y_comp, hop_length=512)[0]
    ref_env = (ref_env - ref_env.min()) / (ref_env.max() - ref_env.min() + 1e-10)
    comp_env = (comp_env - comp_env.min()) / (comp_env.max() - comp_env.min() + 1e-10)
    
    corr = signal.correlate(comp_env, ref_env, mode='full')
    lag = np.argmax(corr) - (len(ref_env) - 1)
    return round(float(lag * 512 / sr * 1000), 2)

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    work_dir = os.path.join(DATA_DIR, session_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        ref_file = request.files['reference']
        ref_path = os.path.join(work_dir, secure_filename(ref_file.filename))
        ref_file.save(ref_path)
        
        y_ref, sr = librosa.load(ref_path, sr=22050)
        ref_density = get_lip_flap_score(y_ref, sr)
        ref_meta = {"sr": sr, "duration": round(len(y_ref)/sr, 2)}

        results = []
        for f in request.files.getlist('comparison[]'):
            if not f.filename: continue
            f_path = os.path.join(work_dir, secure_filename(f.filename))
            f.save(f_path)
            
            y_comp, _ = librosa.load(f_path, sr=22050)
            
            # Metrics Calculation
            offset = get_offset(y_ref[:20*sr], y_comp[:20*sr], sr)
            comp_density = get_lip_flap_score(y_comp, sr)
            
            # Sync Score based on density similarity
            lip_sync = 100 - (abs(ref_density - comp_density) / (ref_density + 1e-6) * 100)
            tone_score = get_tone_lock(y_ref, y_comp)
            
            # Visual Alignment Waveform
            plt.figure(figsize=(10, 2))
            librosa.display.waveshow(y_ref[:10*sr], sr=sr, alpha=0.5, label="Master")
            librosa.display.waveshow(y_comp[:10*sr], sr=sr, alpha=0.5, label="Dub")
            plt.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            results.append({
                'filename': f.filename,
                'offset_ms': offset,
                'lip_flap_sync': round(max(0, lip_sync), 2),
                'tone_lock': round(tone_score, 2),
                'visual': base64.b64encode(buf.getvalue()).decode('utf-8'),
                'needs_review': lip_sync < 75 or abs(offset) > 50,
                'meta': {"sr": sr, "duration": round(len(y_comp)/sr, 2)}
            })

        return jsonify({'reference': ref_file.filename, 'results': results, 'ref_meta': ref_meta})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
