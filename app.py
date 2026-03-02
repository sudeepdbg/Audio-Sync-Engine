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
import soundfile as sf
import threading
from scipy import signal
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template
from audio_separator.separator import Separator

app = Flask(__name__)

# --- CONFIGURATION ---
app.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_VOLATILE_PATH = os.path.join(BASE_DIR, "data")
os.makedirs(MEDIA_VOLATILE_PATH, exist_ok=True)

# --- NEW IMPACT ANALYSIS FUNCTIONS ---

def get_vocal_density(y, sr):
    """Measures syllables/speech density for Lip-Flap Sync."""
    if len(y) == 0: return 0
    # Use onset strength to find 'mouth open/close' events
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    # Normalized density: Syllables per second
    density = np.sum(onset_env > (np.max(onset_env) * 0.2)) / (duration + 1e-6)
    return float(density)

def get_tone_lock_score(y_ref, y_comp):
    """Compares spectral contrast to verify emotional texture matching."""
    # Spectral contrast captures the 'brightness' and 'texture' of the performance
    ref_contrast = np.mean(librosa.feature.spectral_contrast(y=y_ref))
    comp_contrast = np.mean(librosa.feature.spectral_contrast(y=y_comp))
    diff = abs(ref_contrast - comp_contrast)
    score = 100 - (diff * 12) # Empirical scaling for impact
    return float(max(0, min(100, score)))

def run_vocal_split(file_path, output_dir):
    """Isolates vocals using local Spleeter-ONNX."""
    sep = Separator()
    sep.load_model(model_filename='2stem_vocal_remover.onnx')
    output_files = sep.separate(file_path)
    # Construct the path to the generated vocal file
    vocal_file_path = os.path.join(output_dir, output_files[0])
    return vocal_file_path

# --- EXISTING UTILS ---

def get_file_metadata(path):
    try:
        info = sf.info(path)
        return {"sr": f"{info.samplerate} Hz", "duration": info.duration, "duration_str": f"{round(info.duration, 2)}s", "bit_depth": str(info.subtype), "channels": int(info.channels)}
    except:
        duration = librosa.get_duration(path=path)
        return {"sr": "N/A", "duration": duration, "duration_str": f"{round(duration, 2)}s", "bit_depth": "N/A", "channels": "N/A"}

def get_offset_at_time(y_ref, y_comp, sr, hop_length=512):
    ref_env = librosa.feature.rms(y=y_ref, hop_length=hop_length)[0]
    comp_env = librosa.feature.rms(y=y_comp, hop_length=hop_length)[0]
    ref_env = (ref_env - ref_env.min()) / (ref_env.max() - ref_env.min() + 1e-10)
    comp_env = (comp_env - comp_env.min()) / (comp_env.max() - comp_env.min() + 1e-10)
    correlation = signal.correlate(comp_env, ref_env, mode='full')
    lag_frame = np.argmax(correlation) - (len(ref_env) - 1)
    return round(float(lag_frame * hop_length / sr * 1000), 2)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    work_dir = os.path.join(MEDIA_VOLATILE_PATH, session_id)
    os.makedirs(work_dir, exist_ok=True)
    
    use_vocal_split = request.form.get('vocalSplit') == 'true'
    
    try:
        # Save Reference
        ref_file = request.files['reference']
        ref_path = os.path.join(work_dir, secure_filename(ref_file.filename))
        ref_file.save(ref_path)
        ref_meta = get_file_metadata(ref_path)

        # Pre-process Reference (Vocal Split)
        y_ref_full, sr = librosa.load(ref_path, sr=22050)
        y_ref_analysis = y_ref_full
        
        if use_vocal_split:
            vocal_path = run_vocal_split(ref_path, work_dir)
            y_ref_analysis, _ = librosa.load(vocal_path, sr=22050)

        results = []
        for f in request.files.getlist('comparison[]'):
            if not f.filename: continue
            f_path = os.path.join(work_dir, secure_filename(f.filename))
            f.save(f_path)
            
            y_comp, _ = librosa.load(f_path, sr=22050)
            comp_meta = get_file_metadata(f_path)
            
            # 1. Sync Analysis (Existing)
            start_offset = get_offset_at_time(y_ref_analysis[:20*sr], y_comp[:20*sr], sr)
            
            # 2. Impact Analysis (New)
            dens_ref = get_vocal_density(y_ref_analysis, sr)
            dens_comp = get_vocal_density(y_comp, sr)
            # Compare density: Closer to 100 is better
            lip_flap_score = 100 - (abs(dens_ref - dens_comp) / (dens_ref + 1e-6) * 100)
            tone_lock_score = get_tone_lock_score(y_ref_analysis, y_comp)
            
            # 3. Visualization
            plt.figure(figsize=(10, 2))
            librosa.display.waveshow(y_ref_analysis[:10*sr], sr=sr, alpha=0.5, label="Master")
            librosa.display.waveshow(y_comp[:10*sr], sr=sr, alpha=0.5, label="Dub")
            plt.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', transparent=True)
            plt.close()

            results.append({
                'filename': f.filename,
                'offset_ms': start_offset,
                'lip_flap_sync': round(max(0, lip_flap_score), 2),
                'tone_lock': round(tone_lock_score, 2),
                'visual': base64.b64encode(buf.getvalue()).decode('utf-8'),
                'ref_meta': ref_meta,
                'comp_meta': comp_meta,
                'needs_review': lip_flap_score < 80 or abs(start_offset) > 50
            })
            
        return jsonify({'reference': ref_file.filename, 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
