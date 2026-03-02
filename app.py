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
app.config['MAX_CONTENT_LENGTH'] = 2048 * 1024 * 1024 # Supports 2GB Files

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- ADVANCED CALCULATION ENGINES ---

def get_timecode(ms, fps=24):
    """Translates milliseconds to broadcast timecode HH:MM:SS:FF."""
    abs_ms = abs(ms)
    total_sec = abs_ms / 1000.0
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    f = int((total_sec % 1) * fps)
    return f"{'-' if ms < 0 else ''}{h:02}:{m:02}:{s:02}:{f:02}"

def analyze_broadcaster_specs(path):
    """Deep scan for loudness, phase, and hardware metadata."""
    info = sf.info(path)
    # Technical Metadata
    meta = {
        "sr": f"{info.samplerate} Hz",
        "bit_depth": info.subtype,
        "channels": "Stereo" if info.channels > 1 else "Mono",
        "duration": f"{round(info.duration, 2)}s"
    }
    
    # Advanced Compliance
    lufs = "N/A"
    phase = 1.0
    try:
        data, rate = sf.read(path)
        if HAS_LOUDNORM:
            meter = fln.Meter(rate)
            lufs = f"{round(meter.integrated_loudness(data), 2)} LUFS"
        
        if info.channels >= 2:
            # Correlation: +1 = Mono/In-Phase, -1 = Phase Inverted
            phase = round(float(np.corrcoef(data[:, 0], data[:, 1])[0, 1]), 3)
    except: pass

    return meta, lufs, phase

def get_sync_metrics(y_ref, y_comp, sr):
    """Calculates temporal offset and lip-flap sync density."""
    hop = 512
    # Temporal Offset via Cross-Correlation
    ref_env = librosa.feature.rms(y=y_ref[:30*sr], hop_length=hop)[0]
    comp_env = librosa.feature.rms(y=y_comp[:30*sr], hop_length=hop)[0]
    corr = signal.correlate(comp_env, ref_env, mode='full')
    lag = np.argmax(corr) - (len(ref_env) - 1)
    offset = round(float(lag * hop / sr * 1000), 2)

    # Lip-Flap (Speech Density Match)
    ref_ons = librosa.onset.onset_strength(y=y_ref, sr=sr)
    comp_ons = librosa.onset.onset_strength(y=y_comp, sr=sr)
    ref_dens = np.sum(ref_ons > (np.max(ref_ons)*0.2))
    comp_dens = np.sum(comp_ons > (np.max(comp_ons)*0.2))
    lip_flap = 100 - (abs(ref_dens - comp_dens) / (ref_dens + 1e-6) * 100)

    return offset, max(0, round(lip_flap, 2))

@app.route('/')
def index(): return render_template('index.html')

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
        
        # Load Reference (Mono/16k for speed)
        y_r, sr = librosa.load(ref_path, sr=16000)
        ref_meta, ref_lufs, _ = analyze_broadcaster_specs(ref_path)

        results = []
        for f in request.files.getlist('comparison[]'):
            if not f.filename: continue
            f_path = os.path.join(work_dir, secure_filename(f.filename))
            f.save(f_path)
            
            y_c, _ = librosa.load(f_path, sr=16000)
            comp_meta, comp_lufs, phase = analyze_broadcaster_specs(f_path)
            offset, lip_flap = get_sync_metrics(y_r, y_c, sr)

            # Build Summary Message
            summary = "✅ Technical Pass."
            if abs(offset) > (1000/fps): summary = f"⚠️ Sync Warning: {get_timecode(offset, fps)} drift."
            if phase < 0: summary = "❌ Phase Issue: Audio may cancel in mono."

            # Visual
            plt.figure(figsize=(10, 2))
            librosa.display.waveshow(y_r[:15*sr], sr=sr, alpha=0.3, color='gray')
            librosa.display.waveshow(y_c[:15*sr], sr=sr, alpha=0.6, color='#2563eb')
            plt.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            results.append({
                'filename': f.filename,
                'offset_ms': offset,
                'timecode': get_timecode(offset, fps),
                'lip_flap': lip_flap,
                'summary': summary,
                'lufs': comp_lufs,
                'phase': phase,
                'meta': comp_meta,
                'visual': base64.b64encode(buf.getvalue()).decode('utf-8'),
                'passed': abs(offset) < (1000/fps) and phase > 0
            })

        return jsonify({'results': results, 'master_lufs': ref_lufs})
    except Exception:
        return jsonify({'error': str(traceback.format_exc())}), 500
