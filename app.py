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
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
ANALYSIS_FOLDER = 'analysis'
SUPPORTED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'aac', 'mp4'}
[os.makedirs(d, exist_ok=True) for d in [UPLOAD_FOLDER, ANALYSIS_FOLDER]]

try:
    import demucs.separate
    HAS_DEMUCS = True
except ImportError:
    HAS_DEMUCS = False
    print("⚠️ Warning: Demucs not found. Vocal DNA mode disabled.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in SUPPORTED_EXTENSIONS

# --- CORE LOGIC ---

def isolate_vocals(file_path, output_root):
    """Isolates vocals using Demucs with path verification."""
    if not HAS_DEMUCS:
        return file_path, False
    
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        # Demucs version-specific pathing can vary; we check the most common htdemucs output
        vocal_path = os.path.join(output_root, "htdemucs", base_name, "vocals.mp3")
        
        if not os.path.exists(vocal_path):
            print(f"🧬 Running Demucs on: {base_name}...")
            subprocess.run(["demucs", "--mp3", "-o", output_root, file_path], check=True, capture_output=True)
        
        if os.path.exists(vocal_path):
            return vocal_path, True
        else:
            print(f"⚠️ Warning: Demucs output not found at {vocal_path}. Falling back.")
            return file_path, False
    except Exception as e:
        print(f"❌ Demucs Error: {e}")
        return file_path, False

def get_offset(y_ref, y_comp, sr):
    correlation = np.correlate(y_ref, y_comp, mode='full')
    return (np.argmax(correlation) - len(y_comp) + 1) / sr * 1000

def analyze_sync(ref_path, comp_path):
    """Calculates offset and drift with explicit error handling."""
    try:
        # Load Start (First 20s)
        y_ref_start, sr = librosa.load(ref_path, duration=20)
        y_comp_start, _ = librosa.load(comp_path, duration=20)
        start_offset = round(get_offset(y_ref_start, y_comp_start, sr), 2)
        
        # Load End (Last 20s)
        try:
            duration_ref = librosa.get_duration(path=ref_path)
            duration_comp = librosa.get_duration(path=comp_path)
            
            # Only attempt drift if file is long enough
            if min(duration_ref, duration_comp) > 40:
                y_ref_end, _ = librosa.load(ref_path, offset=duration_ref-20, duration=20)
                y_comp_end, _ = librosa.load(comp_path, offset=duration_comp-20, duration=20)
                end_offset = get_offset(y_ref_end, y_comp_end, sr)
                drift_ms = round(abs(end_offset - start_offset), 2)
            else:
                drift_ms = 0.0
        except Exception as e:
            print(f"⚠️ End-offset analysis failed: {e}")
            drift_ms = None # Signal "N/A" to UI
            
        return start_offset, drift_ms
    except Exception as e:
        print(f"❌ Sync Analysis Error: {e}")
        return 0.0, None

def classify_audio_quality(file_path, chunk_duration=30):
    """Memory-efficient quality analysis using chunked reading."""
    try:
        info = sf.info(file_path)
        max_frames = int(info.samplerate * chunk_duration)
        
        with sf.SoundFile(file_path) as f:
            data = f.read(frames=min(max_frames, len(f)), dtype='float32')
        
        rms = np.sqrt(np.mean(data**2))
        lufs = 20 * np.log10(rms + 1e-9) # Simplified loudness
        peak = np.max(np.abs(data))
        
        return {
            "quality_label": "Verified" if lufs > -27 else "Low Volume",
            "peak_status": "Clean Peaks" if peak < 0.98 else "CLIPPING",
            "dynamic_range": f"{round(lufs, 2)} LUFS"
        }
    except Exception as e:
        print(f"❌ Quality Scan Error: {e}")
        return {"quality_label": "Error", "peak_status": "Unknown", "dynamic_range": "N/A"}

def cleanup_session(path):
    if os.path.exists(path):
        shutil.rmtree(path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    session_id = str(uuid.uuid4())
    analysis_root = os.path.join(ANALYSIS_FOLDER, session_id)
    os.makedirs(analysis_root, exist_ok=True)
    
    try:
        ref_file = request.files.get('reference')
        comp_files = request.files.getlist('comparison[]')
        deep_analysis = request.form.get('deepAnalysis') == 'true'

        if not ref_file or not allowed_file(ref_file.filename):
            return jsonify({'error': 'Invalid Master file'}), 400

        ref_path = os.path.join(analysis_root, "master_" + ref_file.filename)
        ref_file.save(ref_path)

        # 1. Pre-process Reference for Deep Mode (Once per request)
        active_ref, ref_is_vocal = isolate_vocals(ref_path, analysis_root) if (deep_analysis and HAS_DEMUCS) else (ref_path, False)

        results = []
        for f in comp_files:
            if not f.filename or not allowed_file(f.filename):
                continue
                
            comp_path = os.path.join(analysis_root, f.filename)
            f.save(comp_path)
            
            # 2. Process Dub for Deep Mode
            active_comp, comp_is_vocal = isolate_vocals(comp_path, analysis_root) if (deep_analysis and HAS_DEMUCS) else (comp_path, False)
            
            # 3. Analyze
            offset, drift = analyze_sync(active_ref, active_comp)
            quality = classify_audio_quality(comp_path)
            
            # Content DNA (Match Confidence)
            y_ref, sr = librosa.load(active_ref, duration=10)
            y_comp, _ = librosa.load(active_comp, duration=10)
            conf = round(np.corrcoef(librosa.feature.melspectrogram(y=y_ref, sr=sr).flatten(), 
                                    librosa.feature.melspectrogram(y=y_comp, sr=sr).flatten())[0,1] * 100, 2)

            results.append({
                'filename': f.filename,
                'offset_ms': offset,
                'drift_ms': drift if drift is not None else "undefined",
                'match_confidence': max(0, conf),
                'quality': quality,
                'needs_review': abs(offset) > 50 or (drift is not None and drift > 30) or conf < 15,
                'deep_mode_active': comp_is_vocal
            })

        return jsonify({'reference': ref_file.filename, 'results': results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup can be tricky if you want to keep files for the UI to show. 
        # Usually, we'd clean up after a timeout, but for now, we'll keep the session 
        # until the next 'Clear Cache' button press or manual wipe.
        pass

if __name__ == '__main__':
    app.run(port=5001, debug=True)
