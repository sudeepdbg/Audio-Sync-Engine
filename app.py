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
import threading
import gc
import json
from scipy import signal
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- SYSTEM CONFIGURATION ---
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 
SUPPORTED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'aac', 'mp4'}

# --- QUALITY THRESHOLDS ---
EXACT_MATCH_THRESHOLD = 95    
DUB_MATCH_THRESHOLD = 15      
MAX_DRIFT_MS = 30             
MAX_START_OFFSET_MS = 50      
LOUDNESS_TARGET = -23.0        # EBU R128 broadcast standard
LOUDNESS_TOLERANCE = 2.0       # ±2 LUFS tolerance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_VOLATILE_PATH = os.path.join(BASE_DIR, "data")
if not os.path.exists(MEDIA_VOLATILE_PATH):
    os.makedirs(MEDIA_VOLATILE_PATH)

FINGERPRINT_CACHE = {}
_cache_lock = threading.Lock()

# --- UTILITIES ---

def allowed_file(filename):
    """Ensures only valid audio/video containers are processed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in SUPPORTED_EXTENSIONS

def get_file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def cleanup_session(path, delay=600):
    def _delete():
        time.sleep(delay)
        gc.collect() 
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    threading.Thread(target=_delete, daemon=True).start()

def get_file_metadata(path):
    """Retrieves technical specifications of the audio file."""
    try:
        info = sf.info(path)
        return {
            "sr": f"{info.samplerate} Hz",
            "sr_raw": info.samplerate,
            "duration": info.duration,
            "duration_str": f"{round(info.duration, 2)}s",
            "bit_depth": info.subtype,
            "channels": info.channels,
            "format": info.format
        }
    except Exception:
        duration = librosa.get_duration(path=path)
        y, sr = librosa.load(path, sr=None, duration=1)
        return {
            "sr": f"{sr} Hz",
            "sr_raw": sr,
            "duration": duration,
            "duration_str": f"{round(duration, 2)}s",
            "bit_depth": "Compressed",
            "channels": "Unknown",
            "format": "Unknown"
        }

def calculate_loudness(file_path):
    """
    Calculate Integrated LUFS loudness using ffmpeg ebur128 filter.
    Returns integrated loudness in LUFS and true peak.
    """
    try:
        # Use ffmpeg to calculate loudness (most accurate)
        cmd = [
            'ffmpeg', '-i', file_path, '-af', 'ebur128=peak=true',
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Parse the output for LUFS values
        output = result.stderr
        
        # Extract Integrated loudness
        import re
        integrated_match = re.search(r'I:\s*(-?\d+\.?\d*)\s+LUFS', output)
        integrated = float(integrated_match.group(1)) if integrated_match else None
        
        # Extract True Peak
        peak_match = re.search(r'Peak:\s*(-?\d+\.?\d*)\s+dBFS', output)
        true_peak = float(peak_match.group(1)) if peak_match else None
        
        # Extract LRA (Loudness Range)
        lra_match = re.search(r'LRA:\s*(-?\d+\.?\d*)\s+LU', output)
        lra = float(lra_match.group(1)) if lra_match else None
        
        return {
            "integrated_lufs": integrated,
            "true_peak_dbfs": true_peak,
            "lra": lra,
            "compliant": integrated and abs(integrated - LOUDNESS_TARGET) <= LOUDNESS_TOLERANCE if integrated else False
        }
    except Exception as e:
        print(f"Loudness calculation failed: {e}")
        # Fallback: Use pyloudnorm if ffmpeg fails
        try:
            import pyloudnorm as pyln
            y, sr = librosa.load(file_path, sr=None, duration=60)  # Analyze first 60s
            meter = pyln.Meter(sr)
            integrated = meter.integrated_loudness(y)
            return {
                "integrated_lufs": round(integrated, 2),
                "true_peak_dbfs": None,
                "lra": None,
                "compliant": abs(integrated - LOUDNESS_TARGET) <= LOUDNESS_TOLERANCE
            }
        except:
            return {
                "integrated_lufs": None,
                "true_peak_dbfs": None,
                "lra": None,
                "compliant": None
            }

# --- CORE LOGIC ---

def compare_fingerprints(fp_a, fp_b):
    try:
        if not fp_a or not fp_b: return 0.0
        fp_a, fp_b = fp_a.strip(), fp_b.strip()
        if fp_a == fp_b: return 100.0
        if ',' in fp_a:
            list_a = [int(x) for x in fp_a.split(',') if x.strip()]
            list_b = [int(x) for x in fp_b.split(',') if x.strip()]
            min_len = min(len(list_a), len(list_b))
            if min_len == 0: return 0.0
            matches = sum(1 for a, b in zip(list_a[:min_len], list_b[:min_len]) if a == b)
            return round((matches / min_len) * 100, 2)
        from difflib import SequenceMatcher
        return round(SequenceMatcher(None, fp_a, fp_b).ratio() * 100, 2)
    except (ValueError, TypeError, AttributeError): 
        return 0.0

def get_efficient_fingerprint(file_path):
    file_hash = get_file_hash(file_path)
    with _cache_lock:
        if file_hash in FINGERPRINT_CACHE: return FINGERPRINT_CACHE[file_hash]
    fpcalc_path = shutil.which("fpcalc") or "/usr/local/bin/fpcalc"
    try:
        cmd = [fpcalc_path, "-plain", file_path]
        fp = subprocess.check_output(cmd, timeout=30).decode().strip()
        with _cache_lock: FINGERPRINT_CACHE[file_hash] = fp
        return fp
    except Exception: return None

def get_offset_at_time(y_ref, y_comp, sr, hop_length):
    ref_env = librosa.feature.rms(y=y_ref, hop_length=hop_length)[0]
    comp_env = librosa.feature.rms(y=y_comp, hop_length=hop_length)[0]
    ref_env = (ref_env - ref_env.min()) / (ref_env.max() - ref_env.min() + 1e-10)
    comp_env = (comp_env - comp_env.min()) / (comp_env.max() - comp_env.min() + 1e-10)
    correlation = signal.correlate(comp_env, ref_env, mode='full')
    lag_frame = np.argmax(correlation) - (len(ref_env) - 1)
    return round(float(lag_frame * hop_length / sr * 1000), 2)

def analyze_lip_flap_sync(y_ref, y_comp, sr):
    """
    Analyze lip-flap synchronization by detecting transients.
    Lip-flap sync measures how well dialogue onsets align.
    """
    # Detect onsets (speech starts)
    onset_ref = librosa.onset.onset_detect(y=y_ref, sr=sr, units='time')
    onset_comp = librosa.onset.onset_detect(y=y_comp, sr=sr, units='time')
    
    if len(onset_ref) == 0 or len(onset_comp) == 0:
        return {"score": 100.0, "avg_delay": 0.0, "status": "No speech detected"}
    
    # Match onsets (simplified - take first 10 for comparison)
    min_len = min(10, len(onset_ref), len(onset_comp))
    delays = []
    for i in range(min_len):
        # Find closest onset in comp to ref onset
        ref_t = onset_ref[i]
        closest = min(onset_comp, key=lambda x: abs(x - ref_t))
        delays.append((closest - ref_t) * 1000)  # Convert to ms
    
    avg_delay = np.mean(delays)
    std_delay = np.std(delays)
    
    # Score: 100 - (avg_delay/2 + std_delay) bounded to 0-100
    score = max(0, min(100, 100 - (abs(avg_delay)/2 + std_delay)))
    
    return {
        "score": round(score, 2),
        "avg_delay_ms": round(avg_delay, 2),
        "jitter_ms": round(std_delay, 2),
        "status": "Good" if score > 80 else "Fair" if score > 60 else "Poor"
    }

def analyze_tone_lock(y_ref, y_comp, sr):
    """
    Analyze spectral balance consistency (Tone Lock).
    Measures how well frequency characteristics match.
    """
    # Compute spectral centroids
    cent_ref = librosa.feature.spectral_centroid(y=y_ref, sr=sr)[0]
    cent_comp = librosa.feature.spectral_centroid(y=y_comp, sr=sr)[0]
    
    # Compute spectral rolloff
    rolloff_ref = librosa.feature.spectral_rolloff(y=y_ref, sr=sr)[0]
    rolloff_comp = librosa.feature.spectral_rolloff(y=y_comp, sr=sr)[0]
    
    # Normalize and compare
    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
    
    cent_ref_norm = normalize(cent_ref)
    cent_comp_norm = normalize(cent_comp)
    rolloff_ref_norm = normalize(rolloff_ref)
    rolloff_comp_norm = normalize(rolloff_comp)
    
    # Correlation scores
    cent_corr = np.corrcoef(cent_ref_norm[:min(len(cent_ref_norm), len(cent_comp_norm))], 
                            cent_comp_norm[:min(len(cent_ref_norm), len(cent_comp_norm))])[0,1]
    rolloff_corr = np.corrcoef(rolloff_ref_norm[:min(len(rolloff_ref_norm), len(rolloff_comp_norm))], 
                               rolloff_comp_norm[:min(len(rolloff_ref_norm), len(rolloff_comp_norm))])[0,1]
    
    # Handle NaN
    cent_corr = 0.0 if np.isnan(cent_corr) else cent_corr
    rolloff_corr = 0.0 if np.isnan(rolloff_corr) else rolloff_corr
    
    # Composite tone lock score (0-100)
    score = round((cent_corr * 50 + rolloff_corr * 50), 2)
    
    return {
        "score": score,
        "spectral_balance": round(cent_corr * 100, 2),
        "frequency_range": round(rolloff_corr * 100, 2),
        "status": "Excellent" if score > 85 else "Good" if score > 70 else "Fair" if score > 50 else "Poor"
    }

def generate_impact_dashboard(start_offset, end_offset, drift, match_score, 
                              lip_flap, tone_lock, loudness):
    """
    Creates a visual dashboard of all quality metrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.patch.set_facecolor('#f8fafc')
    
    # 1. Temporal Offset (Start vs End)
    axes[0,0].bar(['Start', 'End'], [start_offset, end_offset], 
                  color=['#3b82f6', '#f59e0b'], alpha=0.7)
    axes[0,0].axhline(y=MAX_START_OFFSET_MS, color='red', linestyle='--', alpha=0.5)
    axes[0,0].set_title('Temporal Offset (ms)', fontsize=10)
    axes[0,0].set_ylabel('ms')
    
    # 2. Drift Over Duration
    axes[0,1].bar(['Drift'], [drift], color='#ef4444' if drift > MAX_DRIFT_MS else '#10b981')
    axes[0,1].axhline(y=MAX_DRIFT_MS, color='red', linestyle='--', alpha=0.5)
    axes[0,1].set_title(f'Drift (Max {MAX_DRIFT_MS}ms)', fontsize=10)
    
    # 3. Content DNA Match
    axes[0,2].bar(['DNA Match'], [match_score], color='#8b5cf6')
    axes[0,2].axhline(y=DUB_MATCH_THRESHOLD, color='red', linestyle='--', alpha=0.5)
    axes[0,2].set_title('Content DNA (%)', fontsize=10)
    axes[0,2].set_ylim([0, 100])
    
    # 4. Lip-Flap Sync
    lip_score = lip_flap['score'] if lip_flap else 0
    axes[1,0].bar(['Lip-Flap'], [lip_score], color='#ec4899')
    axes[1,0].set_title(f'Lip-Flap Sync: {lip_flap["status"] if lip_flap else "N/A"}', fontsize=10)
    axes[1,0].set_ylim([0, 100])
    
    # 5. Tone Lock
    tone_score = tone_lock['score'] if tone_lock else 0
    axes[1,1].bar(['Tone Lock'], [tone_score], color='#14b8a6')
    axes[1,1].set_title(f'Tone Lock: {tone_lock["status"] if tone_lock else "N/A"}', fontsize=10)
    axes[1,1].set_ylim([0, 100])
    
    # 6. Loudness Compliance
    loudness_val = abs(loudness['integrated_lufs']) if loudness and loudness['integrated_lufs'] else 0
    loudness_status = "Compliant" if loudness and loudness['compliant'] else "Non-compliant"
    loudness_color = '#10b981' if loudness and loudness['compliant'] else '#ef4444'
    axes[1,2].bar(['Loudness'], [loudness_val], color=loudness_color)
    axes[1,2].set_title(f'LUFS: {loudness_status}', fontsize=10)
    axes[1,2].axhline(y=LOUDNESS_TARGET, color='blue', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_summary(match_score, drift, start_offset, lip_flap, tone_lock, loudness):
    """Generates the text verdict using defined thresholds."""
    # Content analysis
    if match_score >= EXACT_MATCH_THRESHOLD:
        content_txt = "✅ Exact match detected (same source)."
    elif match_score >= DUB_MATCH_THRESHOLD:
        content_txt = "🔊 Acoustically related (consistent with localized dub)."
    else:
        content_txt = "⚠️ Content DNA mismatch (unrelated audio or different version)."
    
    # Sync analysis    
    if drift > MAX_DRIFT_MS:
        sync_txt = f"⚠️ Significant temporal drift ({drift}ms variation)."
    elif abs(start_offset) > MAX_START_OFFSET_MS:
        sync_txt = f"⏱️ Constant delay of {start_offset}ms detected."
    else:
        sync_txt = "✅ Sync is frame-accurate."
    
    # Lip-Flap analysis
    if lip_flap:
        if lip_flap['score'] > 80:
            lip_txt = "✅ Lip-flap sync excellent."
        elif lip_flap['score'] > 60:
            lip_txt = f"🔄 Lip-flap fair (avg delay {lip_flap['avg_delay_ms']}ms)."
        else:
            lip_txt = f"⚠️ Poor lip-flap sync (jitter: {lip_flap['jitter_ms']}ms)."
    else:
        lip_txt = ""
    
    # Tone Lock analysis
    if tone_lock:
        if tone_lock['score'] > 85:
            tone_txt = "✅ Spectral balance perfectly matched."
        elif tone_lock['score'] > 70:
            tone_txt = "🎚️ Good tone consistency."
        else:
            tone_txt = "⚠️ Noticeable tonal mismatch."
    else:
        tone_txt = ""
    
    # Loudness analysis
    if loudness and loudness['integrated_lufs']:
        if loudness['compliant']:
            loud_txt = f"✅ Loudness compliant ({loudness['integrated_lufs']} LUFS)."
        else:
            loud_txt = f"⚠️ Loudness off-spec ({loudness['integrated_lufs']} LUFS vs target {LOUDNESS_TARGET} LUFS)."
    else:
        loud_txt = ""
    
    # Combine all parts
    parts = [content_txt, sync_txt, lip_txt, tone_txt, loud_txt]
    return " ".join([p for p in parts if p])

def analyze_sync(anchor_path, rendition_path, ref_meta, sr=22050, hop_length=512):
    """
    Analyzes temporal sync between two files. 
    ref_meta is passed in to avoid redundant disk I/O.
    """
    fp_a = get_efficient_fingerprint(anchor_path)
    fp_b = get_efficient_fingerprint(rendition_path)
    match_score = compare_fingerprints(fp_a, fp_b)
    
    comp_meta = get_file_metadata(rendition_path)
    duration = min(ref_meta['duration'], comp_meta['duration'])
    
    y_ref_start, _ = librosa.load(anchor_path, sr=sr, duration=60)
    y_comp_start, _ = librosa.load(rendition_path, sr=sr, duration=60)
    
    start_offset = get_offset_at_time(y_ref_start, y_comp_start, sr, hop_length)
    
    end_offset = start_offset
    if duration > 120:
        y_ref_end, _ = librosa.load(anchor_path, sr=sr, offset=duration-60, duration=60)
        y_comp_end, _ = librosa.load(rendition_path, sr=sr, offset=duration-60, duration=60)
        end_offset = get_offset_at_time(y_ref_end, y_comp_end, sr, hop_length)
    
    drift = round(abs(start_offset - end_offset), 2)
    
    # New advanced analyses
    lip_flap = analyze_lip_flap_sync(y_ref_start, y_comp_start, sr)
    tone_lock = analyze_tone_lock(y_ref_start, y_comp_start, sr)
    loudness = calculate_loudness(rendition_path)
    
    summary = generate_summary(match_score, drift, start_offset, lip_flap, tone_lock, loudness)
    
    issues = []
    if abs(start_offset) > MAX_START_OFFSET_MS: 
        issues.append(f"Start Offset: {start_offset}ms")
    if drift > MAX_DRIFT_MS: 
        issues.append(f"Drift Detected: {drift}ms variance")
    if match_score < DUB_MATCH_THRESHOLD: 
        issues.append(f"DNA Match Low ({match_score}%)")
    if lip_flap and lip_flap['score'] < 60:
        issues.append(f"Lip-Flap Sync Poor ({lip_flap['score']}%)")
    if tone_lock and tone_lock['score'] < 70:
        issues.append(f"Tone Lock Fair ({tone_lock['score']}%)")
    if loudness and not loudness['compliant'] and loudness['integrated_lufs']:
        issues.append(f"Loudness Non-compliant ({loudness['integrated_lufs']} LUFS)")
    
    # Generate impact dashboard
    dashboard = generate_impact_dashboard(start_offset, end_offset, drift, match_score,
                                         lip_flap, tone_lock, loudness)
    
    # Visualization: Waveform comparison
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y_ref_start[:sr*10], sr=sr, color='blue', alpha=0.7)
    plt.title("Reference (First 10s)")
    
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y_comp_start[:sr*10], sr=sr, color='orange', alpha=0.7)
    plt.title(f"Comparison (Offset: {start_offset}ms)")
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    return {
        "start_offset": start_offset, "end_offset": end_offset, "drift": drift,
        "match_score": match_score, "summary": summary, "issues": issues,
        "visual": base64.b64encode(buf.getvalue()).decode('utf-8'),
        "dashboard": dashboard,
        "ref_meta": ref_meta, "comp_meta": comp_meta,
        "lip_flap": lip_flap, "tone_lock": tone_lock, "loudness": loudness
    }

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    try:
        gc.collect() 
        time.sleep(0.5)
        with _cache_lock:
            FINGERPRINT_CACHE.clear()

        if os.path.exists(MEDIA_VOLATILE_PATH):
            for item in os.listdir(MEDIA_VOLATILE_PATH):
                item_path = os.path.join(MEDIA_VOLATILE_PATH, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    else:
                        os.remove(item_path)
                except Exception as e:
                    print(f"Skipping {item}: {e}")
                    continue

        return jsonify({'status': 'Volatile storage and cache cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    analysis_root = os.path.join(MEDIA_VOLATILE_PATH, session_id)
    os.makedirs(analysis_root, exist_ok=True)
    try:
        ref_file = request.files['reference']
        if not ref_file or not allowed_file(ref_file.filename):
            return jsonify({'error': 'Reference file missing or unsupported format'}), 400

        ref_path = os.path.join(analysis_root, secure_filename(ref_file.filename))
        ref_file.save(ref_path)
        
        # Optimization: Scan reference once
        ref_meta = get_file_metadata(ref_path)
        
        comp_files = request.files.getlist('comparison[]')
        results = []
        for f in comp_files:
            if not f.filename or not allowed_file(f.filename):
                continue
            
            f_path = os.path.join(analysis_root, secure_filename(f.filename))
            f.save(f_path)
            
            analysis = analyze_sync(ref_path, f_path, ref_meta)
            results.append({
                'filename': f.filename,
                'offset_ms': analysis['start_offset'],
                'end_offset_ms': analysis['end_offset'],
                'drift_ms': analysis['drift'],
                'match_confidence': analysis['match_score'],
                'summary': analysis['summary'],
                'issues': analysis['issues'],
                'visual': analysis['visual'],
                'dashboard': analysis['dashboard'],
                'ref_meta': analysis['ref_meta'],
                'comp_meta': analysis['comp_meta'],
                'lip_flap': analysis['lip_flap'],
                'tone_lock': analysis['tone_lock'],
                'loudness': analysis['loudness'],
                'needs_review': len(analysis['issues']) > 0
            })
        
        cleanup_session(analysis_root)
        return jsonify({'reference': ref_file.filename, 'results': results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
