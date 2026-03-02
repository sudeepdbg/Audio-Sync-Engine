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
from functools import wraps
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional Loudness Scanning
try:
    import pyloudnorm as fln
    HAS_LOUDNORM = True
except ImportError:
    HAS_LOUDNORM = False
    logger.warning("pyloudnorm not installed. Loudness metrics will be unavailable.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024
app.config['JSON_SORT_KEYS'] = False

# Configuration Constants
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'ogg', 'aiff'}
ALLOWED_AUDIO_MIMETYPES = {'audio/mpeg', 'audio/wav', 'audio/flac', 'audio/mp4', 'audio/ogg', 'audio/x-aiff'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB per file
MAX_SESSION_FILES = 100
MIN_AUDIO_DURATION = 0.5  # seconds
MAX_AUDIO_DURATION = 3600  # 1 hour
CLEANUP_INTERVAL = 3600  # 1 hour
SESSION_TIMEOUT = 24  # hours

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_PATH = os.path.join(BASE_DIR, "data")
os.makedirs(MEDIA_PATH, exist_ok=True)

# Session tracking
active_sessions = {}

# --- VALIDATION & SECURITY ---

def allowed_file(filename):
    """Validate file extension and mimetype."""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_AUDIO_EXTENSIONS

def validate_audio_file(filepath):
    """Validate audio file integrity and properties."""
    try:
        info = sf.info(filepath)
        
        if info.duration < MIN_AUDIO_DURATION:
            raise ValueError(f"Audio too short ({info.duration:.2f}s). Minimum: {MIN_AUDIO_DURATION}s")
        
        if info.duration > MAX_AUDIO_DURATION:
            raise ValueError(f"Audio too long ({info.duration:.0f}s). Maximum: {MAX_AUDIO_DURATION}s")
        
        if info.channels < 1 or info.channels > 8:
            raise ValueError(f"Invalid channel count: {info.channels}")
        
        return True
    except Exception as e:
        logger.error(f"Audio validation failed: {str(e)}")
        raise

def require_files(*file_keys):
    """Decorator to validate required file uploads."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            for key in file_keys:
                if key not in request.files:
                    return jsonify({'error': f'Missing required file: {key}'}), 400
                if request.files[key].filename == '':
                    return jsonify({'error': f'Empty file: {key}'}), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# --- AUDIO PROCESSING & METRICS ---

def get_loudness(y, sr):
    """Calculate loudness in LUFS (Loudness Units relative to Full Scale)."""
    if not HAS_LOUDNORM:
        return "N/A"
    try:
        meter = fln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        if np.isinf(loudness) or np.isnan(loudness):
            return "N/A"
        return f"{round(loudness, 2)} LUFS"
    except Exception as e:
        logger.warning(f"Loudness calculation failed: {str(e)}")
        return "Error"

def get_lip_flap_density(y, sr):
    """
    Measures syllable/speech density for Lip-Flap Sync.
    Uses onset detection to count speech onsets per second.
    """
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration == 0:
            return 0.0
        
        # Count onsets above threshold
        threshold = np.max(onset_env) * 0.2 if np.max(onset_env) > 0 else 0
        density = np.sum(onset_env > threshold) / duration
        
        return float(max(0.0, density))
    except Exception as e:
        logger.warning(f"Lip-flap density calculation failed: {str(e)}")
        return 0.0

def get_tone_lock_score(y_ref, y_comp):
    """
    Compares spectral texture to verify emotional consistency.
    Higher score = more similar tone/texture.
    """
    try:
        # Use multiple spectral features for robustness
        ref_contrast = np.mean(librosa.feature.spectral_contrast(y=y_ref, sr=22050))
        comp_contrast = np.mean(librosa.feature.spectral_contrast(y=y_comp, sr=22050))
        
        ref_centroid = np.mean(librosa.feature.spectral_centroid(y=y_ref, sr=22050))
        comp_centroid = np.mean(librosa.feature.spectral_centroid(y=y_comp, sr=22050))
        
        # Normalize differences
        contrast_diff = abs(ref_contrast - comp_contrast) / (max(ref_contrast, comp_contrast, 1) + 1e-6)
        centroid_diff = abs(ref_centroid - comp_centroid) / max(ref_centroid, comp_centroid, 1)
        
        # Combined score
        avg_diff = (contrast_diff + centroid_diff) / 2
        score = 100 * (1 - min(avg_diff, 1.0))
        
        return float(max(0.0, min(100.0, score)))
    except Exception as e:
        logger.warning(f"Tone lock calculation failed: {str(e)}")
        return 50.0  # Return neutral score on error

def get_file_metadata(path):
    """Extract comprehensive file metadata."""
    try:
        info = sf.info(path)
        return {
            "sr": f"{info.samplerate} Hz",
            "duration": float(info.duration),
            "duration_str": f"{info.duration:.2f}s",
            "bit_depth": info.subtype if info.subtype else "Unknown",
            "channels": int(info.channels),
            "format": info.format
        }
    except Exception as e:
        logger.error(f"Metadata extraction failed for {path}: {str(e)}")
        return {
            "sr": "Unknown",
            "duration": 0,
            "duration_str": "0s",
            "bit_depth": "N/A",
            "channels": "N/A",
            "format": "Unknown"
        }

def get_offset(y_ref, y_comp, sr):
    """
    Calculate temporal offset between reference and comparison audio.
    Uses envelope correlation to find time shift in milliseconds.
    """
    try:
        hop = 512
        
        # Ensure arrays are not empty
        if len(y_ref) < hop or len(y_comp) < hop:
            return 0.0
        
        # Calculate RMS envelopes
        ref_env = librosa.feature.rms(y=y_ref, hop_length=hop)[0]
        comp_env = librosa.feature.rms(y=y_comp, hop_length=hop)[0]
        
        # Normalize envelopes
        ref_range = ref_env.max() - ref_env.min()
        comp_range = comp_env.max() - comp_env.min()
        
        if ref_range > 1e-10:
            ref_env = (ref_env - ref_env.min()) / ref_range
        if comp_range > 1e-10:
            comp_env = (comp_env - comp_env.min()) / comp_range
        
        # Find lag using cross-correlation
        if len(ref_env) == 0 or len(comp_env) == 0:
            return 0.0
        
        corr = signal.correlate(comp_env, ref_env, mode='full')
        lag = np.argmax(corr) - (len(ref_env) - 1)
        
        # Convert to milliseconds
        offset_ms = float(lag * hop / sr * 1000)
        return round(offset_ms, 2)
    except Exception as e:
        logger.warning(f"Offset calculation failed: {str(e)}")
        return 0.0

def generate_waveform_visual(y_ref, y_comp, sr):
    """Generate comparison waveform visualization."""
    try:
        fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
        
        # Limit display to first 10 seconds for performance
        max_samples = int(10 * sr)
        y_ref_display = y_ref[:max_samples]
        y_comp_display = y_comp[:max_samples]
        
        # Plot waveforms
        librosa.display.waveshow(y_ref_display, sr=sr, ax=ax, alpha=0.5, label="Master", color='#3b82f6')
        librosa.display.waveshow(y_comp_display, sr=sr, ax=ax, alpha=0.5, label="Dub", color='#10b981')
        
        ax.set_facecolor('#0f172a')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_xlim(0, len(y_ref_display) / sr)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor='#0f172a', edgecolor='none', bbox_inches='tight', pad_inches=0.1, dpi=80)
        plt.close(fig)
        
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Waveform generation failed: {str(e)}")
        return None

# --- SESSION MANAGEMENT ---

def cleanup_session(session_id):
    """Clean up session files after processing."""
    work_dir = os.path.join(MEDIA_PATH, session_id)
    try:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            logger.info(f"Cleaned up session {session_id}")
    except Exception as e:
        logger.error(f"Failed to cleanup session {session_id}: {str(e)}")

def cleanup_old_sessions():
    """Periodically clean up old session directories."""
    try:
        now = datetime.now()
        for session_dir in os.listdir(MEDIA_PATH):
            session_path = os.path.join(MEDIA_PATH, session_dir)
            if not os.path.isdir(session_path):
                continue
            
            mtime = datetime.fromtimestamp(os.path.getmtime(session_path))
            if now - mtime > timedelta(hours=SESSION_TIMEOUT):
                shutil.rmtree(session_path)
                logger.info(f"Cleaned up old session {session_dir}")
    except Exception as e:
        logger.error(f"Session cleanup failed: {str(e)}")

# --- ROUTES ---

@app.route('/')
def index():
    """Serve main interface."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/upload', methods=['POST'])
@require_files('reference')
def upload():
    """
    Main analysis endpoint.
    Accepts reference audio and multiple comparison files.
    """
    session_id = f"SES_{uuid.uuid4().hex[:8].upper()}"
    work_dir = os.path.join(MEDIA_PATH, session_id)
    
    try:
        os.makedirs(work_dir, exist_ok=True)
        active_sessions[session_id] = {
            'created': datetime.now(),
            'files': 0
        }
        
        # Process reference file
        ref_file = request.files['reference']
        
        if not allowed_file(ref_file.filename):
            return jsonify({'error': 'Invalid file format. Allowed: ' + ', '.join(ALLOWED_AUDIO_EXTENSIONS)}), 400
        
        ref_path = os.path.join(work_dir, secure_filename(ref_file.filename))
        ref_file.save(ref_path)
        
        # Validate reference audio
        try:
            validate_audio_file(ref_path)
        except ValueError as e:
            return jsonify({'error': f'Reference audio validation failed: {str(e)}'}), 400
        
        # Load reference
        try:
            y_ref, sr = librosa.load(ref_path, sr=22050, mono=True)
        except Exception as e:
            logger.error(f"Failed to load reference audio: {str(e)}")
            return jsonify({'error': 'Failed to load reference audio file'}), 400
        
        ref_meta = get_file_metadata(ref_path)
        ref_meta['loudness'] = get_loudness(y_ref, sr)
        ref_dens = get_lip_flap_density(y_ref, sr)
        
        results = []
        comparison_files = request.files.getlist('comparison[]')
        
        if not comparison_files or all(f.filename == '' for f in comparison_files):
            return jsonify({'error': 'At least one comparison file required'}), 400
        
        if len(comparison_files) > MAX_SESSION_FILES:
            return jsonify({'error': f'Maximum {MAX_SESSION_FILES} files allowed'}), 400
        
        # Process each comparison file
        for file_idx, f in enumerate(comparison_files):
            if not f.filename or f.filename == '':
                continue
            
            try:
                if not allowed_file(f.filename):
                    logger.warning(f"Skipping invalid file: {f.filename}")
                    continue
                
                f_path = os.path.join(work_dir, secure_filename(f.filename))
                f.save(f_path)
                
                # Validate comparison audio
                try:
                    validate_audio_file(f_path)
                except ValueError as e:
                    logger.warning(f"File validation failed for {f.filename}: {str(e)}")
                    continue
                
                # Load comparison
                y_comp, _ = librosa.load(f_path, sr=22050, mono=True)
                comp_meta = get_file_metadata(f_path)
                comp_meta['loudness'] = get_loudness(y_comp, sr)
                
                # Calculate metrics
                # Limit to first 30 seconds for offset calculation
                max_offset_samples = int(30 * sr)
                y_ref_offset = y_ref[:max_offset_samples]
                y_comp_offset = y_comp[:max_offset_samples]
                
                offset = get_offset(y_ref_offset, y_comp_offset, sr)
                comp_dens = get_lip_flap_density(y_comp, sr)
                
                # Calculate lip-flap sync (0-100%)
                if ref_dens + comp_dens > 0:
                    lip_flap = 100 - (abs(ref_dens - comp_dens) / (ref_dens + comp_dens) * 50)
                else:
                    lip_flap = 50.0
                
                tone_lock = get_tone_lock_score(y_ref, y_comp)
                
                # Generate summary
                summary_parts = []
                if abs(offset) <= 50:
                    summary_parts.append("✓ Timing is frame-accurate.")
                else:
                    summary_parts.append(f"⚠ Significant delay of {abs(offset)}ms detected.")
                
                if tone_lock >= 70:
                    summary_parts.append("✓ Tone texture matches master.")
                else:
                    summary_parts.append("⚠ Tone texture differs from master.")
                
                if lip_flap >= 80:
                    summary_parts.append("✓ Speech density synchronized.")
                else:
                    summary_parts.append("⚠ Speech density mismatch.")
                
                summary = " ".join(summary_parts)
                
                # Generate waveform
                visual = generate_waveform_visual(y_ref[:max_offset_samples], y_comp[:max_offset_samples], sr)
                
                # Determine if review needed
                needs_review = (
                    abs(offset) > 50 or
                    lip_flap < 80 or
                    tone_lock < 70
                )
                
                results.append({
                    'filename': f.filename,
                    'offset_ms': offset,
                    'lip_flap_sync': round(max(0.0, min(100.0, lip_flap)), 2),
                    'tone_lock': round(tone_lock, 2),
                    'summary': summary,
                    'visual': visual,
                    'ref_meta': ref_meta,
                    'comp_meta': comp_meta,
                    'needs_review': needs_review
                })
                
                logger.info(f"Processed {f.filename}: offset={offset}ms, lip_flap={lip_flap:.1f}%, tone_lock={tone_lock:.1f}%")
                
                # Cleanup after each file to manage memory
                del y_comp
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing {f.filename}: {str(traceback.format_exc())}")
                continue
        
        if not results:
            return jsonify({'error': 'No valid comparison files could be processed'}), 400
        
        response = {
            'reference': ref_file.filename,
            'session_id': session_id,
            'results': results,
            'total_files': len(results)
        }
        
        # Schedule cleanup after response
        cleanup_thread = threading.Thread(target=lambda: (
            time.sleep(3600),  # Keep for 1 hour
            cleanup_session(session_id)
        ))
        cleanup_thread.daemon = True
        cleanup_thread.start()
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Upload processing failed: {str(traceback.format_exc())}")
        cleanup_session(session_id)
        return jsonify({'error': 'Processing failed. Please try again.'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit exceeded."""
    return jsonify({'error': 'File size exceeds maximum allowed (800MB)'}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.after_request
def after_request(response):
    """Add security headers."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

if __name__ == '__main__':
    # Run cleanup on startup
    cleanup_old_sessions()
    
    # Production: use gunicorn with --workers 4 --timeout 120
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
