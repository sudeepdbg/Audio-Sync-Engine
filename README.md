# Audio Sync Engine Pro


A professional-grade web tool designed for broadcast quality control (QC) and localization workflows. This application uses digital signal processing (DSP) to compare a master audio reference against multiple dubs, detecting temporal drift, start offsets, and content DNA matches.

🚀 Key FeaturesAcoustic Fingerprinting: Utilizes fpcalc (Chromaprint) to generate content DNA and verify if localized dubs are acoustically related to the master.
#Temporal Sync Analysis: Calculates sub-millisecond start and end offsets using RMS envelope cross-correlation.
#Drift Detection: Identifies pitch or speed variances between tracks by comparing alignment at both the head and tail of the files.
#Technical Metadata Validation: Automatically flags mismatches in Sample Rate, Channel count, and Bit Depth between the reference and comparison tracks.
#Visual QC: Generates side-by-side waveform subplots for manual verification of signal alignment.
#Performance Optimized: Features MD5-based fingerprint caching and single-pass reference scanning to handle large batch uploads efficiently.

🛠 Tech StackBackend: Python, FlaskSignal Processing: Librosa, NumPy, SciPy, SoundFile
Visualization: Matplotlib (Agg backend)Fingerprinting: Chromaprint (fpcalc)Frontend: Modern Vanilla JS, CSS3 (Inter font-stack)

📋 PrerequisitesBefore running the application, ensure you have the following installed:Python 3.8+FFmpeg: Required by Librosa for decoding various audio formats.
Chromaprint: The fpcalc binary must be in your system PATH.

📖 Usage GuideSelect Master: Upload your high-resolution reference file (e.g., English Master).Upload Dubs: Select one or more localized tracks to validate.Analyze: Click "Start Analysis" to process the signals.Review:Green Labels: Sync is within acceptable broadcast thresholds (Drift < 30ms, Offset < 50ms).

Red Labels: Significant discrepancies detected that require manual review.

Format Details: Expand the "Format Details" to check for Sample Rate mismatches.

📊 Quality ThresholdsThe system is pre-configured with industry-standard thresholds for automated QC:
MetricThresholdStatusStart Offset< 50msFrame AccurateTotal Drift< 30msBroadcast ReadyDNA Match> 15%Acoustically Related
