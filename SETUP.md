# 🔧 Setup & Installation Guide

Complete step-by-step guide to set up and run the Player Detection System.

---

## 📋 Prerequisites

### System Requirements
- **OS:** Linux, macOS, or Windows
- **Python:** 3.10 or higher
- **RAM:** 8GB minimum (16GB recommended for 4K video)
- **Disk Space:** 10GB+ (for dependencies and sample data)
- **GPU:** NVIDIA GPU with CUDA support (optional but recommended for real-time performance)

### Software Dependencies
- Git
- pip (Python package manager)
- Virtual environment (venv or conda)

---

## 🐍 Step 1: Set Up Python Environment

### Option A: Using venv (Recommended)

```bash
# Clone the repository
git clone https://github.com/SumitDendage/player-detection-system.git
cd player-detection-system

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows (PowerShell):
# venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
# venv\Scripts\activate

# Verify activation (you should see (venv) in terminal)
which python  # or 'where python' on Windows
```

### Option B: Using Conda

```bash
# Create conda environment
conda create -n player-detection python=3.10

# Activate environment
conda activate player-detection
```

---

## 📦 Step 2: Install Dependencies

### Step 2a: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 2b: Install Project Dependencies

```bash
# From project root directory
pip install -r requirements.txt
```

### Step 2c: Install GPU Support (Optional but Recommended)

**For NVIDIA GPU with CUDA:**

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"  # Should print: True
```

**For CPU Only (Slower, but works):**

The default installation works on CPU. GPU installation above is optional.

---

## 🤖 Step 3: Download YOLOv11 Weights

### Automated Download

```bash
# Run the weight downloader
python src/models/download_weights.py

# This will download YOLOv11n (nano) by default (~6MB)
# Available models: nano, small, medium, large, xlarge
```

### Manual Download

```bash
# Create weights directory
mkdir -p weights

# Download specific model size:
# nano (~6MB, fastest)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11n.pt -O weights/yolov11.pt

# small (~13MB)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11s.pt -O weights/yolov11.pt

# medium (~26MB, recommended)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11m.pt -O weights/yolov11.pt

# large (~49MB, more accurate)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11l.pt -O weights/yolov11.pt

# xlarge (~99MB, best accuracy but slowest)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11x.pt -O weights/yolov11.pt
```

---

## ✅ Step 4: Verify Installation

```bash
# Test Python packages
python -c "
import torch
import cv2
import ultralytics
from ultralytics import YOLO
print('✅ All packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'OpenCV version: {cv2.__version__}')
"

# Test YOLO model loading
python -c "
from ultralytics import YOLO
model = YOLO('weights/yolov11.pt')
print('✅ YOLO model loaded successfully!')
print(f'Model: {model.model_name}')
"

# Test project import
python -c "
from src.models import PlayerDetector
from src.tracking import PlayerTracker
print('✅ Project modules imported successfully!')
"
```

---

## 🎥 Step 5: Prepare Input Videos

### Video Format Requirements

- **Codec:** H.264 or H.265
- **Container:** MP4, AVI, MOV
- **Resolution:** 1080p or higher recommended
- **Frame Rate:** 24-60 FPS
- **Color Space:** RGB or BGR

### Organize Video Files

```bash
# Create data directory
mkdir -p data/videos/{broadcast,tactical}

# Copy your videos
cp /path/to/broadcast.mp4 data/videos/broadcast/
cp /path/to/tactical.mp4 data/videos/tactical/

# Expected structure:
# data/videos/
# ├── broadcast/
# │   └── broadcast.mp4
# └── tactical/
#     └── tacticam.mp4
```

### Synchronize Videos (Important!)

Videos should be temporally aligned. Use ffmpeg to trim/synchronize:

```bash
# Check video duration
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1:novalue=1 broadcast.mp4

# Trim to same duration
ffmpeg -i broadcast.mp4 -t 120 broadcast_trimmed.mp4
ffmpeg -i tacticam.mp4 -t 120 tacticam_trimmed.mp4

# Re-encode if needed
ffmpeg -i broadcast.mp4 -c:v libx264 -c:a aac broadcast_encoded.mp4
```

---

## 🚀 Step 6: Run the System

### Option A: Command Line Interface

```bash
# Basic run
python player_mapper.py \
  --broadcast data/videos/broadcast/broadcast.mp4 \
  --tacticam data/videos/tactical/tacticam.mp4 \
  --weights weights/yolov11.pt

# With custom parameters
python player_mapper.py \
  --broadcast data/videos/broadcast/broadcast.mp4 \
  --tacticam data/videos/tactical/tacticam.mp4 \
  --weights weights/yolov11.pt \
  --confidence 0.4 \
  --iou-threshold 0.5 \
  --device 0 \
  --output output/

# View all options
python player_mapper.py --help
```

### Option B: Web GUI Interface

```bash
# Start the NiceGUI application
python main_app.py

# Access at http://localhost:8000 in your browser
# Upload videos through the web interface
# Click "Run Analysis"
# Download results
```

### Option C: Python Script

```python
# Create run_analysis.py
from src.models import PlayerDetector
from src.tracking import PlayerTracker, CrossCameraMapper
from src.tracking.feature_extractor import extract_features
import json

# Configuration
CONFIG = {
    'broadcast_video': 'data/videos/broadcast/broadcast.mp4',
    'tactical_video': 'data/videos/tactical/tacticam.mp4',
    'weights': 'weights/yolov11.pt',
    'device': 0,  # GPU device ID, or 'cpu'
    'confidence': 0.4,
    'iou_threshold': 0.5,
}

# Initialize
detector = PlayerDetector(CONFIG['weights'], device=CONFIG['device'])
tracker = PlayerTracker(iou_threshold=CONFIG['iou_threshold'])
mapper = CrossCameraMapper()

# Process broadcast feed
print("Processing broadcast feed...")
broadcast_detections = detector.detect_video(CONFIG['broadcast_video'])
broadcast_tracks = tracker.track(broadcast_detections)
broadcast_features = extract_features(broadcast_tracks, CONFIG['broadcast_video'])

# Process tactical feed
print("Processing tactical feed...")
tactical_detections = detector.detect_video(CONFIG['tactical_video'])
tactical_tracks = tracker.track(tactical_detections)
tactical_features = extract_features(tactical_tracks, CONFIG['tactical_video'])

# Map players
print("Matching players across cameras...")
player_mapping = mapper.match(broadcast_features, tactical_features)

# Save results
with open('output/player_mappings.json', 'w') as f:
    json.dump(player_mapping, f, indent=2)

print("✅ Analysis complete!")
print(f"Found {len(player_mapping)} player correspondences")
```

---

## 📊 Step 7: View Results

### Output Files

```
output/
├── broadcast_annotated.mp4      # Annotated broadcast video
├── tacticam_annotated.mp4       # Annotated tactical video
├── player_mappings.json         # Player ID mappings
└── statistics.json              # Performance statistics
```

### Examine Results

```bash
# View JSON mapping
cat output/player_mappings.json | python -m json.tool

# Check video properties
ffprobe output/broadcast_annotated.mp4

# Play annotated videos
ffplay output/broadcast_annotated.mp4
```

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file in project root:

```env
# YOLO Settings
YOLO_DEVICE=0                   # GPU device ID (0, 1, 2, etc.) or 'cpu'
YOLO_CONFIDENCE=0.4             # Detection confidence threshold
YOLO_IOU_THRESHOLD=0.5          # Tracking IOU threshold

# Feature Extraction
HIST_BINS=8                     # Color histogram bins per channel
FEATURE_SIZE=64,128             # Crop size for feature extraction

# Output Settings
OUTPUT_DIR=output               # Output directory
SAVE_ANNOTATED_VIDEOS=true      # Save annotated videos
SAVE_JSON_MAPPING=true          # Save JSON mapping
VIDEO_FPS=30                    # Output video FPS
```

### Load Configuration

```python
from dotenv import load_dotenv
import os

load_dotenv()
DEVICE = os.getenv('YOLO_DEVICE', 'cpu')
CONFIDENCE = float(os.getenv('YOLO_CONFIDENCE', 0.4))
```

---

## 🐛 Troubleshooting

### Issue: CUDA not available

```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or use CPU (slower):
# Modify code to use device='cpu'
```

### Issue: Out of Memory (OOM)

```bash
# Use smaller model
python player_mapper.py \
  --broadcast broadcast.mp4 \
  --tacticam tacticam.mp4 \
  --weights weights/yolov11n.pt  # Use nano instead of large

# Reduce batch size
# Reduce input resolution
# Use --batch-size 1
```

### Issue: Videos not synchronized

```bash
# Check duration
ffprobe -show_entries format=duration broadcast.mp4
ffprobe -show_entries format=duration tacticam.mp4

# Trim to same length
ffmpeg -i broadcast.mp4 -t 120 broadcast_trim.mp4
ffmpeg -i tacticam.mp4 -t 120 tacticam_trim.mp4
```

### Issue: No players detected

```bash
# Lower confidence threshold
python player_mapper.py \
  --confidence 0.3  # Default is 0.4

# Check video quality
# Ensure adequate lighting
# Verify video codec support
```

---

## 📚 Additional Resources

- **[YOLOv11 Documentation](https://docs.ultralytics.com/)** - Official YOLO docs
- **[OpenCV Tutorials](https://docs.opencv.org/master/index.html)** - Computer vision basics
- **[SciPy Optimization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)** - Hungarian algorithm
- **[FFmpeg Guide](https://ffmpeg.org/ffmpeg.html)** - Video processing

---

## ✅ Verification Checklist

Before running analysis, verify:

- [ ] Python 3.10+ installed
- [ ] Virtual environment activated
- [ ] All packages installed (`pip list`)
- [ ] GPU working (if applicable)
- [ ] YOLO weights downloaded
- [ ] Input videos prepared and synchronized
- [ ] Output directory exists
- [ ] Sufficient disk space available

---

## 🆘 Getting Help

If you encounter issues:

1. Check [TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)
2. Review [GitHub Issues](https://github.com/SumitDendage/player-detection-system/issues)
3. Create a new issue with:
   - Python version
   - OS information
   - Error messages
   - Video specifications
   - Hardware details

---

**Ready to run! Follow the steps above and start analyzing.** 🚀
