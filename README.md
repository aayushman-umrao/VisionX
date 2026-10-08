# VisionX TensorFlow Sentinel

Real-time AI-powered object detection & analytics system using TensorFlow Lite and OpenCV.

## Features

- **Real-time Object Detection**: Uses TensorFlow Lite (SSD MobileNet / EfficientDet-Lite0) for live webcam detection
- **Object Tracking**: Persistent object tracking with unique IDs (PRO MAX edition)
- **Restricted Zone Alerts**: Draw zones and get alerts when objects enter them
- **Analytics & Reporting**: Live HUD with FPS, object counts, trend graphs; instant PDF reports
- **Screenshot Capture**: Save evidence frames with timestamps
- **Detection Logging**: Full CSV history of all detections with timestamps and confidence
- **Cross-model Support**: Auto-fallback from EfficientDet-Lite0 to SSD MobileNet V1

## Versions

| File | Description |
|------|-------------|
| `v3.py` | **PRO MAX** - Full featured: tracking, zones, trends, alerts, dwell time |
| `visionx_sentinel.py` | Original Sentinel - Detection, logging, screenshots, PDF reports |
| `visionx_sentinel/visionx_sentinel.py` | Standalone sub-project version |

## Quick Start

### Prerequisites
- Python 3.8+
- Webcam

### Installation
```bash
git clone <this-repo>
cd VisionX
pip install -r requirements.txt
```

### Run PRO MAX Edition (recommended)
```bash
python v3.py
```

### Run Original Sentinel
```bash
python visionx_sentinel.py
```

### Controls (while webcam window is focused)

| Key | Action |
|-----|--------|
| `S` | Save screenshot |
| `R` | Generate PDF analytics report |
| `Z` | Toggle restricted-zone drawing mode (PRO MAX) |
| `C` | Clear restricted zone (PRO MAX) |
| `Q` | Quit (auto-saves logs) |

## Model Setup

On first run, the script automatically downloads:
- **Primary**: EfficientDet-Lite0 (~5.6 MB) - Better accuracy (~25.7% mAP)
- **Fallback**: SSD MobileNet V1 (~4.3 MB) - Original model

Label map (COCO 80 classes) is downloaded with the fallback model.

## Output Files

All runtime outputs go to `output/`:
- `detections_log.csv` - Full detection history
- `alerts_log.csv` - Zone breach alerts (PRO MAX)
- `screenshots/` - Captured frames
- `session_report.pdf` - Generated analytics report

## Requirements

```
opencv-python>=4.8.0
numpy>=1.24.0
tensorflow>=2.15.0
reportlab>=4.0.0
```

## Project Structure

```
VisionX/
├── v3.py                          # PRO MAX edition (main)
├── visionx_sentinel.py            # Original sentinel
├── Check.py                       # Labelmap diagnostic tool
├── check2.py                      # Raw model debug tool
├── requirements.txt
├── models/
│   ├── detect.tflite              # SSD MobileNet (downloaded)
│   ├── efficientdet_lite0.tflite  # EfficientDet-Lite0 (downloaded)
│   └── labelmap.txt               # COCO labels (downloaded)
├── output/                        # Runtime outputs (gitignored)
│   ├── detections_log.csv
│   ├── alerts_log.csv
│   ├── screenshots/
│   └── session_report.pdf
└── visionx_sentinel/              # Sub-project
    ├── visionx_sentinel.py
    ├── requirements.txt
    ├── models/
    └── output/
```

## License

MIT License - Feel free to use and modify.

---

**Author**: Aayushman