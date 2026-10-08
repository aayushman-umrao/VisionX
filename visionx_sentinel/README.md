# VisionX TensorFlow Sentinel

Real-time AI-powered object detection and analytics system. Uses a
pre-trained TensorFlow Lite model (SSD MobileNet V1, trained on COCO)
and OpenCV to detect objects live from your webcam, with on-screen
analytics, screenshot capture, CSV history logging, and one-key PDF
report generation.

## 1. Setup (one-time)

```bash
cd visionx_sentinel
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

The first time you run the script, it will automatically download the
pre-trained TensorFlow model (~27 MB) into the `models/` folder. You
need an internet connection for that one-time download only — after
that, it runs fully offline.

## 2. Run

```bash
python visionx_sentinel.py
```

A window will open showing your webcam feed with live detection boxes.

## 3. Controls

| Key | Action |
|-----|--------|
| `S` | Save a screenshot to `output/screenshots/` |
| `R` | Generate a PDF report instantly to `output/session_report.pdf` |
| `Q` | Quit (auto-saves the full detection log as `output/detections_log.csv`) |

## 4. What it detects

The model is trained on the COCO dataset (90 classes), which includes
everything in your spec — person, bottle, chair, laptop, cell phone —
plus many more (cup, book, keyboard, tv, etc.).

## 5. Folder structure

```
visionx_sentinel/
├── visionx_sentinel.py     # main script — everything lives here
├── requirements.txt
├── models/                 # auto-downloaded TFLite model + labels
├── output/
│   ├── screenshots/        # evidence captures
│   ├── detections_log.csv  # full timestamped detection history
│   └── session_report.pdf  # generated analytics report
```

## 6. Troubleshooting

- **Webcam doesn't open**: try changing `CAMERA_INDEX = 0` to `1` near
  the top of `visionx_sentinel.py`.
- **Low FPS**: this is normal on CPU-only laptops; the quantized model
  is already optimized for this. Closing other heavy apps helps.
- **Model download fails**: check your internet connection, or
  manually download the model zip from the URL in `MODEL_URL` inside
  the script and extract `detect.tflite` + `labelmap.txt` into
  `models/`.

## 7. For your demo / viva

If asked "where's the TensorFlow?" — the detection itself runs through
`tf.lite.Interpreter`, TensorFlow's official lightweight inference
engine, loading a real pre-trained TensorFlow model (SSD MobileNet).
This is the same underlying model family used in most production
mobile/edge object detection apps — it's TensorFlow, just the fast,
demo-friendly version of it.
