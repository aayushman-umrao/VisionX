"""
VisionX TensorFlow Sentinel
============================
Real-time AI-powered object detection & analytics system.
Uses TensorFlow Lite (SSD MobileNet) + OpenCV for live webcam detection,
with object counting, history logging, screenshot capture, and PDF reporting.

Controls (while the webcam window is focused):
    S  -> Save a screenshot of the current frame
    R  -> Generate a PDF analytics report instantly
    Q  -> Quit (auto-saves the CSV detection log)

Author: Aayushman
"""

import os
import csv
import time
import urllib.request
from datetime import datetime
from collections import Counter

import cv2
import numpy as np

# TensorFlow Lite runtime (part of the TensorFlow ecosystem)
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    # Fallback to the lightweight standalone tflite-runtime package
    from tflite_runtime.interpreter import Interpreter


# =====================================================================
# 1. CONFIG
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "screenshots")

MODEL_PATH = os.path.join(MODEL_DIR, "detect.tflite")
LABELMAP_PATH = os.path.join(MODEL_DIR, "labelmap.txt")
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, "detections_log.csv")

# Pre-trained SSD MobileNet V1 (COCO) TFLite model, hosted by TensorFlow team
MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"

CONFIDENCE_THRESHOLD = 0.50   # ignore detections below 50% confidence
CAMERA_INDEX = 0              # change if you have multiple webcams
INPUT_SIZE = 300               # SSD MobileNet V1 expects 300x300 input

# Colors (BGR) for overlay
COLOR_BOX = (0, 255, 140)
COLOR_TEXT_BG = (20, 20, 20)
COLOR_TEXT = (255, 255, 255)
COLOR_HUD_BG = (15, 15, 15)
COLOR_ACCENT = (0, 212, 170)   # teal accent matching the dashboard theme

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# =====================================================================
# 2. MODEL LOADER
# =====================================================================
def ensure_model_files():
    """Download + extract the TFLite model and labelmap if not already present."""
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELMAP_PATH):
        return

    print("[SETUP] Model files not found locally. Downloading pre-trained model...")
    import zipfile
    import tempfile

    zip_path = os.path.join(tempfile.gettempdir(), "ssd_mobilenet.zip")
    urllib.request.urlretrieve(MODEL_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(MODEL_DIR)

    # The zip contains 'detect.tflite' and 'labelmap.txt' directly
    print("[SETUP] Model ready.")


def load_labels(path):
    with open(path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    # First entry in this labelmap is often '???' (background) - keep as-is,
    # we index directly using the class id returned by the model.
    return labels


def load_interpreter():
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


# =====================================================================
# 3. DETECTOR
# =====================================================================
class Detector:
    """Wraps the TFLite interpreter: preprocess -> invoke -> decode results."""

    def __init__(self, interpreter, labels):
        self.interpreter = interpreter
        self.labels = labels
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        self.input_h = self.input_details[0]["shape"][1]
        self.input_w = self.input_details[0]["shape"][2]

    def detect(self, frame):
        """Run detection on a single BGR frame. Returns list of detections:
        [{box: (xmin,ymin,xmax,ymax), label: str, score: float}, ...]
        """
        h, w, _ = frame.shape

        # Preprocess: resize to model input size, convert to RGB, add batch dim
        img_resized = cv2.resize(frame, (self.input_w, self.input_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0)

        # Quantized model expects uint8 input directly
        if self.input_details[0]["dtype"] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]

        results = []
        for i in range(len(scores)):
            if scores[i] < CONFIDENCE_THRESHOLD:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)

            class_id = int(classes[i])
            label = self.labels[class_id + 1] if class_id + 1 < len(self.labels) else "unknown"

            results.append({
                "box": (xmin, ymin, xmax, ymax),
                "label": label,
                "score": float(scores[i]),
            })

        return results


# =====================================================================
# 4. ANALYTICS STATE
# =====================================================================
class AnalyticsState:
    """Tracks running counts and full detection history for the session."""

    def __init__(self):
        self.total_counts = Counter()      # label -> total times seen across all frames
        self.history = []                  # list of dicts: timestamp, label, score
        self.screenshots_taken = 0
        self.session_start = datetime.now()

    def record(self, detections):
        seen_this_frame = set()
        for det in detections:
            self.total_counts[det["label"]] += 1
            seen_this_frame.add(det["label"])
            self.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "label": det["label"],
                "confidence": round(det["score"] * 100, 1),
            })
        return seen_this_frame

    def current_frame_counts(self, detections):
        c = Counter(d["label"] for d in detections)
        return c


# =====================================================================
# 5. DRAW OVERLAY
# =====================================================================
def draw_detections(frame, detections):
    for det in detections:
        xmin, ymin, xmax, ymax = det["box"]
        label = det["label"]
        score_pct = int(det["score"] * 100)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), COLOR_BOX, 2)

        text = f"{label} {score_pct}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (xmin, ymin - th - 12), (xmin + tw + 8, ymin), COLOR_BOX, -1)
        cv2.putText(frame, text, (xmin + 4, ymin - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def draw_hud(frame, analytics, frame_counts, fps):
    """Top-left analytics HUD panel showing live stats."""
    h, w, _ = frame.shape
    panel_w = 260
    panel_h = 40 + 24 * (len(frame_counts) + 3) + 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), COLOR_HUD_BG, -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    y = 35
    cv2.putText(frame, "VISIONX SENTINEL", (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ACCENT, 2)
    y += 26
    cv2.putText(frame, f"FPS: {fps:.1f}", (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    y += 22
    cv2.putText(frame, f"Total detections: {sum(analytics.total_counts.values())}",
                (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    y += 24

    if frame_counts:
        cv2.putText(frame, "Live objects:", (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1)
        y += 22
        for label, count in frame_counts.most_common():
            cv2.putText(frame, f"  {label}: {count}", (22, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
            y += 20
    else:
        cv2.putText(frame, "No objects detected", (22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # Bottom hint bar
    cv2.putText(frame, "[S] Screenshot   [R] Report   [Q] Quit",
                (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


# =====================================================================
# 6. LOGGING
# =====================================================================
def write_csv_log(analytics):
    file_exists = os.path.exists(CSV_LOG_PATH)
    with open(CSV_LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "label", "confidence"])
        writer.writeheader()
        writer.writerows(analytics.history)
    print(f"[LOG] Detection history saved -> {CSV_LOG_PATH}")


# =====================================================================
# 7. SCREENSHOT
# =====================================================================
def save_screenshot(frame, analytics):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    path = os.path.join(SCREENSHOT_DIR, filename)
    cv2.imwrite(path, frame)
    analytics.screenshots_taken += 1
    print(f"[SCREENSHOT] Saved -> {path}")
    return path


# =====================================================================
# 8. REPORT GENERATOR
# =====================================================================
def generate_pdf_report(analytics, last_screenshot_path=None):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    report_path = os.path.join(OUTPUT_DIR, "session_report.pdf")
    doc = SimpleDocTemplate(report_path, pagesize=letter,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    accent = colors.HexColor("#00d4aa")

    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=accent)
    heading_style = ParagraphStyle("HeadX", parent=styles["Heading2"], textColor=colors.HexColor("#222222"))

    story = []
    story.append(Paragraph("VisionX TensorFlow Sentinel", title_style))
    story.append(Paragraph("Real-Time Object Detection & Analytics Report", styles["Normal"]))
    story.append(Spacer(1, 16))

    duration = datetime.now() - analytics.session_start
    minutes, seconds = divmod(int(duration.total_seconds()), 60)

    meta_table_data = [
        ["Session started", analytics.session_start.strftime("%Y-%m-%d %H:%M:%S")],
        ["Report generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Session duration", f"{minutes}m {seconds}s"],
        ["Total detections logged", str(sum(analytics.total_counts.values()))],
        ["Unique object types", str(len(analytics.total_counts))],
        ["Screenshots captured", str(analytics.screenshots_taken)],
    ]
    meta_table = Table(meta_table_data, colWidths=[200, 250])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Object Detection Summary", heading_style))
    story.append(Spacer(1, 8))

    if analytics.total_counts:
        summary_data = [["Object", "Times Detected", "% of Total"]]
        total = sum(analytics.total_counts.values())
        for label, count in analytics.total_counts.most_common():
            pct = f"{(count / total) * 100:.1f}%"
            summary_data.append([label.title(), str(count), pct])

        summary_table = Table(summary_data, colWidths=[200, 130, 120])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), accent),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(summary_table)
    else:
        story.append(Paragraph("No objects were detected during this session.", styles["Normal"]))

    story.append(Spacer(1, 20))

    # Recent detection timeline (last 15 entries)
    story.append(Paragraph("Recent Detection Timeline", heading_style))
    story.append(Spacer(1, 8))
    recent = analytics.history[-15:]
    if recent:
        timeline_data = [["Time", "Object", "Confidence"]]
        for entry in reversed(recent):
            timeline_data.append([entry["timestamp"], entry["label"].title(), f"{entry['confidence']}%"])
        timeline_table = Table(timeline_data, colWidths=[170, 180, 100])
        timeline_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222222")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ]))
        story.append(timeline_table)
    else:
        story.append(Paragraph("No timeline data available.", styles["Normal"]))

    # Embed a sample screenshot if one was taken
    if last_screenshot_path and os.path.exists(last_screenshot_path):
        story.append(Spacer(1, 20))
        story.append(Paragraph("Evidence Screenshot", heading_style))
        story.append(Spacer(1, 8))
        story.append(Image(last_screenshot_path, width=4.5 * inch, height=3.0 * inch))

    doc.build(story)
    print(f"[REPORT] PDF report generated -> {report_path}")
    return report_path


# =====================================================================
# 9. MAIN LOOP
# =====================================================================
def main():
    print("=" * 60)
    print(" VisionX TensorFlow Sentinel - Starting up")
    print("=" * 60)

    ensure_model_files()
    labels = load_labels(LABELMAP_PATH)
    interpreter = load_interpreter()
    detector = Detector(interpreter, labels)
    analytics = AnalyticsState()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not access webcam. Check CAMERA_INDEX in config.")
        return

    print("[READY] Webcam feed active. Press Q to quit, S to screenshot, R for report.")

    last_screenshot_path = None
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from webcam.")
                break

            detections = detector.detect(frame)
            analytics.record(detections)
            frame_counts = analytics.current_frame_counts(detections)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now

            draw_detections(frame, detections)
            draw_hud(frame, analytics, frame_counts, fps)

            cv2.imshow("VisionX TensorFlow Sentinel", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                last_screenshot_path = save_screenshot(frame, analytics)
            elif key == ord("r"):
                generate_pdf_report(analytics, last_screenshot_path)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        write_csv_log(analytics)
        print("[EXIT] Session ended. All logs saved to /output.")


if __name__ == "__main__":
    main()
