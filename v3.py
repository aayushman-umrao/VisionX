"""
VisionX TensorFlow Sentinel — PRO MAX EDITION
================================================
Real-time AI-powered object detection, tracking & analytics system.

New in this version (on top of the original Sentinel):
    - Persistent object tracking with IDs (Person #1, Person #2, ...)
    - Restricted Zone alert system (draw a zone, get visual + logged alerts
      when any object enters it)
    - Detection smoothing (kills single-frame flicker / false positives)
    - Live trend mini-graph in the HUD (object count over the last 30s)
    - New-object-type alert flash (first time an object type appears)
    - Per-object dwell time tracking (how long each tracked object has
      been visible)

Uses the same TensorFlow Lite model + OpenCV pipeline as the original —
no new model download, no new dependencies. Everything below is purely
additive logic on top of detection.

Controls (while the webcam window is focused):
    S  -> Save a screenshot of the current frame
    R  -> Generate a PDF analytics report instantly
    Z  -> Toggle restricted-zone drawing mode (click-drag with mouse,
          then press Z again to lock it in)
    C  -> Clear the current restricted zone
    Q  -> Quit (auto-saves the CSV detection log + alert log)

Author: Aayushman
"""

import os
import csv
import time
import math
import urllib.request
from datetime import datetime
from collections import Counter, deque

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

MODEL_PATH = os.path.join(MODEL_DIR, "efficientdet_lite0.tflite")
LABELMAP_PATH = os.path.join(MODEL_DIR, "labelmap.txt")
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, "detections_log.csv")
ALERT_LOG_PATH = os.path.join(OUTPUT_DIR, "alerts_log.csv")

# EfficientDet-Lite0: a genuine accuracy upgrade over SSD MobileNet V1
# (~25.7% mAP vs ~21% mAP on COCO 2017 validation) while staying in the
# same "lightweight enough for an old CPU laptop" class. Trained on the
# same COCO 80-class vocabulary, so it does NOT add brand-new object
# categories (a pen still isn't a COCO class) — it's more accurate at
# recognizing the same set of objects, with fewer flickery/missed
# detections, especially on small/cluttered scenes.
MODEL_URL = "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1?lite-format=tflite"

# Fallback: the original SSD MobileNet V1 model, used automatically if
# the EfficientDet download fails (e.g. no internet) or the model file
# turns out to be corrupted/incompatible on a particular machine.
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "detect_fallback.tflite")
FALLBACK_MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"

CONFIDENCE_THRESHOLD = 0.50    # default — ignore raw detections below 50% confidence

# Some objects are small / low-texture and the lightweight model is
# naturally less confident about them even when it's correct (mouse,
# book, remote, keyboard, cell phone are common examples). Rather than
# lowering the threshold globally (which lets in more false positives
# for everything), we lower it only for these specific labels.
PER_CLASS_THRESHOLDS = {
    "mouse": 0.30,
    "book": 0.32,
    "remote": 0.35,
    "keyboard": 0.35,
    "cell phone": 0.38,
}

CAMERA_INDEX = 0               # change if you have multiple webcams

# --- Tracking tuning ---
MAX_TRACK_DISTANCE = 80         # px — max centroid movement to still count as "same object"
MAX_FRAMES_MISSING = 15         # frames a tracked object can vanish before its ID is dropped
SMOOTHING_MIN_HITS = 2          # a track must be seen this many times before it's "confirmed"

# --- Trend graph ---
TREND_WINDOW_SECONDS = 30
TREND_SAMPLE_INTERVAL = 1.0     # sample total live count once per second

# Colors (BGR) for overlay
COLOR_BOX = (0, 255, 140)
COLOR_TEXT_BG = (20, 20, 20)
COLOR_TEXT = (255, 255, 255)
COLOR_HUD_BG = (15, 15, 15)
COLOR_ACCENT = (0, 212, 170)    # teal accent matching the dashboard theme
COLOR_ALERT = (40, 40, 255)     # red (BGR) for zone breach alerts
COLOR_ZONE = (0, 165, 255)      # orange for the restricted zone outline
COLOR_NEW_OBJECT = (255, 220, 0)  # cyan-ish flash for first-seen object type

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# =====================================================================
# 2. MODEL LOADER
# =====================================================================
# EfficientDet-Lite0 ships with its labels embedded in the model file's
# metadata (no separate labelmap.txt needed). However, to keep this
# script self-contained and to support the fallback model (which DOES
# need a separate labelmap.txt), we still maintain our own labelmap
# file locally — extracted from the fallback model's zip, which uses
# the same standard 90-line COCO labelmap format either way.
def ensure_model_files():
    """Download the primary (EfficientDet-Lite0) model. Falls back to the
    original SSD MobileNet V1 model if the download fails for any reason
    (no internet, blocked URL, etc.) so the script never hard-crashes on
    setup.
    """
    if os.path.exists(LABELMAP_PATH) and (os.path.exists(MODEL_PATH) or os.path.exists(FALLBACK_MODEL_PATH)):
        return  # already set up from a previous run

    # Always make sure we have a labelmap, since the fallback model needs
    # one and it's the simplest way to label EfficientDet's output too.
    if not os.path.exists(LABELMAP_PATH):
        print("[SETUP] Fetching standard COCO labelmap...")
        _ensure_fallback_model()  # this also extracts the labelmap

    if not os.path.exists(MODEL_PATH):
        print("[SETUP] Downloading EfficientDet-Lite0 (better accuracy model)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            # Sanity check: a real model file should be at least ~1 MB.
            # If we got an error page or redirect stub instead, the file
            # will be tiny — catch that here rather than crashing later
            # deep inside the TFLite interpreter with a confusing error.
            if os.path.getsize(MODEL_PATH) < 500_000:
                print("[SETUP] Warning: downloaded file looks too small to be a real model.")
                os.remove(MODEL_PATH)
                raise RuntimeError("Downloaded EfficientDet file is suspiciously small.")
            print("[SETUP] EfficientDet-Lite0 ready.")
        except Exception as e:
            print(f"[SETUP] Could not download EfficientDet-Lite0 ({e}).")
            print("[SETUP] Falling back to the original SSD MobileNet V1 model.")
            _ensure_fallback_model()


def _ensure_fallback_model():
    """Downloads the original SSD MobileNet V1 model + labelmap (used as
    a safety net if EfficientDet-Lite0 can't be downloaded or fails to load).
    """
    if os.path.exists(FALLBACK_MODEL_PATH) and os.path.exists(LABELMAP_PATH):
        return

    import zipfile
    import tempfile

    zip_path = os.path.join(tempfile.gettempdir(), "ssd_mobilenet.zip")
    urllib.request.urlretrieve(FALLBACK_MODEL_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(MODEL_DIR)

    # The zip extracts as 'detect.tflite' + 'labelmap.txt'; rename the
    # model file so it doesn't collide with the EfficientDet filename.
    extracted_model = os.path.join(MODEL_DIR, "detect.tflite")
    if os.path.exists(extracted_model) and not os.path.exists(FALLBACK_MODEL_PATH):
        os.rename(extracted_model, FALLBACK_MODEL_PATH)


def get_active_model_path():
    """Returns whichever model file actually exists and should be used:
    EfficientDet-Lite0 if present and valid, otherwise the fallback.
    """
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 500_000:
        return MODEL_PATH, "EfficientDet-Lite0"
    if os.path.exists(FALLBACK_MODEL_PATH):
        return FALLBACK_MODEL_PATH, "SSD MobileNet V1 (fallback)"
    raise FileNotFoundError("No usable model file found. Check your internet connection and try again.")


def load_labels(path):
    with open(path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def load_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# =====================================================================
# 3. DETECTOR
# =====================================================================
class Detector:
    """Wraps the TFLite interpreter: preprocess -> invoke -> decode results.

    Auto-detects which output convention the loaded model uses:
      - "raw" style (SSD MobileNet V1 / most older TFLite detection models):
        4 separate output tensors accessed via get_tensor(index), typically
        ordered [boxes, classes, scores, count].
      - "signature" style (EfficientDet-Lite metadata models): a named
        signature function returning a dict with keys like output_0..3
        (count, scores, classes, boxes — note the different order).

    This avoids hard-coding one model's exact tensor order, since that's
    the single most common cause of "confident boxes, wrong labels" bugs
    when swapping models.
    """

    def __init__(self, interpreter, labels, model_name="model"):
        self.interpreter = interpreter
        self.labels = labels
        self.model_name = model_name
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        self.input_h = self.input_details[0]["shape"][1]
        self.input_w = self.input_details[0]["shape"][2]

        self.signature_fn = None
        self.output_mode = "raw"

        # Try to use a named signature first (this is how EfficientDet-Lite
        # metadata models are documented to work). If the model has no
        # signatures defined, fall back to raw indexed tensor access.
        try:
            signatures = interpreter.get_signature_list()
            if signatures:
                self.signature_fn = interpreter.get_signature_runner()
                self.output_mode = "signature"
                print(f"[MODEL] '{model_name}' loaded using signature-runner output mode.")
            else:
                print(f"[MODEL] '{model_name}' loaded using raw tensor output mode.")
        except Exception:
            print(f"[MODEL] '{model_name}' loaded using raw tensor output mode (no signature support).")

    def detect(self, frame):
        """Run detection on a single BGR frame. Returns list of detections:
        [{box: (xmin,ymin,xmax,ymax), label: str, score: float}, ...]
        """
        h, w, _ = frame.shape

        img_resized = cv2.resize(frame, (self.input_w, self.input_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0)

        if self.input_details[0]["dtype"] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            input_data = input_data.astype(self.input_details[0]["dtype"])

        if self.output_mode == "signature":
            boxes, classes, scores = self._detect_signature(input_data)
        else:
            boxes, classes, scores = self._detect_raw(input_data)

        results = []
        for i in range(len(scores)):
            score = float(scores[i])

            class_id = int(classes[i])
            label = self.labels[class_id + 1] if 0 <= class_id + 1 < len(self.labels) else "unknown"

            threshold = PER_CLASS_THRESHOLDS.get(label, CONFIDENCE_THRESHOLD)
            if score < threshold:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)

            # Clip to frame bounds defensively (EfficientDet can sometimes
            # return slightly out-of-range coordinates near edges)
            xmin, xmax = max(0, xmin), min(w, xmax)
            ymin, ymax = max(0, ymin), min(h, ymax)

            results.append({
                "box": (xmin, ymin, xmax, ymax),
                "label": label,
                "score": score,
            })

        return results

    def _detect_raw(self, input_data):
        """SSD-MobileNet-style raw indexed tensor output."""
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        return boxes, classes, scores

    def _detect_signature(self, input_data):
        """EfficientDet-Lite-style signature-runner output (documented as
        output_0=count, output_1=scores, output_2=classes, output_3=boxes).
        We look up keys defensively by name rather than assuming order,
        since the exact key names can vary slightly between exports.
        """
        output = self.signature_fn(images=input_data)

        # Try documented key names first; fall back to positional guess
        # if the model uses different key names than expected.
        scores = output.get("output_1")
        classes = output.get("output_2")
        boxes = output.get("output_3")

        if scores is None or classes is None or boxes is None:
            # Defensive fallback: sort available outputs by their shape to
            # guess which is which (boxes has a trailing dim of 4).
            for key, val in output.items():
                val = np.squeeze(val)
                if val.ndim == 2 and val.shape[-1] == 4:
                    boxes = val
                elif boxes is None or val is not boxes:
                    if scores is None:
                        scores = val
                    elif classes is None:
                        classes = val

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes)
        scores = np.squeeze(scores)
        return boxes, classes, scores


# =====================================================================
# 4. TRACKER  (lightweight centroid tracker, no extra dependencies)
# =====================================================================
class Track:
    """A single tracked object across frames."""
    _next_id = 1

    def __init__(self, label, box, score):
        self.id = Track._next_id
        Track._next_id += 1
        self.label = label
        self.box = box
        self.score = score
        self.centroid = self._compute_centroid(box)
        self.hits = 1                 # how many frames this track has been seen
        self.frames_missing = 0       # how many consecutive frames it's been missing
        self.first_seen = time.time()
        self.confirmed = False
        self.alerted_zone = False     # prevents spamming zone alerts every frame

    @staticmethod
    def _compute_centroid(box):
        xmin, ymin, xmax, ymax = box
        return ((xmin + xmax) // 2, (ymin + ymax) // 2)

    def update(self, box, score):
        self.box = box
        self.score = score
        self.centroid = self._compute_centroid(box)
        self.hits += 1
        self.frames_missing = 0
        if self.hits >= SMOOTHING_MIN_HITS:
            self.confirmed = True

    def mark_missing(self):
        self.frames_missing += 1

    def dwell_seconds(self):
        return time.time() - self.first_seen


class CentroidTracker:
    """Matches new detections to existing tracks by nearest centroid distance.
    Simple, dependency-free (no scipy) — good enough for a handful of objects
    in a classroom demo setting.
    """

    def __init__(self):
        self.tracks = []  # list of Track

    def update(self, detections):
        # Mark all existing tracks as "missing" by default; we'll un-mark
        # the ones that get matched below.
        unmatched_tracks = list(self.tracks)
        unmatched_detections = list(detections)
        matched_pairs = []

        # Greedy nearest-centroid matching, restricted to same label
        for track in list(unmatched_tracks):
            best_match = None
            best_dist = MAX_TRACK_DISTANCE + 1

            for det in unmatched_detections:
                if det["label"] != track.label:
                    continue
                cx, cy = Track._compute_centroid(det["box"])
                tx, ty = track.centroid
                dist = math.hypot(cx - tx, cy - ty)
                if dist < best_dist:
                    best_dist = dist
                    best_match = det

            if best_match is not None and best_dist <= MAX_TRACK_DISTANCE:
                matched_pairs.append((track, best_match))
                unmatched_detections.remove(best_match)
                unmatched_tracks.remove(track)

        # Update matched tracks
        for track, det in matched_pairs:
            track.update(det["box"], det["score"])

        # Mark unmatched (vanished) tracks as missing, drop if too stale
        for track in unmatched_tracks:
            track.mark_missing()
        self.tracks = [t for t in self.tracks if t.frames_missing <= MAX_FRAMES_MISSING]

        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            self.tracks.append(Track(det["label"], det["box"], det["score"]))

        # Return only currently-visible, confirmed tracks for drawing/logging
        visible = [t for t in self.tracks if t.frames_missing == 0]
        return visible


# =====================================================================
# 5. RESTRICTED ZONE
# =====================================================================
class RestrictedZone:
    """A user-drawn rectangular zone. Triggers an alert when a tracked
    object's centroid enters it.
    """

    def __init__(self):
        self.rect = None          # (x1, y1, x2, y2) once locked in
        self.drawing_mode = False
        self.drag_start = None
        self.drag_current = None

    def mouse_callback(self, event, x, y, flags, param):
        if not self.drawing_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            self.drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            self.drag_current = (x, y)
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            self.rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def contains(self, point):
        if self.rect is None:
            return False
        x, y = point
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def clear(self):
        self.rect = None
        self.drag_start = None
        self.drag_current = None
        self.drawing_mode = False


# =====================================================================
# 6. ANALYTICS STATE
# =====================================================================
class AnalyticsState:
    """Tracks running counts, full detection history, alerts, and trend data."""

    def __init__(self):
        self.total_counts = Counter()
        self.history = []
        self.alerts = []
        self.screenshots_taken = 0
        self.session_start = datetime.now()
        self.known_labels = set()

        # Trend graph: rolling (timestamp, count) samples
        self.trend_samples = deque(maxlen=TREND_WINDOW_SECONDS)
        self._last_trend_sample_time = 0.0

        self.new_object_flash_until = 0.0
        self.new_object_flash_label = ""

    def record_tracks(self, tracks):
        """Logs each confirmed, visible track once per frame it's seen."""
        for t in tracks:
            if not t.confirmed:
                continue
            self.total_counts[t.label] += 1
            self.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "label": t.label,
                "confidence": round(t.score * 100, 1),
                "track_id": t.id,
            })

            if t.label not in self.known_labels:
                self.known_labels.add(t.label)
                self.new_object_flash_until = time.time() + 2.0
                self.new_object_flash_label = t.label

    def record_alert(self, label, track_id):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label": label,
            "track_id": track_id,
            "message": f"{label} (ID {track_id}) entered restricted zone",
        }
        self.alerts.append(entry)
        print(f"[ALERT] {entry['message']}")

    def sample_trend(self, current_live_count):
        now = time.time()
        if now - self._last_trend_sample_time >= TREND_SAMPLE_INTERVAL:
            self.trend_samples.append(current_live_count)
            self._last_trend_sample_time = now

    def current_frame_counts(self, tracks):
        return Counter(t.label for t in tracks if t.confirmed)


# =====================================================================
# 7. DRAW OVERLAY
# =====================================================================
def draw_tracks(frame, tracks, zone):
    for t in tracks:
        if not t.confirmed:
            continue  # don't draw flickery unconfirmed detections

        xmin, ymin, xmax, ymax = t.box
        in_zone = zone.contains(t.centroid)
        box_color = COLOR_ALERT if in_zone else COLOR_BOX

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.circle(frame, t.centroid, 3, box_color, -1)

        score_pct = int(t.score * 100)
        dwell = int(t.dwell_seconds())
        text = f"{t.label} #{t.id}  {score_pct}%  ({dwell}s)"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (xmin, ymin - th - 12), (xmin + tw + 8, ymin), box_color, -1)
        cv2.putText(frame, text, (xmin + 4, ymin - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if in_zone:
            cv2.putText(frame, "!! ZONE BREACH !!", (xmin, ymax + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ALERT, 2)


def draw_zone(frame, zone):
    if zone.drawing_mode and zone.drag_start and zone.drag_current:
        cv2.rectangle(frame, zone.drag_start, zone.drag_current, COLOR_ZONE, 2)
        cv2.putText(frame, "Drawing zone... release to set, press Z to lock",
                    (zone.drag_start[0], max(20, zone.drag_start[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ZONE, 1)
    elif zone.rect:
        x1, y1, x2, y2 = zone.rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ZONE, 2)
        cv2.putText(frame, "RESTRICTED ZONE", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ZONE, 2)


def draw_trend_graph(frame, analytics, origin_x, origin_y, width=220, height=60):
    """Small live line-chart of total visible object count over the trend window."""
    cv2.rectangle(frame, (origin_x, origin_y), (origin_x + width, origin_y + height),
                  (35, 35, 35), -1)
    cv2.putText(frame, "Live Count Trend", (origin_x + 6, origin_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_ACCENT, 1)

    samples = list(analytics.trend_samples)
    if len(samples) < 2:
        return

    max_val = max(max(samples), 1)
    n = len(samples)
    step_x = width / max(n - 1, 1)

    points = []
    for i, val in enumerate(samples):
        x = int(origin_x + i * step_x)
        y = int(origin_y + height - (val / max_val) * (height - 10) - 5)
        points.append((x, y))

    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], COLOR_ACCENT, 2)

    # Current value label
    cv2.putText(frame, str(samples[-1]), (origin_x + width - 24, origin_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1)


def draw_hud(frame, analytics, frame_counts, fps, zone, alert_active):
    h, w, _ = frame.shape
    panel_w = 260
    panel_h = 40 + 24 * (len(frame_counts) + 3) + 20
    trend_h = 60
    trend_gap = 14

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), COLOR_HUD_BG, -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    y = 35
    title_color = COLOR_ALERT if alert_active else COLOR_ACCENT
    title_text = "VISIONX SENTINEL [ALERT]" if alert_active else "VISIONX SENTINEL PRO"
    cv2.putText(frame, title_text, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, title_color, 2)
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

    # Trend graph, placed just under the main panel (uses panel_h so it
    # never overlaps the live-objects list above it, regardless of how
    # many object types are currently listed)
    draw_trend_graph(frame, analytics, origin_x=10, origin_y=10 + panel_h + trend_gap + 18,
                      width=panel_w, height=trend_h)

    # New-object-type flash banner (top-right, clear of the left HUD panel)
    if time.time() < analytics.new_object_flash_until:
        msg = f"NEW: {analytics.new_object_flash_label.upper()}"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bx = w - tw - 30
        if bx > 10 + panel_w + 10:  # only draw if it won't collide with the HUD panel
            cv2.rectangle(frame, (bx - 12, 10), (bx + tw + 12, 42), COLOR_NEW_OBJECT, -1)
            cv2.putText(frame, msg, (bx, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Bottom hint bar
    cv2.putText(frame, "[S] Screenshot  [R] Report  [Z] Draw Zone  [C] Clear Zone  [Q] Quit",
                (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)


# =====================================================================
# 8. LOGGING
# =====================================================================
def write_csv_log(analytics):
    with open(CSV_LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "label", "confidence", "track_id"])
        writer.writeheader()
        writer.writerows(analytics.history)
    print(f"[LOG] Detection history saved -> {CSV_LOG_PATH}")

    with open(ALERT_LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "label", "track_id", "message"])
        writer.writeheader()
        writer.writerows(analytics.alerts)
    print(f"[LOG] Alert log saved -> {ALERT_LOG_PATH}")


# =====================================================================
# 9. SCREENSHOT
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
# 10. REPORT GENERATOR
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
    alert_color = colors.HexColor("#e02424")

    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=accent)
    heading_style = ParagraphStyle("HeadX", parent=styles["Heading2"], textColor=colors.HexColor("#222222"))

    story = []
    story.append(Paragraph("VisionX TensorFlow Sentinel — Pro Max", title_style))
    story.append(Paragraph("Real-Time Object Detection, Tracking & Security Report", styles["Normal"]))
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
        ["Restricted-zone alerts", str(len(analytics.alerts))],
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

    # Restricted zone alerts section
    story.append(Paragraph("Restricted Zone Alerts", heading_style))
    story.append(Spacer(1, 8))
    if analytics.alerts:
        alert_data = [["Time", "Object", "Track ID", "Event"]]
        for entry in analytics.alerts[-15:]:
            alert_data.append([entry["timestamp"], entry["label"].title(),
                                str(entry["track_id"]), "Zone Breach"])
        alert_table = Table(alert_data, colWidths=[150, 130, 70, 100])
        alert_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), alert_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fdecea")]),
        ]))
        story.append(alert_table)
    else:
        story.append(Paragraph("No restricted-zone alerts were triggered.", styles["Normal"]))

    story.append(Spacer(1, 20))

    # Recent detection timeline (last 15 entries)
    story.append(Paragraph("Recent Detection Timeline", heading_style))
    story.append(Spacer(1, 8))
    recent = analytics.history[-15:]
    if recent:
        timeline_data = [["Time", "Object", "Track ID", "Confidence"]]
        for entry in reversed(recent):
            timeline_data.append([entry["timestamp"], entry["label"].title(),
                                   str(entry.get("track_id", "-")), f"{entry['confidence']}%"])
        timeline_table = Table(timeline_data, colWidths=[150, 150, 70, 100])
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

    if last_screenshot_path and os.path.exists(last_screenshot_path):
        story.append(Spacer(1, 20))
        story.append(Paragraph("Evidence Screenshot", heading_style))
        story.append(Spacer(1, 8))
        story.append(Image(last_screenshot_path, width=4.5 * inch, height=3.0 * inch))

    doc.build(story)
    print(f"[REPORT] PDF report generated -> {report_path}")
    return report_path


# =====================================================================
# 11. MAIN LOOP
# =====================================================================
def main():
    print("=" * 60)
    print(" VisionX TensorFlow Sentinel PRO MAX - Starting up")
    print("=" * 60)

    ensure_model_files()
    active_model_path, model_name = get_active_model_path()
    labels = load_labels(LABELMAP_PATH)
    interpreter = load_interpreter(active_model_path)
    detector = Detector(interpreter, labels, model_name=model_name)
    print(f"[MODEL] Using: {model_name}")
    analytics = AnalyticsState()
    tracker = CentroidTracker()
    zone = RestrictedZone()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not access webcam. Check CAMERA_INDEX in config.")
        return

    window_name = "VisionX TensorFlow Sentinel - PRO MAX"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, zone.mouse_callback)

    print("[READY] Webcam feed active.")
    print("        S = Screenshot | R = Report | Z = Draw Zone | C = Clear Zone | Q = Quit")

    last_screenshot_path = None
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from webcam.")
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            analytics.record_tracks(tracks)
            frame_counts = analytics.current_frame_counts(tracks)
            analytics.sample_trend(sum(frame_counts.values()))

            # Zone breach detection (debounced per-track so it doesn't spam)
            alert_active = False
            for t in tracks:
                if not t.confirmed:
                    continue
                if zone.contains(t.centroid):
                    alert_active = True
                    if not t.alerted_zone:
                        t.alerted_zone = True
                        analytics.record_alert(t.label, t.id)
                else:
                    t.alerted_zone = False

            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now

            draw_tracks(frame, tracks, zone)
            draw_zone(frame, zone)
            draw_hud(frame, analytics, frame_counts, fps, zone, alert_active)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                last_screenshot_path = save_screenshot(frame, analytics)
            elif key == ord("r"):
                generate_pdf_report(analytics, last_screenshot_path)
            elif key == ord("z"):
                zone.drawing_mode = not zone.drawing_mode
                state = "ON - click and drag to draw" if zone.drawing_mode else "locked"
                print(f"[ZONE] Drawing mode {state}")
            elif key == ord("c"):
                zone.clear()
                print("[ZONE] Cleared.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        write_csv_log(analytics)
        print("[EXIT] Session ended. All logs saved to /output.")


if __name__ == "__main__":
    main()