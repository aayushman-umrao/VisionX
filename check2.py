"""
VisionX Debug Tool — Raw Model Diagnosis
==========================================
This is a TEMPORARY diagnostic script, not part of your final project.
It shows the model's TOP 3 raw guesses (with exact confidence %) for
every detection box, printed live in your terminal.

This tells us whether the model is:
 (a) genuinely confused between similar-looking categories (a real,
     expected limitation), or
 (b) doing something structurally wrong (a real bug)

Run this exactly like the main script:
    python debug_top3.py

Press Q in the video window to quit.
"""

import os
import urllib.request
import cv2
import numpy as np

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "detect.tflite")
LABELMAP_PATH = os.path.join(BASE_DIR, "models", "labelmap.txt")


def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def main():
    labels = load_labels(LABELMAP_PATH)
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_h = input_details[0]["shape"][1]
    input_w = input_details[0]["shape"][2]

    print("Output tensor order check:")
    for i, d in enumerate(output_details):
        print(f"  output[{i}]: shape={d['shape']}, dtype={d['dtype']}, name={d.get('name', '?')}")
    print()
    print("If output[0] is NOT boxes (4 values per detection), or output[2]")
    print("is NOT scores (values between 0 and 1), that's the bug.")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[READY] Point camera at an object. Press Q to quit.")
    print("Printing top-3 raw guesses for the highest-confidence detection each frame.")
    print("-" * 70)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        h, w, _ = frame.shape
        img_resized = cv2.resize(frame, (input_w, input_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0)

        if input_details[0]["dtype"] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            input_data = input_data.astype(np.uint8)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]["index"])[0]
        classes = interpreter.get_tensor(output_details[1]["index"])[0]
        scores = interpreter.get_tensor(output_details[2]["index"])[0]

        # Only print every 15 frames so the terminal isn't flooded
        if frame_count % 15 == 0:
            # Sort detection slots by score, show top 3
            order = np.argsort(scores)[::-1][:3]
            print(f"\n[Frame {frame_count}] Top 3 raw detections:")
            for idx in order:
                class_id = int(classes[idx])
                label = labels[class_id + 1] if class_id + 1 < len(labels) else f"UNKNOWN(id={class_id})"
                print(f"   class_id={class_id:3d}  label='{label}'  score={scores[idx]*100:5.1f}%  box={boxes[idx]}")

        # Draw all detections above 25% for visual reference
        for i in range(len(scores)):
            if scores[i] < 0.25:
                continue
            class_id = int(classes[i])
            label = labels[class_id + 1] if class_id + 1 < len(labels) else "?"
            ymin, xmin, ymax, xmax = boxes[i]
            xmin, xmax = int(xmin * w), int(xmax * w)
            ymin, ymax = int(ymin * h), int(ymax * h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {int(scores[i]*100)}%", (xmin, max(15, ymin - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("VisionX DEBUG - Top3 Raw Predictions", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()