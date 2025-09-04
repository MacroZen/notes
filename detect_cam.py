#!/usr/bin/env python3
import time
import numpy as np
import cv2
from pathlib import Path

# Prefer the lightweight tflite-runtime on Raspberry Pi
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    # Fallback if you actually have full TensorFlow installed
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            lab = line.strip()
            if lab: labels.append(lab)
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ssd_mobilenet_v1/detect.tflite", help="path to TFLite model")
    ap.add_argument("--labels", default="ssd_mobilenet_v1/labelmap.txt", help="path to labels file")
    ap.add_argument("--cam", type=str, default="0", help="camera index (e.g., 0) or device path (/dev/video0)")
    ap.add_argument("--score", type=float, default=0.5, help="score threshold")
    ap.add_argument("--size", type=int, default=320, help="inference size (e.g., 320)")
    args = ap.parse_args()

    model_path = Path(args.model)
    labels_path = Path(args.labels)
    assert model_path.exists(), f"Model not found: {model_path}"
    assert labels_path.exists(), f"Labels not found: {labels_path}"

    labels = load_labels(labels_path)

    # Load TFLite model
    interpreter = Interpreter(model_path=str(MODEL))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()

    # Determine camera source (index or /dev/video path)
    try:
        cam_index = int(args.cam)
        source = cam_index
    except ValueError:
        source = args.cam  # e.g., "/dev/video0"

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera: {args.cam}")

    # UVC cams on Pi often like MJPG at a set size
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    win = "MobileNet-SSD (ESC to quit)"
    fps_t0, frames = time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARN: frame read failed")
            break

        # Prepare input
        ih, iw = args.size, args.size
        img = cv2.resize(frame, (iw, ih))
        # Model is uint8 quantized
        inp = np.expand_dims(img, 0).astype(np.uint8)
        interpreter.set_tensor(in_det['index'], inp)

        t0 = time.time()
        interpreter.invoke()
        inf_ms = (time.time() - t0) * 1000.0

        # Typical TFLite SSD outputs: boxes, classes, scores, count
        boxes  = interpreter.get_tensor(out_det[0]['index'])[0]
        clzs   = interpreter.get_tensor(out_det[1]['index'])[0].astype(int)
        scores = interpreter.get_tensor(out_det[2]['index'])[0]
        count  = int(interpreter.get_tensor(out_det[3]['index'])[0])

        H, W = frame.shape[:2]
        for i in range(count):
            s = float(scores[i])
            if s < args.score: 
                continue
            y1, x1, y2, x2 = boxes[i]
            x1i, y1i = int(x1 * W), int(y1 * H)
            x2i, y2i = int(x2 * W), int(y2 * H)
            cls = clzs[i]
            name = labels[cls] if 0 <= cls < len(labels) else str(cls)
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {s:.2f}", (x1i, max(0, y1i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                        cv2.LINE_AA)

        # FPS overlay
        frames += 1
        if frames >= 10:
            fps = frames / (time.time() - fps_t0)
            fps_t0, frames = time.time(), 0
        else:
            fps = None

        info = f"Infer: {inf_ms:.1f} ms"
        if fps is not None:
            info += f" | FPS: {fps:.1f}"
        cv2.putText(frame, info, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 220, 30), 2, cv2.LINE_AA)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
