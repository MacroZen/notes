#!/usr/bin/env python3
"""
detect_cam.py — simple, robust TFLite live detection
#########################################################################################################
##########################################                    v1.1.0                    ##########################################
#########################################################################################################
Features
- Works with tflite_runtime OR tensorflow.lite (auto-fallback)
- Optional EdgeTPU delegate if available (--edgetpu)
- Looks for model in multiple locations, helpful error messages
- Built-in COCO-80 labels if you don't provide a labels file
- Camera listing, resolution control, FPS overlay
- Save screenshots (press 's') or record video (--record out.mp4)

Keys in window:
  q: quit
  s: save a frame as PNG in ./frames/
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

# --------------------------
# Label handling (built-in COCO-80)
# --------------------------
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

def load_labels(labels_path: Optional[Path]) -> Tuple[list, bool]:
    """
    Returns (labels, has_background_zero)
    has_background_zero=True if label[0] is a background placeholder like '???'
    """
    if labels_path and labels_path.is_file():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [ln.strip() for ln in f if ln.strip()]
        # Heuristic: some labelmaps start with '???'
        has_bg = bool(labels and labels[0] in ("???", "background", "__background__"))
        return labels, has_bg
    # Fallback to built-in COCO-80 (no explicit background)
    return COCO80, False

# --------------------------
# TFLite interpreter loader
# --------------------------
def try_import_interpreter(edgetpu: bool):
    """
    Try tflite_runtime first, then tensorflow as fallback.
    Returns (InterpreterClass, delegates_list)
    """
    delegates = []
    # Try tflite_runtime
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        if edgetpu:
            try:
                delegates = [load_delegate("libedgetpu.so.1")]
                print("[info] EdgeTPU delegate loaded.")
            except Exception as e:
                print(f"[warn] Could not load EdgeTPU delegate: {e}")
        return Interpreter, delegates
    except Exception:
        pass

    # Fallback to full TensorFlow
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        # EdgeTPU with TF (rare), typically not supported this way
        return Interpreter, []
    except Exception as e:
        print("[error] Could not import TFLite interpreter from tflite_runtime or tensorflow.")
        print("Install one of:\n  pip3 install tflite-runtime\n  OR\n  pip3 install tensorflow")
        raise e

# --------------------------
# Model + label path resolution
# --------------------------
def resolve_path(user_path: Optional[str], default_name: str) -> Optional[Path]:
    """
    Return the first existing candidate path or None if not found.
    Search order:
      1) direct user_path
      2) CWD
      3) script_dir
      4) script_dir/models
      5) CWD/models
    """
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()

    candidates = []
    if user_path:
        p = Path(user_path)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(here / user_path)
            candidates.append(here / "models" / user_path)
            candidates.append(cwd / user_path)
            candidates.append(cwd / "models" / user_path)
    else:
        candidates.extend([
            cwd / default_name,
            cwd / "models" / default_name,
            here / default_name,
            here / "models" / default_name,
        ])

    for c in candidates:
        if c.is_file():
            return c
    return None

# --------------------------
# SSD-style postprocessing
# --------------------------
def extract_output(interpreter):
    """
    Return a dict with 'boxes','classes','scores','count' tensor indices (by best-effort name match).
    """
    details = interpreter.get_output_details()
    keys = {"boxes": None, "classes": None, "scores": None, "count": None}
    # Best-effort by name:
    for i, d in enumerate(details):
        name = d.get("name", "").lower()
        if "box" in name:
            keys["boxes"] = i
        elif "class" in name:
            keys["classes"] = i
        elif "score" in name:
            keys["scores"] = i
        elif "num" in name or "count" in name:
            keys["count"] = i
    # If not found by name, assume canonical order
    if any(v is None for v in keys.values()):
        if len(details) >= 4:
            keys = {"boxes": 0, "classes": 1, "scores": 2, "count": 3}
    return keys

def draw_detections(frame, detections, labels, has_bg_zero, threshold):
    h, w = frame.shape[:2]
    for y_min, x_min, y_max, x_max, cls_id, score in detections:
        if score < threshold:
            continue
        # Classes can be float; cast safely
        c = int(cls_id)
        # Adjust index if labels start at 1 or 0
        if has_bg_zero:
            # labels[0] is background, classes are typically 1-based
            label_idx = c
        else:
            # Many COCO-80 label files are 1..80, but some are 0..79.
            # If index equals len(labels), clamp; else map best-effort:
            label_idx = c
            if label_idx >= len(labels):
                label_idx = max(0, min(len(labels) - 1, c - 1))
        name = labels[label_idx] if 0 <= label_idx < len(labels) else f"id:{c}"

        # Convert normalized boxes -> pixels
        x1, y1 = int(x_min * w), int(y_min * h)
        x2, y2 = int(x_max * w), int(y_max * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        txt = f"{name} {score:.2f}"
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), (0, 180, 255), -1)
        cv2.putText(frame, txt, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)

# --------------------------
# Camera utilities
# --------------------------
def list_cameras(max_index=5):
    print("Scanning cameras...")
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                print(f"  [{i}] OK  {w}x{h}")
                found.append(i)
            else:
                print(f"  [{i}] Opened but no frame")
        cap.release()
    if not found:
        print("No cameras found. If you're on Raspberry Pi with libcamera, ensure V4L2 compatibility is enabled.")

def open_camera(_src: str, _width: int, _height: int, _fps: int, *args, **_kwargs):
    pipeline = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=1280, height=720, framerate=30/1 ! "
        "videoconvert ! appsink"
    )
    print["using hard coded line"]
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap

    #    idx = int(src)
    #    cap = cv2.VideoCapture(idx)
    #except ValueError:
    #    # Treat as path/URL
    #    cap = cv2.VideoCapture(src)

    #if width > 0 and height > 0:
    #    cap.set(cv2.CAP_PROP_FRAME_WIDTh, float(width))
    #    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    #if fps > 0:
    #    cap.set(cv2.CAP_PROP_FPS, float(fps))
    #return cap

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-m", "--model", type=str, default="ssd_mobilenet_detect.tflite",
                    help="Path to TFLite model (absolute or relative)")
    ap.add_argument("-l", "--labels", type=str, default="labelmap.txt",
                    help="Path to labels file (optional; built-in COCO-80 used if missing)")
    ap.add_argument("-s", "--source", type=str, default="0",
                    help="Camera index (e.g., '0') or video file/URL")
    ap.add_argument("--width", type=int, default=1280, help="Capture width")
    ap.add_argument("--height", type=int, default=720, help="Capture height")
    ap.add_argument("--cam-fps", type=int, default=30, help="Requested camera FPS")
    ap.add_argument("-t", "--threshold", type=float, default=0.5, help="Score threshold [0..1]")
    ap.add_argument("--edgetpu", action="store_true", help="Try to use EdgeTPU delegate (Coral USB)")
    ap.add_argument("--record", type=str, default="", help="Record output to video file (e.g., out.mp4)")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="Rotate display (deg)")
    ap.add_argument("--flip", type=str, default="", choices=["", "h", "v", "hv"], help="Flip image")
    ap.add_argument("--list-cams", action="store_true", help="List available cameras and exit")
    args = ap.parse_args()

    if args.list_cams:
        list_cameras(8)
        return

    # Resolve model/labels
    model_path = Path("~/vision/notes/ssd_mobilenet_v1/detect.tflite").expanduser()
    labels_path = Path("~/vision/notes/ssd_mobilenet_v1/labelmap.txt").expanduser()

    if not model_path:
        print("❌ Could not find a TFLite model.\n"
              "Searched (relative to CWD and script):\n"
              f"  - {args.model}\n  - ./models/{args.model}\n  - ./ssd_mobilenet_detect.tflite\n  - ./models/ssd_mobilenet_detect.tflite\n")
        print("👉 Fix it by:\n"
              "   1) Passing --model /full/path/to/your_model.tflite\n"
              "   2) Or placing your model at ./models/ssd_mobilenet_detect.tflite\n")
        sys.exit(1)

    labels, has_bg_zero = load_labels(labels_path)
    if labels_path and labels_path.is_file():
        print(f"[info] Using label file: {labels_path}")
    else:
        print("[info] No labels file found; using built-in COCO-80 labels.")

    # Interpreter
    Interpreter, delegates = try_import_interpreter(args.edgetpu)
    interpreter = Interpreter(model_path=str(model_path), experimental_delegates=delegates)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()
    ih, iw = in_details[0]["shape"][1], in_details[0]["shape"][2]
    in_dtype = in_details[0]["dtype"]
    in_quant = in_details[0].get("quantization", (0.0, 0))  # (scale, zero_point)

    print(f"[info] Model: {model_path.name} | input {iw}x{ih} dtype={in_dtype} quant={in_quant}")
    outs = extract_output(interpreter)

    # Camera
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    pipeline = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=1280, height=720, framerate=30/1 ! "
        "videoconvert ! appsink"
    )
    #cap = open_camera(args.source, args.width, args.height, args.cam_fps)
    #if not cap.isOpened():
    #    print(f"[error] Could not open video source: {args.source}")
    #    sys.exit(1)

    # Video writer (optional)
    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely supported
        writer = cv2.VideoWriter(args.record, fourcc, float(args.cam_fps if args.cam_fps > 0 else 30),
                                 (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if not writer.isOpened():
            print(f"[warn] Could not open writer for {args.record}; disabling recording.")
            writer = None
        else:
            print(f"[info] Recording to {args.record}")

    # Main loop
    last_t = time.time()
    fps = 0.0
    frame_count = 0
    Path("frames").mkdir(exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[warn] No frame from camera.")
            break

        # Optional rotation/flip for convenience
        if args.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif args.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if args.flip:
            if "h" in args.flip:
                frame = cv2.flip(frame, 1)  # horizontal
            if "v" in args.flip:
                frame = cv2.flip(frame, 0)  # vertical

        # Prepare input
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)

        if in_dtype == np.uint8:
            inp = np.asarray(inp, dtype=np.uint8)
        else:
            # Most float models expect [0,1]
            inp = (np.asarray(inp, dtype=np.float32) / 255.0)

        # Invoke
        interpreter.set_tensor(in_details[0]["index"], np.expand_dims(inp, 0))
        interpreter.invoke()

        # Fetch outputs
        boxes = interpreter.get_tensor(out_details[outs["boxes"]]["index"])[0]
        classes = interpreter.get_tensor(out_details[outs["classes"]]["index"])[0]
        scores = interpreter.get_tensor(out_details[outs["scores"]]["index"])[0]
        # Some models provide a count tensor; not strictly needed
        # count = int(interpreter.get_tensor(out_details[outs["count"]]["index"])[0]) if outs["count"] is not None else len(scores)

        detections = []
        for i in range(len(scores)):
            y_min, x_min, y_max, x_max = boxes[i]
            detections.append((y_min, x_min, y_max, x_max, classes[i], scores[i]))

        # Draw
        draw_detections(frame, detections, labels, has_bg_zero, args.threshold)

        # FPS
        frame_count += 1
        now = time.time()
        if now - last_t >= 0.5:
            fps = frame_count / (now - last_t)
            frame_count = 0
            last_t = now
        cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)

        # Show
        cv2.imshow("detect_cam", frame)

        # Record
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            out = Path("frames") / f"frame_{int(time.time())}.png"
            cv2.imwrite(str(out), frame)
            print(f"[info] Saved {out}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
