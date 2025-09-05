#!/usr/bin/env python3
"""
detect_cam.py — robust TFLite live detection (USB webcam, Raspberry Pi friendly)

v1.2.0

Features
- Hard-coded GStreamer pipeline for /dev/video0 with safe V4L2 fallback
- Works with tflite_runtime OR tensorflow.lite (auto-fallback)
- Robust model/labels path resolution (expands ~, searches ./, ./models, script dir, and ~/vision/notes/ssd_mobilenet_v1)
- Built-in COCO-80 labels if no label file is found
- Camera listing, FPS overlay, rotate/flip, optional MP4 recording
- Optional --pipeline to override the default GStreamer pipeline
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2


# --------------------------
# Built-in COCO-80 labels (fallback)
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


def load_labels(labels_path: Optional[Path]) -> Tuple[List[str], bool]:
    """
    Returns (labels, has_background_zero)
    has_background_zero=True if label[0] is a background placeholder like '???'
    """
    if labels_path and labels_path.is_file():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [ln.strip() for ln in f if ln.strip()]
        has_bg = bool(labels and labels[0] in ("???", "background", "__background__"))
        return labels, has_bg
    # Fallback to built-in COCO-80 (no explicit background)
    return COCO80, False


# --------------------------
# TFLite interpreter loader (tflite_runtime → tensorflow fallback)
# --------------------------
def try_import_interpreter(edgetpu: bool):
    """
    Try tflite_runtime first, then tensorflow as fallback.
    Returns (InterpreterClass, delegates_list)
    """
    delegates = []
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate  # type: ignore
        if edgetpu:
            try:
                delegates = [load_delegate("libedgetpu.so.1")]
                print("[info] EdgeTPU delegate loaded.")
            except Exception as e:
                print(f"[warn] Could not load EdgeTPU delegate: {e}")
        return Interpreter, delegates
    except Exception:
        pass

    try:
        from tensorflow.lite import Interpreter  # type: ignore
        # EdgeTPU via TF is uncommon; skip delegate here
        return Interpreter, []
    except Exception as e:
        print("[error] Could not import TFLite interpreter from tflite_runtime or tensorflow.")
        print("Install one of:\n  pip3 install tflite-runtime\n  OR\n  pip3 install tensorflow")
        raise e


# --------------------------
# Path resolution helpers
# --------------------------
def resolve_path(user_path: Optional[str], default_name: str, extra_dirs: Optional[List[Path]] = None) -> Optional[Path]:
    """
    Search order:
      - user-provided path (expanded)
      - CWD, ./models, script dir, script_dir/models
      - any extra_dirs passed in
      - default_name in the same search roots
    """
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    roots = [cwd, cwd / "models", here, here / "models"]
    if extra_dirs:
        roots.extend(extra_dirs)

    if user_path:
        p = Path(user_path).expanduser()
        if p.is_file():
            return p
        for r in roots:
            c = (r / user_path).expanduser()
            if c.is_file():
                return c

    for r in roots:
        c = (r / default_name).expanduser()
        if c.is_file():
            return c

    return None


def resolve_model(model_arg: Optional[str]) -> Path:
    """
    Find a usable TFLite model by checking:
      - user-provided path
      - typical locations
      - ~/vision/notes/ssd_mobilenet_v1/{detect.tflite, ssd_mobilenet_detect.tflite}
    """
    extra = [Path("~/vision/notes/ssd_mobilenet_v1").expanduser()]
    # Prefer Dylan's default model name first
    candidates = []
    if model_arg:
        candidates.append(Path(model_arg).expanduser())

    # Default names we will try
    common_names = ["detect.tflite", "ssd_mobilenet_detect.tflite"]

    # Build candidate list from search dirs
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    search_dirs = [cwd, cwd / "models", here, here / "models"] + extra

    for d in search_dirs:
        for name in common_names:
            candidates.append(d / name)

    for c in candidates:
        if c.is_file():
            return c

    # If nothing found, construct a helpful message
    tried = "\n  ".join(str(c) for c in candidates[:16])
    raise SystemExit(
        "❌ Could not find a TFLite model. Tried:\n  " + tried +
        "\n👉 Fix by passing --model /full/path/to/your_model.tflite "
        "or placing the file at ./models/detect.tflite"
    )


def resolve_labels(labels_arg: Optional[str]) -> Optional[Path]:
    extra = [Path("~/vision/notes/ssd_mobilenet_v1").expanduser()]
    p = resolve_path(labels_arg, "labelmap.txt", extra_dirs=extra)
    return p if p and p.is_file() else None


# --------------------------
# SSD-style output mapping
# --------------------------
def get_output_indices(interpreter):
    details = interpreter.get_output_details()
    keys = {"boxes": None, "classes": None, "scores": None, "count": None}
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
    if any(v is None for v in keys.values()):
        if len(details) >= 4:
            keys = {"boxes": 0, "classes": 1, "scores": 2, "count": 3}
    return keys


def draw_dets(frame, dets, labels, has_bg_zero, thr):
    h, w = frame.shape[:2]
    for y1, x1, y2, x2, cls_id, score in dets:
        if score < thr:
            continue
        c = int(cls_id)
        # Map class index to label sensibly
        if has_bg_zero:
            idx = c
        else:
            idx = c if c < len(labels) else max(0, min(len(labels) - 1, c - 1))
        name = labels[idx] if 0 <= idx < len(labels) else f"id:{c}"

        X1, Y1 = int(x1 * w), int(y1 * h)
        X2, Y2 = int(x2 * w), int(y2 * h)
        cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 180, 255), 2)
        txt = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (X1, Y1 - th - 6), (X1 + tw + 2, Y1), (0, 180, 255), -1)
        cv2.putText(frame, txt, (X1 + 1, Y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)


# --------------------------
# Camera helpers (GStreamer + V4L2 fallback)
# --------------------------
DEFAULT_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,width=1280,height=720,framerate=30/1 ! "
    "videoconvert ! appsink"
)
# For MJPEG webcams, this can be smoother:
# DEFAULT_PIPELINE = (
#     "v4l2src device=/dev/video0 ! "
#     "image/jpeg,width=1280,height=720,framerate=30/1 ! "
#     "jpegdec ! videoconvert ! appsink"
# )


def list_cameras(max_index=8):
    print("Scanning cameras (V4L2 indices)…")
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                print(f"  [{i}] OK  {w}x{h}")
                found.append(i)
        cap.release()
    if not found:
        print("No V4L2 cameras detected. If you have a Pi Camera, use libcamera or a GStreamer pipeline.")


def open_camera(pipeline: Optional[str], width: int, height: int, fps: int):
    use_pipeline = (pipeline or DEFAULT_PIPELINE)

    if "appsink" not in use_pipeline:
        print("[warn] Supplied pipeline missing 'appsink' at the end; OpenCV may not read frames.")

    print("[info] Trying GStreamer pipeline:")
    print("       ", use_pipeline)
    cap = cv2.VideoCapture(use_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap

    print("[warn] GStreamer pipeline failed to open. Falling back to V4L2 /dev/video0")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if width > 0 and height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Default model set to Dylan's exact path/name
    ap.add_argument(
        "-m", "--model",
        type=str,
        default=str(Path("~/vision/notes/ssd_mobilenet_v1/detect.tflite").expanduser()),
        help="Path to TFLite model (absolute or relative). By default, uses your detect.tflite."
    )
    ap.add_argument("-l", "--labels", type=str, default="labelmap.txt",
                    help="Path to labels file (optional; built-in COCO-80 is used if missing)")
    ap.add_argument("--pipeline", type=str, default="",
                    help="GStreamer pipeline to use (optional). If empty, a robust default is used.")
    ap.add_argument("--width", type=int, default=1280, help="Capture width (V4L2 fallback only)")
    ap.add_argument("--height", type=int, default=720, help="Capture height (V4L2 fallback only)")
    ap.add_argument("--cam-fps", type=int, default=30, help="Requested camera FPS (V4L2 fallback only)")
    ap.add_argument("-t", "--threshold", type=float, default=0.5, help="Score threshold [0..1]")
    ap.add_argument("--edgetpu", action="store_true", help="Try to use EdgeTPU delegate (Coral USB)")
    ap.add_argument("--record", type=str, default="", help="Record output to MP4 (e.g., out.mp4)")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="Rotate display (deg)")
    ap.add_argument("--flip", type=str, default="", choices=["", "h", "v", "hv"], help="Flip image")
    ap.add_argument("--list-cams", action="store_true", help="List available V4L2 cameras and exit")
    args = ap.parse_args()

    if args.list_cams:
        list_cameras()
        return

    # Resolve model & labels
    model_path = resolve_model(args.model)
    labels_path = resolve_labels(args.labels)
    labels, has_bg_zero = load_labels(labels_path)
    if labels_path:
        print(f"[info] Using label file: {labels_path}")
    else:
        print("[info] Using built-in COCO-80 labels.")

    # Load TFLite model
    Interpreter, delegates = try_import_interpreter(args.edgetpu)
    interpreter = Interpreter(model_path=str(model_path), experimental_delegates=delegates)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()
    ih, iw = in_details[0]["shape"][1], in_details[0]["shape"][2]
    in_dtype = in_details[0]["dtype"]
    in_quant = in_details[0].get("quantization", (0.0, 0))
    outs = get_output_indices(interpreter)
    print(f"[info] Model: {model_path} | input {iw}x{ih} dtype={in_dtype} quant={in_quant}")

    # Camera
    cap = open_camera(args.pipeline, args.width, args.height, args.cam_fps)
    if not cap.isOpened():
        print("[error] Could not open camera via GStreamer pipeline or V4L2 fallback.")
        sys.exit(1)

    # Optional recorder
    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = float(args.cam_fps if args.cam_fps > 0 else 30)
        writer = cv2.VideoWriter(args.record, fourcc, out_fps, (out_w, out_h))
        if writer.isOpened():
            print(f"[info] Recording to {args.record}")
        else:
            print(f"[warn] Failed to open writer for {args.record}; recording disabled.")
            writer = None

    # Inference loop
    Path("frames").mkdir(exist_ok=True)
    last_t = time.time()
    fps = 0.0
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[warn] No frame from camera.")
            break

        if args.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif args.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if args.flip:
            if "h" in args.flip:
                frame = cv2.flip(frame, 1)
            if "v" in args.flip:
                frame = cv2.flip(frame, 0)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
        if str(in_dtype).endswith("uint8"):
            inp = np.asarray(inp, dtype=np.uint8)
        else:
            inp = (np.asarray(inp, dtype=np.float32) / 255.0)

        interpreter.set_tensor(in_details[0]["index"], np.expand_dims(inp, 0))
        interpreter.invoke()
        boxes = interpreter.get_tensor(out_details[outs["boxes"]]["index"])[0]
        classes = interpreter.get_tensor(out_details[outs["classes"]]["index"])[0]
        scores = interpreter.get_tensor(out_details[outs["scores"]]["index"])[0]

        dets = [(b[0], b[1], b[2], b[3], classes[i], scores[i]) for i, b in enumerate(boxes)]
        draw_dets(frame, dets, labels, has_bg_zero, args.threshold)

        # FPS overlay
        n += 1
        now = time.time()
        if now - last_t >= 0.5:
            fps = n / (now - last_t)
            n = 0
            last_t = now
        cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)

        # Show & record
        cv2.imshow("detect_cam", frame)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            out = Path("frames") / f"frame_{int(time.time())}.png"
            cv2.imwrite(str(out), frame)
            print(f"[info] Saved {out}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
