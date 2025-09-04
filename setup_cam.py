#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# One-shot setup + run for MobileNet-SSD object detection on Raspberry Pi
# - Creates venv named 'opencv' (or reuses it)
# - Installs OpenCV + TFLite Runtime (fallback to OpenCV-DNN Caffe)
# - Downloads model
# - Writes detector with AUTO camera selection
# Usage:
#   ./setup_detect.sh
# Optional env:
#   CAM="auto|/dev/videoN|index" SIZE=320 THRESH=0.5 VENV_NAME=opencv ./setup_detect.sh
# -------------------------------------------------------------------

CAM="${CAM:-auto}"         # "auto" scans /dev/video*, else index or path
SIZE="${SIZE:-320}"        # inference size
THRESH="${THRESH:-0.5}"    # confidence threshold
VENV_NAME="${VENV_NAME:-opencv}"

echo "[1/7] System prerequisites..."
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip unzip wget \
                        v4l-utils ffmpeg libgl1 libglib2.0-0

WORKDIR="$(pwd)/vision"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "[2/7] Virtual environment: $VENV_NAME"
if [[ ! -d "$VENV_NAME" ]]; then
  python3 -m venv "$VENV_NAME"
fi
# shellcheck disable=SC1091
source "$VENV_NAME/bin/activate"

echo "[3/7] Python deps..."
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade opencv-python numpy

echo "[3b] Checking for tflite-runtime..."
USE_TFLITE=1
python - <<'PY' || USE_TFLITE=0
try:
    import tflite_runtime.interpreter as tflite
    print("tflite-runtime: OK")
except Exception as e:
    raise SystemExit(1)
PY

if [[ "$USE_TFLITE" -eq 0 ]]; then
  echo "Installing tflite-runtime..."
  if ! pip install --upgrade tflite-runtime >/dev/null 2>&1; then
    USE_TFLITE=0
    echo "tflite-runtime unavailable; will fall back to OpenCV-DNN."
  else
    USE_TFLITE=1
  fi
fi

if [[ "$USE_TFLITE" -eq 1 ]]; then
  echo "[4/7] Downloading TFLite MobileNet-SSD..."
  mkdir -p ssd_mobilenet_v1
  if [[ ! -f ssd_mobilenet_v1/detect.tflite ]]; then
    wget -q https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -O model.zip
    unzip -o model.zip -d ssd_mobilenet_v1 >/dev/null
    rm -f model.zip
  fi

  echo "[5/7] Writing detect_cam.py (TFLite + AUTO camera)..."
  cat > detect_cam.py <<'PY'
#!/usr/bin/env python3
import time, sys, os, glob
from pathlib import Path
import numpy as np
import cv2

# Prefer tflite-runtime (lightweight)
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except Exception:
    sys.stderr.write("ERROR: tflite-runtime not available in this environment.\n")
    sys.exit(1)

MODEL = Path("ssd_mobilenet_v1/detect.tflite")
LABELS = Path("ssd_mobilenet_v1/labelmap.txt")

def load_labels(path: Path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def build_camera_candidates():
    # Prefer indices 0..5, then /dev/video0..9 (skip 10+ ISP nodes)
    cands = list(range(0, 6))
    for i in range(0, 10):
        p = f"/dev/video{i}"
        if os.path.exists(p):
            cands.append(p)
    # Deduplicate preserving order
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def try_open(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap.release()
        return None
    # Many UVC cams prefer MJPG + explicit size
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap

def auto_open_camera(preferred):
    # preferred: "auto" | "" | None | index | "/dev/videoX"
    if preferred not in (None, "", "auto"):
        # Try the user-specified source first
        try:
            # numeric index
            src = int(preferred)
        except ValueError:
            src = preferred
        cap = try_open(src)
        if cap: 
            return cap, src

    # Auto probe
    for cand in build_camera_candidates():
        cap = try_open(cand)
        if cap:
            return cap, cand
    return None, None

def main():
    cam_arg = os.environ.get("CAM", "auto")
    size    = int(os.environ.get("SIZE", "320"))
    thresh  = float(os.environ.get("THRESH", "0.5"))

    if not MODEL.exists() or not LABELS.exists():
        raise SystemExit("Model or labels not found in ssd_mobilenet_v1.")

    labels = load_labels(LABELS)

    cap, selected = auto_open_camera(cam_arg)
    if cap is None:
        sys.stderr.write(f"Could not open camera (CAM={cam_arg}).\n")
        devs = ' '.join(sorted(glob.glob('/dev/video*')))
        sys.stderr.write(f"Detected devices: {devs or '(none)'}\n")
        sys.exit(2)
    print(f"[INFO] Using camera source: {selected}")

    # TFLite interpreter
    interpreter = Interpreter(model_path=str(MODEL))
    interpreter.allocate_tensors()
    in_det  = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()
    in_index = in_det['index']
    in_dtype = in_det['dtype']

    win = "MobileNet-SSD (ESC to quit)"
    t0, n = time.time(), 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok: 
            print("WARN: frame read failed")
            break

        img = cv2.resize(frame, (size, size))
        if in_dtype == np.uint8:
            inp = np.expand_dims(img, 0).astype(np.uint8)
        else:
            inp = (np.expand_dims(img, 0).astype(np.float32)) / 255.0

        t_infer0 = time.time()
        interpreter.set_tensor(in_index, inp)
        interpreter.invoke()
        infer_ms = (time.time() - t_infer0) * 1000.0

        outs = [interpreter.get_tensor(o['index']) for o in out_det]
        boxes = classes = scores = count = None
        for arr in outs:
            a = np.squeeze(arr)
            if a.ndim == 2 and a.shape[1] == 4: boxes = a
            elif a.ndim == 1 and np.issubdtype(a.dtype, np.integer) and a.size <= 100: classes = a.astype(int)
            elif a.ndim == 1 and np.issubdtype(a.dtype, np.floating) and a.size <= 100: scores = a
            elif np.issubdtype(a.dtype, np.integer) and a.size == 1: count = int(a.ravel()[0])
        if count is None and scores is not None:
            count = len(scores)

        H, W = frame.shape[:2]
        if boxes is not None and classes is not None and scores is not None and count is not None:
            for i in range(count):
                if i >= len(scores): break
                s = float(scores[i])
                if s < thresh: continue
                y1, x1, y2, x2 = boxes[i]
                p1 = (int(x1*W), int(y1*H))
                p2 = (int(x2*W), int(y2*H))
                cls = classes[i] if 0 <= i < len(classes) else -1
                name = labels[cls] if 0 <= cls < len(labels) else str(cls)
                cv2.rectangle(frame, p1, p2, (0,255,0), 2)
                cv2.putText(frame, f"{name} {s:.2f}", (p1[0], max(15, p1[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)

        n += 1
        if n >= 10:
            fps = n / (time.time() - t0)
            t0, n = time.time(), 0

        cv2.putText(frame, f"Infer {infer_ms:.1f} ms | FPS {fps:.1f}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,220,30), 2, cv2.LINE_AA)
        cv2.imshow(win, frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
PY
  chmod +x detect_cam.py

  echo "[6/7] Run detection (TFLite, CAM=${CAM}, SIZE=${SIZE}, THRESH=${THRESH})..."
  CAM="$CAM" SIZE="$SIZE" THRESH="$THRESH" python3 detect_cam.py

else
  echo "[4/7] Fallback: OpenCV-DNN (Caffe MobileNet-SSD)..."
  mkdir -p caffe_ssd
  cd caffe_ssd
  if [[ ! -f MobileNetSSD_deploy.caffemodel ]]; then
    wget -q https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel
  fi
  if [[ ! -f MobileNetSSD_deploy.prototxt ]]; then
    wget -q https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt
  fi
  cd ..

  echo "[5/7] Writing detect_cam_dnn.py (OpenCV-DNN + AUTO camera)..."
  cat > detect_cam_dnn.py <<'PY'
#!/usr/bin/env python3
import os, sys, cv2, numpy as np, glob

PROTOTXT = "caffe_ssd/MobileNetSSD_deploy.prototxt"
MODEL    = "caffe_ssd/MobileNetSSD_deploy.caffemodel"
CLASSES  = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car",
            "cat","chair","cow","diningtable","dog","horse","motorbike","person",
            "pottedplant","sheep","sofa","train","tvmonitor"]

def build_camera_candidates():
    cands = list(range(0, 6))
    for i in range(0, 10):
        p = f"/dev/video{i}"
        if os.path.exists(p):
            cands.append(p)
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def try_open(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap

def auto_open_camera(preferred):
    if preferred not in (None, "", "auto"):
        try:
            src = int(preferred)
        except ValueError:
            src = preferred
        cap = try_open(src)
        if cap: return cap, src
    for cand in build_camera_candidates():
        cap = try_open(cand)
        if cap: return cap, cand
    return None, None

def main():
    cam_arg = os.environ.get("CAM", "auto")
    thresh  = float(os.environ.get("THRESH", "0.5"))

    if not (os.path.isfile(PROTOTXT) and os.path.isfile(MODEL)):
        sys.exit("Missing Caffe model or prototxt.")

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

    cap, selected = auto_open_camera(cam_arg)
    if cap is None:
        sys.stderr.write(f"Could not open camera (CAM={cam_arg}).\n")
        devs = ' '.join(sorted(glob.glob('/dev/video*')))
        sys.stderr.write(f"Detected devices: {devs or '(none)'}\n")
        sys.exit(2)
    print(f"[INFO] Using camera source: {selected}")

    win = "OpenCV DNN MobileNet-SSD (ESC to quit)"
    while True:
        ok, frame = cap.read()
        if not ok: break
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 0.007843, (300,300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            conf = float(detections[0,0,i,2])
            if conf < thresh: continue
            idx = int(detections[0,0,i,1])
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (x1,y1,x2,y2) = box.astype("int")
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            label = f"{CLASSES[idx] if 0<=idx<len(CLASSES) else idx} {conf:.2f}"
            cv2.putText(frame,label,(x1,max(15,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv2.imshow(win, frame)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
PY
  chmod +x detect_cam_dnn.py

  echo "[6/7] Run detection (OpenCV-DNN, CAM=${CAM}, THRESH=${THRESH})..."
  CAM="$CAM" THRESH="$THRESH" python3 detect_cam_dnn.py
fi
echo "[7/7] Done."
