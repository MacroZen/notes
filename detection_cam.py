# detect_cam.py
from __future__ import annotations
import time
import sys
from typing import List, Optional

import cv2

from presence_trigger import PresenceTrigger, Detection
import presence_trigger as pt_config

# --- NEW: robot controls ---
from robot_controls import RobotController

# =========================
# BASIC APP CONFIG
# =========================
VIDEO_SOURCE = 0
WINDOW_TITLE = "Detect Cam + Presence Trigger + Robot Keys"
DRAW = True

# =========================
# FAST MODE TOGGLE
# =========================
FAST_MODE = True

IMG_SIZE = 640
CAM_WIDTH, CAM_HEIGHT = (1280, 720)
FRAME_SKIP = 0
YOLO_CLASSES: Optional[list[int]] = None
MAX_DET = 300
TARGET_FPS: Optional[int] = None
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

if FAST_MODE:
    IMG_SIZE = 320
    CAM_WIDTH, CAM_HEIGHT = (640, 360)
    FRAME_SKIP = 2
    YOLO_CLASSES = [0]          # person only
    MAX_DET = 10
    TARGET_FPS = 10
    CONF_THRESHOLD = 0.35
    IOU_THRESHOLD = 0.50

# =========================
# KEY → POSE MAP
# =========================
# Press keys in the OpenCV window to move the robot.
KEYMAP = {
    ord('h'): "HOME",
    ord('s'): "SAFE",
    ord('1'): "LEFT",
    ord('2'): "RIGHT",
    # add more as needed
}
EMERGENCY_STOP_KEYS = {ord('x'), 27}  # 'x' or ESC -> stop robot immediately


def parse_yolo_results_to_detections(results, names) -> List[Detection]:
    detections: List[Detection] = []
    r = results[0]
    boxes = getattr(r, "boxes", None)
    if not boxes:
        return detections
    for b in boxes:
        try:
            cls_id = int(b.cls.item())
        except Exception:
            cls_id = int(float(b.cls))
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        try:
            conf = float(b.conf.item())
        except Exception:
            conf = float(b.conf)
        xyxy = b.xyxy[0].tolist() if hasattr(b.xyxy, "__iter__") else list(b.xyxy)
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        detections.append({'label': label, 'conf': conf, 'bbox': (x1, y1, x2, y2)})
    return detections


def main() -> int:
    # --- Start robot controller (non-blocking worker thread) ---
    rc = RobotController()
    rc.start()

    # --- Optional: hook presence trigger so ENTER -> SAFE pose ---
    def _on_enter(dets: List[Detection]):
        print("[EVENT] Person ENTER -> Robot SAFE")
        try:
            rc.enqueue("SAFE")
        except Exception as e:
            print(f"[Robot][WARN] SAFE enqueue failed: {e}")

    def _on_exit():
        print("[EVENT] Person EXIT")

    # Monkey-patch presence hook functions (looked up at call time)
    pt_config.on_person_enter = _on_enter
    pt_config.on_person_exit = _on_exit

    # --- YOLO model ---
    yolo_model = None
    model_names = None
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
        model_names = yolo_model.names
        print("[INFO] YOLOv8 model loaded (yolov8n.pt)")
    except Exception as e:
        print(f"[WARN] Could not init YOLOv8: {e}")
        return 2

    presence = PresenceTrigger()

    # --- Camera ---
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {VIDEO_SOURCE}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    print("[INFO] Controls: h=HOME, s=SAFE, 1=LEFT, 2=RIGHT, x=EMERGENCY STOP, q/ESC=quit")

    fps_avg = 0.0
    alpha = 0.05
    frame_idx = 0

    try:
        while True:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Empty frame. Exiting.")
                break

            do_infer = True
            if FRAME_SKIP and (frame_idx % (FRAME_SKIP + 1) != 0):
                do_infer = False

            detections: List[Detection] = []
            events: List[str] = []

            if do_infer:
                results = yolo_model(
                    frame,
                    imgsz=IMG_SIZE,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    classes=YOLO_CLASSES,
                    max_det=MAX_DET,
                    device='cpu',
                    verbose=False
                )
                detections = parse_yolo_results_to_detections(results, model_names)
                events = presence.update(detections)

            # Draw
            if DRAW:
                if do_infer:
                    for d in detections:
                        x1, y1, x2, y2 = map(int, d['bbox'])
                        label = str(d['label'])
                        conf = float(d['conf'])
                        color = (0, 255, 0) if label.lower() == 'person' else (0, 255, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if pt_config.ROI is not None:
                    rx1, ry1, rx2, ry2 = pt_config.ROI
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)
                    cv2.putText(frame, "ROI", (rx1, max(0, ry1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                dt = time.time() - loop_start
                fps = 1.0 / dt if dt > 0 else 0.0
                fps_avg = (1 - alpha) * fps_avg + alpha * fps
                cv2.putText(frame, f"FPS: {fps_avg:.1f}  (FAST_MODE={FAST_MODE})",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if do_infer and events:
                    cv2.putText(frame, f"EVENT: {','.join(events)}", (10, 46),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(WINDOW_TITLE, frame)

            # --- Keys: move the robot without blocking detection ---
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC to quit
                break
            if key in EMERGENCY_STOP_KEYS:
                rc.emergency_stop()
            elif key in KEYMAP:
                pose = KEYMAP[key]
                rc.enqueue(pose)

            if TARGET_FPS:
                elapsed = time.time() - loop_start
                time.sleep(max(0, (1.0 / TARGET_FPS) - elapsed))

            frame_idx += 1

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        rc.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
