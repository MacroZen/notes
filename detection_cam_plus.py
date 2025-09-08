# detect_cam.py
from __future__ import annotations
import time, sys
from typing import List, Optional, Tuple
import cv2

from person_trigger import PresenceTrigger, Detection
import person_trigger as pt_config
from robo_control_plus import RobotController

# =========================
# TOGGLES
# =========================
FAST_MODE = True
USE_TRACKING_IDS = False   # keep False (we're doing single-target CSRT lock)
DRAW = True

# =========================
# CAMERA / YOLO CONFIG
# =========================
VIDEO_SOURCE = 0
WINDOW_TITLE = "Detect Cam + Single-Target Lock + Robot Follow"

IMG_SIZE = 640
CAM_WIDTH, CAM_HEIGHT = (1280, 720)
FRAME_SKIP = 0
YOLO_CLASSES: Optional[list[int]] = [0]   # detect 'person' only
MAX_DET = 50
TARGET_FPS: Optional[int] = None
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.50

if FAST_MODE:
    IMG_SIZE = 320
    CAM_WIDTH, CAM_HEIGHT = (640, 360)
    FRAME_SKIP = 2
    MAX_DET = 10
    TARGET_FPS = 10

# =========================
# LOCK/Follow settings
# =========================
LOCK_POLICY = "largest"   # 'largest' | 'center' | 'nearest_roi' (ROI from presence_trigger)
LOST_REACQUIRE = False    # auto-reacquire when lost (largest person)
TRACKER_TYPE = "CSRT"     # "CSRT" (robust) or "KCF" (lighter)

# helper: create tracker
def create_tracker():
    if TRACKER_TYPE.upper() == "KCF":
        # KCF fallback if contrib install is heavy
        return cv2.legacy.TrackerKCF_create()
    # CSRT (requires opencv-contrib-python)
    return cv2.legacy.TrackerCSRT_create()

def choose_target(dets: List[Detection], frame_size: Tuple[int,int]) -> Optional[Detection]:
    """Select one 'person' detection to lock onto based on LOCK_POLICY."""
    persons = [d for d in dets if str(d['label']).lower() == 'person']
    if not persons:
        return None
    W, H = frame_size

    if LOCK_POLICY == "largest":
        def area(d):
            x1,y1,x2,y2 = d['bbox']; return (x2-x1)*(y2-y1)
        return max(persons, key=area)

    if LOCK_POLICY == "center":
        cx, cy = W/2.0, H/2.0
        def dist2(d):
            x1,y1,x2,y2 = d['bbox']
            mx, my = 0.5*(x1+x2), 0.5*(y1+y2)
            return (mx-cx)**2 + (my-cy)**2
        return min(persons, key=dist2)

    if LOCK_POLICY == "nearest_roi" and pt_config.ROI is not None:
        rx1, ry1, rx2, ry2 = pt_config.ROI
        rcx, rcy = 0.5*(rx1+rx2), 0.5*(ry1+ry2)
        def dist2_roi(d):
            x1,y1,x2,y2 = d['bbox']
            mx, my = 0.5*(x1+x2), 0.5*(y1+y2)
            return (mx-rcx)**2 + (my-rcy)**2
        return min(persons, key=dist2_roi)

    return persons[0]

def parse_results_to_detections(results, names) -> List[Detection]:
    out: List[Detection] = []
    r = results[0]
    boxes = getattr(r, "boxes", None)
    if not boxes:
        return out
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
        out.append({'label': label, 'conf': conf, 'bbox': (x1, y1, x2, y2)})
    return out

def norm_error_x(bbox, W):
    """Return normalized horizontal error in [-1..1]: + means target right of center."""
    x1,y1,x2,y2 = bbox
    cx = 0.5*(x1+x2)
    return float((cx - (W/2.0)) / (W/2.0))

def main() -> int:
    # --- Robot ---
    rc = RobotController()
    rc.start()

    # --- Presence (unchanged) ---
    presence = PresenceTrigger()

    # --- YOLO ---
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        names = model.names
        print("[INFO] YOLOv8 model loaded (yolov8n.pt)")
    except Exception as e:
        print(f"[ERROR] Could not init YOLO: {e}")
        return 2

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
    print("[INFO] Keys: t=lock, u=unlock, g=follow toggle, x=EMERGENCY STOP, q/ESC=quit")

    # --- Lock/track state ---
    locked = False
    tracker = None
    lock_bbox_xywh: Optional[Tuple[int,int,int,int]] = None

    fps_avg, alpha = 0.0, 0.05
    frame_idx = 0

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            H, W = frame.shape[:2]

            # YOLO every Nth frame
            do_infer = not FRAME_SKIP or (frame_idx % (FRAME_SKIP + 1) == 0)
            detections: List[Detection] = []
            events: List[str] = []

            if do_infer:
                res = model(
                    frame,
                    imgsz=IMG_SIZE,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    classes=YOLO_CLASSES,
                    max_det=MAX_DET,
                    device='cpu',
                    verbose=False
                )
                detections = parse_results_to_detections(res, names)
                events = presence.update(detections)

            # ---- update single-object tracker if locked ----
            if locked and tracker is not None:
                ok_track, box = tracker.update(frame)
                if ok_track:
                    x, y, w, h = map(int, box)
                    lock_bbox_xywh = (x, y, w, h)
                    # robot follow: compute normalized horizontal error
                    err_x = norm_error_x((x, y, x+w, y+h), W)
                    rc.update_follow_error(err_x)
                else:
                    # lost tracking
                    locked = False
                    rc.set_follow_enabled(False)
                    lock_bbox_xywh = None
                    tracker = None
                    print("[LOCK] Lost target. Follow disabled.")
                    # optional auto-reacquire
                    if LOST_REACQUIRE and do_infer:
                        cand = choose_target(detections, (W, H))
                        if cand:
                            x1,y1,x2,y2 = cand['bbox']
                            box = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                            tracker = create_tracker()
                            tracker.init(frame, box)
                            locked = True
                            print("[LOCK] Reacquired target.")

            # ---- drawing ----
            if DRAW:
                # draw detections sparsely (when we ran YOLO)
                if do_infer:
                    for d in detections:
                        x1,y1,x2,y2 = map(int, d['bbox'])
                        label = str(d['label']); conf = float(d['conf'])
                        color = (0,255,0) if label.lower() == 'person' else (0,255,255)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(0,y1-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # draw lock box
                if locked and lock_bbox_xywh is not None:
                    x,y,w,h = lock_bbox_xywh
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,140,255), 2)
                    cv2.putText(frame, "LOCKED", (x, max(0, y-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,140,255), 2)

                # ROI box (if any)
                if pt_config.ROI is not None:
                    rx1, ry1, rx2, ry2 = pt_config.ROI
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255,0,0), 1)
                    cv2.putText(frame, "ROI", (rx1, max(0, ry1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

                # FPS
                dt = time.time() - t0
                fps = 1.0 / dt if dt > 0 else 0.0
                fps_avg = (1 - alpha) * fps_avg + alpha * fps
                cv2.putText(frame, f"FPS:{fps_avg:.1f} FAST:{FAST_MODE} LOCK:{locked}",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                if do_infer and events:
                    cv2.putText(frame, f"EVENT:{','.join(events)}",
                                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cv2.imshow(WINDOW_TITLE, frame)

            # optional throttle
            if TARGET_FPS:
                elapsed = time.time() - t0
                time.sleep(max(0, (1.0 / TARGET_FPS) - elapsed))

            # ---- keys ----
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):   # q/ESC quit
                break
            if key == ord('x'):         # emergency stop
                rc.emergency_stop()
                locked = False
                tracker = None
                lock_bbox_xywh = None
            if key == ord('u'):         # unlock
                locked = False
                rc.set_follow_enabled(False)
                tracker = None
                lock_bbox_xywh = None
                print("[LOCK] Unlocked.")
            if key == ord('g'):         # follow toggle
                # Only enables follow if locked
                enable = not rc._follow_enabled
                rc.set_follow_enabled(enable and locked)
            if key == ord('t'):         # lock on current best person
                # Need a person detection from the most recent YOLO run
                cand = choose_target(detections, (W, H)) if do_infer else None
                if cand:
                    x1,y1,x2,y2 = cand['bbox']
                    box = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                    tracker = create_tracker()
                    tracker.init(frame, box)
                    locked = True
                    lock_bbox_xywh = box
                    print("[LOCK] Target locked. Press 'g' to follow, 'u' to unlock.")
                else:
                    print("[LOCK] No person to lock.")
            # J/L: manual yaw jog test (+/- 10 degrees on joint-1)
            if key == ord('l'):  # expect yaw to turn right in the camera view
                try:
                    mc = rc._mc
                    if mc:
                        j = mc.get_angles()
                        if isinstance(j, list) and len(j) == 6:
                            j[0] +=10
                            mc.send_angles(j, 20)
                            print("[TEST] J1 += 10° (should pan RIGHT in view)")
                except Exception as e:
                    print("[TEST] jog failed:", e)

            if key == ord('j'):  # expect yaw to turn left in the camera view
                try:
                    mc = rc._mc
                    if mc:
                        j = mc.get_angles()
                        if isinstance(j, list) and len(j) == 6:
                            j[0] -=10
                            mc.send_angles(j, 20)
                            print("[TEST] J1 -= 10° (should pan LEFT in view)")
                except Exception as e:
                    print("[TEST] jog failed:", e)

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        rc.stop()

    return 0

if __name__ == "__main__":
    sys.exit(main())
