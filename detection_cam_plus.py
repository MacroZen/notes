# detect_cam.py
from __future__ import annotations
import time, sys
from typing import List, Optional, Tuple, Dict
import cv2

from person_trigger import PresenceTrigger, Detection
import person_trigger as pt_config
from robo_control_plus import RobotController, INITIAL_POSE

# =========================
# TOGGLES
# =========================
FAST_MODE = True
USE_TRACKING_IDS = True   # keep False (we're doing single-target CSRT lock)
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
LOCK_POLICY = "nearest_roi"   # 'largest' | 'center' | 'nearest_roi' (ROI from presence_trigger)
LOST_REACQUIRE = False    # auto-reacquire when lost (largest person)
TRACKER_TYPE = "CSRT"     # "CSRT" (robust) or "KCF" (lighter)

JOG_STEP = 4.0  # degrees per key press for manual jogs

# Detection/tracker sanity filters
MIN_BOX_AREA_RATIO = 0.2   # ignore boxes smaller than this fraction of frame area
INIT_CONF = 0.35            # minimum confidence required to initialize tracker
MIN_ASPECT = 0.15           # min width/height
MAX_ASPECT = 1.8            # max width/height
IOU_REVALIDATE_EVERY = 5    # frames between revalidation checks (when detections are available)
IOU_MIN = 0.15              # minimum IoU between detector and tracker to consider still valid
HIST_SIM_THRESH = 0.4       # histogram similarity threshold (cv2.HISTCMP_CORREL -> [-1..1], expect >0.4)

# Switch/hysteresis tuning
SWITCH_CONF_MARGIN = 0.4   # candidate must exceed current_conf by this margin to be considered
SWITCH_PERSISTENCE = 3      # consecutive frames required to accept a switch
SWITCH_COOLDOWN_S = 2.0     # seconds after a switch during which further switches are suppressed
IOU_SWITCH_LOW = 0.25       # if candidate IoU < this and conf margin met, candidate may be considered
IOU_SWITCH_HIGH = 0.55      # if candidate IoU > this, consider it same target

# helper: create tracker
def create_tracker():
    if TRACKER_TYPE.upper() == "KCF":
        # Try legacy namespace first, fallback to main cv2 namespace
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create()
        elif hasattr(cv2, 'TrackerKCF_create'):
            return cv2.TrackerKCF_create()
        else:
            raise RuntimeError("KCF tracker not available in your OpenCV installation.")
    # CSRT (requires opencv-contrib-python)
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    elif hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    else:
        raise RuntimeError("CSRT tracker not available in your OpenCV installation.")

def bbox_area(b):
    x1,y1,x2,y2 = b
    return max(0.0, float(x2 - x1) * float(y2 - y1))


def bbox_iou(a, b):
    # a and b are (x1,y1,x2,y2)
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area(a) + bbox_area(b) - inter
    return (inter / union) if union > 0 else 0.0


def bbox_aspect(b):
    x1,y1,x2,y2 = b
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return float(w) / float(h)


def make_hist(img, bbox):
    # compute a small HSV histogram for the bbox region
    x1,y1,x2,y2 = [int(v) for v in bbox]
    h,w = img.shape[:2]
    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [30,32], [0,180,0,256])
    cv2.normalize(hist, hist)
    return hist

def choose_target(dets: List[Detection], frame_size: Tuple[int,int]) -> Optional[Detection]:
    """Select one 'person' detection to lock onto based on LOCK_POLICY.
    Applies sanity filters (confidence, area, aspect) to avoid bright-spot false positives.
    """
    persons = [d for d in dets if str(d['label']).lower() == 'person']
    if not persons:
        return None
    W, H = frame_size

    # filter by basic criteria
    def valid(d):
        try:
            conf = float(d.get('conf', 0.0))
            if conf < INIT_CONF:
                return False
            x1,y1,x2,y2 = d['bbox']
            area = bbox_area((x1,y1,x2,y2))
            if area < (MIN_BOX_AREA_RATIO * float(W) * float(H)):
                return False
            asp = bbox_aspect((x1,y1,x2,y2))
            if asp < MIN_ASPECT or asp > MAX_ASPECT:
                return False
            return True
        except Exception:
            return False

    filtered = [p for p in persons if valid(p)]
    if not filtered:
        # no high-quality person detections
        return None

    if LOCK_POLICY == "largest":
        def area_fn(d):
            x1,y1,x2,y2 = d['bbox']; return (x2-x1)*(y2-y1)
        return max(filtered, key=area_fn)

    if LOCK_POLICY == "center":
        cx, cy = W/2.0, H/2.0
        def dist2(d):
            x1,y1,x2,y2 = d['bbox']
            mx, my = 0.5*(x1+x2), 0.5*(y1+y2)
            return (mx-cx)**2 + (my-cy)**2
        return min(filtered, key=dist2)

    if LOCK_POLICY == "nearest_roi" and pt_config.ROI is not None:
        rx1, ry1, rx2, ry2 = pt_config.ROI
        rcx, rcy = 0.5*(rx1+rx2), 0.5*(ry1+ry2)
        def dist2_roi(d):
            x1,y1,x2,y2 = d['bbox']
            mx, my = 0.5*(x1+x2), 0.5*(y1+y2)
            return (mx-rcx)**2 + (my-rcy)**2
        return min(filtered, key=dist2_roi)

    return filtered[0]

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

def set_lock_led(rc, locked: bool):
    """Set the robot arm LED: red when locked, green when unlocked.
    Guarded by RobotController.dry_run and presence of the low-level _mc object.
    """
    try:
        if rc is None:
            return
        # RobotController may expose a dry_run flag to avoid hardware calls
        if getattr(rc, 'dry_run', False):
            return
        mc = getattr(rc, '_mc', None)
        if mc is None:
            return
        if locked:
            # red
            mc.set_color(255, 0, 0)
        else:
            # green
            mc.set_color(0, 255, 0)
    except Exception:
        # best-effort only; don't raise from LED failures
        pass

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
    track_hist = None
    track_revalidate_counter = 0
    last_detector_detections: List[Detection] = []
    next_det_id = 1
    # small registry for detection IDs: {id: {'bbox':(x1,y1,x2,y2),'last_seen':frame_idx,'conf':float}}
    det_registry: Dict[int, Dict] = {}
    REGISTRY_MAX_AGE_FRAMES = 6
     # sticky-switch state
    _current_locked_det: Optional[Detection] = None
    _current_locked_conf: float = 0.0
    _switch_counter: int = 0
    _last_switch_ts: float = 0.0

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
                # registry-based matching: match detections to det_registry by IoU (greedy)
                assigned: List[Detection] = []
                used_ids = set()
                for d in detections:
                    best_id = None
                    best_iou = 0.0
                    for tid, entry in list(det_registry.items()):
                        try:
                            iou = bbox_iou(d['bbox'], entry['bbox'])
                        except Exception:
                            iou = 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_id = tid
                    # adopt match if IoU high enough
                    if best_id is not None and best_iou >= 0.5 and best_id not in used_ids:
                        d['track_id'] = best_id
                        # update registry entry
                        det_registry[best_id]['bbox'] = d['bbox']
                        det_registry[best_id]['last_seen'] = frame_idx
                        det_registry[best_id]['conf'] = float(d.get('conf', 0.0))
                        used_ids.add(best_id)
                    else:
                        # create new id
                        tid = next_det_id
                        next_det_id += 1
                        d['track_id'] = tid
                        det_registry[tid] = {'bbox': d['bbox'], 'last_seen': frame_idx, 'conf': float(d.get('conf', 0.0))}
                        used_ids.add(tid)
                    assigned.append(d)
                last_detector_detections = assigned
                # cleanup stale registry entries
                stale = [tid for tid, e in det_registry.items() if (frame_idx - int(e.get('last_seen', frame_idx))) > REGISTRY_MAX_AGE_FRAMES]
                for tid in stale:
                    det_registry.pop(tid, None)
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

                    # revalidation: every N frames when detector results exist, ensure tracker still matches detections
                    track_revalidate_counter += 1
                    if do_infer and (track_revalidate_counter % IOU_REVALIDATE_EVERY == 0):
                        # find best detector bbox IoU with tracker
                        tbbox = (x, y, x+w, y+h)
                        best = None
                        best_iou = 0.0
                        for d in last_detector_detections:
                            if str(d.get('label')).lower() != 'person':
                                continue
                            iou = bbox_iou(tbbox, d['bbox'])
                            if iou > best_iou:
                                best_iou = iou
                                best = d
                        if best is None or best_iou < IOU_MIN:
                            print(f"[TRACK] Revalidation failed (best_iou={best_iou:.2f}) — stopping track")
                            locked = False
                            rc.set_follow_enabled(False)
                            tracker = None
                            lock_bbox_xywh = None
                            track_hist = None
                            _current_locked_det = None
                            _current_locked_conf = 0.0
                            _switch_counter = 0
                            track_revalidate_counter = 0
                            # update LED to indicate unlocked
                            set_lock_led(rc, False)
                            continue
                        # also check histogram similarity if available
                        if track_hist is not None:
                            cur_hist = make_hist(frame, best['bbox'])
                            if cur_hist is None:
                                print("[TRACK] couldn't compute detector hist — stopping")
                                locked = False
                                rc.set_follow_enabled(False)
                                tracker = None
                                lock_bbox_xywh = None
                                track_hist = None
                                _current_locked_det = None
                                _current_locked_conf = 0.0
                                _switch_counter = 0
                                track_revalidate_counter = 0
                                # update LED to indicate unlocked
                                set_lock_led(rc, False)
                                continue
                            sim = cv2.compareHist(track_hist, cur_hist, cv2.HISTCMP_CORREL)
                            if sim < HIST_SIM_THRESH:
                                print(f"[TRACK] Histogram similarity low ({sim:.2f}) — stopping")
                                locked = False
                                rc.set_follow_enabled(False)
                                tracker = None
                                lock_bbox_xywh = None
                                track_hist = None
                                _current_locked_det = None
                                _current_locked_conf = 0.0
                                _switch_counter = 0
                                track_revalidate_counter = 0
                                # update LED to indicate unlocked
                                set_lock_led(rc, False)
                                continue
                        # --- switching logic: consider strong detector candidates to replace current lock ---
                        try:
                            # build candidate list with basic quality checks
                            candidates = []
                            for d in last_detector_detections:
                                if str(d.get('label')).lower() != 'person':
                                    continue
                                conf = float(d.get('conf', 0.0))
                                x1,y1,x2,y2 = d['bbox']
                                area = bbox_area((x1,y1,x2,y2))
                                if conf < INIT_CONF:
                                    continue
                                if area < (MIN_BOX_AREA_RATIO * float(W) * float(H)):
                                    continue
                                asp = bbox_aspect(d['bbox'])
                                if asp < MIN_ASPECT or asp > MAX_ASPECT:
                                    continue
                                candidates.append(d)
                            if candidates:
                                best_cand = max(candidates, key=lambda z: float(z.get('conf', 0.0)))
                                cand_iou = bbox_iou(tbbox, best_cand['bbox'])
                                cand_conf = float(best_cand.get('conf', 0.0))
                               
                                # if we have tracking ids enabled and the candidate matches the current id, treat as same

                            if USE_TRACKING_IDS and _current_locked_det is not None and best_cand.get('track_id') == _current_locked_det.get('track_id'):
                                _current_locked_det = best_cand
                                _current_locked_conf = max(_current_locked_conf, cand_conf)
                                _switch_counter = 0
                            else:
                                 # if high IoU -> same target; refresh stored confidence
                                    if cand_iou > IOU_SWITCH_HIGH:
                                        _current_locked_det = best_cand
                                        _current_locked_conf = max(_current_locked_conf, cand_conf)
                                        _switch_counter = 0
                                    else:
                                        # candidate appears better than current by margin and is not overlapping
                                        if cand_conf > (_current_locked_conf + SWITCH_CONF_MARGIN):
                                            _switch_counter += 1
                                            if _switch_counter >= SWITCH_PERSISTENCE and (time.time() - _last_switch_ts) >= SWITCH_COOLDOWN_S:
                                                # perform switch
                                                print(f"[TRACK] Switching lock to new candidate conf={cand_conf:.2f} iou={cand_iou:.2f}")
                                                x1,y1,x2,y2 = best_cand['bbox']
                                                box_new = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                                                tracker = create_tracker()
                                                tracker.init(frame, box_new)
                                                locked = True
                                                lock_bbox_xywh = box_new
                                                _current_locked_det = best_cand
                                                _current_locked_conf = cand_conf
                                                _last_switch_ts = time.time()
                                                _switch_counter = 0
                                                track_hist = make_hist(frame, best_cand['bbox'])
                                                # switched lock -> LED red
                                                set_lock_led(rc, True)
                                        else:
                                            _switch_counter = 0
                        except Exception:
                            pass
                else:
                    # lost tracking
                    locked = False
                    rc.set_follow_enabled(False)
                    lock_bbox_xywh = None
                    tracker = None
                    track_hist = None
                    track_revalidate_counter = 0
                    print("[LOCK] Lost target. Follow disabled.")
                    # indicate unlocked LED
                    set_lock_led(rc, False)
                    # optional auto-reacquire
                    if LOST_REACQUIRE and do_infer:
                        cand = choose_target(detections, (W, H))
                        if cand:
                            x1,y1,x2,y2 = cand['bbox']
                            box = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                            tracker = create_tracker()
                            tracker.init(frame, box)
                            # compute initial histogram signature
                            track_hist = make_hist(frame, cand['bbox'])
                            locked = True
                            print("[LOCK] Reacquired target.")
                            # indicate locked LED
                            set_lock_led(rc, True)

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
                set_lock_led(rc, False)
            if key == ord('u'):         # unlock
                locked = False
                rc.set_follow_enabled(False)
                tracker = None
                lock_bbox_xywh = None
                print("[LOCK] Unlocked.")
                set_lock_led(rc, False)
            if key == ord('g'):         # follow toggle
                # Toggle follow only when locked; otherwise ensure disabled
                if locked:
                    rc.set_follow_enabled(not rc._follow_enabled)
                else:
                    rc.set_follow_enabled(False)
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
                    # initialize sticky-switch state
                    _current_locked_det = cand
                    _current_locked_conf = float(cand.get('conf', 0.0))
                    track_revalidate_counter = 0
                    _last_switch_ts = time.time()
                    _switch_counter = 0
                    track_hist = make_hist(frame, cand['bbox'])
                    print("[LOCK] Target locked. Press 'g' to follow, 'u' to unlock.")
                    # indicate locked LED
                    set_lock_led(rc, True)
                else:
                    print("[LOCK] No person to lock.")
            # J/L: manual yaw jog test (+/- JOG_STEP degrees on joint-0)
            if key == ord('l'):  # expect yaw to turn right in the camera view
                try:
                    mc = rc._mc
                    if mc:
                        # disable follow while doing a manual jog to avoid competing commands
                        rc.set_follow_enabled(False)
                        j = mc.get_angles()
                        if isinstance(j, list) and len(j) == 6:
                            delta = float(JOG_STEP)
                           # apply delta to joint-0 and keep the rest at INITIAL_POSE
                            new_angles = [j[0] + delta] + INITIAL_POSE[1:]
                            rc.enqueue_angles(new_angles)
                            print(f"[TEST] J1 += {delta}° -> {new_angles}")
                except Exception as e:
                    print("[TEST] jog failed:", e)

            if key == ord('k'):  # expect yaw to turn left in the camera view
                try:
                    mc = rc._mc
                    if mc:
                        # disable follow while doing a manual jog to avoid competing commands
                        rc.set_follow_enabled(False)
                        j = mc.get_angles()
                        if isinstance(j, list) and len(j) == 6:
                            delta = -float(JOG_STEP)
                            new_angles = [j[0] + delta] + INITIAL_POSE[1:]
                            rc.enqueue_angles(new_angles)
                            print(f"[TEST] J1 += {delta}° -> {new_angles}")
                except Exception as e:
                    print("[TEST] jog failed:", e)

            if key == ord('c'):         # move camera into postion
                try:
                    mc = rc._mc
                    if mc:
                        rc.set_follow_enabled(False)
                        rc.enqueue_angles([100,0,0,70,90,0], 20)  #point cam
                except Exception as e:
                    print("[WARN] error", e)
            
            if key == ord('a'):
                try:
                    mc = rc._mc
                    if mc:
                        j = mc.get_angles()
                        if isinstance(j, list) and len(j) == 6:
                            j_new = j[:] # copy
                            #keep j1 as is
                            # ensure the rest are (0,0,0,70,90,0)
                            MIN_POS = 1.0 #OPTIONAL DEGREES THRESHOLD
                            if j_new[1] <= 0.0:
                                j_new[1] = MIN_POS
                            if j_new[2] <= 0.0:
                                j_new[2] = MIN_POS
                            # assign the desired pose for remaining joints
                            j_new[3] = 70.0
                            j_new[4] = 90.0
                            j_new[5] = 0.0
                            # disable follow and enqueue the move so it doesn't get immediately overridden
                            rc.set_follow_enabled(False)
                            try:
                                if not rc.dry_run and rc._mc:
                                    rc._mc.focus_all_servos()
                            except Exception:
                                pass
                            rc.enqueue_angles(j_new, 40)  # use higher speed to overcome gravity
                            print("STAND UP")

                except Exception as e:
                    print("[WARN] CANT STAND UP", e)

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        rc.stop()

    return 0

if __name__ == "__main__":
    sys.exit(main())
