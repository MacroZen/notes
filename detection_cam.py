"""
detect_cam.py
-------------
Camera loop that runs detection each frame, converts results into the generic
detections format, and feeds them to PresenceTrigger.

Run:
    python detect_cam.py

Dependencies:
    - OpenCV:        pip install opencv-python
    - (Optional) YOLOv8: pip install ultralytics
"""

import time
import sys
import cv2

from presence_trigger import PresenceTrigger, Detection
import presence_trigger as pt_config  # to access/override ROI & thresholds if you want
# =========================
# CONFIG
# =========================
VIDEO_SOURCE = 0           # 0 for default webcam; or RTSP/URL/file path
WINDOW_TITLE = "Detect Cam + Presence Trigger"
DRAW = True                # draw boxes and labels
USE_YOLOV8 = True          # set False to integrate your own detector path
IMG_SIZE = 640             # YOLOv8 inference size (if used)

# Optional: override ROI here at runtime (instead of editing presence_trigger.py)
# Uncomment and set as needed:
# pt_config.ROI = (100, 100, 540, 380)


# =========================
# Detection adapters
# =========================
def to_generic_detections(labels, confs, boxes_xyxy) -> list[Detection]:
    """
    Convert parallel lists of labels, confidences, and xyxy boxes
    into the generic detections list that PresenceTrigger expects.
    """
    dets: list[Detection] = []
    for label, conf, (x1, y1, x2, y2) in zip(labels, confs, boxes_xyxy):
        dets.append({
            'label': str(label),
            'conf': float(conf),
            'bbox': (float(x1), float(y1), float(x2), float(y2)),
        })
    return dets


def get_detections_from_your_detector(frame) -> list[Detection]:
    """
    Stub for your existing detector.
    Replace the body with your detector call and return a list of detections.

    Expected return:
        [
          {'label': 'person', 'conf': 0.87, 'bbox': (x1, y1, x2, y2)},
          ...
        ]
    """
    # ----- YOUR CODE HERE -----
    # Example:
    # results = my_detector.predict(frame)
    # labels, confs, boxes = parse_my_results(results)
    # return to_generic_detections(labels, confs, boxes)
    return []


def get_detections_from_yolov8(model, frame) -> list[Detection]:
    """
    Run YOLOv8 on the frame and convert the results to generic detections.
    """
    results = model(frame, verbose=False, imgsz=IMG_SIZE)
    detections: list[Detection] = []

    r = results[0]
    boxes = r.boxes
    if boxes is not None and len(boxes) > 0:
        names = model.names  # class id -> label
        for b in boxes:
            cls_id = int(b.cls.item())
            label = names.get(cls_id, str(cls_id))
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            detections.append({'label': label, 'conf': conf, 'bbox': (x1, y1, x2, y2)})

    return detections


# =========================
# Main
# =========================
def main() -> int:
    presence = PresenceTrigger()

    # Try init YOLOv8 if enabled
    yolo_model = None
    if USE_YOLOV8:
        try:
            from ultralytics import YOLO
            yolo_model = YOLO("yolov8n.pt")  # nano model for speed
            print("[INFO] YOLOv8 model loaded (yolov8n.pt)")
        except Exception as e:
            print(f"[WARN] Could not init YOLOv8: {e}")
            print("[WARN] Falling back to your_detector() path.")
            yolo_model = None

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {VIDEO_SOURCE}")
        return 1

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    fps_avg = 0.0
    alpha = 0.05  # smoothing for FPS

    print("[INFO] Press ESC or 'q' to quit.")
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Empty frame. Exiting.")
            break

        # Get detections from the chosen path
        if yolo_model is not None:
            detections = get_detections_from_yolov8(yolo_model, frame)
        else:
            detections = get_detections_from_your_detector(frame)

        # Update presence trigger
        events = presence.update(detections)  # ['enter'], ['exit'], or []

        # Optional: draw results
        if DRAW:
            for d in detections:
                x1, y1, x2, y2 = map(int, d['bbox'])
                label = str(d['label'])
                conf = float(d['conf'])
                color = (0, 255, 0) if label.lower() == 'person' else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw ROI if set
            if pt_config.ROI is not None:
                rx1, ry1, rx2, ry2 = pt_config.ROI
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)
                cv2.putText(frame, "ROI", (rx1, max(0, ry1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Overlay FPS and last event
            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_avg = (1 - alpha) * fps_avg + alpha * fps
            cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if events:
                cv2.putText(frame, f"EVENT: {','.join(events)}", (10, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
