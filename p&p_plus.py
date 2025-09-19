#!/usr/bin/env python3
# pick-and-place macro for MyCobot 320pi
# - safe by default (DRY_RUN=True)
# - simple color-based object detection (tune HSV ranges)
# - map camera pixel -> robot world with a user-configurable linear transform or homography (calibrate!)
# - numbered selection: press 1..9 to pick that object and place to next location

import time
import cv2
import numpy as np
from typing import List, Tuple
from robo_control_plus import RobotController, INITIAL_POSE

# Safety: keep True for testing so hardware isn't commanded accidentally
DRY_RUN = True
SHOW_MASK = True

# Camera / detection params
VIDEO_SOURCE = 0
MIN_CONTOUR_AREA = 400
# HSV range for object color (default tuned for red-ish objects) - feel free to modify
HSV_LOWER = np.array([0, 100, 100])
HSV_UPPER = np.array([10, 255, 255])

# Places (world coordinates in mm) - update to suit your workspace
PLACE_LOCATIONS: List[Tuple[float, float, float]] = [
    (150.0, 0.0, 50.0),
    (200.0, 0.0, 50.0),
    (250.0, 0.0, 50.0),
]

# Robot approach heights (mm)
APPROACH_Z = 120.0
PICK_Z = 40.0
LIFT_Z = 120.0
PLACE_Z = 50.0

# Camera -> Robot mapping (naive linear mapping). You should calibrate this for accurate picks.
# Two-point scale + offset mapping: world_x = px * S_X + O_X ; world_y = py * S_Y + O_Y
# Default values are placeholders and will almost certainly need calibration.
CAM_MAP_SCALE_X = 0.5  # mm per pixel (X)
CAM_MAP_SCALE_Y = -0.5 # mm per pixel (Y) (negative if camera axes differ)
CAM_MAP_OFFSET_X = 150.0
CAM_MAP_OFFSET_Y = 100.0

# Utility: map pixel to robot XY
def pixel_to_world(px: int, py: int) -> Tuple[float, float]:
    wx = float(px) * CAM_MAP_SCALE_X + CAM_MAP_OFFSET_X
    wy = float(py) * CAM_MAP_SCALE_Y + CAM_MAP_OFFSET_Y
    return wx, wy

# Gripper adapter: best-effort methods for common pymycobot variants
class GripperAdapter:
    def __init__(self, mc):
        self.mc = mc
        self._open_cmd = None
        self._close_cmd = None
        # detect likely methods
        if mc is None:
            return
        if hasattr(mc, 'set_gripper'):
            # expects (value, ) or (value, speed)
            self._open_cmd = lambda: mc.set_gripper(0)
            self._close_cmd = lambda: mc.set_gripper(50)
        elif hasattr(mc, 'set_gripper_value'):
            self._open_cmd = lambda: mc.set_gripper_value(0)
            self._close_cmd = lambda: mc.set_gripper_value(50)
        elif hasattr(mc, 'set_tool_state'):
            self._open_cmd = lambda: mc.set_tool_state(0)
            self._close_cmd = lambda: mc.set_tool_state(1)
        else:
            # fallback: try servo on a channel if available
            if hasattr(mc, 'set_pin'):
                self._open_cmd = lambda: mc.set_pin(0, 0)
                self._close_cmd = lambda: mc.set_pin(0, 1)

    def open(self):
        if self.mc is None:
            print("[GRIP] (dry) open")
            return
        if self._open_cmd:
            try:
                self._open_cmd()
                time.sleep(0.2)
            except Exception as e:
                print("[GRIP] open failed:", e)
        else:
            print("[GRIP] No known gripper method available on mc")

    def close(self):
        if self.mc is None:
            print("[GRIP] (dry) close")
            return
        if self._close_cmd:
            try:
                self._close_cmd()
                time.sleep(0.2)
            except Exception as e:
                print("[GRIP] close failed:", e)
        else:
            print("[GRIP] No known gripper method available on mc")


def detect_objects(frame) -> List[Tuple[int,int,int,int]]:
    """Return list of bounding boxes (x,y,w,h) for detected objects (improved red handling)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (7, 7), 0)

    # red wraps around hue=0 -> use two ranges
    lower1 = np.array([0, 60, 60])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 60, 60])
    upper2 = np.array([180, 255, 255])
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    # morphological: close to fill holes, open to remove small noise, then dilate to merge fragments
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)

    try:
        cv2.imshow('Mask', mask)
    except Exception:
        pass

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    areas = [int(cv2.contourArea(c)) for c in contours]
    if len(contours) == 0:
        print("[DETECT] no contours found")
    for cnt, area in zip(contours, areas):
        if area < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        out.append((x, y, w, h))

    if len(out) == 0 and len(areas) > 0:
        print(f"[DETECT] contours present but filtered by MIN_CONTOUR_AREA={MIN_CONTOUR_AREA}; areas={areas[:6]}")

    out.sort(key=lambda b: b[0])
    return out


def perform_pick_and_place(mc, gripper: GripperAdapter, world_x: float, world_y: float,
                           pick_z: float, place_pos: Tuple[float,float,float], speed=50):
    """High-level pick and place sequence using either send_coords or queued joint moves.
    mc: low-level MyCobot instance (may be None in DRY_RUN)
    gripper: GripperAdapter
    world_x, world_y: coordinates in robot/world space
    pick_z: target Z for picking
    place_pos: tuple (x,y,z)
    """
    # helper to move to coords if supported
    def move_to_coords(x,y,z, speed=speed):
        if DRY_RUN or mc is None:
            print(f"[ROBOT] (dry) move_to_coords {x:.1f},{y:.1f},{z:.1f}")
            return
        # many pymycobot variants expose send_coords(x,y,z, rx, ry, rz, speed)
        if hasattr(mc, 'send_coords'):
            try:
                mc.send_coords(x, y, z, 0, 0, 0, speed)
                time.sleep(0.6)
                return
            except Exception as e:
                print("[ROBOT] send_coords failed:", e)
        # fallback: try move_arm or other names
        if hasattr(mc, 'move_coords'):
            try:
                mc.move_coords(x, y, z, 0, 0, 0, speed)
                time.sleep(0.6)
                return
            except Exception as e:
                print("[ROBOT] move_coords failed:", e)
        # last resort: no coordinate move available — user must provide calibration to joint angles
        print("[ROBOT] No coords API available on mc; cannot move in cartesian space.")

    # approach above object
    move_to_coords(world_x, world_y, APPROACH_Z)
    move_to_coords(world_x, world_y, pick_z)
    # close gripper
    gripper.close()
    time.sleep(0.4)
    # lift
    move_to_coords(world_x, world_y, LIFT_Z)
    # move to place
    px, py, pz = place_pos
    move_to_coords(px, py, APPROACH_Z)
    move_to_coords(px, py, pz)
    gripper.open()
    time.sleep(0.3)
    move_to_coords(px, py, APPROACH_Z)
    print("[P&P] Done")


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERR] Camera open failed")
        return

    # prepare robot controller / low-level mc
    rc = None
    mc = None
    gripper = GripperAdapter(None)
    if RobotController is not None:
        try:
            rc = RobotController(dry_run=DRY_RUN)
            rc.start()
            mc = rc._mc
            gripper = GripperAdapter(mc)
            print("[ROBOT] RobotController started")
        except Exception as e:
            print("[ROBOT] Could not start RobotController:", e)
            rc = None
    else:
        print("[ROBOT] RobotController unavailable; running in dry/mock mode")

    place_idx = 0

    print("Press number key corresponding to object (1..9) to pick it and place to next location. 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = detect_objects(frame)

        # draw boxes and numbers
        for i, (x,y,w,h) in enumerate(boxes[:9]):
            cx = int(x + w/2)
            cy = int(y + h/2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{i+1}", (x+4, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.circle(frame, (cx,cy), 3, (0,0,255), -1)

        cv2.imshow("PickPlace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        # numeric keys 1..9
        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(boxes):
                x,y,w,h = boxes[idx]
                cx = int(x + w/2)
                cy = int(y + h/2)
                wx, wy = pixel_to_world(cx, cy)
                print(f"[PICK] Selected object #{idx+1} px=({cx},{cy}) -> world=({wx:.1f},{wy:.1f})")
                # pick and place
                place_pos = PLACE_LOCATIONS[place_idx % len(PLACE_LOCATIONS)]
                perform_pick_and_place(mc, gripper, wx, wy, PICK_Z, place_pos)
                place_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    if rc is not None:
        rc.stop()


if __name__ == '__main__':
    main()