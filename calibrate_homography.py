import cv2
import json
import numpy as np
import sys
from typing import List, Tuple

OUT_FILE = "cam2world.json"
CAM_INDEX = 0

print("Homography calibration tool\n"
      "Place calibration markers at known world XY (mm) on a flat plane.\n"
      "You'll be asked to enter world coords for each marker, then click the markers in the camera view in the same order.\n")

def ask_world_points(n: int) -> List[Tuple[float,float]]:
    pts = []
    for i in range(n):
        s = input(f"Enter world X,Y for marker #{i+1} in mm (e.g. 150,50): ").strip()
        try:
            x,y = [float(p) for p in s.split(',')]
        except Exception:
            print("Invalid format, try again.")
            return ask_world_points(n)
        pts.append((x,y))
    return pts

def collect_clicks(n: int, cap) -> List[Tuple[int,int]]:
    clicks: List[Tuple[int,int]] = []
    win = "Calib Clicks"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((int(x), int(y)))
            print(f"Clicked #{len(clicks)}: ({x},{y})")

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(f"Click the {n} marker centers in the camera window in the SAME ORDER as the world points entered.")
    while len(clicks) < n:
        ret, frame = cap.read()
        if not ret:
            continue
        for i,(cx,cy) in enumerate(clicks):
            cv2.circle(frame, (cx,cy), 6, (0,255,0), -1)
            cv2.putText(frame, str(i+1), (cx+6,cy+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow(win, frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyWindow(win)
    return clicks

def compute_and_save(src_px, dst_world):
    src = np.array(src_px, dtype=np.float64)
    dst = np.array(dst_world, dtype=np.float64)
    if src.shape[0] < 4:
        print("Need at least 4 non-collinear points for a homography.")
        return False
    H, mask = cv2.findHomography(src, dst, 0)
    if H is None:
        print("findHomography failed.")
        return False
    # compute reprojection error
    pts = src.reshape(-1,1,2).astype(np.float64)
    reproj = cv2.perspectiveTransform(pts, H).reshape(-1,2)
    err = np.linalg.norm(reproj - dst, axis=1)
    print("Reprojection errors (mm):", err.tolist())
    print("RMS error: %.3f mm" % np.sqrt(np.mean(err**2)))
    out = {"H": H.tolist(), "world_points": dst.tolist()}
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved homography -> {OUT_FILE}")
    return True

def visualize_result(cap, H):
    win = "Calib Verify"
    print("Visual verification: click image points to see mapped world coords (ESC to quit).")
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        # allow clicks to be processed by OpenCV; use mouse on separate window if desired
    cv2.destroyWindow(win)

def main():
    try:
        n = int(input("Number of markers (>=4): ").strip())
    except Exception:
        print("Invalid count.")
        return
    if n < 4:
        print("Need >=4 points for homography.")
        return
    print("Enter the world coordinates (X,Y) in mm for each marker (order matters).")
    world_pts = ask_world_points(n)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Camera open failed.")
        return

    px_pts = collect_clicks(n, cap)
    if len(px_pts) != n:
        print("Not enough clicks collected.")
        cap.release()
        return

    ok = compute_and_save(px_pts, world_pts)
    if ok:
        print("Calibration complete. You can now use cam2world.json in your pick-and-place script.")
    cap.release()

if __name__ == "__main__":
    main()