# robot_controls.py
from __future__ import annotations
import threading
import queue
import time
from typing import Dict, List, Optional

# =========================
# CONFIG
# =========================
# Adjust the port/baud for your MyCobot 320 Pi setup
ROBOT_PORT = "/dev/ttyAMA0"   # or "/dev/ttyUSB0" if using USB adapter
ROBOT_BAUD = 115200

# Define your named poses (joint angles in degrees)
POSES: Dict[str, List[float]] = {
    "DOOR": [0, 0, 0, 80, 90, 0],
    "DYLAN": [131, 0, 0, 70, 90, 0],      # <-- Replace with your validated safe pose
    "SAL": [68, 0, 0, 80, 90, 0],    # <-- Example
    "LEO": [122, 0, 0, 80, 100, 50],    # <-- Example
    # Add more named poses here
}

SPEED = 5        # joint speed (0-100); keep slow while testing
DRY_RUN = True   # True = don't talk to robot, just log (safe for first tests)


class RobotController:
    """
    Threaded controller to serialize robot motions and keep detect loop responsive.
    Use: rc.enqueue('HOME') or rc.enqueue_angles([...])
    """
    def __init__(self, port: str = ROBOT_PORT, baud: int = ROBOT_BAUD, dry_run: bool = DRY_RUN):
        self.dry_run = dry_run
        self._q: "queue.Queue[tuple[str, Optional[List[float]]]]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._mc = None
        if not self.dry_run:
            try:
                from pymycobot.mycobot import MyCobot
                self._mc = MyCobot(port, baud)
                print(f"[Robot] Connected on {port} @ {baud}")
            except Exception as e:
                print(f"[Robot][WARN] Could not init MyCobot: {e}. Running DRY_RUN mode.")
                self.dry_run = True

    def start(self):
        self._worker.start()

    def stop(self):
        self._stop_evt.set()
        # Optionally send a stop command
        try:
            if self._mc:
                self._mc.stop()
        except Exception:
            pass

    def enqueue(self, name: str):
        """Queue a named pose from POSES dict."""
        self._q.put(("pose", POSES.get(name)))
        print(f"[Robot] Enqueued pose: {name}")

    def enqueue_angles(self, angles: List[float]):
        """Queue a raw angles move."""
        self._q.put(("angles", angles))
        print(f"[Robot] Enqueued angles: {angles}")

    def emergency_stop(self):
        """Try to halt motion immediately."""
        try:
            if self._mc and not self.dry_run:
                self._mc.stop()
        except Exception:
            pass
        # Flush queue
        with self._q.mutex:
            self._q.queue.clear()
        print("[Robot] EMERGENCY STOP (queue cleared)")

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                kind, payload = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if kind == "pose":
                    if payload is None:
                        print("[Robot][WARN] Pose not found, skipping.")
                        continue
                    self._send_angles(payload)
                elif kind == "angles":
                    self._send_angles(payload or [])
            except Exception as e:
                print(f"[Robot][ERR] Move failed: {e}")
            finally:
                self._q.task_done()

    def _send_angles(self, angles: List[float]):
        if len(angles) != 6:
            print(f"[Robot][WARN] Expected 6 joints, got {len(angles)}")
            return
        if self.dry_run:
            print(f"[Robot][DRY] send_angles({angles}, speed={SPEED})")
            time.sleep(0.1)
            return
        self._mc.send_angles(angles, SPEED)
        # Optional wait until stopped or a short delay:
        time.sleep(0.05)  # don't hog the thread
