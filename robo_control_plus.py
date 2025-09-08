# robot_controls.py
from __future__ import annotations
import threading
import queue
import time
from typing import Dict, List, Optional
#import keyboard  # for emergency stop key detection

# =========================
# ROBOT CONNECTION CONFIG
# =========================
ROBOT_PORT = "/dev/ttyAMA0"   # or "/dev/ttyUSB0"
ROBOT_BAUD = 115200

# Define your named poses (joint angles in degrees)
POSES: Dict[str, List[float]] = {
#    "HOME": [0, 0, 0, 80, 90, 0],
#    "SAFE": [131, 0, 0, 70, 90, 0],      # <-- Replace with tested safe pose
#    "LEFT": [68, 0, 0, 80, 89, 0],
#    "RIGHT": [121, 0, 0, 80, 100, 50],
}

SPEED = 5         # keep slow for safety (0-100)
DRY_RUN = False     # <-- START HERE TRUE; set False when you're ready

# =========================
# FOLLOW CONTROLLER CONFIG
# =========================
FOLLOW_RATE_HZ = 3.0          # how often we send corrections
YAW_KP_DEG_PER_ERR = 18.0     # deg of joint-1 per unit normalized error ([-1..1])
YAW_MAX_STEP_DEG = 6.0        # max deg per control tick
YAW_LIMITS_DEG = (-130.0, 130.0)  # safety clamp for joint-1 range
FOLLOW_ERR_DEADBAND = 0.10   # ignore small errors 10% of frame width
ERR_SMOOTH_ALPHA = 0.3       # LOWPASS filter alpha for error smoothing
MIN_EFFECTIVE_STEP_DEG = 1.0  # if step smaller than this, skip it


class RobotController:
    """
    Threaded controller to serialize robot motions (queued) and a follow controller
    that applies small yaw corrections at a fixed rate. DRY_RUN skips hardware calls.
    """
    def __init__(self, port: str = ROBOT_PORT, baud: int = ROBOT_BAUD, dry_run: bool = DRY_RUN):
        self.dry_run = dry_run
        self._q: "queue.Queue[tuple[str, Optional[List[float]]]]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._worker = threading.Thread(target=self._run_queue, daemon=True)

        # Follow loop state
        self._follow_enabled = False
        self._follow_err_x = 0.0     # latest normalized horizontal error [-1..1]
        self._follow_lock = threading.Lock()
        self._follow_thread = threading.Thread(target=self._follow_loop, daemon=True)

        self._mc = None
        if not self.dry_run:
            from pymycobot.mycobot import MyCobot
            self._mc = MyCobot(port, baud)
            time.sleep (0.5)  # wait for connection
            try:
                self._mc.power_on()
                time.sleep (0.5)
                self._mc.focus_all_servos()
                print(f"[robot] power on: {self._mc.is_power_on()}")
                print("[robot] ready")
            except Exception as e:
                print(f"[Robot][WARN] Could not init MyCobot: {e}. DRY_RUN enabled.")
                self.dry_run = True
        
        #if keyboard.is_pressed('1'):
        #    g = [121.5,0,0,80.5,100,50]


    # ---------- lifecycle ----------
    def start(self):
        self._worker.start()
        self._follow_thread.start()

    def stop(self):
        self._stop_evt.set()
        try:
            if self._mc:
                self._mc.stop()
        except Exception:
            pass

    # ---------- queue API ----------
    def enqueue(self, name: str):
        """Queue a named pose from POSES dict."""
        self._q.put(("pose", POSES.get(name)))
        print(f"[Robot] Enqueued pose: {name}")

    def enqueue_angles(self, angles: List[float]):
        """Queue an absolute joint move."""
        self._q.put(("angles", angles))
        print(f"[Robot] Enqueued angles: {angles}")

    def emergency_stop(self):
        """Immediate stop + clear queues + disable follow."""
        self.set_follow_enabled(False)
        try:
            if self._mc and not self.dry_run:
                self._mc.stop()
        except Exception:
            pass
        with self._q.mutex:
            self._q.queue.clear()
        print("[Robot] EMERGENCY STOP (queue cleared, follow disabled)")

    # ---------- follow API (called from vision thread) ----------
    def set_follow_enabled(self, enabled: bool):
        with self._follow_lock:
            self._follow_enabled = enabled
        print(f"[Robot] Follow {'ENABLED' if enabled else 'DISABLED'}")

    def update_follow_error(self, err_x: float):
        """err_x: normalized horizontal error [-1..1], + = target to the right of center."""
        with self._follow_lock:
            self._follow_err_x = max(-1.0, min(1.0, float(err_x)))

    # ---------- internals ----------
    def _run_queue(self):
        while not self._stop_evt.is_set():
            try:
                kind, payload = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if kind == "pose":
                    if payload is None:
                        print("[Robot][WARN] Pose not found")
                    else:
                        self._send_angles(payload)
                elif kind == "angles":
                    self._send_angles(payload or [])
            except Exception as e:
                print(f"[Robot][ERR] Move failed: {e}")
            finally:
                self._q.task_done()

    def _follow_loop(self):
        # (inside self._follow_loop)
                period = 1.0 / FOLLOW_RATE_HZ
                err_filt = 0.0
                while not self._stop_evt.is_set():
                    time.sleep(period)
                    with self._follow_lock:
                        enabled = self._follow_enabled
                        err = self._follow_err_x
                    if not enabled:
                        continue
                    # Smooth the error
                    err_filt = (1.0 - ERR_SMOOTH_ALPHA) * err_filt + ERR_SMOOTH_ALPHA * err

                    # Deadband: ignore small errors near center
                    if abs(err_filt) < FOLLOW_ERR_DEADBAND:
                        continue
                    
                    # Proportional step
                    delta = YAW_KP_DEG_PER_ERR * err_filt
                    delta = max(-YAW_MAX_STEP_DEG, min(YAW_MAX_STEP_DEG, delta))

                    # Skip tiny steps (avoid jitter)
                    if abs(delta) < MIN_EFFECTIVE_STEP_DEG:
                        continue
                    
                    if self.dry_run or not self._mc:
                        print(f"[Robot][FOLLOW DRY] err={err:+.2f} filt={err_filt:+.2f} -> Δyaw={delta:+.2f}°")
                        continue
                    
                    try:
                        joints = self._mc.get_angles()
                        if not joints or len(joints) != 6:
                            print("[Robot][WARN] get_angles() failed")
                            continue
                        j1 = joints[0] + delta
                        j1 = max(YAW_LIMITS_DEG[0], min(YAW_LIMITS_DEG[1], j1))
                        #new_angles = [j1, joints[1], joints[2], joints[3], joints[4], joints[5]]
                        new_angles = [j1, joints[1], joints[2], 75, 90, joints[5]]
                        self._mc.send_angles(new_angles, SPEED)
                    except Exception as e:
                        print(f"[Robot][WARN] Follow step failed: {e}")


    def _send_angles(self, angles: List[float]):
        if len(angles) != 6:
            print(f"[Robot][WARN] Expected 6 joints, got {len(angles)}")
            return
        if self.dry_run or not self._mc:
            print(f"[Robot][DRY] send_angles({angles}, speed={SPEED})")
            time.sleep(0.1)
            return
        self._mc.send_angles(angles, SPEED)
