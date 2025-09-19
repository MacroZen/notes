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

SPEED = 50         # keep slow for safety (0-100)
DRY_RUN = False     # <-- START HERE TRUE; set False when you're ready

# Ensure robot moves to an initial camera pose on start to avoid sag after manual moves
INITIAL_POSE = [0,0,0,0,-100,0]
INITIAL_SPEED = 30
#0,90,-90,0,-90,0]


# =========================
# FOLLOW CONTROLLER CONFIG
# =========================
FOLLOW_RATE_HZ = 8.0          # how often we send corrections
YAW_KP_DEG_PER_ERR = 18.0     # deg of joint-1 per unit normalized error ([-1..1])
YAW_MAX_STEP_DEG = 6.0        # max deg per control tick
YAW_LIMITS_DEG = (-160.0, 160.0)  # safety clamp for joint-1 range
FOLLOW_ERR_DEADBAND = 0.10   # ignore small errors 10% of frame width
ERR_SMOOTH_ALPHA = 0.3       # LOWPASS filter alpha for error smoothing
MIN_EFFECTIVE_STEP_DEG = 1.0  # if step smaller than this, skip it

# safety joint limits per-joint (deg): [(min,max), ...]
JOINT_LIMITS_DEG = [
    YAW_LIMITS_DEG,    # joint-0 (yaw)
    (-90.0, 90.0),     # joint-1
    (-90.0, 90.0),     # joint-2
    (-90.0, 90.0),      # joint-3
    (-90.0, 90.0),      # joint-4
    (-90.0, 90.0),     # joint-5
]


class RobotController:
    """
    Threaded controller to serialize robot motions (queued) and a follow controller
    that applies small yaw corrections at a fixed rate. DRY_RUN skips hardware calls.
    """
    def __init__(self, port: str = ROBOT_PORT, baud: int = ROBOT_BAUD, dry_run: bool = DRY_RUN):
        self.dry_run = dry_run
        # LED color state (R,G,B) - keep green by default (unlocked)
        self._led_color = (0, 255, 0)
        # use an untyped Queue to avoid fragile generic typing mismatches
        self._q = queue.Queue()
        self._stop_evt = threading.Event()
        self._worker = threading.Thread(target=self._run_queue, daemon=True)

        # Follow loop state
        self._follow_enabled = False
        self._follow_err_x = 0.0     # latest normalized horizontal error [-1..1]
        self._follow_lock = threading.Lock()
        self._follow_thread = threading.Thread(target=self._follow_loop, daemon=True)

        # last enqueued joint-0 angle (to avoid spamming the same command when HW hasn't updated yet)
        self._last_enqueued_j0: Optional[float] = None

        self._mc = None
        if not self.dry_run:
            from pymycobot.mycobot import MyCobot
            self._mc = MyCobot(port, baud) # type: ignore
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
        # Power/focus servos and move to a known initial pose to ensure holding torque
        try:
            if self._mc and not self.dry_run:
                try:
                    self._mc.power_on()
                except Exception:
                    pass
                try:
                    self._mc.focus_all_servos()
                except Exception:
                    pass
        except Exception as e:
            print(f"[Robot][WARN] startup focus/power failed: {e}")

        # Disable follow while moving to initial pose and enqueue the pose
        self.set_follow_enabled(False)
        self.enqueue_angles(INITIAL_POSE, INITIAL_SPEED)
        # apply initial LED color (best-effort)
        try:
            self.set_status_color(*self._led_color)
        except Exception:
            pass

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

    def enqueue_angles(self, angles: List[float], speed: Optional[int] = None):
        """Queue an absolute joint move. Optional speed overrides default SPEED for this move."""
        self._q.put(("angles", (angles, speed)))
        print(f"[Robot] Enqueued angles: {angles} speed={speed or SPEED}")
        # remember the last requested joint-0 so follow loop doesn't repeatedly enqueue the same target
        try:
            if isinstance(angles, (list, tuple)) and len(angles) >= 1:
                with self._follow_lock:
                    self._last_enqueued_j0 = float(angles[0])
        except Exception:
            pass

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
                        self._send_angles(payload, None)
                elif kind == "angles":
                    angs, sp = payload if isinstance(payload, tuple) else (payload, None)
                    self._send_angles(angs or [], sp)
            except Exception as e:
                print(f"[Robot][ERR] Move failed: {e}")
            finally:
                self._q.task_done()

    def _follow_loop(self):
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
            
            # Proportional step (note sign): positive err -> target is right of center
            # we invert sign so positive error causes corrective pan toward the target
            delta = -YAW_KP_DEG_PER_ERR * err_filt
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
                old_j0 = float(joints[0])
                new_j0 = old_j0 + delta
                new_j0 = max(YAW_LIMITS_DEG[0], min(YAW_LIMITS_DEG[1], new_j0))
                # build the command using INITIAL_POSE for joints 1..5 and updated joint-0
                try:
                    base = list(INITIAL_POSE)
                except Exception:
                    base = [0.0, 0.0, 0.0, 70.0, 90.0, 0.0]
                new_angles = [new_j0, base[1], base[2], base[3], base[4], base[5]]
                # concise debug to help tune control loop
                print(f"[Robot][FOLLOW] err={err:+.2f} filt={err_filt:+.2f} Δ={delta:+.2f} -> j0 {old_j0:+.2f}->{new_j0:+.2f} enqueuing {new_angles}")
                # avoid spamming the same j0 if we already enqueued it recently
                do_enqueue = True
                with self._follow_lock:
                    last = self._last_enqueued_j0
                    if last is not None and abs(last - new_j0) <= max(0.5, MIN_EFFECTIVE_STEP_DEG):
                        do_enqueue = False
                if not do_enqueue:
                    print(f"[Robot][FOLLOW] Skipping enqueue: already requested j0={last:+.2f}")
                    continue

                # enqueue the new target using the same queue path (this keeps motion serialized)
                try:
                    self.enqueue_angles(new_angles, speed=20)
                except Exception as e:
                    print(f"[Robot][ERR] Failed to enqueue follow move: {e}")
            except Exception as e:
                # catch any unexpected errors in the follow loop's outer try
                print(f"[Robot][ERR] Follow loop error: {e}")
                continue

    # helper: clamp angles to JOINT_LIMITS_DEG and ensure 6-element list
    def _clamp_angles(self, angles: List[float]) -> List[float]:
        out = []
        base = list(INITIAL_POSE)
        for i in range(6):
            v = float(angles[i]) if i < len(angles) else float(base[i])
            if i < len(JOINT_LIMITS_DEG):
                lo, hi = JOINT_LIMITS_DEG[i]
                v = max(lo, min(hi, v))
            out.append(v)
        return out

    def _send_angles(self, angles: List[float], speed: Optional[int] = None):
        """Send a clamped 6-joint command to the robot. Ensures power/focus and handles DRY_RUN."""
        if angles is None:
            return
        try:
            angs = [float(x) for x in angles]
        except Exception:
            print(f"[Robot][ERR] invalid angles payload: {angles}")
            return

        # ensure 6 joints and clamp to limits
        angs = self._clamp_angles(angs)

        sp = int(speed) if speed is not None else SPEED

        if self.dry_run or not self._mc:
            print(f"[Robot][DRY] would send angles: {angs} speed={sp}")
            return

        # Ensure servos are powered and focused before commanding (best-effort)
        try:
            try:
                self._mc.power_on()
            except Exception:
                pass
            try:
                self._mc.focus_all_servos()
            except Exception:
                pass
            # send to hardware
            try:
                # pymycobot.MyCobot.send_angles expects (angles, speed)
                self._mc.send_angles(angs, sp)  # type: ignore
                print(f"[Robot] Sent angles: {angs} speed={sp}")
                # reapply stored LED color after motion (some firmwares reset it)
                try:
                    if hasattr(self._mc, 'set_color') and self._led_color is not None:
                        r,g,b = self._led_color
                        try:
                            self._mc.set_color(int(r), int(g), int(b))
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception as e:
                print(f"[Robot][ERR] send_angles failed: {e}")
        except Exception as e:
            print(f"[Robot][ERR] _send_angles unexpected error: {e}")
    
    def set_status_color(self, r: int, g: int, b: int):
        """Store desired LED color and apply it to the robot (best-effort)."""
        try:
            self._led_color = (int(r), int(g), int(b))
        except Exception:
            return
        if self.dry_run or not self._mc:
            # don't attempt hardware if dry_run
            return
        try:
            if hasattr(self._mc, 'set_color'):
                try:
                    self._mc.set_color(int(r), int(g), int(b))
                except Exception:
                    pass
        except Exception:
            pass