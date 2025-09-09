"""
presence_trigger.py
-------------------
A reusable 'presence trigger' that fires callbacks when a PERSON appears
or disappears in the camera view.

Expected detector output per frame:
    detections = [
        {'label': 'person', 'conf': 0.87, 'bbox': (x1, y1, x2, y2)},
        {'label': 'dog', 'conf': 0.56, 'bbox': (..)},
        ...
    ]

How it works:
- Debounces: requires N consecutive frames with person before firing 'enter'
- Debounces: requires M consecutive frames without person before firing 'exit'
- Optional cooldown: limits how often 'enter' can fire
- Optional ROI: only trigger if person’s bbox center falls within ROI
"""

from __future__ import annotations
import time
from typing import Dict, List, Tuple, Optional

# =========================
# CONFIG (tune as needed)
# =========================
# Acceptable label names for "person" from your model:
PERSON_CLASS_NAMES = {"person", "Person", "human"}

CONF_THRESH: float = 0.5      # minimum confidence to consider the detection valid
ENTER_PERSISTENCE: int = 3    # frames with person required to confirm ENTER
EXIT_PERSISTENCE: int = 5     # frames without person required to confirm EXIT
COOLDOWN_S: float = 10.0      # min seconds between ENTER events (prevents re-trigger spam)

# Optional Region of Interest (x1, y1, x2, y2). If None, ROI is disabled.
ROI: Optional[Tuple[int, int, int, int]] = None
# Example: ROI = (100, 100, 540, 380)

# =========================
# Types
# =========================
Detection = Dict[str, object]  # {'label': str, 'conf': float, 'bbox': (x1,y1,x2,y2)}


def bbox_center(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def in_roi(bbox: Tuple[float, float, float, float], roi: Optional[Tuple[int, int, int, int]]) -> bool:
    if roi is None:
        return True
    cx, cy = bbox_center(bbox)
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2


# =========================
# Your action hooks
# =========================
def on_person_enter(detections: List[Detection]) -> None:
    """
    Put your 'do something' here (fires once when a person FIRST appears).
    Examples:
      - Toggle a GPIO pin
      - Send an HTTP request
      - Start recording
      - Move a robot to a safe pose

    Args:
        detections: the full detections list from this frame (could contain more than just 'person').
    """
    print("[EVENT] Person ENTER")

    # ---- Example GPIO (Raspberry Pi) ----
    # from gpiozero import LED
    # led = LED(17)
    # led.on()

    # ---- Example: MyCobot move (safety first, validate before enabling) ----
    # robot_move_to_safe_pose()

    # ---- Example: Webhook ----
    # import requests
    # try:
    #     requests.post("http://localhost:5000/notify", json={"event": "enter"})
    # except Exception as e:
    #     print(f"[WARN] Webhook failed: {e}")


def on_person_exit() -> None:
    """
    Optional: fires once when the person has been absent for EXIT_PERSISTENCE frames.
    """
    print("[EVENT] Person EXIT")

    # ---- Example GPIO ----
    # led = LED(17)
    # led.off()

    # ---- Example: Webhook ----
    # import requests
    # try:
    #     requests.post("http://localhost:5000/notify", json={"event": "exit"})
    # except Exception as e:
    #     print(f"[WARN] Webhook failed: {e}")


# =========================
# (Optional) Robot helper
# =========================
def robot_move_to_safe_pose() -> None:
    """
    Example: move MyCobot to a safe pose at slow speed.
    Adjust port/baud/angles for your setup; validate safety before enabling.
    """
    try:
        from pymycobot.mycobot import MyCobot
        mc = MyCobot('/dev/ttyAMA0', 115200)  # type: ignore # adapt if using USB or different port
        #safe_angles = [0, 0, 0, 0, 0, 0]      # <-- replace with your tested safe angles
        #mc.send_angles(safe_angles, 20)       # move slowly
    except Exception as e:
        print(f"[WARN] Robot move skipped: {e}")


# =========================
# Core trigger logic
# =========================
class PresenceTrigger:
    """
    Tracks whether a 'person' is present based on per-frame detections
    and fires enter/exit events with debouncing and cooldown.
    """
    def __init__(self):
        self.present_counter = 0
        self.absent_counter = 0
        self.person_present = False
        self.last_enter_ts = 0.0

    def update(self, detections: List[Detection]) -> List[str]:
        """
        Process detections for the current frame.

        Args:
            detections: list of dicts:
                {
                  'label': str,
                  'conf': float,
                  'bbox': (x1, y1, x2, y2)
                }

        Returns:
            events: list of events fired this call. e.g. [], ['enter'], ['exit']
        """
        # Is a qualifying person detected in this frame?
        person_now = any(
            (str(d.get('label')) in PERSON_CLASS_NAMES) and
            (float(d.get('conf', 0.0)) >= CONF_THRESH) and # type: ignore
            in_roi(tuple(d.get('bbox')), ROI)  # type: ignore[arg-type]
            for d in detections
            if 'bbox' in d
        )

        events: List[str] = []

        if person_now:
            self.present_counter += 1
            self.absent_counter = 0
        else:
            self.absent_counter += 1
            self.present_counter = 0

        # Rising edge (ENTER)
        if not self.person_present and self.present_counter >= ENTER_PERSISTENCE:
            now = time.time()
            if now - self.last_enter_ts >= COOLDOWN_S:
                self.person_present = True
                self.last_enter_ts = now
                on_person_enter(detections)
                events.append('enter')

        # Falling edge (EXIT)
        if self.person_present and self.absent_counter >= EXIT_PERSISTENCE:
            self.person_present = False
            on_person_exit()
            events.append('exit')

        return events
