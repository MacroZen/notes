"""
Simplified interactive recorder for world points (robot-frame X,Y in mm).

Usage:
 - Run with robot connected.
 - Keys:
    w/s : +Y / -Y
    a/d : -X / +X
    r/f : +Z / -Z
    q/e : +Rz / -Rz (deg)
    p   : print full pose
    SPACE/ENTER : record current (X,Y) and save
    h   : help
    x/ESC : exit (saves if any)
 - Adjust TRANSLATE_STEP, ROT_STEP, SPEED for safety.
"""
import json
import sys
import time
import termios
import tty
from typing import List, Tuple
from pymycobot.mycobot import MyCobot

OUT = "world_points.json"
PORT = "/dev/ttyAMA0"
BAUD = 115200

TRANSLATE_STEP = 10.0
ROT_STEP = 10.0
SPEED = 50  # integer speed

def save_points(pts: List[Tuple[float, float]]):
    with open(OUT, "w") as f:
        json.dump({"world_pts": pts}, f, indent=2)
    print(f"\n[SAVE] {len(pts)} points -> {OUT}")

def print_help():
    print("""Controls:
  w/s : +Y / -Y
  a/d : -X / +X
  r/f : +Z / -Z
  q/e : +Rz / -Rz (deg)
  p   : print current pose
  SPACE / ENTER : record current X,Y and save
  h   : show this help
  x / ESC : exit
""")

def read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def try_send_coords(mc, *args):
    """Try common variants of send_coords; include array-style ([x,y,z,rx,ry,rz], speed, mode)."""
    if mc is None:
        return
    args = list(args)
    # coerce speed to int if present as last arg
    speed_int = None
    if len(args) >= 1:
        try:
            speed_int = int(round(float(args[-1])))
        except Exception:
            speed_int = None

    # prepare coords list
    coords = None
    if len(args) >= 6:
        coords = [float(args[0]), float(args[1]), float(args[2]),
                  float(args[3]), float(args[4]), float(args[5])]

    variants = []
    # array-style variants many MyCobot libs use
    if coords is not None:
        if speed_int is not None:
            variants.append((coords, speed_int))         # send_coords([coords], speed)
            variants.append((coords, speed_int, 0))      # send_coords([coords], speed, mode)
        variants.append((coords,))                       # send_coords([coords])

    # fall back to positional variants
    variants += [
        tuple(args),                                     # as given
        tuple(args[:-1]) if len(args) > 1 else (),       # drop last (speed)
        (args[0], args[1], args[2], int(SPEED)) if len(args) >= 3 else (),
        (args[0], args[1], args[2]) if len(args) >= 3 else (),
    ]

    last_exc = None
    for v in variants:
        if not v:
            continue
        try:
            return mc.send_coords(*v)
        except TypeError as e:
            last_exc = e
            continue
        except Exception:
            raise
    raise TypeError(f"send_coords failed; tried variants. Last: {last_exc}")

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0

def interactive_record(port: str = PORT, baud: int = BAUD):
    try:
        mc = MyCobot(port, baud)
    except Exception as e:
        print(f"[ERR] open {port}@{baud} failed: {e}")
        return

    pts: List[Tuple[float, float]] = []
    try:
        with open(OUT, "r") as f:
            data = json.load(f)
            pts = [tuple(x) for x in data.get("world_pts", [])]
            if pts:
                print(f"[LOAD] {len(pts)} existing points")
    except Exception:
        pass

    print_help()
    try:
        while True:
            try:
                coords = mc.get_coords()  # [x,y,z,rx,ry,rz]
                x = safe_float(coords[0]); y = safe_float(coords[1])
                z = safe_float(coords[2]); rx = safe_float(coords[3])
                ry = safe_float(coords[4]); rz = safe_float(coords[5])
            except Exception as e:
                print(f"[WARN] get_coords failed: {e}")
                time.sleep(0.5)
                continue

            print(f"[POS] X={x:.1f} Y={y:.1f} Z={z:.1f} Rz={rz:.1f}", end="\r", flush=True)
            k = read_key()
            if k in ('h','H'):
                print_help()
            elif k == 'p':
                print(f"\n[POSE] X={x:.1f} Y={y:.1f} Z={z:.1f} rx={rx:.1f} ry={ry:.1f} rz={rz:.1f}")
            elif k in ('w','W'):
                y += TRANSLATE_STEP; try_send_coords(mc, x, y, z, rx, ry, rz, int(SPEED))
            elif k in ('s','S'):
                y -= TRANSLATE_STEP; try_send_coords(mc, x, y, z, rx, ry, rz, int(SPEED))
            elif k in ('a','A'):
                x -= TRANSLATE_STEP; try_send_coords(mc, x, y, z, rx, ry, rz, int(SPEED))
            elif k in ('d','D'):
                x += TRANSLATE_STEP; try_send_coords(mc, x, y, z, rx, ry, rz, int(SPEED))
            elif k in ('r','R'):
                z += TRANSLATE_STEP; try_send_coords(mc, x, y, z, rx, ry, rz, int(SPEED))
            elif k in ('f','F'):
                z -= TRANSLATE_STEP; try_send_coords(mc, x, y, z, rx, ry, rz, int(SPEED))
            elif k in ('q','Q'):
                rz += ROT_STEP; try_send_coords(mc, x, y, z, rx, ry, rz, int(SPEED))
            elif k in ('e','E'):
                rz -= ROT_STEP; try_send_coords(mc, x, y, z, rx, ry, rz, int(SPEED))
            elif k in (' ', '\r', '\n'):
                pts.append((round(x,2), round(y,2))); save_points(pts); print(f"[REC] #{len(pts)} X={x:.1f} Y={y:.1f}")
            elif k in ('x','X','\x1b'):
                print("\n[EXIT]"); 
                if pts: save_points(pts)
                break
            else:
                pass
            time.sleep(0.05)
    finally:
        try:
            mc.release_all_servos()
        except Exception:
            pass

if __name__ == "__main__":
    interactive_record()