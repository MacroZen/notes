import json
import time
import sys
import termios
import tty
from typing import List, Tuple
import inspect

from pymycobot.mycobot import MyCobot

OUT = "world_points.json"

# jog params (tune for safety)
TRANSLATE_STEP = 10.0   # mm per key press
ROT_STEP = 10.0         # deg per key press
SPEED = 50              # motion SPEED (0-100)

def save_points(pts: List[Tuple[float, float]]):
    with open(OUT, "w") as f:
        json.dump({"world_pts": pts}, f, indent=2)
    print(f"[SAVE] {len(pts)} points -> {OUT}")

def print_help():
    print("""
Controls:
  w/s : +Y / -Y
  a/d : -X / +X
  r/f : +Z / -Z
  q/e : +Rz / -Rz (degrees)
  p   : print current coords
  SPACE / ENTER : record current X,Y and save
  h   : show this help
  x / ESC : exit
""")

def read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # read potential escape sequence
        if ch == '\x1b':
            # try to read two more bytes (arrow keys), but we don't use them here
            ch += sys.stdin.read(2)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0

def safe_send_coords(mc, x, y, z, rx, ry, rz, speed):
    """
    Robust caller for mc.send_coords. Coerces speed to int and tries keyword
    and positional variants until one succeeds.
    """
    if mc is None:
        return
    fn = getattr(mc, "send_coords", None)
    if fn is None:
        return

    # coerce speed to int to satisfy firmwares that require integer speed
    try:
        speed_int = int(round(float(speed)))
    except Exception:
        try:
            speed_int = int(speed)
        except Exception:
            speed_int = 0

    # Prepare common positional variants (most to least specific)
    variants = [
        (x, y, z, rx, ry, rz, speed_int),  # full 7-arg form
        (x, y, z, rx, ry, rz),             # without speed
        (x, y, z, rx, ry),                 # 5 args
        (x, y, z, rx),                     # 4 args
        (x, y, z, speed_int),              # coords + speed (4 args)
        (x, y, z),                         # 3 args
    ]

    # Try keyword form first if send_coords accepts a 'speed' kwarg
    try:
        sig = inspect.signature(fn)
        params = [p for p in sig.parameters.keys() if p != 'self']
        if 'speed' in params:
            try:
                return fn(x, y, z, rx, ry, rz, speed=speed_int)
            except TypeError:
                # some implementations might expect fewer positional args plus speed kw
                try_args = (x, y, z)
                try:
                    return fn(*try_args, speed=speed_int)
                except TypeError:
                    pass
    except Exception:
        # ignore signature introspection errors and fall back to positional attempts
        pass

    # Try positional variants in order
    last_exc = None
    for args in variants:
        try:
            return fn(*args)
        except TypeError as e:
            last_exc = e
            continue
        except Exception:
            # propagate hardware/runtime exceptions
            raise

    # If we reach here nothing matched; raise a helpful error
    raise TypeError(f"send_coords call failed; tried variants and speed={speed_int}. Last error: {last_exc}")

def interactive_record(port="/dev/ttyAMA0", baud=115200, start_n=4):
    mc = None
    try:
        mc = MyCobot(port, baud)
    except Exception as e:
        print(f"[WARN] Can't open MyCobot on {port}@{baud}: {e}")
        return

    print("Connected to MyCobot. Press 'h' for help.")
    pts: List[Tuple[float, float]] = []
    # load existing if present
    try:
        with open(OUT, "r") as f:
            data = json.load(f)
            pts = [tuple(x) for x in data.get("world_pts", [])]
            if pts:
                print(f"[LOAD] {len(pts)} existing points loaded from {OUT}")
    except Exception:
        pass

    print_help()
    try:
        while True:
            # get current coords
            try:
                coords = mc.get_coords()  # [x,y,z,rx,ry,rz]
                x = safe_float(coords[0])
                y = safe_float(coords[1])
                z = safe_float(coords[2])
                rx = safe_float(coords[3])
                ry = safe_float(coords[4])
                rz = safe_float(coords[5])
            except Exception as e:
                print(f"[ERR] get_coords failed: {e}")
                time.sleep(0.5)
                continue

            print(f"[POS] X={x:.1f} Y={y:.1f} Z={z:.1f} Rz={rz:.1f}", end="\r", flush=True)
            key = read_key()

            if key in ('h', 'H'):
                print_help()
            elif key == 'p':
                print(f"\n[POS] X={x:.1f} Y={y:.1f} Z={z:.1f} rx={rx:.1f} ry={ry:.1f} rz={rz:.1f}")
            elif key in ('w', 'W'):
                y += TRANSLATE_STEP
                safe_send_coords(mc, x, y, z, rx, ry, rz, SPEED)
            elif key in ('s', 'S'):
                y -= TRANSLATE_STEP
                safe_send_coords(mc, x, y, z, rx, ry, rz, SPEED)
            elif key in ('a', 'A'):
                x -= TRANSLATE_STEP
                safe_send_coords(mc, x, y, z, rx, ry, rz, SPEED)
            elif key in ('d', 'D'):
                x += TRANSLATE_STEP
                safe_send_coords(mc, x, y, z, rx, ry, rz, SPEED)
            elif key in ('r', 'R'):
                z += TRANSLATE_STEP
                safe_send_coords(mc, x, y, z, rx, ry, rz, SPEED)
            elif key in ('f', 'F'):
                z -= TRANSLATE_STEP
                safe_send_coords(mc, x, y, z, rx, ry, rz, SPEED)
            elif key in ('q', 'Q'):
                rz += ROT_STEP
                safe_send_coords(mc, x, y, z, rx, ry, rz, SPEED)
            elif key in ('e', 'E'):
                rz -= ROT_STEP
                safe_send_coords(mc, x, y, z, rx, ry, rz, SPEED)
            elif key in (' ', '\r', '\n'):  # record current X,Y
                pts.append((round(x, 2), round(y, 2)))
                save_points(pts)
                print(f"[REC] Recorded #{len(pts)}: X={x:.1f} Y={y:.1f}")
            elif key in ('x', 'X', '\x1b'):  # exit on x or ESC
                print("\nExiting.")
                if pts:
                    save_points(pts)
                break
            else:
                # ignore other keys
                pass

            # short delay so get_coords has time to update
            time.sleep(0.05)

    finally:
        try:
            mc.release_all_servos()
        except Exception:
            pass

if __name__ == "__main__":
    interactive_record()