import atexit, subprocess
atexit.register(lambda: subprocess.run("killall scgrab 2>/dev/null || true", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))


DO_WE_WANT_CALLOUTS = True #Audio queues on moves 

if DO_WE_WANT_CALLOUTS:
    import subprocess, sys; ANN_PROC = subprocess.Popen([sys.executable, "/Users/marcus/Documents/GitHub/Ai-plays-SubwaySurfers/alpha/announcer.py"], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import subprocess, shlex; scgrab_proc = subprocess.Popen(shlex.split("./scgrab --x 644 --y 77 --w 505 --h 906 --fps 60 --out /tmp/scap.ring --slots 3 --scale 2"), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

import os, sys, argparse, builtins, warnings, subprocess, time
BASE = "/Users/marcus/Documents/GitHub/Ai-plays-SubwaySurfers/alpha/arrow_save_to_transcend.py"
proc = subprocess.Popen([sys.executable, BASE, "start"])


_mute_parser = argparse.ArgumentParser(add_help=False)
_mute_parser.add_argument("--quiet",  action="store_true",
                          help="Mute stdout/stderr (no terminal I/O).")
_mute_parser.add_argument("--silent", action="store_true",
                          help="Stronger mute: also replace print() with no-op.")
_mute_args, _ = _mute_parser.parse_known_args()

SILENT_MODE = bool(_mute_args.silent)
QUIET_MODE  = bool(_mute_args.quiet or _mute_args.silent)

def _redirect_to_devnull():
    # Python-level redirection
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    # OS-level redirection (catches C/C++ prints)
    try:
        dn_fd = os.open(os.devnull, os.O_WRONLY)
        for fd in (1, 2):  # stdout, stderr
            try:
                os.dup2(dn_fd, fd)
            except OSError:
                pass
        os.close(dn_fd)
    except Exception:
        pass

# Global muting for libraries / warnings / logging
if QUIET_MODE:
    _redirect_to_devnull()
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    try:
        import logging
        logging.disable(100)  # beyond CRITICAL
        root = logging.getLogger()
        root.handlers[:] = []
        root.setLevel(100)
    except Exception:
        pass

# Replace print itself (prevents function call overhead, but note: f-strings are
if SILENT_MODE:
    builtins.print = lambda *a, **k: None

# Lazy, zero-compute debug printer for hotspots:
def dprint(msg=None, *a, **k):
    """
    Use as: dprint(lambda: f'heavy {expensive()} string')
    Nothing is computed when --quiet/--silent are active.
    """
    if not QUIET_MODE:
        if callable(msg):
            msg = msg()
        return builtins.__dict__.get("print", lambda *aa, **kk: None)(msg, *a, **k)
# ============================================================================



import os, time, math, subprocess
import cv2, torch, numpy as np
from pathlib import Path
from mss import mss
import pyautogui
from pynput import keyboard
from ultralytics import YOLO
from threading import Timer
from threading import Thread
import time
start_time = time.perf_counter()
from ring_grab import get_frame_bgr_from_ring  # or place the helper above and import nothing
from typing import Optional


# ===== Boot-time save toggle =====
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Force saving overlays/top-logic")
args, _ = parser.parse_known_args()

save_frames = False

# macOS Quartz (works if available)
try:
    from Quartz import CGEventSourceKeyState, kCGEventSourceStateHIDSystemState
    _is_g_held_at_boot = lambda: bool(CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, 5))  # 5 = 'G' on US layout
except Exception:
    _is_g_held_at_boot = lambda: False

# quick cross-platform fallback: brief listener window at startup
def _enable_save_if_g_within_window(win_s=0.8):
    from pynput import keyboard
    hit = {"g": False}
    def on_press(k):
        try:
            if getattr(k, "char", "").lower() == "g":
                hit["g"] = True
                return False
        except Exception:
            pass
    L = keyboard.Listener(on_press=on_press)
    L.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < win_s and L.running:
        time.sleep(0.02)
    try: L.stop()
    except: pass
    return hit["g"]

if args.save or _is_g_held_at_boot() or _enable_save_if_g_within_window():
    save_frames = True
    print("[BOOT] Saving enabled (arg/Quartz/boot-window)")



# purple triangles funciton config
_SE_OPEN_5x9 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))


# One-pixel ON_TOP probe: screen coords relative to each lane anchor (x,y)
# Tune dy for your layout; -360..-460 is typical for train tops.
ONTOP_PROBE_OFFSETS = (
    (0, 0),  # lane 0: left
    (0, 0),  # lane 1: mid
    (0, 0),  # lane 2: right
)


# ---------- decision logging ----------
DECISIONS_VERBOSE = True

def dlog(tag: str, **kv):
    """Compact one-liner logs for decisions/actions."""
    if not DECISIONS_VERBOSE:
        return
    ts = time.perf_counter()
    payload = " ".join(f"{k}={v}" for k, v in kv.items() if v is not None)
    print(f"[DECIDE] t={ts:.6f} {tag} {payload}")
# --------------------------------------


# === micro-profiler (drop-in) =========================================
from collections import OrderedDict
_PROF_CUR = OrderedDict()

def _prof_reset():
    _PROF_CUR.clear()

PROF_PRINT_IMMEDIATE = False  # toggle

def _record_prof(label, t0):
    dt = (time.perf_counter() - t0) * 1000.0
    _PROF_CUR[label] = dt
    if PROF_PRINT_IMMEDIATE:
        print(f"[PROF] {label}: {dt:.2f} ms")


def __PROF(label: str):
    """Usage: __p = __PROF('tag');  ...work... ; __p()"""
    t0 = time.perf_counter()
    return lambda: _record_prof(label, t0)

def prof_summary(frame_idx: int):
    if not _PROF_CUR:
        print(f"[PROF_SUM] frame {frame_idx}: (no samples)")
        return
    total = sum(_PROF_CUR.values())
    print(f"[PROF_SUM] frame {frame_idx}: total={total:.2f} ms")
    for k, v in _PROF_CUR.items():
        print(f"   {k:<36} {v:7.2f} ms")
# ======================================================================

_RGS_CACHE = {"shape": None}
def _rgs_ensure(H, W):
    if _RGS_CACHE.get("shape") != (H, W):
        _RGS_CACHE["shape"] = (H, W)
        _RGS_CACHE["r32"]   = np.empty((H, W), np.float32)
        _RGS_CACHE["g32"]   = np.empty((H, W), np.float32)
        _RGS_CACHE["diff"]  = np.empty((H, W), np.float32)
        _RGS_CACHE["norm"]  = np.empty((H, W), np.float32)
        _RGS_CACHE["u8"]    = np.empty((H, W), np.uint8)

def _as_u8_view(a: np.ndarray) -> np.ndarray:
    """Zero-copy bool->uint8 view when possible; else cheap fallback."""
    if a.dtype == np.bool_:
        v = a.view(np.uint8)  # 0/1 bytes, no copy
    elif a.dtype == np.uint8:
        v = a  
    else:
        v = a.astype(np.uint8, copy=False)  # already small; avoids extra copy if possible
    return v if v.flags["C_CONTIGUOUS"] else np.ascontiguousarray(v)


# --- swallow AI-generated keypresses in the listener for a short window ---
SYNTHETIC_SUPPRESS_S = 0.15  # 150 ms is plenty
_synth_block_until = 0.0     # simple, explicit init avoids IDE warnings

try:
    _synth_block_until
except NameError:
    _synth_block_until = 0.0

# ======================= Quick supreesion to prevent instant bailouts =======================

# --- allows 0.5s of movement, then mute for 2.5s, then restore ---
# Save originals
__press_orig   = pyautogui.press
__keyDown_orig = pyautogui.keyDown
__keyUp_orig   = pyautogui.keyUp
__hotkey_orig  = pyautogui.hotkey

# near your other globals, after imports
MOVEMENT_ENABLED = True

def __mute_keys():
    global MOVEMENT_ENABLED
    MOVEMENT_ENABLED = False
    pyautogui.press  = lambda *a, **k: None
    pyautogui.keyDown = lambda *a, **k: None
    pyautogui.keyUp   = lambda *a, **k: None
    pyautogui.hotkey  = lambda *a, **k: None
    print("[BOOT] movement muted")

def __unmute_keys():
    global MOVEMENT_ENABLED
    MOVEMENT_ENABLED = True
    pyautogui.press   = __press_orig
    pyautogui.keyDown = __keyDown_orig
    pyautogui.keyUp   = __keyUp_orig
    pyautogui.hotkey  = __hotkey_orig
    print("[BOOT] movement unmuted")


# Allow movement immediately; after 0.5s, mute; after 6.0s total, unmute
Timer(0.5, __mute_keys).start()
Timer(6.0, __unmute_keys).start()


# =======================
# Config
# =======================

# --- Pillar-guard scan state (edge-triggered) ---
PILLAR_SCAN = {
    "active": False,           # currently watching for red_ahead to flip
    "last_red": None,          # red_ahead previous frame (bool)
    "start_frame": -1,         # for logs
    "cooldown_until": 0.0,     # monotonic ts; ignore re-triggers during cooldown
}
PILLAR_SCAN_COOLDOWN_S = 0.50  # ignore new scans for this long after a double move



home       = os.path.expanduser("~")
weights    = f"{home}/models/jakes-loped/jakes-finder-mk1/1/weights.pt"

# SAVE HERE
out_dir    = Path(home) / "Documents" / "GitHub" / "Ai-plays-SubwaySurfers" / "out_live_overlays"
out_dir.mkdir(parents=True, exist_ok=True)

# --- Replay dump setup (sibling to out_live_overlays) ---
REPLAYS_DIR = out_dir.parent / "replays"
REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
NEON_GREEN = (57, 255, 20)  # BGR neon green

#================================================================================================================================================================

# === Saving toggles ===    
#save_top_frames = True         # NEW: TL (top-logic) analysed frames -> OVERWRRITEN BY G PRESS LOGIC

# Existing overlay dir:
out_dir = Path(home) / "Documents" / "GitHub" / "Ai-plays-SubwaySurfers" / "out_live_overlays"
out_dir.mkdir(parents=True, exist_ok=True)

# NEW: top-logic output dir
TOP_OUT_DIR = out_dir.parent / "out_top_logic"
TOP_OUT_DIR.mkdir(parents=True, exist_ok=True)



from TL_modular import run_top_logic_on_frame, tl_sticky_until

# --- helper inside the main script (no extra work; passes precomputed results) ---
def do_top_logic_from_result(
    frame_bgr: np.ndarray,
    yolo_result,
    lane_2: int,
    *,
    save_analysed: bool = True,
    save_path: Optional[str] = None,
    print_prefix: str = ""
):
    """
    Extracts masks/classes from an existing YOLO 'segment' result and runs the exact
    'script2' logic in TL_modular without reloading the model or re-running predict().

    Args:
        frame_bgr: original frame (BGR).
        yolo_result: ultralytics result for this frame (already computed).
        lane_2: current lane state carried over from your main loop.
        save_analysed: toggle saving the annotated frame.
        save_path: where to save (required if save_analysed=True).
        print_prefix: optional console print prefix.

    Returns:
        decision dict from TL_modular.run_top_logic_on_frame(...)
    """
    if yolo_result is None:
        masks_np = None
        classes_np = None
    else:
        if getattr(yolo_result, "masks", None) is not None and getattr(yolo_result.masks, "data", None) is not None:
            masks_np = yolo_result.masks.data.detach().cpu().numpy()
        else:
            masks_np = None

        if getattr(yolo_result, "boxes", None) is not None and getattr(yolo_result.boxes, "cls", None) is not None:
            classes_np = yolo_result.boxes.cls.detach().cpu().numpy().astype(int)
        else:
            classes_np = None

    return run_top_logic_on_frame(
        frame_bgr,
        masks_np,
        classes_np,
        lane_2,
        save_frames=save_analysed,
        save_path=save_path,
        print_prefix=print_prefix
    )

def _stamp_lane_badge(img, lane_idx: int):
    name = ("LEFT","MID","RIGHT")[lane_idx] if 0 <= lane_idx <= 2 else str(lane_idx)
    # outline + text so it pops on any background
    cv2.putText(img, f"LANE: {name}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"LANE: {name}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    
#=================================================================================================================PILLAR LOGIC)

# --- IDs ---
ID_PILLAR = 7
ID_RAILS  = 9
ID_RAMP   = 8
TRAIN_TOPS = {1, 6, 11}  # GREY/ORANGE/YELLOW trains

PILLAR_ALLOWED = TRAIN_TOPS | {ID_RAILS, ID_PILLAR}          # trains + rails + pillar only


def unit_from_theta(theta_deg: float):
    """Return a unit (dx,dy) pointing 'theta_deg' from vertical, up = -y."""
    t = float(theta_deg)
    v = TRIG_TABLE.get(t)
    if v is not None:
        return v
    r = math.radians(t)
    return (math.sin(r), -math.cos(r))

# --- Ramp lock: suppress lateral moves while Jake's probe sees a ramp ahead in same lane
RAMP_LOCK = {"active": False, "lane": None}

def _lane_from_x(x: int) -> int:
    x0, x1, x2 = LANE_LEFT[0], LANE_MID[0], LANE_RIGHT[0]
    d0 = abs(x - x0); d1 = abs(x - x1); d2 = abs(x - x2)
    return 0 if d0 <= d1 and d0 <= d2 else (1 if d1 <= d2 else 2)

def _update_ramp_lock_from_jake(jake_tri, jake_band_y: int):
    """
    Enter lock iff Jake's own triangle ray hits a RAMP (ID_RAMP) that is ahead in
    the same lane. Release when ON_TOP flips true or the condition no longer holds.
    """
    global RAMP_LOCK, lane, ON_TOP, ID_RAMP

    # Safe reads (hit_class may be None)
    cid = jake_tri.get("hit_class") if jake_tri else None
    pos = jake_tri.get("pos")       if jake_tri else None

    # Short-circuit order avoids extra work when not needed
    want_lock = (
        (cid == ID_RAMP) and                         # no int() → handles None
        (pos is not None) and
        ((jake_band_y - int(pos[1])) > 0) and        # ahead of Jake
        (_lane_from_x(int(pos[0])) == lane)          # same lane
    )

    if RAMP_LOCK["active"]:
        if ON_TOP or not want_lock or RAMP_LOCK.get("lane") != lane:
            RAMP_LOCK.update(active=False, lane=None)
    else:
        if want_lock:
            RAMP_LOCK.update(active=True, lane=lane)


def _allowed_and_pillar_stats(classes_np):
    """
    Returns (allowed_only, pillar_count, uniq_ids)
      - allowed_only: all *present* classes ∈ PILLAR_ALLOWED
      - pillar_count: number of instances with class == ID_PILLAR
      - uniq_ids:     unique class ids present (np.ndarray[int])
    """
    if classes_np is None or classes_np.size == 0:
        return False, 0, np.array([], dtype=int)

    arr = classes_np.astype(int, copy=False).ravel()
    uniq, counts = np.unique(arr, return_counts=True)
    allowed_only = np.all(np.isin(uniq, list(PILLAR_ALLOWED)))
    pillar_count = int(counts[uniq == ID_PILLAR].sum()) if np.any(uniq == ID_PILLAR) else 0
    return bool(allowed_only), pillar_count, uniq


# --- minimal helper: pause if the largest RAMP mask is > 28% of the screen ---
def pause_if_big_ramp(yres, ramp_id=ID_RAMP, thr_frac=0.28, sleep_s=1.0) -> bool:
    """
    Checks the largest RAMP instance area on the mask grid (proportional to screen area).
    If coverage > thr_frac, sleep(sleep_s). Returns True if slept.
    """
    try:
        mobj = getattr(yres, "masks", None)
        if mobj is None or getattr(mobj, "data", None) is None:
            return False

        # classes for each mask (prefer masks.cls if present; else boxes.cls)
        if getattr(mobj, "cls", None) is not None:
            cls_np = mobj.cls.detach().cpu().numpy().astype(int, copy=False)
        else:
            cls_np = yres.boxes.cls.detach().cpu().numpy().astype(int, copy=False)

        ramp_idxs = np.where(cls_np == int(ramp_id))[0]
        if ramp_idxs.size == 0:
            return False

        m = mobj.data  # torch.Tensor [n, mh, mw] (device: cpu/cuda/mps)
        import torch
        idx = torch.as_tensor(ramp_idxs, device=m.device, dtype=torch.long)
        sub = m.index_select(0, idx)

        # threshold and count on-device; compare to mask grid area
        max_area = (sub > 0.5).sum(dim=(1, 2)).amax().item()
        mh, mw = sub.shape[-2], sub.shape[-1]
        frac = float(max_area) / float(mh * mw)

        if frac > thr_frac:  # strictly greater than 28%
            print(f"[RAMP-PAUSE] Largest RAMP covers {frac*100:.1f}% > {thr_frac*100:.0f}% → pausing {sleep_s:.2f}s")
            time.sleep(sleep_s)
            return True
        return False
    except Exception as _e:
        # stay silent on failure; keep loop going
        return False


def red_ahead_from_overlay(best_idx, tri_hit_classes, tri_colours) -> bool:
    if best_idx is None:
        return False
    # 1) If classifier said the Jake triangle hits a DANGER_RED class
    if 0 <= best_idx < len(tri_hit_classes):
        cid = tri_hit_classes[best_idx]
        if cid is not None and int(cid) in DANGER_RED:
            return True
    # 2) Respect any post flips that made it red in the overlay
    if 0 <= best_idx < len(tri_colours):
        # COLOR_RED is (0, 0, 255) in your code
        return tuple(tri_colours[best_idx]) == COLOR_RED
    return False

def pillar_evasion_check_and_act(H: int, W: int,
                                 masks_np: np.ndarray,
                                 classes_np: np.ndarray,
                                 JAKE_POINT: tuple[int,int],
                                 *,
                                 red_ahead_overlay: bool | None = None) -> bool:
    """
    New semantics:
      • When (allowed_only & pillar>=1 & in_side_lane & movement enabled) holds AND red_ahead=False,
        enter 'scan' mode (do nothing but watch).
      • When that same guard set holds AND red_ahead flips False -> True (rising edge),
        perform a double sidestep (L⟶R or R⟶L) and exit scan (with a short cooldown).
      • If guards break, reset scan state.
    Returns True iff we executed the double sidestep this frame.
    """
    global lane, PILLAR_SCAN

    # --- quick guards / readings ---
    allowed_only, pillar_cnt, _ = _allowed_and_pillar_stats(classes_np)
    in_side_lane = lane in (0, 2)
    move_ok      = MOVEMENT_ENABLED
    red_now      = bool(red_ahead_overlay)
    targets_ok   = allowed_only and (pillar_cnt >= 1)
    now_mono     = time.monotonic()


    if now_mono < PILLAR_SCAN.get("cooldown_until", 0.0):
        return False
    # If we become eligible while red is already ahead and scan isn't armed, fire immediately
    if targets_ok and in_side_lane and move_ok and red_now and not PILLAR_SCAN["active"]:
        print("[PILLAR] red already ahead on entry → DOUBLE SIDESTEP")
        time.sleep(0.3) 
        acted = _double_sidestep()
        PILLAR_SCAN.update(active=False, last_red=None, start_frame=-1,
                        cooldown_until=now_mono + PILLAR_SCAN_COOLDOWN_S)
        return bool(acted)

    dlog("PILLAR_GUARDS",
         only_targets=allowed_only,
         pillar_cnt=pillar_cnt,
         pillars_gt1=(pillar_cnt >= 2),
         targets_ok=targets_ok,
         in_side_lane=in_side_lane,
         red_ahead=red_now,
         move_enabled=move_ok,
         scan_active=PILLAR_SCAN["active"])

    # If we’re in a cooldown after an evade, do nothing.
    if now_mono < PILLAR_SCAN.get("cooldown_until", 0.0):
        return False

    # If the fundamental guards aren’t satisfied, reset the scan state and bail.
    if not (targets_ok and in_side_lane and move_ok):
        if PILLAR_SCAN["active"]:
            print("[PILLAR] scan reset (guards broke)")
        PILLAR_SCAN.update(active=False, last_red=None, start_frame=-1)
        return False

    # At this point, guards are true and we are in a side lane with pillars-only scene.

    # Start or continue the scan while red is NOT ahead.
    if not red_now:
        if not PILLAR_SCAN["active"]:
            # enter scan mode
            PILLAR_SCAN.update(active=True,
                               last_red=False,
                               start_frame=int(globals().get("frame_idx", -1)))
            print(f"[PILLAR] scanning... (frame={PILLAR_SCAN['start_frame']})")
        else:
            # still scanning; update memory
            PILLAR_SCAN["last_red"] = False
        return False

    # red_now == True here
    if PILLAR_SCAN["active"] and PILLAR_SCAN.get("last_red") is False:
        # Rising edge detected → perform the double sidestep.
        print("[PILLAR] RED edge detected → DOUBLE SIDESTEP")
        acted = _double_sidestep()

        # End scan and set a short cooldown to avoid re-triggering immediately.
        PILLAR_SCAN.update(active=False,
                           last_red=None,
                           start_frame=-1,
                           cooldown_until=now_mono + PILLAR_SCAN_COOLDOWN_S)

        return bool(acted)

    # If red was already true before scan start (or state desynced), treat as no-op this frame.
    PILLAR_SCAN["last_red"] = True
    return False

def _double_sidestep(from_pillar: bool = True) -> bool:
    """
    Emergency dodge: two taps to the opposite side lane, ignoring mid-lane logic.
    Returns True if keys were sent.
    """
    global lane, last_move_ts, _synth_block_until, REENTRY_BAN

    if not MOVEMENT_ENABLED:
        return False

    prev_lane = lane
    now = time.perf_counter()

    # keep vision steady and swallow synthetic key echos
    pause_inference(0.30)
    _synth_block_until = time.monotonic() + SYNTHETIC_SUPPRESS_S

    def _tap(k):
        try: pyautogui.press(k)
        except Exception: 
            pass
        time.sleep(0.045)

    if lane == 0:
        _tap('right'); _tap('right'); lane = 2
        print("[PILLAR EVADE] RIGHT, RIGHT → lane=2")
        time.sleep(0.1)
    elif lane == 2:
        _tap('left'); _tap('left'); lane = 0
        print("[PILLAR EVADE] LEFT, LEFT → lane=0")
        time.sleep(0.1)
    else:
        _tap('right'); _tap('right'); lane = 2
        print("[PILLAR EVADE] MID → RIGHT, RIGHT → lane=2")
        time.sleep(0.1)

    last_move_ts = now

    if from_pillar:
        # ensure any existing re-entry ban can’t block the immediate follow-up
        REENTRY_BAN.update(lane=None, counter=0, last_intent_dir=None,
                           last_intent_frame=None, expiry_frame=None)
    else:
        _register_red_evasion_ban(prev_lane)

    return True

#======================================================================================================================================================================================

# =======================
# Lane/keyboard state
# =======================
lane = 1
MIN_LANE = 0
MAX_LANE = 2
running = True
# --- Lane re-entry hysteresis (avoid bouncing back after RED) ---
REENTRY_NEED = 2  # require N consecutive "enter" intents before re-entering banned lane
if "REENTRY_BAN" not in globals():
    REENTRY_BAN = {"lane": None, "counter": 0, "last_intent_dir": None, "needed": REENTRY_NEED}


REENTRY_BAN_FRAMES = 2                 # ban only lasts this many frames after creation

if "REENTRY_BAN" not in globals():
    REENTRY_BAN = {
        "lane": None,
        "counter": 0,
        "last_intent_dir": None,
        "last_intent_frame": None,     # NEW: to enforce adjacency (sequential frames)
        "needed": REENTRY_NEED,
        "expiry_frame": None,          # NEW: auto-expire window
        "created_at_frame": None,      # optional: for logging
    }


def _register_red_evasion_ban(prev_lane: int):
    global REENTRY_BAN  # keep
    # NOTE: we read the global frame_idx (no assignment -> no 'global' needed)
    REENTRY_BAN["lane"] = int(prev_lane)
    REENTRY_BAN["counter"] = 0
    REENTRY_BAN["last_intent_dir"] = None
    REENTRY_BAN["last_intent_frame"] = None
    REENTRY_BAN["needed"] = int(REENTRY_NEED)
    REENTRY_BAN["created_at_frame"] = int(frame_idx)
    REENTRY_BAN["expiry_frame"] = int(frame_idx) + int(REENTRY_BAN_FRAMES)
    print(f"[REENTRY] ban set: lane {REENTRY_BAN['lane']} "
          f"(need {REENTRY_BAN['needed']} consecutive frames, "
          f"expires after frame {REENTRY_BAN['expiry_frame']})")
    
def _reentry_gate_allow(jx: int, tx: int) -> bool:
    global REENTRY_BAN, lane

    banned = REENTRY_BAN.get("lane")
    if banned is None:
        return True

    # --- auto-expire after the window of N frames ---
    exp = REENTRY_BAN.get("expiry_frame")
    if (exp is None) or (frame_idx > int(exp)):  # strictly after expiry
        # clear the ban
        REENTRY_BAN["lane"] = None
        REENTRY_BAN["counter"] = 0
        REENTRY_BAN["last_intent_dir"] = None
        REENTRY_BAN["last_intent_frame"] = None
        return True

    # If triangle aligned with Jake (no lateral intent), clear streak and allow
    if tx == jx:
        REENTRY_BAN["counter"] = 0
        REENTRY_BAN["last_intent_dir"] = None
        REENTRY_BAN["last_intent_frame"] = None
        return True

    # Determine intent direction (+1 right, -1 left)
    intent_dir = 1 if tx > jx else -1

    # Are we trying to move back *towards* the banned lane?
    towards_banned = (intent_dir > 0 and banned > lane) or (intent_dir < 0 and banned < lane)
    if not towards_banned:
        # Moving away or sideways -> allow and reset streak
        REENTRY_BAN["counter"] = 0
        REENTRY_BAN["last_intent_dir"] = None
        REENTRY_BAN["last_intent_frame"] = None
        return True

    # --- within the ban window AND towards banned lane:
    # Require *sequential frames* for the counter
    last_dir   = REENTRY_BAN.get("last_intent_dir")
    last_frame = REENTRY_BAN.get("last_intent_frame")

    if last_dir == intent_dir and last_frame is not None and frame_idx == (int(last_frame) + 1):
        # Adjacent frame, same direction -> increment
        REENTRY_BAN["counter"] += 1
    else:
        # Not adjacent or changed direction -> start new streak
        REENTRY_BAN["counter"] = 1
        REENTRY_BAN["last_intent_dir"] = intent_dir

    REENTRY_BAN["last_intent_frame"] = int(frame_idx)

    need = int(REENTRY_BAN.get("needed", REENTRY_NEED))
    have = int(REENTRY_BAN["counter"])

    if have < need:
        print(f"[REENTRY] veto {have}/{need} (within window till f{REENTRY_BAN['expiry_frame']})")
        return False  # veto this move

    # Achieved required consecutive intents within window -> clear ban & allow
    print(f"[REENTRY] allowed after {need} consecutive frames → clearing ban")
    REENTRY_BAN["lane"] = None
    REENTRY_BAN["counter"] = 0
    REENTRY_BAN["last_intent_dir"] = None
    REENTRY_BAN["last_intent_frame"] = None
    return True


# --- Left probe offset (toggle) --- Skew our left probe off 
LEFT_PROBE_OFFSET_ENABLED = False      # set False to disable at runtime if you like
LEFT_PROBE_OFFSET_DEG     = 0.0      # anticlockwise = toward the left (negative in this codebase)


SIDEWALK_ID = 10
SIDEWALK_JUMP_THEN_DUCK_DELAY_S = 0.40  # down after jump whiel sideqlks prsetn


# ===== Side-ray → middle-triangle spacing threshold =====
SIDE_MID_FLIP_DIST_PX = 1500.0 #EXPLODE NUMBER orig 15 so that we dont convert off falsely


# Crop + click (set by ad layout)
advertisement = True
if advertisement:
    snap_coords = (644, 77, (1149-644), (981-75))  # (left, top, width, height)
    start_click = (1030, 900)
else:
    snap_coords = (483, 75, (988-483), (981-75))
    start_click = (870, 895)

RAIL_ID    = 9
IMG_SIZE   = 512
CONF, IOU  = 0.30, 0.45
MAX_DET    = 30

# Color/region filter
TARGET_COLORS_RGB  = [(119,104,67), (81,42,45)]
TOLERANCE          = 20.0
MIN_REGION_SIZE    = 30
MIN_REGION_HEIGHT  = 150

# Heat/triangle
HEAT_BLUR_KSIZE     = 51
RED_SCORE_THRESH    = 220
EXCLUDE_TOP_FRAC    = 0.40
EXCLUDE_BOTTOM_FRAC = 0.15
MIN_DARK_RED_AREA   = 1200
MIN_DARK_FRACTION   = 0.15
TRI_SIZE_PX         = 18

# Sampling ray length
SAMPLE_UP_PX        = 200
RAY_STEP_PX         = 20   # walk the ray every 20 px

# ===== Bend degrees (tune here) =====
BEND_LEFT_STATE_RIGHT_DEG  = -20.0  # N1
BEND_MID_STATE_RIGHT_DEG   = -20.0  # N2
BEND_MID_STATE_LEFT_DEG    = +20.0  # N3
BEND_RIGHT_STATE_LEFT_DEG  = +20.0  # N4

# Colours (BGR)
COLOR_GREEN  = (0, 255, 0)
COLOR_PINK   = (203, 192, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED    = (0, 0, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_CYAN   = (255, 255, 0)
COLOR_BLACK  = (0, 0, 0)


# ===== Hypergreen: treat these like "better than green" for lane picking =====
HYPERGREEN_CLASSES = {3, 8}       # 3:JUMP, 5:LOWBARRIER2 (jump). Add 10 if you want Sidewalk too.
MIN_HYPERGREEN_AHEAD_PX = 0     # like MIN_YELLOW_AHEAD_PX but a bit looser; tune 250–400


if "_ONTOP_CACHE" not in globals():
    _ONTOP_CACHE = {
        "on_top_now": globals().get("ON_TOP", False),
        "seen": set(),
        "ot_ms": 0.0,
    }


# =======================
# Jake lane points + dynamic JAKE_POINT
# =======================
LANE_LEFT   = (300, 1340)
LANE_MID    = (490, 1340)
LANE_RIGHT  = (680, 1340)
LANE_POINTS = (LANE_LEFT, LANE_MID, LANE_RIGHT)  # index by lane (0,1,2)
JAKE_POINT  = LANE_MID  # will be set each frame from 'lane'

LANE_TARGET_DEG = {"left": -10.7, "mid": +1.5, "right": +15.0}

def lane_name_from_point(p):
    if p == LANE_LEFT:  return "left"
    if p == LANE_MID:   return "mid"
    if p == LANE_RIGHT: return "right"
    return "mid"


# ===== Movement logic (modular) HELPER FUNCTIOSNS==============================================================================================================


# --- inference pause gate (after imports/globals) ---
PAUSE_AFTER_MOVE_S = 0.40

try:
    PAUSE_UNTIL
except NameError:
    PAUSE_UNTIL = 0.0  # monotonic timestamp

def pause_inference(sec: float = PAUSE_AFTER_MOVE_S):
    """Freeze the main loop for `sec` seconds from NOW."""
    global PAUSE_UNTIL
    PAUSE_UNTIL = time.monotonic() + sec


# One-shot gating
try:
    IMPACT_TOKEN
except NameError:
    IMPACT_TOKEN = None  # (lane, class_id)

def _fire_action_key(key: str, token_snapshot):
    global IMPACT_TOKEN
    if MOVEMENT_ENABLED:
        pyautogui.press(key)
        print(f"[TIMER FIRE] pressed {key}")
    # allow instant re-arm for the next identical obstacle
    if IMPACT_TOKEN == token_snapshot:
        IMPACT_TOKEN = None


# Only arm timer when distance is strictly inside this window (px)
IMPACT_MIN_PX = 100
IMPACT_MAX_PX = 650

# ===== Impact delay lookup (distance px -> seconds) =====
# Fill these with your *monotone ascending* distances (px) and corresponding delays (seconds).
# Example placeholders; REPLACE with your numbers:
LUT_PX = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800], dtype=float)

# immutable base (no shortening applied)
LUT_BASE_S = np.array(
    [0.0259, 0.0303, 0.0353, 0.0412, 0.0481, 0.0561, 0.0655, 0.0765, 0.0893,
     0.1042, 0.1216, 0.1419, 0.1656, 0.1933, 0.2256, 0.2633, 0.3073, 0.3586,
     0.4185, 0.4885, 0.5701, 0.6654, 0.7765, 0.9063, 1.0578, 1.2345, 1.4408,
     1.6815, 1.9625], dtype=float)

SHORTEN_S = 0.40
MIN_DELAY_S = 0.03
MAX_DELAY_S = 2.00

def _compute_lut(shorten: float) -> np.ndarray:
    return np.clip(LUT_BASE_S - float(shorten), MIN_DELAY_S, MAX_DELAY_S)

LUT_S = _compute_lut(SHORTEN_S)

def first_mask_hit_starburst_then_ray_for_set(
    jake_point, tri_pos, theta_deg, masks_np, classes_np, H, W,
    allowed_classes, up2_px=SAMPLE_UP_PX, step_px=2
):
    """
    Same as first_mask_hit_starburst_then_ray, but only counts hits whose class ∈ allowed_classes.
    Returns (dist_px_from_jake, (x_hit, y_hit), class_id) or (None, None, None).
    """
    if masks_np is None or classes_np is None or masks_np.size == 0:
        return (None, None, None)

    mh, mw = masks_np.shape[1], masks_np.shape[2]
    sx = (mw - 1) / max(1, (W - 1))
    sy = (mh - 1) / max(1, (H - 1))

    allowed = set(int(c) for c in allowed_classes)
    idxs = [i for i, c in enumerate(classes_np) if int(c) in allowed]
    if not idxs:
        return (None, None, None)

    def _hit_at(xs, ys):
        mx = _clampi(int(round(xs * sx)), 0, mw-1)
        my = _clampi(int(round(ys * sy)), 0, mh-1)
        for i in idxs:
            if masks_np[i][my, mx] > 0.5:
                return int(classes_np[i])
        return None

    x0, y0 = map(int, jake_point)
    x1, y1 = map(int, tri_pos)
    x0 = _clampi(x0, 0, W-1); y0 = _clampi(y0, 0, H-1)
    x1 = _clampi(x1, 0, W-1); y1 = _clampi(y1, 0, H-1)

    dx = x1 - x0; dy = y1 - y0
    seg1_len = max(1e-6, math.hypot(dx, dy))
    n1 = max(1, int(seg1_len // max(1, step_px)))
    for k in range(1, n1 + 1):
        t = min(1.0, (k * step_px) / seg1_len)
        xs = _clampi(int(round(x0 + dx * t)), 0, W-1)
        ys = _clampi(int(round(y0 + dy * t)), 0, H-1)
        cls_hit = _hit_at(xs, ys)
        if cls_hit is not None:
            dist = math.hypot(xs - x0, ys - y0)
            return (float(dist), (int(xs), int(ys)), int(cls_hit))

    dxr, dyr = unit_from_theta(theta_deg)
    n2 = max(1, int(up2_px // max(1, step_px)))
    for k in range(1, n2 + 1):
        t = k * step_px
        xs = _clampi(int(round(x1 + dxr * t)), 0, W-1)
        ys = _clampi(int(round(y1 + dyr * t)), 0, H-1)
        cls_hit = _hit_at(xs, ys)
        if cls_hit is not None:
            dist = seg1_len + math.hypot(xs - x1, ys - y1)
            return (float(dist), (int(xs), int(ys)), int(cls_hit))

    return (None, None, None)


# ===== Impact-timer overhaul (single-triangle action) =====
# Classes to act on (exact mapping)
IMPACT_CLASSES = {2, 3, 4, 5}  # 2:HIGHBARRIER1, 3:JUMP, 4:LOWBARRIER1, 5:LOWBARRIER2
ACTION_BY_CLASS = {3: "up", 5: "up", 4: "down", 2: "down"}  # per spec

# Global timer handle (overwritten when re-arming)
try:
    IMPACT_TIMER
except NameError:
    IMPACT_TIMER = None


def _cancel_impact_timer(reason=None):
    global IMPACT_TIMER
    if IMPACT_TIMER is not None and getattr(IMPACT_TIMER, "is_alive", lambda: False)():
        print("[TIMER] cancelled" + (f" ({reason})" if reason else ""))
        try:
            IMPACT_TIMER.cancel()
        except Exception:
            pass
    IMPACT_TIMER = None



def _impact_delay_seconds(dist_px: float) -> float:
    """
    O(1) lookup + linear interpolation from a monotone table (px -> seconds).
    - Dist is cropped to [IMPACT_MIN_PX, IMPACT_MAX_PX] to preserve your windowing.
    - Result is clamped to [MIN_DELAY_S, MAX_DELAY_S] for Timer safety.
    """
    if not math.isfinite(dist_px):
        return MIN_DELAY_S

    # Respect your arming window; crop inside it so behavior matches old gating.
    d = max(IMPACT_MIN_PX, min(float(dist_px), IMPACT_MAX_PX))

    # Interpolate within the table’s range
    lo = float(LUT_PX[0]); hi = float(LUT_PX[-1])
    d_clamped = max(lo, min(d, hi))

    delay = float(np.interp(d_clamped, LUT_PX, LUT_S))
    # Final safety clamp
    return max(MIN_DELAY_S, min(delay, MAX_DELAY_S))

def _dist_point_to_segment(p, a, b) -> float:
    """Euclidean distance from point p to segment a-b (screen pixels)."""
    px, py = float(p[0]), float(p[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    seg_len2 = vx*vx + vy*vy
    if seg_len2 <= 1e-12:
        # a and b are the same point
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx*vx + wy*vy) / seg_len2))
    projx = ax + t * vx
    projy = ay + t * vy
    return math.hypot(px - projx, py - projy)



def _arm_impact_timer(dist_px: float, cls_id: int):
    """
    Overwrite-or-set the global timer if dist is in (400, 800) px and class is in IMPACT_CLASSES.
    Prints whether we armed a NEW timer or UPDATED (overwrote) an existing one.
    """
    if cls_id not in IMPACT_CLASSES:
        return

    if not (IMPACT_MIN_PX < dist_px < IMPACT_MAX_PX):
        # Optional debug: show why we didn't arm
        print(f"[TIMER] skip: dist {dist_px:.1f}px outside ({IMPACT_MIN_PX},{IMPACT_MAX_PX}) for {LABELS.get(int(cls_id), cls_id)}")
        return

    key = ACTION_BY_CLASS.get(int(cls_id))
    if not key:
        return

    delay_s = _impact_delay_seconds(dist_px)

    if not math.isfinite(delay_s) or delay_s <= 0.0:
        print(f"[TIMER] skip: invalid delay {delay_s} for dist={dist_px:.1f}px, cls={LABELS.get(int(cls_id), cls_id)}")
        return

    # detect whether we are overwriting a live timer
    # detect whether we are overwriting a live timer
    global IMPACT_TIMER, IMPACT_TOKEN
    was_live = (IMPACT_TIMER is not None and getattr(IMPACT_TIMER, "is_alive", lambda: False)())

    _cancel_impact_timer()  # overwrite existing timer if any

    # --- REPLACEMENT: arm with a token so post-fire re-arm is instant ---
    new_token = (lane, int(cls_id))            # <-- build token for THIS arm
    from threading import Timer
    IMPACT_TIMER = Timer(delay_s, _fire_action_key, args=(key, new_token))
    IMPACT_TIMER.daemon = True
    IMPACT_TIMER.start()

    IMPACT_TOKEN = new_token                   # remember what we armed
    status = "updated" if was_live else "armed"
    print(f"[TIMER] {status}: key={key} in {delay_s:.3f}s  (dist={dist_px:.1f}px, cls={LABELS.get(int(cls_id), cls_id)})")


def first_mask_hit_starburst_then_ray(
    jake_point, tri_pos, theta_deg, masks_np, classes_np, H, W,
    up2_px=SAMPLE_UP_PX, step_px=2, exclude_classes=(RAIL_ID,), danger_only=False
):
    """
    Follow the path used in viz:
      1) straight line JAKE_POINT -> tri_pos,
      2) then continue from tri_pos along the angled probe ray (theta_deg) for up to `up2_px`.
    Return (dist_px_from_jake, (x_hit, y_hit), class_id) for the first mask hit
    (skipping rails or restricted to DANGER_RED if danger_only=True). If none, return (None, None, None).
    """
    if masks_np is None or classes_np is None or masks_np.size == 0:
        return (None, None, None)

    # choose class indices to test
    if danger_only:
        test_idxs = [i for i,c in enumerate(classes_np) if int(c) in DANGER_RED]
    else:
        test_idxs = [i for i,c in enumerate(classes_np) if int(c) not in exclude_classes]
    if not test_idxs:
        return (None, None, None)

    # scale factors from frame to mask grid
    mh, mw = masks_np.shape[1], masks_np.shape[2]
    sx = (mw - 1) / max(1, (W - 1))
    sy = (mh - 1) / max(1, (H - 1))

    def _hit_at(xs, ys):
        mx = _clampi(int(round(xs * sx)), 0, mw-1)
        my = _clampi(int(round(ys * sy)), 0, mh-1)
        for i in test_idxs:
            if masks_np[i][my, mx] > 0.5:
                return int(classes_np[i])
        return None

    # ---- segment 1: JAKE_POINT -> triangle apex
    x0, y0 = map(int, jake_point)
    x1, y1 = map(int, tri_pos)
    x0 = _clampi(x0, 0, W-1); y0 = _clampi(y0, 0, H-1)
    x1 = _clampi(x1, 0, W-1); y1 = _clampi(y1, 0, H-1)

    dx = x1 - x0; dy = y1 - y0
    seg1_len = max(1e-6, math.hypot(dx, dy))
    n1 = max(1, int(seg1_len // max(1, step_px)))
    for k in range(1, n1 + 1):
        t = min(1.0, (k * step_px) / seg1_len)
        xs = _clampi(int(round(x0 + dx * t)), 0, W-1)
        ys = _clampi(int(round(y0 + dy * t)), 0, H-1)
        cls_hit = _hit_at(xs, ys)
        if cls_hit is not None:
            dist = math.hypot(xs - x0, ys - y0)  # Euclidean from Jake
            return (float(dist), (int(xs), int(ys)), int(cls_hit))

    # ---- segment 2: continue from triangle along angled probe (same as classify rays)
    dxr, dyr = unit_from_theta(theta_deg)
    n2 = max(1, int(up2_px // max(1, step_px)))
    for k in range(1, n2 + 1):
        t = k * step_px
        xs = _clampi(int(round(x1 + dxr * t)), 0, W-1)
        ys = _clampi(int(round(y1 + dyr * t)), 0, H-1)
        cls_hit = _hit_at(xs, ys)
        if cls_hit is not None:
            dist = seg1_len + math.hypot(xs - x1, ys - y1)  # piecewise length from Jake
            return (float(dist), (int(xs), int(ys)), int(cls_hit))

    return (None, None, None)


def first_mask_hit_along_segment(jake_point, tri_pos, masks_np, classes_np,
                                 H, W, exclude_classes=(RAIL_ID,), step_px=1):
    """
    Walk the straight segment from JAKE_POINT -> Jake's triangle apex.
    Return (distance_px, (x_hit, y_hit), class_id) for the first mask hit,
    skipping any classes in `exclude_classes`. If none, return (None, None, None).
    """
    if masks_np is None or masks_np.size == 0 or classes_np is None or len(tri_pos) != 2:
        return (None, None, None)

    x0, y0 = map(int, jake_point)
    x1, y1 = map(int, tri_pos)
    # clamp
    x0 = _clampi(x0, 0, W-1); y0 = _clampi(y0, 0, H-1)
    x1 = _clampi(x1, 0, W-1); y1 = _clampi(y1, 0, H-1)

    dx = x1 - x0
    dy = y1 - y0
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-6:
        return (None, None, None)

    # map-to-mask scale factors
    mh, mw = masks_np.shape[1], masks_np.shape[2]
    sx = (mw - 1) / max(1, (W - 1))
    sy = (mh - 1) / max(1, (H - 1))

    # prebuild indices of classes we actually test (skip excluded, e.g., RAIL_ID)
    test_idxs = [i for i, c in enumerate(classes_np) if int(c) not in exclude_classes]
    if not test_idxs:
        return (None, None, None)

    # step along the line; start at step 1 so we don't immediately "hit" Jake's pixel
    n_steps = max(1, int(seg_len // max(1, step_px)))
    for k in range(1, n_steps + 1):
        t = (k * step_px) / seg_len
        if t > 1.0: t = 1.0
        xs = _clampi(int(round(x0 + dx * t)), 0, W-1)
        ys = _clampi(int(round(y0 + dy * t)), 0, H-1)

        mx = _clampi(int(round(xs * sx)), 0, mw-1)
        my = _clampi(int(round(ys * sy)), 0, mh-1)

        for i in test_idxs:
            # masks_np is float in [0,1]; >0.5 treated as hit (consistent with rest of code)
            if masks_np[i][my, mx] > 0.5:
                # distance in pixels along the segment to this sample
                dist_px = math.hypot(xs - x0, ys - y0)
                return (float(dist_px), (int(xs), int(ys)), int(classes_np[i]))

    return (None, None, None)

# --- helper: % of frame matching RGBA target with ±2% per-channel tolerance ---
import numpy as np, cv2, time

def percent_of_color_rgba(img, rgba=(210, 36, 35, 255), tol_frac=0.05):
    """
    img: OpenCV image (BGR or BGRA uint8)
    rgba: target color in RGBA
    tol_frac: per-channel tolerance fraction of 255 (0.02 -> ±5)
    returns: percentage of pixels within tolerance (0..100)
    """
    if img.ndim != 3 or img.dtype != np.uint8:
        raise ValueError("img must be uint8 BGR/BGRA")

    tol = int(round(255 * tol_frac))  # ±5 for 2%

    if img.shape[2] == 3:  # BGR frame (typical OpenCV)
        target = np.array([rgba[2], rgba[1], rgba[0]], dtype=np.int16)  # RGBA -> BGR
    elif img.shape[2] == 4:  # BGRA
        target = np.array([rgba[2], rgba[1], rgba[0], rgba[3]], dtype=np.int16)  # RGBA -> BGRA
    else:
        raise ValueError("Unsupported number of channels")

    lo = np.clip(target - tol, 0, 255).astype(np.uint8)
    hi = np.clip(target + tol, 0, 255).astype(np.uint8)

    t0 = time.perf_counter()
    mask = cv2.inRange(img, lo, hi)              # 255 where inside the tolerance, else 0
    pct = 100.0 * (cv2.countNonZero(mask) / mask.size)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return pct, elapsed_ms



# --- tunnel wall color gate (HSV) ---
LOWBARRIER1_ID   = 4
ORANGETRAIN_ID   = 6
WALL_STRIP_PX    = 14          # vertical strip height checked just above the barrier
WALL_MATCH_FRAC  = 0.135       # % of “wall” pixels required to relabel
WALL_ORANGE_LO = np.array([5,  80,  60], dtype=np.uint8)   # H,S,V (lo)
WALL_ORANGE_HI = np.array([35, 255, 255], dtype=np.uint8)  # H,S,V (hi)


def promote_lowbarrier_when_wall(frame_bgr, masks_np, classes_np,
                                 strip_px=WALL_STRIP_PX, frac_thresh=WALL_MATCH_FRAC):
    """
    If a LOWBARRIER1 has an orange 'tunnel wall' strip right behind it,
    relabel that instance to ORANGETRAIN (treated as RED).
    """
    if masks_np is None or classes_np is None or masks_np.size == 0:
        return classes_np

    H, W = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    wall_u8 = cv2.inRange(hsv, WALL_ORANGE_LO, WALL_ORANGE_HI)  # 0/255

    # iterate only over LOWBARRIER1 instances
    for i, cls in enumerate(classes_np):
        if int(cls) != LOWBARRIER1_ID:
            continue

        m = masks_np[i]
        # upsample to frame size if needed
        if m.shape != (H, W):
            m_full = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            m_full = m.astype(bool, copy=False)

        ys, xs = np.where(m_full)
        if xs.size == 0:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, _  = ys.min(), ys.max()

        # check a strip immediately above the barrier (toward smaller y)
        yb0 = max(0, y0 - strip_px)
        yb1 = y0
        if yb1 <= yb0:
            continue

        strip = wall_u8[yb0:yb1, x0:x1+1]
        if strip.size == 0:
            continue

        frac = float(cv2.countNonZero(strip)) / strip.size
        print(f'SCANNING FOR ORANGE WALL : % ORANGE DETECTED IS {frac}')
        if frac <= frac_thresh:
            print('UPGRADING CLASS TO RED')
            classes_np[i] = ORANGETRAIN_ID  # promote to a RED class
        else:
            print('SIGNAL NOT STRONG ENOUGH')

    return classes_np


# extra classes/sets
WARN_FOR_MOVE = {2, 3, 4, 5, 8}      # yellow set that should try to sidestep if a green exists
JUMP_SET      = {3, 5, 10}           # Jump, LowBarrier2, Sidewalk
DUCK_SET      = {2, 4}               # HighBarrier1, LowBarrier1

# action keys (change if your emulator uses different binds)
JUMP_KEY = "up"
DUCK_KEY = "down"

# --- "white-ish" lane probe (5x5 box counts) ---
# tune these if your Jake sprite/board highlight isn't pure white
WHITE_MIN = np.array([220, 220, 220], dtype=np.uint8)  # BGR lower bound
WHITE_MAX = np.array([255, 255, 255], dtype=np.uint8)  # BGR upper bound
BOX_RAD   = 2  # 5x5 => radius 2

def _count_white_around(img_bgr, pt, box_rad=BOX_RAD):
    H, W = img_bgr.shape[:2]
    x, y = pt
    x0 = max(0, x - box_rad); x1 = min(W, x + box_rad + 1)
    y0 = max(0, y - box_rad); y1 = min(H, y + box_rad + 1)
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0
    mask = cv2.inRange(roi, WHITE_MIN, WHITE_MAX)
    return int(cv2.countNonZero(mask))

def _detect_lane_by_whiteness(img_bgr):
    # returns lane index 0/1/2 chosen by the largest white count;
    # if all zero, returns None to keep previous lane
    counts = [
        _count_white_around(img_bgr, LANE_LEFT),
        _count_white_around(img_bgr, LANE_MID),
        _count_white_around(img_bgr, LANE_RIGHT),
    ]
    best_idx = int(np.argmax(counts))
    return best_idx if counts[best_idx] > 0 else None


# action cooldown so we don't spam jump/duck
try:
    last_action_ts
except NameError:
    last_action_ts = 0.31
ACTION_COOLDOWN_S = 0.31

# distance threshold (pixels) from Jake to triangle apex for action decisions
ACTION_DIST_PX = 30

def _is_warn(cls_id: int | None) -> bool:
    return (cls_id is not None) and (int(cls_id) in WARN_FOR_MOVE)

def _schedule(fn, *args, **kwargs):
    Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()

MIN_GREEN_AHEAD_PX = 400
def _filter_green_far(cands, jake_band_y: int, min_ahead_px: int = MIN_GREEN_AHEAD_PX):
    """Keep only green triangles that are at least `min_ahead_px` above Jake's y band."""
    out = []
    for c in cands:
        _, yt = c["pos"]
        if (jake_band_y - yt) >= min_ahead_px:  # keep if ≥ 400 px ahead
            out.append(c)
    return out

def first_red_hit_y(pos, masks_np, classes_np, H, W, band_px=6, step_px=5, max_up=SAMPLE_UP_PX):
    """Return the screen y of the first RED pixel straight above `pos`, or None."""
    if masks_np is None or masks_np.size == 0: return None
    mh, mw = masks_np.shape[1], masks_np.shape[2]
    sx = (mw - 1) / max(1, (W - 1)); sy = (mh - 1) / max(1, (H - 1))
    
    red_idx = [i for i, c in enumerate(classes_np) if int(c) in DANGER_RED]
    if not red_idx: return None

    x0, y0 = int(pos[0]), int(pos[1])
    x0 = _clampi(x0, 0, W-1); y0 = _clampi(y0, 0, H-1)

    for t in range(step_px, max_up + 1, step_px):
        y = _clampi(y0 - t, 0, H-1)
        for dx in range(-band_px, band_px + 1):
            x = _clampi(x0 + dx, 0, W-1)
            mx = _clampi(int(round(x * sx)), 0, mw-1)
            my = _clampi(int(round(y * sy)), 0, mh-1)
            for i in red_idx:
                if masks_np[i][my, mx] > 0.5:
                    return y
    return None

def first_hit_y(pos, masks_np, classes_np, H, W, class_set, band_px=6, step_px=5, max_up=SAMPLE_UP_PX):
    """Return the screen y of the first pixel (straight up) whose class ∈ class_set."""
    if masks_np is None or masks_np.size == 0: return None
    mh, mw = masks_np.shape[1], masks_np.shape[2]
    sx = (mw - 1) / max(1, (W - 1)); sy = (mh - 1) / max(1, (H - 1))
    idxs = [i for i, c in enumerate(classes_np) if int(c) in class_set]
    if not idxs: return None

    x0, y0 = int(pos[0]), int(pos[1])
    x0 = _clampi(x0, 0, W-1); y0 = _clampi(y0, 0, H-1)

    for t in range(step_px, max_up + 1, step_px):
        y = _clampi(y0 - t, 0, H-1)
        for dx in range(-band_px, band_px + 1):
            x = _clampi(x0 + dx, 0, W-1)
            mx = _clampi(int(round(x * sx)), 0, mw-1)
            my = _clampi(int(round(y * sy)), 0, mh-1)
            for i in idxs:
                if masks_np[i][my, mx] > 0.5:
                    return y
    return None



# Only step from RED into a YELLOW lane if its triangle is far enough ahead
MIN_YELLOW_AHEAD_PX = 400
def _filter_yellow_far(cands, jake_band_y: int, min_ahead_px: int = MIN_YELLOW_AHEAD_PX):
    """Keep only yellow triangles that are at least `min_ahead_px` above Jake's y band."""
    out = []
    for c in cands:
        _, yt = c["pos"]
        if (jake_band_y - yt) >= min_ahead_px:
            out.append(c)
    return out


def _try_duck():
    if not MOVEMENT_ENABLED:
        return
    global last_action_ts
    now = time.perf_counter()
    if now - last_action_ts >= ACTION_COOLDOWN_S:
        last_action_ts = now
        _schedule(pyautogui.press, DUCK_KEY)
try:
    last_move_ts
except NameError:
    last_move_ts = 0.0

MOVE_COOLDOWN_S = 0.10  # 100 ms

def _is_danger(cls_id: int | None) -> bool:
    return (cls_id is not None) and (int(cls_id) in DANGER_RED)

def _is_safe(cls_id: int | None) -> bool:
    return not _is_danger(cls_id)

def _filter_by_lane(cands, jx: int, lane_idx: int):
    """Prune triangles based on current lane:
       - lane 0 (left): drop triangles with x < jx
       - lane 2 (right): drop triangles with x > jx
       - lane 1 (mid): keep all
    """
    if lane_idx == 0:
        return [c for c in cands if c["pos"][0] >= jx]
    if lane_idx == 2:
        return [c for c in cands if c["pos"][0] <= jx]
    return cands

def _pick_best_safe_triangle(cands, jx: int):
    """Prefer triangles with hit_class == None; otherwise any non-danger.
       Break ties by smallest |x - jx|.
    """
    if not cands:
        return None
    none_hits  = [c for c in cands if c["hit_class"] is None]
    safe_hits  = [c for c in cands if c["hit_class"] is not None and _is_safe(c["hit_class"])]
    pool = none_hits if none_hits else safe_hits
    if not pool:
        return None
    # exclude triangles exactly aligned with Jake in x (no direction)
    pool = [c for c in pool if c["pos"][0] != jx] or pool
    return min(pool, key=lambda c: abs(c["pos"][0] - jx))

def _issue_move_towards_x(jx: int, tx: int, *, sidewalk_present: bool = False):
    global lane, last_move_ts, _synth_block_until
    if not MOVEMENT_ENABLED:
        return

    now = time.perf_counter()
    dt = now - last_move_ts
    if dt < MOVE_COOLDOWN_S:
        remaining = MOVE_COOLDOWN_S - dt
        print(f"[COOLDOWN] Lane move blocked: {remaining*1000:.0f} ms remaining -> Please expect delays")
        return

    # Re-entry hysteresis gate — block until we’ve seen N consecutive intents
    if not _reentry_gate_allow(jx, tx):
        return
    def _jump_then_duck_if_sidewalk():
        if sidewalk_present:
            print("[SIDEWALK] jump → lane-change → duck (0.2s)")
            # Jump now
            _schedule(pyautogui.press, JUMP_KEY)
            # Duck after a short delay
            Timer(SIDEWALK_JUMP_THEN_DUCK_DELAY_S, lambda: MOVEMENT_ENABLED and pyautogui.press(DUCK_KEY)).start()

    if tx < jx and lane > MIN_LANE:
        _jump_then_duck_if_sidewalk()
        pause_inference()  # freeze briefly to avoid mid-lane frames
        _synth_block_until = time.monotonic() + SYNTHETIC_SUPPRESS_S
        pyautogui.press('left')
        lane = max(MIN_LANE, lane - 1)
        print(f"[AI MOVE] left -> Lane {lane}")
        last_move_ts = now

    elif tx > jx and lane < MAX_LANE:
        _jump_then_duck_if_sidewalk()
        pause_inference()
        _synth_block_until = time.monotonic() + SYNTHETIC_SUPPRESS_S
        pyautogui.press('right')
        lane = min(MAX_LANE, lane + 1)
        print(f"[AI MOVE] right -> Lane {lane}")
        last_move_ts = now

    else:
        print('WE ARE COOKED')


#============================================================================================================================================


# =======================
# Lane/keyboard state
# =======================

# ===== Debounce / cooldown =====
COOLDOWN_MS = 20
_last_press_ts = 0.0  # monotonic seconds

def on_press(key):
    global lane, running, _last_press_ts, _synth_block_until, save_frames
    now = time.monotonic()

    # swallow AI-generated lane key events during the suppression window
    if key in (keyboard.Key.left, keyboard.Key.right) and now < _synth_block_until:
        return

    if key != keyboard.Key.esc and (now - _last_press_ts) * 1000.0 < COOLDOWN_MS:
        return

    try:
        if key == keyboard.Key.left:
            lane = max(MIN_LANE, lane - 1)
            _last_press_ts = now
            print(f"Moved Left into → Lane {lane}")

        elif key == keyboard.Key.right:
            lane = min(MAX_LANE, lane + 1)
            _last_press_ts = now
            print(f"Moved Right into → Lane {lane}")

        # NEW: press 'g' at any time to toggle saving
        elif getattr(key, "char", "").lower() == "g":
            save_frames = not save_frames
            state = "ENABLED" if save_frames else "DISABLED"
            print(f"[G] Saving {state}")

        elif key == keyboard.Key.esc:
            running = False
            return False
    except Exception as e:
        print(f"Error: {e}")

def _is_jake_on_sidewalk(jake_point, masks_np, classes_np, H, W) -> bool:
    """
    True iff the SIDEWALK mask covers the EXACT screen pixel under Jake.
    Uses already-computed masks/classes. O(1).
    """
    if masks_np is None or classes_np is None or masks_np.size == 0:
        return False

    idxs = np.where(classes_np.astype(int) == SIDEWALK_ID)[0]
    if idxs.size == 0:
        return False

    mh, mw = masks_np.shape[1], masks_np.shape[2]
    sx = (mw - 1) / max(1, (W - 1))
    sy = (mh - 1) / max(1, (H - 1))

    x, y = int(jake_point[0]), int(jake_point[1])
    mx = _clampi(int(round(x * sx)), 0, mw - 1)
    my = _clampi(int(round(y * sy)), 0, mh - 1)

    for i in idxs:
        if masks_np[i][my, mx] > 0.5:
            return True
    return False

# =======================
# System/Backends
# =======================
cv2.setUseOptimized(True)
try: cv2.setNumThreads(max(1, (os.cpu_count() or 1) - 1))
except Exception: pass

if torch.cuda.is_available():
    device, half = 0, True
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device, half = "mps", False
else:
    device, half = "cpu", False

# =======================
# Model
# =======================
model = YOLO(weights)
try: model.fuse()
except Exception: pass

# warmup
_dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
_ = model.predict(_dummy, task="segment", imgsz=IMG_SIZE, device=device,
                  conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET)

# =======================
# Precomputed
# =======================
TARGETS_BGR_F32 = np.array([(r,g,b)[::-1] for (r,g,b) in TARGET_COLORS_RGB], dtype=np.float32)
TOL2            = TOLERANCE * TOLERANCE

# Class buckets for probe classification
DANGER_RED   = {1, 6, 7, 11}
WARN_YELLOW  = {2, 3, 4, 5, 8}
BOOTS_PINK   = {0}

CLASS_COLOURS = {
    0:(255,255,0),1:(192,192,192),2:(0,128,255),3:(0,255,0),
    4:(255,0,255),5:(0,255,255),6:(255,128,0),7:(128,0,255),
    8:(0,0,128),9:(0,0,255),10:(128,128,0),11:(255,255,102)
}
LABELS = {
    0:"BOOTS",1:"GREYTRAIN",2:"HIGHBARRIER1",3:"JUMP",4:"LOWBARRIER1",
    5:"LOWBARRIER2",6:"ORANGETRAIN",7:"PILLAR",8:"RAMP",9:"RAILS",
    10:"SIDEWALK",11:"YELLOWTRAIN"
}

# -------- ON_TOP fast-state (train/ramp/rails) --------
ID_GREYTRAIN   = 1
ID_ORANGETRAIN = 6
ID_YELLOWTRAIN = 11
ID_RAMP        = 8
ID_RAILS       = 9

TRAIN_TOPS = {ID_GREYTRAIN, ID_ORANGETRAIN, ID_YELLOWTRAIN}
INTEREST   = TRAIN_TOPS | {ID_RAMP, ID_RAILS}

ONTOP_BUDGET_MS = 4.0   # bail if > 3 ms

class OnTopTracker:
    """
    IGNORE airtime. Rules:
      - Any train top -> on_top = True instantly
      - RAMP seen in 2 consecutive frames -> on_top = True
      - RAILS seen in 2 consecutive frames -> on_top = False
    """
    __slots__ = ("on_top", "ramp_streak", "rails_streak")
    def __init__(self, start_on_top: bool = False):
        self.on_top = start_on_top
        self.ramp_streak = 0
        self.rails_streak = 0

    def update(self, has_train: bool, has_ramp: bool, has_rails: bool) -> bool:
        if has_train:                   # immediate lock to top
            self.on_top = True
            self.ramp_streak = 0
            self.rails_streak = 0
            return self.on_top

        # 2-frame debounce for ramp / rails
        if has_ramp:
            self.ramp_streak += 1
            if self.ramp_streak >= 1:
                self.on_top = True
        else:
            self.ramp_streak = 0

        if has_rails:
            self.rails_streak += 1
            if self.rails_streak >= 1:
                self.on_top = False
        else:
            self.rails_streak = 0

        return self.on_top

# global tracker & last computed state
ONTOP_TRACKER = OnTopTracker(start_on_top=False)
ON_TOP        = False   # last stable state


# ====== tiny helpers ======
def _clampi(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)
def _fmt_px(v):
    return f"{v:.1f}px" if v is not None else "n/a"

# --- tiny cache for window indices to avoid per-frame recompute ---
_OT_WIN_CACHE = {}  # key: (mh,mw,H,W,center_xy,r) -> (my_idx_t, mx_idx_t)

def _ot_get_window_indices(mh, mw, H, W, center_xy, r, device):
    key = (mh, mw, H, W, center_xy, r)
    if key in _OT_WIN_CACHE:
        return _OT_WIN_CACHE[key]
    cx, cy = center_xy
    cx = max(0, min(W-1, int(cx)))
    cy = max(0, min(H-1, int(cy)))
    sx = (mw - 1) / max(1, (W - 1))
    sy = (mh - 1) / max(1, (H - 1))
    mx = []
    my = []
    for dy in (-1, 0, 1):
        yy = max(0, min(H-1, cy + dy))
        my.append(int(round(yy * sy)))
    for dx in (-1, 0, 1):
        xx = max(0, min(W-1, cx + dx))
        mx.append(int(round(xx * sx)))
    my_t = torch.tensor(my, dtype=torch.long, device=device)
    mx_t = torch.tensor(mx, dtype=torch.long, device=device)
    _OT_WIN_CACHE[key] = (my_t, mx_t)
    return my_t, mx_t


@torch.inference_mode()
@torch.inference_mode()
def compute_on_top_state_fast(yres, H, W, lane_idx: int, mask_thresh: float = 0.5):
    """
    Super-cheap ON_TOP read:
      - pick ONE screen point based on lane (LANE_POINTS + ONTOP_PROBE_OFFSETS[lane])
      - map it to the mask grid and test only INTEREST classes
      - decide ON_TOP from the class at that pixel (train tops/ramp/rails rules)
    Returns: (on_top_or_last, timed_out_bool, dt_ms, seen_set)
    """
    global ONTOP_TRACKER, ON_TOP

    t0 = time.perf_counter()

    # Minimal guards
    if (yres is None) or (getattr(yres, "masks", None) is None) \
       or (getattr(yres.masks, "data", None) is None) \
       or (getattr(yres, "boxes", None) is None) \
       or (getattr(yres.boxes, "cls", None) is None):
        return ON_TOP, False, (time.perf_counter() - t0) * 1000.0, set()

    # Pick the one probe pixel for this lane (screen space)
    x0, y0 = LANE_POINTS[lane_idx]
    dx, dy = ONTOP_PROBE_OFFSETS[lane_idx]
    xs = _clampi(int(x0 + dx), 0, W - 1)
    ys = _clampi(int(y0 + dy), 0, H - 1)

    # Map to mask grid
    mdata = yres.masks.data            # [n, mh, mw] on device
    mh, mw = mdata.shape[-2:]
    device = mdata.device
    sx = (mw - 1) / max(1, (W - 1))
    sy = (mh - 1) / max(1, (H - 1))
    mx = _clampi(int(round(xs * sx)), 0, mw - 1)
    my = _clampi(int(round(ys * sy)), 0, mh - 1)

    # Classes (CPU), restrict to INTEREST; then get the tiny subset on device
    classes_cpu = yres.boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)
    if classes_cpu.size == 0:
        return ON_TOP, False, (time.perf_counter() - t0) * 1000.0, set()

    interest_mask = np.isin(classes_cpu, list(INTEREST))
    if not np.any(interest_mask):
        # No relevant instances this frame: lightweight debounce update
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if dt_ms > ONTOP_BUDGET_MS:
            return ON_TOP, True, dt_ms, set()
        ON_TOP = ONTOP_TRACKER.update(False, False, False)
        return ON_TOP, False, dt_ms, set()

    interest_idx_cpu = np.nonzero(interest_mask)[0]
    interest_idx = torch.from_numpy(interest_idx_cpu).to(device=device, dtype=torch.long)
    sub = mdata.index_select(0, interest_idx)        # [k, mh, mw]
    vals = sub[:, my, mx]                             # [k] at ONE pixel
    hit = (vals > float(mask_thresh)).detach().cpu().numpy()

    seen_ids = set(classes_cpu[interest_mask][hit].tolist())
    has_train = any(c in TRAIN_TOPS for c in seen_ids)
    has_ramp  = (ID_RAMP  in seen_ids)
    has_rails = (ID_RAILS in seen_ids)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    if dt_ms > ONTOP_BUDGET_MS:
        # Over budget? keep last stable; don't touch debouncers
        return ON_TOP, True, dt_ms, seen_ids

    # Within budget → update tracker (same rules as before)
    ON_TOP = ONTOP_TRACKER.update(has_train=has_train, has_ramp=has_ramp, has_rails=has_rails)
    return ON_TOP, False, dt_ms, seen_ids



# =======================
# Parsec to front + click Start (non-blocking failures)
# =======================
try:
    subprocess.run(["osascript", "-e", 'tell application "Parsec" to activate'], check=False)
    time.sleep(0.4)
except Exception:
    pass

try:
    pyautogui.click(start_click)
except Exception:
    pass

# =======================
# Fast rails green finder
# =======================
def highlight_rails_mask_only_fast(img_bgr, rail_mask):
    __p_fn = __PROF('post.fn.highlight_rails')
    H, W = rail_mask.shape
    if not rail_mask.any():
        return np.zeros((H, W), dtype=bool)

    # 0/255, single channel for cv2.boundingRect (same semantics as before)
    rail_u8 = rail_mask.astype(np.uint8, copy=False) * 255

    __p_rect = __PROF('post.highlight.boundingRect')
    x, y, w, h = cv2.boundingRect(rail_u8)
    __p_rect()

    img_roi  = img_bgr[y:y+h, x:x+w]
    mask_roi = rail_u8[y:y+h, x:x+w]  # still 0/255

    # --- EXACT colour test, but cheaper --------------------------------------
    # Integer arithmetic is exact for our ranges; float32 was exact too, so
    # the <= TOL2 comparison produces identical booleans.
    __p_color = __PROF('post.highlight.color_distance')
    img_i16 = img_roi.astype(np.int16, copy=False)           # [-32768..32767]
    # Targets in BGR as int16 (same values as your TARGETS_BGR_F32)
    targets_i16 = TARGETS_BGR_F32.astype(np.int16, copy=False)
    tol2_i = int(TOL2)

    # OR together the per-target hits (keeps output identical to ".any(-1)")
    colour_hit = np.zeros((h, w), dtype=bool)
    for c in targets_i16:  # typically just 2 colours
        # per-channel diffs as int32 to avoid overflow when squaring
        db = img_i16[..., 0].astype(np.int32) - int(c[0])
        dg = img_i16[..., 1].astype(np.int32) - int(c[1])
        dr = img_i16[..., 2].astype(np.int32) - int(c[2])
        dist2 = db * db + dg * dg + dr * dr
        colour_hit |= (dist2 <= tol2_i)
    __p_color()

    # Combine with the rail ROI exactly as before
    combined = np.logical_and(colour_hit, mask_roi.astype(bool, copy=False))

    comp = combined.astype(np.uint8, copy=False)  # 0/1 is fine for CC

    __p_cc = __PROF('post.highlight.cc_stats')
    n, lbls, stats, _ = cv2.connectedComponentsWithStats(comp, 8)
    __p_cc()

    if n <= 1:
        __p_fn()
        return np.zeros((H, W), dtype=bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    hs    = stats[1:, cv2.CC_STAT_HEIGHT]

    __p_filt = __PROF('post.highlight.filter_regions')
    keep_idx = np.where((areas >= MIN_REGION_SIZE) & (hs >= MIN_REGION_HEIGHT))[0] + 1
    # Vectorized label selection (exact same result as the Python loop)
    good = np.isin(lbls, keep_idx)
    full = np.zeros((H, W), dtype=bool)
    full[y:y+h, x:x+w] = good
    __p_filt()

    __p_fn()
    return full

#GOOGOGOODGOODOD

def red_vs_green_score(red_mask, green_mask):
    ksz = (HEAT_BLUR_KSIZE, HEAT_BLUR_KSIZE)
    H, W = red_mask.shape[:2]
    _rgs_ensure(H, W)

    r32   = _RGS_CACHE["r32"]
    g32   = _RGS_CACHE["g32"]
    diff  = _RGS_CACHE["diff"]
    norm  = _RGS_CACHE["norm"]
    out_u8= _RGS_CACHE["u8"]

    # boxFilter == blur; ddepth=CV_32F gives the same float32 result as your original
    red_u8   = _as_u8_view(red_mask)
    green_u8 = _as_u8_view(green_mask)
    cv2.boxFilter(red_u8,   ddepth=cv2.CV_32F, ksize=ksz, dst=r32,   normalize=True, borderType=cv2.BORDER_DEFAULT)
    cv2.boxFilter(green_u8, ddepth=cv2.CV_32F, ksize=ksz, dst=g32,   normalize=True, borderType=cv2.BORDER_DEFAULT)

    # diff = r - g (exact)
    cv2.subtract(r32, g32, diff)

    # amax = max(abs(diff)) + 1e-6 (exactly matches np.max(np.abs(diff)) + 1e-6)
    mn, mx, _, _ = cv2.minMaxLoc(diff)
    amax = max(abs(mn), abs(mx)) + 1e-6

    # norm = (diff / (2*amax) + 0.5) * 255, then clip and truncate to uint8
    cv2.multiply(diff, 1.0 / (2.0 * amax), norm)  # norm = diff * scale
    cv2.add(norm, 0.5, norm)
    cv2.multiply(norm, 255.0, norm)
    cv2.max(norm, 0.0, norm); cv2.min(norm, 255.0, norm)
    np.copyto(out_u8, norm, casting="unsafe")      # float32 -> uint8 (truncation)

    return out_u8



def purple_triangles(score, H):
    top_ex = int(H * EXCLUDE_TOP_FRAC)
    bot_ex = int(H * EXCLUDE_BOTTOM_FRAC)

    dark = (score >= RED_SCORE_THRESH).astype(np.uint8, copy=False)
    if top_ex: dark[:top_ex, :] = 0
    if bot_ex: dark[-bot_ex:, :] = 0

    # same morphology (kernel precomputed)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, _SE_OPEN_5x9, iterations=1)

    # exact replacement for int(dark.sum()) since dark is 0/1
    total_dark = int(cv2.countNonZero(dark))
    if total_dark == 0:
        return [], None

    frac_thresh = int(np.ceil(MIN_DARK_FRACTION * total_dark))

    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    if n_lbl <= 1:
        return [], None

    tris = []
    for lbl in range(1, n_lbl):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < MIN_DARK_RED_AREA or area < frac_thresh:
            continue

        xL = stats[lbl, cv2.CC_STAT_LEFT]
        yT = stats[lbl, cv2.CC_STAT_TOP]
        w  = stats[lbl, cv2.CC_STAT_WIDTH]

        # examine only the top row inside the bbox (exact same set as xs[ys==y_top])
        row = lbls[yT, xL:xL + w]
        xs_rel = np.flatnonzero(row == lbl)
        if xs_rel.size == 0:
            continue  # should not happen; defensive
        x_mid = int(xs_rel.mean()) + xL

        tris.append((x_mid, int(yT)))

    if not tris:
        return [], None

    best = min(tris, key=lambda xy: xy[1])
    return tris, best


# ===== Bearing-based Jake triangle selection =====
def signed_degrees_from_vertical(dx, dy):
    if dx == 0 and dy == 0: return 0.0
    return -math.degrees(math.atan2(dx, -dy))

def select_triangle_by_bearing(tri_positions, jx, jy, target_deg, min_dy=6):
    best_i, best_deg, best_err = -1, None, None
    for i, (xt, yt) in enumerate(tri_positions):
        dy = yt - jy
        if dy >= -min_dy:  # must be above Jake
            continue
        deg = signed_degrees_from_vertical(xt - jx, dy)
        err = abs(deg - target_deg)
        if (best_err is None) or (err < best_err):
            best_i, best_deg, best_err = i, deg, err
    return best_i, best_deg, best_err

# ===== Lane-aware curved sampling (precompute sin/cos) =====
def _precompute_trig():
    angles = sorted(set([0.0,
        BEND_LEFT_STATE_RIGHT_DEG,
        BEND_MID_STATE_RIGHT_DEG,
        BEND_MID_STATE_LEFT_DEG,
        BEND_RIGHT_STATE_LEFT_DEG
    ]))
    table = {}
    for a in angles:
        r = math.radians(a)
        table[a] = (math.sin(r), -math.cos(r))  # (dx, dy) for unit ray (up = -y)
    return table
TRIG_TABLE = _precompute_trig()

def pick_bend_angle(jake_point, xt, x_ref, idx, best_idx):
    if idx == best_idx:
        return 0.0
    if jake_point == LANE_LEFT:
        return BEND_LEFT_STATE_RIGHT_DEG if xt > x_ref else 0.0
    if jake_point == LANE_RIGHT:
        return BEND_RIGHT_STATE_LEFT_DEG if xt < x_ref else 0.0
    if xt > x_ref: return BEND_MID_STATE_RIGHT_DEG
    if xt < x_ref: return BEND_MID_STATE_LEFT_DEG
    return 0.0

# --------- walk-the-ray classifier (first-hit wins) ----------
def classify_triangles_at_sample_curved(
    tri_positions, masks_np, classes_np, H, W,
    jake_point, x_ref, best_idx, sample_px=SAMPLE_UP_PX, step_px=RAY_STEP_PX
):
    __p_fn = __PROF('post.fn.classify_triangles')
    if masks_np is None or classes_np is None or len(tri_positions) == 0:
        __p_fn()  # <-- call it
        return [], [], [], []


    mh, mw = masks_np.shape[1], masks_np.shape[2]
    sx = (mw - 1) / max(1, (W - 1))
    sy = (mh - 1) / max(1, (H - 1))

    red_idx    = [i for i, c in enumerate(classes_np) if int(c) in DANGER_RED]
    yellow_idx = [i for i, c in enumerate(classes_np) if int(c) in WARN_YELLOW]
    boots_idx  = [i for i, c in enumerate(classes_np) if int(c) in BOOTS_PINK]

    colours, rays, hit_class_ids, hit_distances_px = [], [], [], []
    max_k = max(1, sample_px // max(1, step_px))

    jx, jy = map(int, jake_point)

    for idx, (x0, y0) in enumerate(tri_positions):
        # Always force collinearity for Jake's triangle: extend Jake→triangle as a straight line
        if idx == best_idx:
            theta = signed_degrees_from_vertical(x0 - jx, y0 - jy)  # (was using xt by mistake)
            r = math.radians(theta)
            dx1, dy1 = math.sin(r), -math.cos(r)
        else:
            theta = pick_bend_angle(jake_point, x0, x_ref, idx, best_idx)
            dx1, dy1 = TRIG_TABLE[theta]



        hit_colour = COLOR_GREEN
        hit_cls = None
        hit_dist_px = None

        found = False
        for k in range(1, max_k + 1):
            t  = k * step_px
            xs = _clampi(int(round(x0 + dx1 * t)), 0, W-1)
            ys = _clampi(int(round(y0 + dy1 * t)), 0, H-1)
            mx = _clampi(int(round(xs * sx)), 0, mw-1)
            my = _clampi(int(round(ys * sy)), 0, mh-1)

            # RED first (so if red exists at a point, we record red distance)
            for i in red_idx:
                if masks_np[i][my, mx] > 0.5:
                    hit_colour = COLOR_RED
                    hit_cls = int(classes_np[i])
                    hit_dist_px = float(t)
                    found = True
                    break
            if found: break
            # then YELLOW
            for i in yellow_idx:
                if masks_np[i][my, mx] > 0.5:
                    hit_colour = COLOR_YELLOW
                    hit_cls = int(classes_np[i])
                    hit_dist_px = float(t)
                    found = True
                    break
            if found: break
            # then BOOTS
            for i in boots_idx:
                if masks_np[i][my, mx] > 0.5:
                    hit_colour = COLOR_PINK
                    hit_cls = int(classes_np[i])
                    hit_dist_px = float(t)
                    found = True
                    break
            if found: break

        x1 = _clampi(int(round(x0 + dx1 * sample_px)), 0, W-1)
        y1 = _clampi(int(round(y0 + dy1 * sample_px)), 0, H-1)

        colours.append(hit_colour)
        rays.append(((int(x0), int(y0)), (x1, y1), float(theta)))
        hit_class_ids.append(hit_cls)
        hit_distances_px.append(hit_dist_px)

    __p_fn()

    return colours, rays, hit_class_ids, hit_distances_px

# -----------------------------------------------------------------------

def _ahead_px_of_tri(t, jake_band_y: int) -> float:
    # positive if triangle apex is ahead (smaller y than Jake)
    return jake_band_y - int(t["pos"][1])

def _nearest_ramp_by_y(cands, jake_band_y: int):
    """Among hypergreen ramps (class 8), pick the one closest ahead in Y."""
    ramps = [c for c in cands
             if (c.get("hit_class") is not None and int(c["hit_class"]) == ID_RAMP)]
    if not ramps:
        return None
    # All your *_far filters already ensure "ahead >= threshold"; still safe to sort by ahead distance
    return min(ramps, key=lambda c: _ahead_px_of_tri(c, jake_band_y))


def _prearm_jump_for_triangle(tri, idx, jake_point, tri_rays,
                              masks_np, classes_np, H, W):
    """
    Pre-arm the standard jump timer for a triangle if it's class 3 (JUMP).
    Uses the same starburst->ray distance the Jake-lane timer uses.
    """
    if tri is None:
        return
    cls_id = tri.get("hit_class")
    if cls_id is None or int(cls_id) != 3:
        return  # only class 3 (JUMP)

    # Angle used for this triangle’s sampling ray
    theta_deg = float(tri_rays[idx][2]) if (idx is not None and 0 <= idx < len(tri_rays)) else 0.0

    # Distance along Jake→tri, then along that ray, but only through jump-allowed classes
    dist_px, _, _ = first_mask_hit_starburst_then_ray_for_set(
        jake_point=jake_point,
        tri_pos=tri["pos"],
        theta_deg=theta_deg,
        masks_np=masks_np, classes_np=classes_np, H=H, W=W,
        allowed_classes=JUMP_SET,  # {3,5,10} in your code; class 3 is the one we arm
        up2_px=SAMPLE_UP_PX, step_px=2
    )

    if (dist_px is not None) and (IMPACT_MIN_PX < dist_px < IMPACT_MAX_PX):
        _arm_impact_timer(float(dist_px), 3)  # uses the same Timer/token path


# =======================
# Frame post-processing
# =======================
def process_frame_post(frame_bgr, yolo_res, jake_point):
    """
    Returns (…)
      tri_best_xy, tri_count, mask_count, to_cpu_ms, post_ms,
      masks_np, classes_np, rail_mask, green_mask,
      tri_positions, tri_colours, tri_rays,
      best_idx, best_deg, x_ref,
      tri_hit_classes, tri_summary
    """
    H, W = frame_bgr.shape[:2]
    if yolo_res.masks is None:
        return (None, 0, 0, 0.0, 0.0, None, None, None, None,
                [], [], [], None, None, None, [], [])

    t0 = time.perf_counter()
    __p_cpu = __PROF('post.to_cpu')
    masks_np = yolo_res.masks.data.detach().cpu().numpy()  # [n,h,w]
    if hasattr(yolo_res.masks, "cls") and yolo_res.masks.cls is not None:
        classes_np = yolo_res.masks.cls.detach().cpu().numpy().astype(int)
    else:
        classes_np = yolo_res.boxes.cls.detach().cpu().numpy().astype(int)

    __p_cpu()

    to_cpu_ms = (time.perf_counter() - t0) * 1000.0
    mask_count = int(masks_np.shape[0])
    if mask_count == 0 or classes_np.size == 0:
        return (None, 0, mask_count, to_cpu_ms, 0.0, masks_np, classes_np, None, None,
                [], [], [], None, None, None, [], [])

    __p_prom = __PROF('post.promote_lowbarrier')
    classes_np = promote_lowbarrier_when_wall(frame_bgr, masks_np, classes_np)
    __p_prom()

    rail_sel = (classes_np == RAIL_ID)
    if not np.any(rail_sel):
        return (None, 0, mask_count, to_cpu_ms, 0.0, masks_np, classes_np, None, None,
                [], [], [], None, None, None, [], [])

    t1 = time.perf_counter()
    __p_rail_union = __PROF('post.rail.union+resize')
    rail_masks = masks_np[rail_sel].astype(bool, copy=False)
    union = np.any(rail_masks, axis=0).astype(np.uint8, copy=False)
    rail_mask = cv2.resize(union, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool, copy=False)
    __p_rail_union()


    __p_highlight = __PROF('post.rails.highlight_green')
    green = highlight_rails_mask_only_fast(frame_bgr, rail_mask)
    __p_highlight()
    red   = np.logical_and(rail_mask, np.logical_not(green))

    __p_score = __PROF('post.heat.red_vs_green_score')
    score = red_vs_green_score(red, green)
    __p_score()

    __p_tris = __PROF('post.heat.purple_triangles')
    tri_positions, tri_best = purple_triangles(score, H)
    __p_tris()

    # Jake triangle by bearing
    lane_name = lane_name_from_point(jake_point)
    target_deg = LANE_TARGET_DEG[lane_name]
    xj, yj = jake_point

    MIN_AHEAD_FROM_JAKE_PX = 120  # tune (e.g., 80–160)
    tri_positions = [p for p in tri_positions if (yj - p[1]) >= MIN_AHEAD_FROM_JAKE_PX]

    # Filter triangles by absolute angle from vertical (≤ 45°) and at least 6px above Jake
    ANGLE_MAX_DEG = 45.0
    MIN_DY_ABOVE  = 100

    def _angle_ok(p):
        xt, yt = p
        dy = yt - yj
        if dy >= -MIN_DY_ABOVE:     # must be above Jake
            return False
        deg = signed_degrees_from_vertical(xt - xj, dy)
        return abs(deg) <= ANGLE_MAX_DEG

    tri_positions = [p for p in tri_positions if _angle_ok(p)]

    __p_pick = __PROF('post.pick_triangle_by_bearing')
    best_idx, best_deg, _ = select_triangle_by_bearing(tri_positions, xj, yj, target_deg, min_dy=6)
    __p_pick()

    # x_ref for bending
    if lane_name == "mid" and (best_idx is not None) and (0 <= best_idx < len(tri_positions)):
        x_ref = tri_positions[best_idx][0]
    else:
        x_ref = xj

    __p_classify = __PROF('post.classify_triangles_curved')
    tri_colours, tri_rays, tri_hit_classes, tri_hit_dists = classify_triangles_at_sample_curved(
        tri_positions, masks_np, classes_np, H, W, jake_point, x_ref, best_idx,
        SAMPLE_UP_PX, RAY_STEP_PX
    )
    __p_classify()

    if tri_positions and any(ty >= (H) for _, ty in tri_positions):
    # compute rail_grad/edge_dist and run the loop

        # --- edge-danger override: triangles too close to rail edges in bottom half ---
        #26/08 DISABLED DUE TO BETTER LOGIC GUARD AGAINST VORTEXS
        EDGE_PAD_PX = 0

        # distance-to-rail-edge map (in pixels)
        __p_edge = __PROF('post.edge_distance_override')
        rail_grad = cv2.morphologyEx(rail_mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        edge_bg   = (rail_grad == 0).astype(np.uint8) * 255
        edge_dist = cv2.distanceTransform(edge_bg, cv2.DIST_L2, 5)

        for i, (tx, ty) in enumerate(tri_positions):
            if ty >= (H // 2) and edge_dist[int(ty), int(tx)] <= EDGE_PAD_PX:
                tri_colours[i]     = COLOR_RED
                tri_hit_classes[i] = 1
                tri_hit_dists[i]   = 0.0
        __p_edge()


    post_ms = (time.perf_counter() - t1) * 1000.0

    # near config with other thresholds
    GREEN_NEAR_Y_TO_RED_PX = 615  # N pixels ahead of Jake -> force red -> With CONSEC logic can we forgo the < 615 requirment for greens
    # --- GREEN->RED relabel when too close in Y (ahead of Jake) ---
    if tri_positions:
        xj, yj = jake_point
        for i, (tx, ty) in enumerate(tri_positions):
            if tri_colours[i] == COLOR_GREEN:
                dy_ahead = yj - ty          # >0 if triangle is ahead of Jake
                print()
                print(f'CURRENT PERCIEVED LANE IS {lane_name.upper()}')
                print(f'GREEN IS {dy_ahead} px ahead of Jake')
                print()
                if 0 < dy_ahead < GREEN_NEAR_Y_TO_RED_PX:
                    tri_colours[i]     = COLOR_RED
                    tri_hit_classes[i] = 1          # any class in DANGER_RED works
                    tri_hit_dists[i]   = float(dy_ahead)  # optional annotation

    # --- SIDE-RAY → MIDDLE TRIANGLE rule (ray-tip vs ray-tip) ---
    if lane in (0, 2) and tri_positions and tri_rays:
        jx, jy = jake_point

        # 1) "middle" = apex closest to mid-lane x
        mid_x = LANE_MID[0]
        mid_idx = min(range(len(tri_positions)),
                    key=lambda i: abs(tri_positions[i][0] - mid_x))

        # 2) pick the *first* triangle on the opposite side of Jake (smallest y)
        if lane == 2:  # Jake in RIGHT lane → look LEFT of Jake
            side_idxs = [i for i, (x, y) in enumerate(tri_positions) if x < jx]
        else:          # lane == 0: Jake in LEFT lane → look RIGHT of Jake
            side_idxs = [i for i, (x, y) in enumerate(tri_positions) if x > jx]

        if side_idxs:
            side_idx = min(side_idxs, key=lambda i: tri_positions[i][1])

            # --- NEW: use the *ray endpoints* for both triangles of interest
            # tri_rays[i] = (p0, p1, theta); p1 is the ray end
            p2_mid  = tri_rays[mid_idx][1]
            p2_side = tri_rays[side_idx][1]

            dx = int(p2_side[0]) - int(p2_mid[0])
            dy = int(p2_side[1]) - int(p2_mid[1])
            d = math.hypot(dx, dy)

            # Optional: visualize the two ray tips
            # (uncomment if you want dots on the saved overlay image)
            # cv2.circle(frame_bgr, (int(p2_mid[0]), int(p2_mid[1])),  5, (0, 0, 0), -1, cv2.LINE_AA)
            # cv2.circle(frame_bgr, (int(p2_mid[0]), int(p2_mid[1])),  3, (255, 255, 255), -1, cv2.LINE_AA)
            # cv2.circle(frame_bgr, (int(p2_side[0]), int(p2_side[1])), 5, (0, 0, 0), -1, cv2.LINE_AA)
            # cv2.circle(frame_bgr, (int(p2_side[0]), int(p2_side[1])), 3, (255, 255, 255), -1, cv2.LINE_AA)

            # If your policy is "we want them to be within 25px", keep:
            # - GOOD when d < 25
            # - Flip to RED when d > 25
            if d > SIDE_MID_FLIP_DIST_PX:
                tri_colours[mid_idx]     = COLOR_RED
                tri_hit_classes[mid_idx] = 1  # any DANGER_RED id
                tri_hit_dists[mid_idx]   = float(d)

            print(f"[SIDE→MID] lane={lane} mid={mid_idx} side={side_idx} "
                f"ray-tip distance d={d:.1f}px (thr={SIDE_MID_FLIP_DIST_PX})")

    # ---------------------------------------------------------------------------

    # Minimal movement-friendly summary (pos, hit_class id/label, is_jake)
    __p_summary = __PROF('post.build_tri_summary')
    tri_summary = []
    for i, (x, y) in enumerate(tri_positions):
        cid = tri_hit_classes[i] if i < len(tri_hit_classes) else None
        hdist = tri_hit_dists[i] if i < len(tri_hit_dists) else None
        tri_summary.append({
            "idx": i,  # <— NEW
            "pos": (int(x), int(y)),
            "hit_class": None if cid is None else int(cid),
            "hit_label": None if cid is None else LABELS.get(int(cid), f"C{int(cid)}"),
            "hit_dist_px": None if hdist is None else float(hdist),
            "is_jake": (i == best_idx)
        })

    __p_summary()

    sidewalk_present = (
    classes_np is not None and classes_np.size > 0 and
    np.any(classes_np.astype(int) == SIDEWALK_ID)
    )


    #PATHING LOGIC HERE# =================================================================================================================================================================
    # ===== PATHING / ACTION LOGIC =================================================
    jake_tri = next((t for t in tri_summary if t.get("is_jake")), None)
    if jake_tri:
        jx, jy = jake_tri["pos"]
        jake_hit = jake_tri["hit_class"]

        jake_band_y = jake_point[1]
        _update_ramp_lock_from_jake(jake_tri, jake_band_y)

        # For movement logging: distance to the obstacle ahead of Jake
        __p_redband = __PROF('post.jake.first_red_hit_y')
        y_hit_log = first_red_hit_y(jake_tri["pos"], masks_np, classes_np, H, W, band_px=6, step_px=5)
        __p_redband()
        obstacle_dist_px = (jy - y_hit_log) if y_hit_log is not None else None

        # "Yellow in Jake's lane" == Jake's own triangle has a WARN_YELLOW class.
        jake_cls = jake_tri.get("hit_class", None)
        if (jake_cls is not None) and (int(jake_cls) in WARN_YELLOW) and (int(jake_cls) in IMPACT_CLASSES):
            # theta actually used for Jake’s ray (matches overlay)
            theta_deg = float(tri_rays[best_idx][2]) if (best_idx is not None and 0 <= best_idx < len(tri_rays)) else 0.0
            allowed_set = JUMP_SET if int(jake_cls) in JUMP_SET else DUCK_SET

            __p_starset = __PROF('post.jake.starburst_then_ray_for_set')
            dist_px, _, _ = first_mask_hit_starburst_then_ray_for_set(
                jake_point=JAKE_POINT,
                tri_pos=jake_tri["pos"],
                theta_deg=theta_deg,
                masks_np=masks_np, classes_np=classes_np, H=H, W=W,
                allowed_classes=allowed_set,
                up2_px=SAMPLE_UP_PX, step_px=2
            )
            __p_starset()

            # ---- token: only lane + class ----
            new_token = (lane, int(jake_cls))

            global IMPACT_TOKEN
            # Only arm if no timer for this token yet
            if IMPACT_TOKEN is None:
                if dist_px is not None and (IMPACT_MIN_PX < dist_px < IMPACT_MAX_PX):
                    _arm_impact_timer(float(dist_px), int(jake_cls))
                    IMPACT_TOKEN = new_token
                    print(f"[TIMER] lock token {IMPACT_TOKEN}")
                # else: don’t arm; wait for next frame when it enters the window

            else:
                if IMPACT_TOKEN == new_token:
                    # If the previous timer already fired (no longer alive), allow immediate re-arm
                    if not (IMPACT_TIMER and getattr(IMPACT_TIMER, "is_alive", lambda: False)()):
                        _arm_impact_timer(float(dist_px), int(jake_cls))
                        IMPACT_TOKEN = new_token
                    # else: keep the existing live timer

                    # Same situation → do nothing (no cancel, no re-arm), even if dist jitters/out of window
                    pass
                else:
                    # Situation changed (lane or class) → cancel old and arm once for new (if in window)
                    _cancel_impact_timer("token change")
                    if dist_px is not None and (IMPACT_MIN_PX < dist_px < IMPACT_MAX_PX):
                        _arm_impact_timer(float(dist_px), int(jake_cls))
                        IMPACT_TOKEN = new_token
                        print(f"[TIMER] lock token {IMPACT_TOKEN} (replaced)")
                    else:
                        IMPACT_TOKEN = None  # no valid new timer yet
        else:
            # Jake’s triangle not yellow/impact anymore → usually cancel,
            # but keep a pre-armed hypergreen jump (class 3) alive.
            if IMPACT_TOKEN is not None:
                _, tok_cls = IMPACT_TOKEN
                if int(tok_cls) != 3:
                    _cancel_impact_timer("no longer impact in Jake lane")
                    IMPACT_TOKEN = None


        # --- 2) Lateral pathing decisions (policy: GREEN first) --------------------
        # Build reusable candidate pools (excluding Jake's current triangle)
        __p_filter = __PROF('post.candidates.filtering')
        greens  = [t for t in tri_summary if t["hit_class"] is None]
        yellows = [t for t in tri_summary if (t["hit_class"] is not None and int(t["hit_class"]) in WARN_FOR_MOVE)]
        reds    = [t for t in tri_summary if (t["hit_class"] is not None and int(t["hit_class"]) in DANGER_RED)]
        hypergreens = [t for t in tri_summary if (t["hit_class"] is not None and int(t["hit_class"]) in HYPERGREEN_CLASSES)]

        __p_filter()

        # Lane-based pruning
        greens  = _filter_by_lane(greens,  jx, lane)
        yellows = _filter_by_lane(yellows, jx, lane)
        reds    = _filter_by_lane(reds,    jx, lane)
        hypergreens = _filter_by_lane(hypergreens, jx, lane)

        # Only consider yellow if it's far enough ahead of the Jake band (e.g., 400px)
        jake_band_y   = jake_point[1]  # 1340 with your lane points
        __p_far = __PROF('post.candidates.far_thresholds')
        yellows_far   = _filter_yellow_far(yellows, jake_band_y)  # uses MIN_YELLOW_AHEAD_PX
        greens_far  = _filter_green_far(greens, jake_band_y)
        hypergreens_far = _filter_yellow_far(hypergreens, jake_band_y, min_ahead_px=MIN_HYPERGREEN_AHEAD_PX) 
        __p_far()


        def _nearest_x(cands):
            return min(cands, key=lambda c: abs(c["pos"][0] - jx)) if cands else None

        # Score for "least-bad red": prefer the ray that hits red furthest away.
        # If 'hit_dist_px' isn't present in tri_summary, fall back to apex distance.
        jake_band_y = jake_point[1]

        def _red_score(c):
            y_hit = first_red_hit_y(c["pos"], masks_np, classes_np, H, W, band_px=6, step_px=5)
            if y_hit is None:
                return (float('inf'), -abs(c["pos"][0] - jx))  # no red in range = strictly better
            ahead_px = jake_band_y - y_hit  # larger = farther ahead
            return (ahead_px, -abs(c["pos"][0] - jx))


        # If Jake is already GREEN, stay put.
        if jake_hit is None:
            pass

        elif RAMP_LOCK["active"]:
            print("[MOVE] suppressed: ramp lock active (wait for ON_TOP)")

        # RED ahead: GREEN -> (far) YELLOW -> least-bad RED (all-red fallback)
        elif _is_danger(jake_hit):
            # 1) HYPERGREEN first (jump lanes)
            tgt = _nearest_ramp_by_y(hypergreens_far, jake_band_y) or _nearest_x(hypergreens_far)
            if tgt is not None:
                # NEW: pre-arm a jump if this hypergreen is actually class 3
                _prearm_jump_for_triangle(tgt, tgt["idx"], jake_point, tri_rays,
                                        masks_np, classes_np, H, W)

                if MOVEMENT_ENABLED:
                    print(f"[MOVE] RED ahead → HYPERGREEN (jump): obstacle={LABELS.get(int(jake_hit), str(jake_hit))}, "
                        f"dist={_fmt_px(obstacle_dist_px)}; target_x={tgt['pos'][0]}")
                prev_lane = lane
                _issue_move_towards_x(jx, tgt["pos"][0], sidewalk_present=sidewalk_present)

                if tgt is not None:
                    cid = tgt.get("hit_class")
                    if cid is not None and int(cid) in HYPERGREEN_CLASSES and lane != prev_lane:
                        time.sleep(0.6)

                if lane != prev_lane:
                    _register_red_evasion_ban(prev_lane)


            else:
                # 2) Plain GREEN
                tgt = _nearest_x(greens)
                if tgt is not None:
                    if MOVEMENT_ENABLED:
                        print(f"[MOVE] RED ahead → GREEN: obstacle={LABELS.get(int(jake_hit), str(jake_hit))}, "
                            f"dist={_fmt_px(obstacle_dist_px)}; target_x={tgt['pos'][0]}")
                    prev_lane = lane
                    _issue_move_towards_x(jx, tgt["pos"][0], sidewalk_present=sidewalk_present)
                    if lane != prev_lane:
                        _register_red_evasion_ban(prev_lane)

                else:
                    # 3) Far YELLOW
                    tgt = _nearest_x(yellows_far)
                    if tgt is not None:
                        if MOVEMENT_ENABLED:
                            ahead_px = jake_band_y - tgt["pos"][1]
                            print(f"[MOVE] RED ahead → YELLOW (far): obstacle={LABELS.get(int(jake_hit), str(jake_hit))}, "
                                f"dist={_fmt_px(obstacle_dist_px)}; yellow_ahead={int(ahead_px)}px (≥{MIN_YELLOW_AHEAD_PX})")
                        prev_lane = lane
                        _issue_move_towards_x(jx, tgt["pos"][0], sidewalk_present=sidewalk_present)
                        if lane != prev_lane:
                            _register_red_evasion_ban(prev_lane)
                    else:
                        # 4) All-red fallback (your existing scoring)
                        if reds:
                            __p_redscore = __PROF('post.red_scoring')
                            best_red = max(reds, key=_red_score)
                            __p_redscore()
                            tx = best_red["pos"][0]
                            if tx != jx:
                                prev_lane = lane
                                _issue_move_towards_x(jx, tx, sidewalk_present=sidewalk_present)
                                if lane != prev_lane:
                                    _register_red_evasion_ban(prev_lane)




                    # else: boxed in → no lateral move this frame

        # YELLOW ahead: try GREEN; if none, rely on countermeasures (jump/duck)
        # YELLOW ahead: try GREEN; if none, rely on countermeasures (jump/duck)
        elif _is_warn(jake_hit):
            tgt = (_nearest_ramp_by_y(hypergreens_far, jake_band_y)
                or _nearest_x(hypergreens_far)
                or _nearest_x(greens_far))
            if tgt is not None:
                # Only pre-arm if this chosen target is the hypergreen jump (class 3)
                cls = tgt.get("hit_class")
                if cls is not None and int(cls) == 3:
                    _prearm_jump_for_triangle(tgt, tgt["idx"], jake_point, tri_rays,
                                            masks_np, classes_np, H, W)

                prev_lane = lane                         # ← add this
                _issue_move_towards_x(jx, tgt["pos"][0], sidewalk_present=sidewalk_present)

                # sleep only if we actually moved and the target is hypergreen (3 or 8)
                cid = tgt.get("hit_class") if tgt is not None else None
                if cid is not None and int(cid) in HYPERGREEN_CLASSES and lane != prev_lane:
                    time.sleep(0.6)




# ============================================================================

# ============================================================================
#END
# ============================================================================

    return (tri_best, len(tri_positions), mask_count, to_cpu_ms, post_ms,
            masks_np, classes_np, rail_mask, green,
            tri_positions, tri_colours, tri_rays,
            best_idx, best_deg, x_ref,
            tri_hit_classes, tri_summary)

# =======================
# Viz helpers
# =======================
def _colour_for_point(x, y, masks_np, classes_np, H, W):
    if masks_np is None or classes_np is None or masks_np.size == 0: return COLOR_GREEN
    mh, mw = masks_np.shape[1], masks_np.shape[2]
    sx = (mw - 1) / max(1, (W - 1)); sy = (mh - 1) / max(1, (H - 1))
    mx = _clampi(int(round(x * sx)), 0, mw-1)
    my = _clampi(int(round(y * sy)), 0, mh-1)
    cls_here = None
    for m, c in zip(masks_np, classes_np):
        if m[my, mx] > 0.5: cls_here = int(c); break
    if cls_here in DANGER_RED:   return COLOR_RED
    if cls_here in WARN_YELLOW:  return COLOR_YELLOW
    if cls_here in BOOTS_PINK:   return COLOR_PINK
    return COLOR_GREEN

def draw_triangle(img, x, y, size=TRI_SIZE_PX, colour=COLOR_RED):
    h = int(size * 1.2)
    pts = np.array([[x, y], [x-size, y+h], [x+size, y+h]], np.int32)
    cv2.fillConvexPoly(img, pts, colour)
    cv2.polylines(img, [pts.reshape(-1,1,2)], True, COLOR_BLACK, 1, cv2.LINE_AA)

def triangle_pts(x, y, size=TRI_SIZE_PX):
    h = int(size * 1.2)
    return np.array([[x, y], [x-size, y+h], [x+size, y+h]], np.int32)

def render_overlays(frame_bgr, masks_np, classes_np, rail_mask, green_mask,
                    tri_positions, tri_colours, tri_rays, best_idx, best_deg, x_ref, jake_point):
    out = frame_bgr.copy()
    H, W = out.shape[:2]
    alpha = 0.45

    if masks_np is not None and classes_np is not None and masks_np.size:
        for m, c in zip(masks_np, classes_np):
            m_full = m
            if m.shape != (H, W):
                m_full = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            color = CLASS_COLOURS.get(int(c), (255,255,255))
            out[m_full] = (np.array(color, dtype=np.uint8) * alpha + out[m_full] * (1 - alpha)).astype(np.uint8)
            ys, xs = np.where(m_full)
            if xs.size:
                xc, yc = int(xs.mean()), int(ys.mean())
                label = LABELS.get(int(c), f"C{int(c)}")
                cv2.putText(out, label, (max(5, xc-40), max(20, yc)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 2, cv2.LINE_AA)
                cv2.putText(out, label, (max(5, xc-40), max(20, yc)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    if rail_mask is not None:
        tint = out.copy()
        tint[rail_mask] = (0, 0, 255)
        out = cv2.addWeighted(tint, 0.30, out, 0.70, 0)
    if green_mask is not None:
        out[green_mask] = (0, 255, 0)

    # tiny scout lines (viz only)
    if tri_positions:
        for (x, y) in tri_positions:
            y_end = max(0, y - SAMPLE_UP_PX)
            for yy in range(y, y_end - 1, -1):
                out[yy, x] = _colour_for_point(x, yy, masks_np, classes_np, H, W)

    # starburst to Jake
    xj, yj = jake_point
    for idx, (xt, yt) in enumerate(tri_positions):
        xt = _clampi(int(xt), 0, W-1); yt = _clampi(int(yt), 0, H-1)
        deg_signed = signed_degrees_from_vertical(xt - xj, yt - yj)
        cv2.line(out, (xj, yj), (xt, yt),
                 COLOR_CYAN if idx == best_idx else COLOR_WHITE, 2, cv2.LINE_AA)
        mx = (xj + xt) // 2; my = (yj + yt) // 2

    # curved sampling rays (viz)
    for (p0, p1, theta) in tri_rays:
        cv2.line(out, p0, p1, (255,255,255), 2, cv2.LINE_AA)
        mx = (p0[0] + p1[0]) // 2; my = (p0[1] + p1[1]) // 2

    for (x, y), col in zip(tri_positions, tri_colours):
        draw_triangle(out, int(x), int(y), colour=col)

    lane_name = lane_name_from_point(jake_point)
    target_deg = LANE_TARGET_DEG[lane_name]
    if best_idx is not None and 0 <= best_idx < len(tri_positions):
        xt, yt = tri_positions[best_idx]

    
        theta_deg = tri_rays[best_idx][2] if (best_idx is not None and best_idx < len(tri_rays)) else 0.0
        dxr, dyr  = unit_from_theta(float(theta_deg))  # use helper, not TRIG_TABLE.get
        xe = _clampi(int(round(xt + dxr * SAMPLE_UP_PX)), 0, W-1)
        ye = _clampi(int(round(yt + dyr * SAMPLE_UP_PX)), 0, H-1)


        dist_px, hit_xy, hit_cls = first_mask_hit_starburst_then_ray(
            jake_point=jake_point,
            tri_pos=(int(xt), int(yt)),
            theta_deg=float(theta_deg),
            masks_np=masks_np, classes_np=classes_np, H=H, W=W,
            up2_px=SAMPLE_UP_PX, step_px=2,
            exclude_classes=(RAIL_ID,),   # skip rails
            danger_only=False             # set True to only consider DANGER_RED
        )

        # draw starburst segment (cyan + thicker)
        xj, yj = jake_point
        cv2.line(out, (xj, yj), (int(xt), int(yt)), COLOR_CYAN, 3, cv2.LINE_AA)

        # draw the angled continuation for viz (cyan + thicker)
        dxr, dyr = TRIG_TABLE.get(float(theta_deg), (0.0, -1.0))
        xe = _clampi(int(round(xt + dxr * SAMPLE_UP_PX)), 0, W-1)
        ye = _clampi(int(round(yt + dyr * SAMPLE_UP_PX)), 0, H-1)
        cv2.line(out, (int(xt), int(yt)), (xe, ye), COLOR_CYAN, 3, cv2.LINE_AA)

        # OPTIONAL: cyan outline around the best triangle so it pops
        cv2.polylines(out, [triangle_pts(int(xt), int(yt)).reshape(-1,1,2)], True, COLOR_CYAN, 3, cv2.LINE_AA)


        if hit_xy is not None:
            cv2.circle(out, hit_xy, 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(out, hit_xy, 4, (255, 255, 255), -1, cv2.LINE_AA)

        dist_text = "∞" if dist_px is None else f"{dist_px:.1f}px"
        if hit_cls is not None:
            dist_text += f" → {LABELS.get(hit_cls, str(hit_cls))}"
        midx = (xj + int(xt)) // 2
        midy = (yj + int(yt)) // 2 - 10
        cv2.putText(out, f"Jake→tri (ray) first-hit: {dist_text}",
                    (max(5, midx - 160), max(24, midy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 2, cv2.LINE_AA)
        cv2.putText(out, f"Jake→tri (ray) first-hit: {dist_text}",
                    (max(5, midx - 160), max(24, midy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        
        # TIME KEEPING

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        time_str = f"{elapsed_ms:.3f} ms"

        # position in bottom-right corner
        (text_w, text_h), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        x_pos = out.shape[1] - text_w - 10  # 10px from right edge
        y_pos = out.shape[0] - 10           # 10px from bottom edge

        # draw text
        cv2.putText(out, time_str, (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(out, time_str, (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------------------------


    # top-left JAKE_POINT state
    cv2.putText(out, f"JAKE_POINT: {lane_name.upper()}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    # bottom-left ON_TOP badge
    cv2.putText(out, f"ON_TOP={int(ON_TOP)}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(out, f"ON_TOP={int(ON_TOP)}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)


    return out

# =======================
# Live loop
# =======================   

listener = keyboard.Listener(on_press=on_press)
listener.start()

sct = mss()
frame_idx = 0

if advertisement:
    CHECK_X, CHECK_Y = 1030, 900
else:
    CHECK_X, CHECK_Y = 870, 895

#===========================================Resource consuption monitoring===========================================

import psutil
import os
import subprocess, threading, re
import subprocess
import threading
import re

process = psutil.Process(os.getpid())

def print_system_usage():
    cpu_percent = psutil.cpu_percent(interval=None)
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 ** 2)  # Resident memory in MB
    print(f"[SYS] CPU: {cpu_percent:.1f}%  |  RAM: {rss_mb:.1f} MB")

    import subprocess, threading, re

def stream_mps_gpu_stats():
    # requires sudo; run your script with:  sudo python your_script.py
    cmd = ["powermetrics", "--samplers", "gpu_power", "-i", "200"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    busy_re = re.compile(r"GPU Busy\s*=\s*(\d+)%")
    power_re = re.compile(r"GPU Power\s*=\s*([\d\.]+)\s*W")
    for line in p.stdout:
        m1 = busy_re.search(line); m2 = power_re.search(line)
        if m1 or m2:
            busy = m1.group(1) if m1 else "?"
            power = m2.group(1) if m2 else "?"
            print(f"[GPU] Busy {busy}% | Power {power} W")

# call once before your while-loop:
threading.Thread(target=stream_mps_gpu_stats, daemon=True).start()

def _selective_hit_from_classes(classes_np, watched: set[int]):
    """
    Returns (hit: bool, ids: list[int]) given a per-instance class array.
    Use this after any class promotion step if you want that respected.
    """
    if classes_np is None:
        return False, []
    try:
        uniq = np.unique(classes_np.astype(int))
    except Exception:
        uniq = [int(c) for c in classes_np]
    hits = [int(c) for c in uniq if int(c) in watched]
    return (len(hits) > 0), hits

def _labels_for(ids):
    """Map class IDs to human-readable labels."""
    return [LABELS.get(int(i), str(int(i))) for i in ids]

#=====================================================================================================================

# =======================

power_metrics = False
active_replay = False
times_collection = []

# === Selective save (save every frame when certain classes are present) ===
SELECTIVE_SAVE_ENABLED = False
WATCH_SAVE_CLASSES = {10, 7}  # tweak as you like


# =======================
lane = 1
while running:
    frame_start_time = time.perf_counter()
    _prof_reset()


    # bump SHORTEN_S after 60s of runtime
    elapsed_s = time.perf_counter() - start_time
    if elapsed_s >= 60.0 and elapsed_s <= 62.0 and SHORTEN_S != 0.50:
        SHORTEN_S = 0.65
        LUT_S = _compute_lut(SHORTEN_S)
        print(f"[LUT] SHORTEN_S -> {SHORTEN_S:.2f} at t={elapsed_s:.1f}s (LUT_S recomputed)")


    _now = time.monotonic()
    if _now < PAUSE_UNTIL:
        time.sleep(PAUSE_UNTIL - _now)
        continue

    frame_idx += 1
    print()
    print(f'===================================== Operating on frame {frame_idx} =====================================')

    # --- Screen grab ---
    t0_grab = time.perf_counter()
    left, top, width, height = snap_coords
   
   # NEW (ring)
    frame_bgr, meta = get_frame_bgr_from_ring(path="/tmp/scap.ring", wait_new=True, timeout_s=0.5)  # HxWx3, uint8, contiguous

    # after you have `frame_bgr` for the current frame:
    pct, ms = percent_of_color_rgba(frame_bgr, rgba=(210, 36, 35, 255), tol_frac=0.02)
    print(f"[COLOR%] RGBA(210,36,35,255) ±2% -> {pct:.2f}% of frame  |  took {ms:.2f} ms")


    if active_replay:
        # --- Replay dump (pre-analysis) ---
        t0_replay = time.perf_counter()

        # draw runtime (since script start) on a COPY so analysis frame stays pristine
        frame_to_save = frame_bgr.copy()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        time_str = f"{elapsed_ms:.3f} ms"

        # bottom-right placement
        (text_w, text_h), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x_pos = frame_to_save.shape[1] - text_w - 10
        y_pos = frame_to_save.shape[0] - 10

        # outline + neon text
        cv2.putText(frame_to_save, time_str, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_to_save, time_str, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, NEON_GREEN, 2, cv2.LINE_AA)

        # save; name by frame index
        replay_path = REPLAYS_DIR / f"replay_{frame_idx:06d}.jpg"
        cv2.imwrite(str(replay_path), frame_to_save)

        replay_ms = (time.perf_counter() - t0_replay) * 1000.0
        print(f"[REPLAY] saved {replay_path.name} in {replay_ms:.2f} ms (elapsed {elapsed_ms:.3f} ms)")


    grab_ms = (time.perf_counter() - t0_grab) * 1000.0

    # same style of debug print
    #print(f"[view] shape: {frame_bgr.shape[1]}x{frame_bgr.shape[0]} px   (BGR)   seq={meta['seq']}")

    # --- ABSOLUTE SCREEN pixel check ---
    t0_check = time.perf_counter()
    arr = np.array(sct.grab({"left": CHECK_X, "top": CHECK_Y, "width": 1, "height": 1}))

    b, g, r, a = arr[0, 0]
    TOL = 20
    target = (61, 156, 93)

    if (abs(b - target[0]) <= TOL and
        abs(g - target[1]) <= TOL and
        abs(r - target[2]) <= TOL) and frame_idx > 10 or (b, g, r) == (24, 24, 24):
        print(f"Kill-switch triggered at ({CHECK_X},{CHECK_Y})")
        running = False

        frames_total_proc = len(times_collection)
        print(f"Average time to process a frame is {sum(times_collection) / frames_total_proc:.2f} ms")
        
        keyboard.Key.esc
        break
    pixel_check_ms = (time.perf_counter() - t0_check) * 1000.0

    # --- Lane detection ---
    t0_lane = time.perf_counter()
    _detected = _detect_lane_by_whiteness(frame_bgr)
    if _detected is not None:
        lane = _detected  # 0/1/2
    JAKE_POINT = LANE_POINTS[lane]
    lane_ms = (time.perf_counter() - t0_lane) * 1000.0

    # --- Inference ---
    t0_inf = time.perf_counter()
    __p_infer = __PROF('infer.model.predict')
    res_list = model.predict(
        [frame_bgr], task="segment", imgsz=IMG_SIZE, device=device,
        conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET, batch=1
    )
    __p_infer()
    infer_ms = (time.perf_counter() - t0_inf) * 1000.0
    yres = res_list[0]

    #pause_if_big_ramp(yres)  # ← sleep 1.0s if the biggest RAMP > 28% of screen


    _p_GROUND_OR_TOP = __PROF('on_top.state')
    H, W = frame_bgr.shape[:2]
    try:
        on_top_now, timed_out, ot_ms, seen = compute_on_top_state_fast(
            yres, H, W, lane_idx=lane, mask_thresh=0.5
        )
        # ---- 0.3s “no drop” gate after TL lateral tap ---- STICKY
        if time.monotonic() < tl_sticky_until():
            # keep running TL; don’t allow TOP -> GROUND flip
            if not on_top_now:
                print("[STATE] TL post-lateral sticky → forcing TOP branch for 0.3s")
            on_top_now = True
            
        if timed_out:
            # use last stable values instead of "bailing"
            on_top_now = _ONTOP_CACHE["on_top_now"]
            seen       = _ONTOP_CACHE["seen"]
            ot_ms      = _ONTOP_CACHE["ot_ms"]
            print(f"[STATE] ON_TOP={int(on_top_now)}  seen={sorted(list(seen))}  ({ot_ms:.2f} ms | CACHED)")
        else:
            # update cache on a good read
            _ONTOP_CACHE["on_top_now"] = bool(on_top_now)
            _ONTOP_CACHE["seen"]       = set(seen)
            _ONTOP_CACHE["ot_ms"]      = float(ot_ms)
            print(f"[STATE] ON_TOP={int(on_top_now)}  seen={sorted(list(seen))}  ({ot_ms:.2f} ms)")

    except Exception as _e_ot:
        # hard error → carry last stable state
        on_top_now = _ONTOP_CACHE["on_top_now"]
        seen       = _ONTOP_CACHE["seen"]
        ot_ms      = _ONTOP_CACHE["ot_ms"]
        timed_out  = True
        print(f"[STATE] ON_TOP={int(on_top_now)}  seen={sorted(list(seen))}  ({ot_ms:.2f} ms | CACHED error: {_e_ot})")

    _p_GROUND_OR_TOP()

    print(f"Currently lane is {lane}")

    if not on_top_now:
        DONT_MOVE = 0
        # --- Postproc ---
        t0_post = time.perf_counter()
        (tri_best_xy, tri_count, mask_count, to_cpu_ms, post_ms,
        masks_np, classes_np, rail_mask, green_mask, tri_positions, tri_colours,
        tri_rays, best_idx, best_deg, x_ref,
        tri_hit_classes, tri_summary) = process_frame_post(frame_bgr, yres, JAKE_POINT)

        red_ahead_overlay = red_ahead_from_overlay(best_idx, tri_hit_classes, tri_colours)

        postproc_ms = (time.perf_counter() - t0_post) * 1000.0

        # ===================== PILLAR-EVASION FAST CHECK =====================
        if pillar_evasion_check_and_act(H, W, masks_np, classes_np, JAKE_POINT,red_ahead_overlay=red_ahead_overlay):

            # Optional: skip the rest of the heavy path this frame to keep overhead minimal.
            times_collection.append((time.perf_counter() - frame_start_time) * 1000.0)
            continue
        # =====================================================================
        if _is_jake_on_sidewalk(JAKE_POINT, masks_np, classes_np, H, W):
            print("[SIDEWALK] Standing on sidewalk → freeze movement & skip frame")
            _cancel_impact_timer("on sidewalk")          # optional: avoids stale jump/duck
            # (Optionally also block synthetic key echoes)
            _synth_block_until = time.monotonic() + SYNTHETIC_SUPPRESS_S
            # Skip the rest of the logic; next frame will re-check
            times_collection.append((time.perf_counter() - frame_start_time) * 1000.0)
            continue

        sidewalk_present = (
        classes_np is not None and classes_np.size > 0 and
        np.any(classes_np.astype(int) == SIDEWALK_ID))


        # --- Selective save (level-triggered; save every frame while watched classes are present) ---
        save_selective = False
        sel_ids = []
        if SELECTIVE_SAVE_ENABLED:
            save_selective, sel_ids = _selective_hit_from_classes(classes_np, WATCH_SAVE_CLASSES)


        total_proc_ms = grab_ms + pixel_check_ms + lane_ms + infer_ms + postproc_ms

        # (timing print; optional — keep only if you want it)
        if save_frames or save_selective:
            elapsed_no_post = time.perf_counter() - frame_start_time
            print(f"Frame {frame_idx} WITHOUT POSTPROC WAS: {elapsed_no_post * 1000:.2f} ms")

        # Render & write overlay
        if save_frames or save_selective:
            t0_overlay = time.perf_counter()
            overlay = render_overlays(frame_bgr, masks_np, classes_np, rail_mask, green_mask,
                                    tri_positions, tri_colours, tri_rays, best_idx, best_deg, x_ref, JAKE_POINT)
            tag = "_sel" if save_selective else ""
            out_path = out_dir / f"live_overlay_{frame_idx:05d}{tag}.jpg"
            cv2.imwrite(str(out_path), overlay)
            overlay_ms = (time.perf_counter() - t0_overlay) * 1000.0
            if save_selective:
                print(f"[SAVE] selective hit: {', '.join(_labels_for(sel_ids))} -> {out_path.name}")
        else:
            overlay_ms = 0.0

        total_elapsed_ms = (time.perf_counter() - frame_start_time) * 1000.0

        if power_metrics:
            print_system_usage()


        if total_elapsed_ms > 60:
            # --- Timing summary ---
            print()
            print(
                f"[TIMINGS] Grab={grab_ms:.2f} ms | PixelChk={pixel_check_ms:.2f} ms | "
                f"LaneDet={lane_ms:.2f} ms | Inference={infer_ms:.2f} ms | "
                f"Postproc={postproc_ms:.2f} ms | Overlay={overlay_ms:.2f} ms | "
                f"TOTAL={total_elapsed_ms:.2f} ms"
            )

            prof_summary(frame_idx)
            print()


        times_collection.append(total_elapsed_ms)

    else:
        if DONT_MOVE < 2:
            print('GUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDEDGUARDED')
            DONT_MOVE +=1
            continue
        
        started_1 = time.perf_counter()
        print("we are on top of train..")

        # Use the same save switch as ground overlays
        seq = meta.get("seq", frame_idx)
        analysed_path = str(TOP_OUT_DIR / f"top_{int(seq):06d}_analysed.jpg")

        decision = do_top_logic_from_result(
            frame_bgr=frame_bgr,
            yolo_result=yres,
            lane_2=lane,
            save_analysed=save_frames,                   # <— respect G / --save
            save_path=analysed_path if save_frames else None,
            print_prefix=f"[TOP f{frame_idx:05d}] ",
        )

        # Fallback: if TL returns an image but didn’t save internally, save it here
        if save_frames and isinstance(decision, dict) and "out_img" in decision and decision["out_img"] is not None:
            out = decision["out_img"].copy()
            _stamp_lane_badge(out, lane)
            cv2.imwrite(analysed_path, out, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"[TOP SAVE] {analysed_path}")

        elapsed_TL = time.perf_counter() - started_1
        print(f"Time taken for logic TL is {elapsed_TL * 1000:.2f} ms")
        continue
    
subprocess.run([sys.executable, BASE, "shutdown"], check=False)

import atexit, subprocess
atexit.register(lambda: subprocess.run("killall scgrab 2>/dev/null || true", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

# Cleanup

try:
    listener.stop()                    # tell it to exit
    listener.join(timeout=0.5)         # never wait forever
except Exception:
    pass

try: ANN_PROC.terminate()
except Exception: pass
pyautogui.press('esc')
print("Script halted.")