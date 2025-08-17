#!/usr/bin/env python3
# Live overlays + lane-aware curved sampling (optimized postproc)
# • Parsec focus + auto click
# • mss live capture of a crop region
# • Arrow keys switch lane (0/1/2) -> JAKE_POINT updates per frame
# • Full overlay rendering + per-frame save
# • Prints compact timing per frame
# • RETURNS per frame: tri_positions, best_idx, tri_hit_classes, tri_summary (for movement logic)

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


# purple triangles funciton config
_SE_OPEN_5x9 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))


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


# Allow movement immediately; after 0.5s, mute; after 3.0s total, unmute
Timer(0.5, __mute_keys).start()
Timer(4.0, __unmute_keys).start()


# =======================
# Config
# =======================
home       = os.path.expanduser("~")
weights    = f"{home}/models/jakes-loped/jakes-finder-mk1/1/weights.pt"

# SAVE HERE
out_dir    = Path(home) / "Documents" / "GitHub" / "Ai-plays-SubwaySurfers" / "out_live_overlays"
out_dir.mkdir(parents=True, exist_ok=True)

# --- Replay dump setup (sibling to out_live_overlays) ---
REPLAYS_DIR = out_dir.parent / "replays"
REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
NEON_GREEN = (57, 255, 20)  # BGR neon green


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

SHORTEN_S = 0.35 #Shortnen by 100ms
# Safety clamps so Timer never explodes or becomes a no-op
MIN_DELAY_S = 0.03   # 30 ms
MAX_DELAY_S = 2.00   # 2 s

LUT_S = np.clip(np.array([0.0259, 0.0303, 0.0353, 0.0412, 0.0481, 0.0561, 0.0655, 0.0765, 0.0893, 0.1042, 0.1216, 0.1419, 0.1656, 0.1933, 0.2256, 0.2633, 0.3073, 0.3586, 0.4185, 0.4885, 0.5701, 0.6654, 0.7765, 0.9063, 1.0578, 1.2345, 1.4408, 1.6815, 1.9625], dtype=float) - SHORTEN_S, MIN_DELAY_S, MAX_DELAY_S)


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

    dxr, dyr = TRIG_TABLE.get(theta_deg, (0.0, -1.0))
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
    dxr, dyr = TRIG_TABLE.get(theta_deg, (0.0, -1.0))  # default straight up
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



# --- tunnel wall color gate (HSV) ---
LOWBARRIER1_ID   = 4
ORANGETRAIN_ID   = 6
WALL_STRIP_PX    = 12          # vertical strip height checked just above the barrier
WALL_MATCH_FRAC  = 0.90         # % of “wall” pixels required to relabel
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
        if frac >= frac_thresh:
            classes_np[i] = ORANGETRAIN_ID  # promote to a RED class

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
    last_action_ts = 0.0
ACTION_COOLDOWN_S = 0.0

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

def _issue_move_towards_x(jx: int, tx: int):
    global lane, last_move_ts, _synth_block_until
    if not MOVEMENT_ENABLED:
        return

    now = time.perf_counter()
    dt = now - last_move_ts
    if dt < MOVE_COOLDOWN_S:
        remaining = MOVE_COOLDOWN_S - dt
        print(f"[COOLDOWN] Lane move blocked: {remaining*1000:.0f} ms remaining -> Please expect delays")
        return

    if tx < jx and lane > MIN_LANE:
        pause_inference()  # 360ms freeze to avoid mid-lane frames
        _synth_block_until = time.monotonic() + SYNTHETIC_SUPPRESS_S
        pyautogui.press('left')
        lane = max(MIN_LANE, lane - 1)
        print(f"[AI MOVE] left -> Lane {lane}")
        last_move_ts = now

    elif tx > jx and lane < MAX_LANE:
        pause_inference()  # 360ms freeze to avoid mid-lane frames
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
lane = 1
MIN_LANE = 0
MAX_LANE = 2
running = True

# ===== Debounce / cooldown =====
COOLDOWN_MS = 20
_last_press_ts = 0.0  # monotonic seconds

def on_press(key):
    global lane, running, _last_press_ts, _synth_block_until
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

        elif key == keyboard.Key.esc:
            running = False
            return False
    except Exception as e:
        print(f"Error: {e}")


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

# ====== tiny helpers ======
def _clampi(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def _fmt_px(v):
    return f"{v:.1f}px" if v is not None else "n/a"

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

    for idx, (x0, y0) in enumerate(tri_positions):
        theta = pick_bend_angle(jake_point, x0, x_ref, idx, best_idx)

        # --- NEW: for Jake's triangle only, if we're in LEFT or RIGHT lane,
        # make the probe an extension of JAKE_POINT -> triangle apex.
        if idx == best_idx and (jake_point == LANE_LEFT or jake_point == LANE_RIGHT):
            jx, jy = jake_point
            dxv = x0 - jx
            dyv = y0 - jy
            L = math.hypot(dxv, dyv)
            if L > 1e-6:
                # normalize the true Jake->tri direction
                dx1 = dxv / L
                dy1 = dyv / L
                # keep a theta value for overlays/first-hit; store the exact vector under that key
                theta = round(signed_degrees_from_vertical(dxv, dyv), 3)
                TRIG_TABLE[theta] = (dx1, dy1)
            else:
                # degenerate: fall back to straight up
                theta = 0.0
                dx1, dy1 = TRIG_TABLE[theta]
        else:
            # all other triangles keep the old bending behavior
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
        EDGE_PAD_PX = 50

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

    # Minimal movement-friendly summary (pos, hit_class id/label, is_jake)
    __p_summary = __PROF('post.build_tri_summary')
    tri_summary = []
    for i, (x, y) in enumerate(tri_positions):
        cid = tri_hit_classes[i] if i < len(tri_hit_classes) else None
        hdist = tri_hit_dists[i] if i < len(tri_hit_dists) else None
        tri_summary.append({
            "pos": (int(x), int(y)),
            "hit_class": None if cid is None else int(cid),
            "hit_label": None if cid is None else LABELS.get(int(cid), f"C{int(cid)}"),
            "hit_dist_px": None if hdist is None else float(hdist),
            "is_jake": (i == best_idx)
        })
    __p_summary()


    #PATHING LOGIC HERE# =================================================================================================================================================================
    # ===== PATHING / ACTION LOGIC =================================================
    jake_tri = next((t for t in tri_summary if t.get("is_jake")), None)
    if jake_tri:
        jx, jy = jake_tri["pos"]
        jake_hit = jake_tri["hit_class"]

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
            # Jake’s triangle not yellow/impact anymore → cancel & unlock
            if IMPACT_TOKEN is not None:
                _cancel_impact_timer("no longer impact in Jake lane")
                IMPACT_TOKEN = None


        # --- 2) Lateral pathing decisions (policy: GREEN first) --------------------
        # Build reusable candidate pools (excluding Jake's current triangle)
        __p_filter = __PROF('post.candidates.filtering')
        greens  = [t for t in tri_summary if t["hit_class"] is None]
        yellows = [t for t in tri_summary if (t["hit_class"] is not None and int(t["hit_class"]) in WARN_FOR_MOVE)]
        reds    = [t for t in tri_summary if (t["hit_class"] is not None and int(t["hit_class"]) in DANGER_RED)]
        __p_filter()

        # Lane-based pruning
        greens  = _filter_by_lane(greens,  jx, lane)
        yellows = _filter_by_lane(yellows, jx, lane)
        reds    = _filter_by_lane(reds,    jx, lane)

        # Only consider yellow if it's far enough ahead of the Jake band (e.g., 400px)
        jake_band_y   = jake_point[1]  # 1340 with your lane points
        __p_far = __PROF('post.candidates.far_thresholds')
        yellows_far   = _filter_yellow_far(yellows, jake_band_y)  # uses MIN_YELLOW_AHEAD_PX
        greens_far  = _filter_green_far(greens, jake_band_y)
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

        # RED ahead: GREEN -> (far) YELLOW -> least-bad RED (all-red fallback)
        elif _is_danger(jake_hit):
            tgt = _nearest_x(greens)
            if tgt is not None:
                if MOVEMENT_ENABLED:
                    print(f"[MOVE] RED ahead → GREEN: obstacle={LABELS.get(int(jake_hit), str(jake_hit))}, dist={_fmt_px(obstacle_dist_px)}; target_x={tgt['pos'][0]}")
                _issue_move_towards_x(jx, tgt["pos"][0])
            else:
                tgt = _nearest_x(yellows_far)   # only yellows ≥ threshold above the band
                if tgt is not None:
                    if MOVEMENT_ENABLED:
                        ahead_px = jake_band_y - tgt["pos"][1]
                        print(f"[MOVE] RED ahead → YELLOW (far): obstacle={LABELS.get(int(jake_hit), str(jake_hit))}, dist={_fmt_px(obstacle_dist_px)}; yellow_ahead={int(ahead_px)}px (≥{MIN_YELLOW_AHEAD_PX})")
                    _issue_move_towards_x(jx, tgt["pos"][0])
                else:
                    if reds:
                        # When choosing best_red:
                        __p_redscore = __PROF('post.red_scoring')
                        best_red = max(reds, key=_red_score)
                        __p_redscore()
                        tx = best_red["pos"][0]
                        if tx != jx:                      # avoid needless move if staying is best
                            _issue_move_towards_x(jx, tx)

                    # else: boxed in → no lateral move this frame

        # YELLOW ahead: try GREEN; if none, rely on countermeasures (jump/duck)
        elif _is_warn(jake_hit):
            tgt = _nearest_x(greens_far)  # only consider far-enough greens
            if tgt is not None:
                _issue_move_towards_x(jx, tgt["pos"][0])
    # else: no safe far green → hold lane; jump/duck handled above

            # else: no green → hold lane; jumps/ducks already handled above
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

        # theta used for that triangle in classify_triangles_at_sample_curved
        theta_deg = tri_rays[best_idx][2] if best_idx < len(tri_rays) else 0.0

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

    return out

# =======================
# Live loop
# =======================   

listener = keyboard.Listener(on_press=on_press)
listener.start()

sct = mss()
frame_idx = 0

from mss import mss

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

#=====================================================================================================================

# =======================

save_frames = False
power_metrics = False
active_replay = False
times_collection = []

# =======================

while running:
    frame_start_time = time.perf_counter()
    _prof_reset()


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
        abs(r - target[2]) <= TOL) or (b, g, r) == (24, 24, 24):
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

    # --- Postproc ---
    t0_post = time.perf_counter()
    (tri_best_xy, tri_count, mask_count, to_cpu_ms, post_ms,
     masks_np, classes_np, rail_mask, green_mask, tri_positions, tri_colours,
     tri_rays, best_idx, best_deg, x_ref,
     tri_hit_classes, tri_summary) = process_frame_post(frame_bgr, yres, JAKE_POINT)
    postproc_ms = (time.perf_counter() - t0_post) * 1000.0

    total_proc_ms = grab_ms + pixel_check_ms + lane_ms + infer_ms + postproc_ms

    if save_frames:
        elapsed_no_post = time.perf_counter() - frame_start_time
        print(f"Frame {frame_idx} WITHOUT POSTPROC WAS: {elapsed_no_post * 1000:.2f} ms")

    if save_frames:
        t0_overlay = time.perf_counter()
        overlay = render_overlays(frame_bgr, masks_np, classes_np, rail_mask, green_mask,
                                  tri_positions, tri_colours, tri_rays, best_idx, best_deg, x_ref, JAKE_POINT)
        out_path = out_dir / f"live_overlay_{frame_idx:05d}.jpg"
        cv2.imwrite(str(out_path), overlay)
        overlay_ms = (time.perf_counter() - t0_overlay) * 1000.0
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

# Cleanup
listener.join()
print("Script halted.")
