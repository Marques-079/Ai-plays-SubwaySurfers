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

# --- tunnel wall color gate (HSV) ---
LOWBARRIER1_ID   = 4
ORANGETRAIN_ID   = 6
WALL_STRIP_PX    = 20           # vertical strip height checked just above the barrier
WALL_MATCH_FRAC  = 0.40         # % of “wall” pixels required to relabel
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
ACTION_COOLDOWN_S = 0.5

# distance threshold (pixels) from Jake to triangle apex for action decisions
ACTION_DIST_PX = 30

def _is_warn(cls_id: int | None) -> bool:
    return (cls_id is not None) and (int(cls_id) in WARN_FOR_MOVE)

def _dist_px(jx: int, jy: int, tx: int, ty: int) -> float:
    return math.hypot(tx - jx, ty - jy)

def _pick_best_green(cands, jx: int):
    """Choose the closest triangle with hit_class == None (no hit along ray)."""
    greens = [c for c in cands if c["hit_class"] is None]
    if not greens:
        return None
    greens = [c for c in greens if c["pos"][0] != jx] or greens
    return min(greens, key=lambda c: abs(c["pos"][0] - jx))

def _schedule(fn, *args, **kwargs):
    Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()

def _do_jump_then_duck(delay_s: float = 0.50):
    pyautogui.press(JUMP_KEY)
    time.sleep(delay_s)
    pyautogui.press(DUCK_KEY)

def _try_jump_then_duck():
    if not MOVEMENT_ENABLED:
        return
    global last_action_ts
    now = time.perf_counter()
    if now - last_action_ts >= ACTION_COOLDOWN_S:
        last_action_ts = now
        _schedule(_do_jump_then_duck, 0.20)

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
    if now - last_move_ts < MOVE_COOLDOWN_S:
        return

    if tx < jx and lane > MIN_LANE:
        _synth_block_until = time.monotonic() + SYNTHETIC_SUPPRESS_S
        pyautogui.press('left')
        lane = max(MIN_LANE, lane - 1)
        print(f"[AI MOVE] left -> Lane {lane}")
        last_move_ts = now

    elif tx > jx and lane < MAX_LANE:
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
    H, W = rail_mask.shape
    if not rail_mask.any():
        return np.zeros((H, W), dtype=bool)

    rail_u8 = rail_mask.view(dtype=np.uint8) * 255
    x, y, w, h = cv2.boundingRect(rail_u8)
    img_roi  = img_bgr[y:y+h, x:x+w]
    mask_roi = rail_u8[y:y+h, x:x+w]

    img_f = img_roi.astype(np.float32, copy=False)
    diff  = img_f[:, :, None, :] - TARGETS_BGR_F32[None, None, :, :]
    dist2 = (diff * diff).sum(-1)
    colour_hit = (dist2 <= TOL2).any(-1)

    combined = np.logical_and(colour_hit, mask_roi.astype(bool))

    comp = combined.astype(np.uint8)
    n, lbls, stats, _ = cv2.connectedComponentsWithStats(comp, 8)
    if n <= 1: return np.zeros((H, W), dtype=bool)

    good = np.zeros_like(combined)
    areas = stats[1:, cv2.CC_STAT_AREA]
    hs    = stats[1:, cv2.CC_STAT_HEIGHT]
    keep  = np.where((areas >= MIN_REGION_SIZE) & (hs >= MIN_REGION_HEIGHT))[0] + 1
    for k in keep: good[lbls == k] = True

    full = np.zeros((H, W), dtype=bool)
    full[y:y+h, x:x+w] = good
    return full

def red_vs_green_score(red_mask, green_mask):
    k = (HEAT_BLUR_KSIZE, HEAT_BLUR_KSIZE)
    r = cv2.blur(red_mask.astype(np.float32, copy=False), k)
    g = cv2.blur(green_mask.astype(np.float32, copy=False), k)
    diff = r - g
    amax = float(np.max(np.abs(diff))) + 1e-6
    norm = (diff / (2.0 * amax) + 0.5)
    return np.clip(norm * 255.0, 0, 255.0).astype(np.uint8, copy=False)

def purple_triangles(score, H):
    top_ex = int(H * EXCLUDE_TOP_FRAC)
    bot_ex = int(H * EXCLUDE_BOTTOM_FRAC)
    dark = (score >= RED_SCORE_THRESH).astype(np.uint8, copy=False)
    if top_ex: dark[:top_ex, :] = 0
    if bot_ex: dark[-bot_ex:, :] = 0

    dark = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9)), iterations=1
    )
    total_dark = int(dark.sum())
    if total_dark == 0: return [], None

    frac_thresh = int(np.ceil(MIN_DARK_FRACTION * total_dark))
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    if n_lbl <= 1: return [], None

    tris = []
    for lbl in range(1, n_lbl):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= MIN_DARK_RED_AREA and area >= frac_thresh:
            ys, xs = np.where(lbls == lbl)
            if ys.size == 0: continue
            y_top = ys.min()
            x_mid = int(xs[ys == y_top].mean())
            tris.append((x_mid, int(y_top)))

    if not tris: return [], None
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
    if masks_np is None or classes_np is None or len(tri_positions) == 0:
        return [], [], [], []  # colours, rays, hit_class_ids, hit_distances_px

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
    masks_np = yolo_res.masks.data.detach().cpu().numpy()  # [n,h,w]
    if hasattr(yolo_res.masks, "cls") and yolo_res.masks.cls is not None:
        classes_np = yolo_res.masks.cls.detach().cpu().numpy().astype(int)
    else:
        classes_np = yolo_res.boxes.cls.detach().cpu().numpy().astype(int)

    to_cpu_ms = (time.perf_counter() - t0) * 1000.0
    mask_count = int(masks_np.shape[0])
    if mask_count == 0 or classes_np.size == 0:
        return (None, 0, mask_count, to_cpu_ms, 0.0, masks_np, classes_np, None, None,
                [], [], [], None, None, None, [], [])

    classes_np = promote_lowbarrier_when_wall(frame_bgr, masks_np, classes_np)

    rail_sel = (classes_np == RAIL_ID)
    if not np.any(rail_sel):
        return (None, 0, mask_count, to_cpu_ms, 0.0, masks_np, classes_np, None, None,
                [], [], [], None, None, None, [], [])

    t1 = time.perf_counter()
    rail_masks = masks_np[rail_sel].astype(bool, copy=False)
    union = np.any(rail_masks, axis=0).astype(np.uint8, copy=False)
    rail_mask = cv2.resize(union, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool, copy=False)

    green = highlight_rails_mask_only_fast(frame_bgr, rail_mask)
    red   = np.logical_and(rail_mask, np.logical_not(green))
    score = red_vs_green_score(red, green)
    tri_positions, tri_best = purple_triangles(score, H)

    # Jake triangle by bearing
    lane_name = lane_name_from_point(jake_point)
    target_deg = LANE_TARGET_DEG[lane_name]
    xj, yj = jake_point
    best_idx, best_deg, _ = select_triangle_by_bearing(tri_positions, xj, yj, target_deg, min_dy=6)

    # x_ref for bending
    if lane_name == "mid" and (best_idx is not None) and (0 <= best_idx < len(tri_positions)):
        x_ref = tri_positions[best_idx][0]
    else:
        x_ref = xj

    tri_colours, tri_rays, tri_hit_classes, tri_hit_dists = classify_triangles_at_sample_curved(
        tri_positions, masks_np, classes_np, H, W, jake_point, x_ref, best_idx,
        SAMPLE_UP_PX, RAY_STEP_PX
    )

    post_ms = (time.perf_counter() - t1) * 1000.0

    # Minimal movement-friendly summary (pos, hit_class id/label, is_jake)
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


    #PATHING LOGIC HERE# =================================================================================================================================================================
    # ===== PATHING / ACTION LOGIC =================================================
    jake_tri = next((t for t in tri_summary if t.get("is_jake")), None)
    if jake_tri:
        jx, jy = jake_tri["pos"]
        jake_hit = jake_tri["hit_class"]

        # For movement logging: distance to the obstacle ahead of Jake

        y_hit_log = first_red_hit_y(jake_tri["pos"], masks_np, classes_np, H, W, band_px=6, step_px=5)
        obstacle_dist_px = (jy - y_hit_log) if y_hit_log is not None else None

        # --- 1) Context actions based on distance to the obstacle ahead of Jake ---
        if jake_hit is not None:
            # choose which classes to probe vertically based on what Jake actually hit
            if int(jake_hit) in DUCK_SET:
                probe_set = DUCK_SET              # {2,4}  (High/LowBarrier1)
            elif int(jake_hit) in JUMP_SET:
                probe_set = JUMP_SET              # {3,5,10} (Jump, LowBarrier2, Sidewalk)
            else:
                probe_set = DANGER_RED            # fallback: red set

            y_hit = first_hit_y(jake_tri["pos"], masks_np, classes_np, H, W, probe_set, band_px=6, step_px=5)
            dpx = (jy - y_hit) if y_hit is not None else None

            if dpx is not None:
                if int(jake_hit) in JUMP_SET and dpx <= ACTION_DIST_PX:
                    if MOVEMENT_ENABLED:
                        print(f"[ACT] jump+duck → reason: {LABELS.get(int(jake_hit), str(jake_hit))} at { _fmt_px(dpx) } (<= {ACTION_DIST_PX}px)")
                    _try_jump_then_duck()
                elif int(jake_hit) in DUCK_SET and dpx <= ACTION_DIST_PX:
                    if MOVEMENT_ENABLED:
                        print(f"[ACT] duck → reason: {LABELS.get(int(jake_hit), str(jake_hit))} at { _fmt_px(dpx) } (<= {ACTION_DIST_PX}px)")
                    _try_duck()


        # --- 2) Lateral pathing decisions (policy: GREEN first) --------------------
        # Build reusable candidate pools (excluding Jake's current triangle)
        greens  = [t for t in tri_summary if t["hit_class"] is None]
        yellows = [t for t in tri_summary if (t["hit_class"] is not None and int(t["hit_class"]) in WARN_FOR_MOVE)]
        reds    = [t for t in tri_summary if (t["hit_class"] is not None and int(t["hit_class"]) in DANGER_RED)]

        # Lane-based pruning
        greens  = _filter_by_lane(greens,  jx, lane)
        yellows = _filter_by_lane(yellows, jx, lane)
        reds    = _filter_by_lane(reds,    jx, lane)

        # Only consider yellow if it's far enough ahead of the Jake band (e.g., 400px)
        jake_band_y   = jake_point[1]  # 1340 with your lane points
        yellows_far   = _filter_yellow_far(yellows, jake_band_y)  # uses MIN_YELLOW_AHEAD_PX
        greens_far  = _filter_green_far(greens, jake_band_y)


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
                        best_red = max(reds, key=_red_score)
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
        txt = f"{deg_signed:.1f}°"
        cv2.putText(out, txt, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BLACK, 2, cv2.LINE_AA)
        cv2.putText(out, txt, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    # curved sampling rays (viz)
    for (p0, p1, theta) in tri_rays:
        cv2.line(out, p0, p1, (255,255,255), 2, cv2.LINE_AA)
        mx = (p0[0] + p1[0]) // 2; my = (p0[1] + p1[1]) // 2
        ttxt = f"{theta:+.1f}°"
        cv2.putText(out, ttxt, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BLACK, 2, cv2.LINE_AA)
        cv2.putText(out, ttxt, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    for (x, y), col in zip(tri_positions, tri_colours):
        draw_triangle(out, int(x), int(y), colour=col)

    lane_name = lane_name_from_point(jake_point)
    target_deg = LANE_TARGET_DEG[lane_name]
    if best_idx is not None and 0 <= best_idx < len(tri_positions):
        xt, yt = tri_positions[best_idx]
        pts = triangle_pts(int(xt), int(yt), size=TRI_SIZE_PX)
        cv2.polylines(out, [pts.reshape(-1,1,2)], True, COLOR_CYAN, 3, cv2.LINE_AA)
        tag = f"JAKE_TRI ({lane_name}: target {target_deg:.1f}°)"
        cv2.putText(out, tag, (max(5, int(xt)-70), max(20, int(yt)-16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BLACK, 2, cv2.LINE_AA)
        cv2.putText(out, tag, (max(5, int(xt)-70), max(20, int(yt)-16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

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

while running:
    # Screen grab
    left, top, width, height = snap_coords
    raw = sct.grab({"left": left, "top": top, "width": width, "height": height})
    frame_bgr = np.array(raw)[:, :, :3]  # BGRA -> BGR

    # Dynamic JAKE_POINT from current lane (O(1))
    # Auto-detect lane by whiteness around the three probes (5x5 box)
    _detected = _detect_lane_by_whiteness(frame_bgr)
    if _detected is not None:
        lane = _detected  # 0/1/2
    JAKE_POINT = LANE_POINTS[lane]


    # Inference
    t0_inf = time.perf_counter()
    res_list = model.predict(
        [frame_bgr], task="segment", imgsz=IMG_SIZE, device=device,
        conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET, batch=1
    )
    infer_ms = (time.perf_counter() - t0_inf) * 1000.0
    yres = res_list[0]

    # Postproc (now returns hit classes + summary)
    (tri_best_xy, tri_count, mask_count, to_cpu_ms, post_ms,
     masks_np, classes_np, rail_mask, green_mask, tri_positions, tri_colours,
     tri_rays, best_idx, best_deg, x_ref,
     tri_hit_classes, tri_summary) = process_frame_post(frame_bgr, yres, JAKE_POINT)

    proc_ms = infer_ms + to_cpu_ms + post_ms

    # Render + save overlay
    overlay = render_overlays(frame_bgr, masks_np, classes_np, rail_mask, green_mask,
                              tri_positions, tri_colours, tri_rays, best_idx, best_deg, x_ref, JAKE_POINT)
    frame_idx += 1
    out_path = out_dir / f"live_overlay_{frame_idx:05d}.jpg"
    cv2.imwrite(str(out_path), overlay)

# Cleanup
listener.join()
print("Script halted.")
