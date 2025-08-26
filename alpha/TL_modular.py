#!/usr/bin/env python3
# TL_modular.py (instrumented)
# Runs the "script2" top-logic using precomputed results from your main loop.
# No extra model loads. No extra predict() calls.

from __future__ import annotations
import cv2, numpy as np, math, time, os
from typing import List, Dict, Tuple, Optional

# ===================== MASTER FEATURE GUARD =====================
# When False: disable ALL overlays (drawing) and disk writes.
# Computation & decisions still run identically.
SAVES_ON = False
# ===============================================================

# --- movement output (pyautogui) for lateral_TL only ---
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    _HAS_PYAUTO = True
except Exception:
    _HAS_PYAUTO = False

# Cooldown STATE strictly for lateral_TL (left/right lane taps)
_LTL_LAST_TAP_TS = 0.0
_LTL_TAP_COOLDOWN_S = 0.330  # seconds

def _ltl_try_tap(key: str) -> bool:
    global _LTL_LAST_TAP_TS, _TL_STICKY_UNTIL
    if not _HAS_PYAUTO:
        return False
    now = time.perf_counter()
    if now - _LTL_LAST_TAP_TS < _LTL_TAP_COOLDOWN_S:
        return False
    pyautogui.press(key)
    _LTL_LAST_TAP_TS = now
    # start the TOP stickiness window
    import time as _t
    _TL_STICKY_UNTIL = _t.monotonic() + STICKY_AFTER_LATERAL_S
    return True


# --- Hold state while carrying out lateral movementsp ---
_TL_STICKY_UNTIL = 0.0          # monotonic seconds
STICKY_AFTER_LATERAL_S = 0.30    # <-- your 0.3s window

def tl_sticky_until() -> float:
    """Monotonic timestamp until which TOP should be forced after a TL lateral tap."""
    return _TL_STICKY_UNTIL


# ===================== PROFILING SWITCHES =====================
PROF_VERBOSE = True          # set False to mute all profiling prints
PROF_SLOW_MS = 10.0          # highlight steps slower than this (ms)
PROF_MEM = True              # if psutil available, print RSS
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:
    PROF_MEM = False
# -------------------------------------------------------------

def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

def _p(prefix: str, msg: str):
    if PROF_VERBOSE:
        print(f"{prefix}{msg}")

def _pmaybe(prefix: str, label: str, dt_ms: float):
    if not PROF_VERBOSE: return
    slow = " ⚠️" if dt_ms >= PROF_SLOW_MS else ""
    print(f"{prefix}[PROF] {label}: {dt_ms:.2f} ms{slow}")

def _mem(prefix: str):
    if PROF_MEM:
        rss_mb = _PROC.memory_info().rss / (1024**2)
        print(f"{prefix}[MEM] RSS={rss_mb:.1f} MB")

# ----------------- CONFIG (mirrors your script exactly) -----------------

# --- suppress lateral taps while standing on a ramp ---
SUPPRESS_LAT_ON_RAMP = True
RAMP_CLASS_ID = 8

# --- yellow-arrow → scanline decisioning (NEW) ---
SCAN_Y             = 1450   # scan band center
SCAN_BAND_PX       = 6      # ± vertical tolerance for y-band
SCAN_PERSIST       = True   # once triggered, keep scanning every frame

# probe thrown vertically from the yellow arrow tip (target)
YELLOW_PROBE_LEN   = 100    # pixels upward from the tip
YELLOW_PROBE_THICK = 2      # half-width (x) around the tip for probe hit test

# sticky state (lives for this process)
_scan_persist_active = False

# Lateral move activation gate
LAT_MOVE_Y_THRESH_PX = 400  # active when arrow length (dy) < this
lat_move_active = False


BRIDGE_GAP_PX = 32   # max gap to bridge (pixels along ray)

# --- endpoint marker (triangles) ---
TRI_SIZE_PX   = 12                 # triangle size
TRI_BLUE      = (255, 0, 0)        # BGR (blue)
TRI_PURPLE    = (255, 0, 255)      # BGR (purple)
TRI_OUTLINE   = (0, 0, 0)          # black outline
TRI_OUT_THICK = 2

# --- lane anchors & selection threshold (screen coords) ---
LANE_LEFT   = (300, 1340)
LANE_MID    = (490, 1340)
LANE_RIGHT  = (680, 1340)
LANE_ANCHORS = {0: LANE_LEFT, 1: LANE_MID, 2: LANE_RIGHT}

# ---- add up to 6 more points here (second dot per lane) ----
EXTRA_DOTS = [
    (300, 1300),  # LEFT 2
    (490, 1300),  # MID 2
    (680, 1300),  # RIGHT 2
]

# Map each lane to its two check points
LANE_DOTS = {
    0: [LANE_LEFT,  EXTRA_DOTS[0]],
    1: [LANE_MID,   EXTRA_DOTS[1]],
    2: [LANE_RIGHT, EXTRA_DOTS[2]],
}

# RGBA color gate config
LANE_DOT_TARGET_RGBA = (252, 245, 226, 255)
LANE_DOT_TOL_FRAC    = 0.90  # 4% per channel

# Primary ray to use when selecting the triangle for a lane
PRIMARY_RAY_FOR_LANE = {0: "LEFT", 1: "MID", 2: "RIGHT"}

ALPHA = 0.60
NEON_GREEN = (57, 255, 20)
OUTLINE_THICKNESS = 2

RAY_STEP_PX = 2       # ← set to 5 for ~5px accuracy if you like
PROBE_BAND  = 0       # 0 = single-pixel ray; >0 = widen horizontally by ±band

EXCLUDE_CLASSES = {9} # rails ignored for "first hit"

TRAIN_CLASSES = {1, 6, 8, 11}   # GREYTRAIN, ORANGETRAIN, RAMP, YELLOWTRAIN

CLASS_COLOURS = {
    0:(255,255,0), 1:(96,96,96), 2:(0,128,255), 3:(0,255,0),
    4:(255,0,255), 5:(0,255,255), 6:(255,128,0), 7:(128,0,255),
    8:(0,0,128), 9:(0,0,255), 10:(128,128,0), 11:(255,255,102)
}
LABELS = {
    0:"BOOTS",1:"GREYTRAIN",2:"HIGHBARRIER1",3:"JUMP",4:"LOWBARRIER1",
    5:"LOWBARRIER2",6:"ORANGETRAIN",7:"PILLAR",8:"RAMP",9:"RAILS",
    10:"SIDEWALK",11:"YELLOWTRAIN"
}

LANE_RAYS = {
    0: [
        {"name": "LEFT",  "start": (375, 1800), "length": 1200, "angle_deg": +2.0},
        {"name": "MID",   "start": (1010, 1800), "length": 1300, "angle_deg": -21.5},
    ],
    1: [
        {"name": "LEFT",  "start": (0, 1600),   "length": 950, "angle_deg": +19.0},
        {"name": "MID",   "start": (490, 1800), "length": 1125, "angle_deg": -1.5},
        {"name": "RIGHT", "start": (1010,1600), "length": 1000, "angle_deg": -23.0},
    ],
    2: [
        {"name": "MID",     "start": (100, 1800), "length": 1100, "angle_deg": +12.3},
        {"name": "RIGHT",   "start": (650, 1800), "length": 1200, "angle_deg":  -8.0},
    ],
}

# ----------------- Helpers (identical logic) -----------------

def _is_on_ramp_now(lane_idx: int,
                    rays_for_lane: List[Dict],
                    segments_by_ray: Dict[str, List[Tuple[float, float, int]]],
                    scan_y: int,
                    band_px: int) -> bool:
    """
    True if the lane's PRIMARY ray is a RAMP at y=scan_y (±band).
    Uses the already-built segments_by_ray, so it's cheap.
    """
    pr_name = PRIMARY_RAY_FOR_LANE.get(lane_idx, "MID")
    ray = next((r for r in rays_for_lane if r.get("name") == pr_name), None)
    if not ray:
        return False
    segs = segments_by_ray.get(pr_name, [])
    for yy in range(scan_y - band_px, scan_y + band_px + 1):
        hit = _intersect_ray_with_y(ray["start"], float(ray["angle_deg"]), float(ray["length"]), yy)
        if not hit:
            continue
        t, _ = hit
        seg = _segment_at_t(segs, t)
        if seg and int(seg[2]) == RAMP_CLASS_ID:
            return True
    return False

def _build_class_mask(H: int, W: int,
                      masks_np: Optional[np.ndarray],
                      classes_np: Optional[np.ndarray],
                      class_ids: set) -> np.ndarray:
    """Boolean HxW mask for any of class_ids."""
    m = np.zeros((H, W), dtype=bool)
    if masks_np is None or classes_np is None or masks_np.size == 0:
        return m
    n = min(masks_np.shape[0], classes_np.shape[0])
    for mk, ck in zip(masks_np[:n], classes_np[:n]):
        if int(ck) not in class_ids:
            continue
        mask = _resize_mask_to_frame(mk, W, H)
        m |= mask
    return m


def _intersect_ray_with_y(start_xy: Tuple[int, int],
                          angle_deg: float,
                          length_px: float,
                          y_target: int) -> Optional[Tuple[float, Tuple[int, int]]]:
    """Return (t, (x,y)) for intersection of a ray segment with horizontal line y=y_target."""
    x0, y0 = map(int, start_xy)
    r = math.radians(angle_deg)
    dx, dy = math.sin(r), -math.cos(r)
    if abs(dy) < 1e-6:
        return None
    t = (y_target - y0) / dy
    if t < 0 or t > float(length_px):
        return None
    x = int(round(x0 + dx * t))
    y = int(round(y0 + dy * t))
    return float(t), (x, y)


def _segment_at_t(segments: List[Tuple[float, float, int]], t: float
                  ) -> Optional[Tuple[float, float, int]]:
    """Find the (t0,t1,cls) segment that contains t, else None."""
    for (t0, t1, c) in segments:
        if t0 <= t <= t1:
            return (t0, t1, int(c))
    return None


def _rails_hit_vertical_probe(rails_map: np.ndarray,
                              tip_xy: Tuple[int, int],
                              length: int,
                              thick: int,
                              *,
                              draw: bool = False,
                              out_img: Optional[np.ndarray] = None) -> bool:
    """
    Shoot a vertical probe from the yellow-arrow tip upward (negative y).
    If any pixel under the probe overlaps rails_map => True.
    thick is the +/- x half-width around the tip.
    """
    H, W = rails_map.shape
    x0, y0 = int(tip_xy[0]), int(tip_xy[1])
    y1 = max(0, y0 - int(length))
    xL = max(0, x0 - int(thick))
    xR = min(W - 1, x0 + int(thick))

    hit = rails_map[y1:y0+1, xL:xR+1].any()

    if draw and out_img is not None:
        # draw the probe in cyan; red dot if hit, white if not
        cv2.line(out_img, (x0, y0), (x0, y1), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(out_img, (xL, y1), (xR, y0), (128, 128, 128), 1, cv2.LINE_AA)
        cv2.circle(out_img, (x0, y1), 5, (0, 0, 255) if hit else (255, 255, 255), -1, cv2.LINE_AA)
    return bool(hit)


def _scan_and_decide_move(lane_idx: int,
                          rays_for_lane: List[Dict],
                          segments_by_ray: Dict[str, List[Tuple[float, float, int]]],
                          band_px: int,
                          *,
                          do_overlays: bool,
                          out_img: Optional[np.ndarray]) -> Tuple[Optional[str], Dict[str, object]]:
    """
    Intersect the SCAN_Y±band with the available hard-coded rays.
    If the intersection lies on a TRAIN/RAMP segment, propose a move.
    If both sides valid, choose the longer segment (t1-t0).
    """
    # which rays we can move onto from each lane
    plan: List[Tuple[str, str]] = []
    if lane_idx == 1:      # MID -> can go LEFT or RIGHT
        plan = [("LEFT", "MOVE_LEFT"), ("RIGHT", "MOVE_RIGHT")]
    elif lane_idx == 0:    # LEFT -> can only go RIGHT (onto MID ray)
        plan = [("MID", "MOVE_RIGHT")]
    else:                  # RIGHT -> can only go LEFT (onto MID ray)
        plan = [("MID", "MOVE_LEFT")]

    r_by_name = {r["name"]: r for r in rays_for_lane}
    print(f"[scan] lane={lane_idx} plan={plan}")

    candidates = []  # (seg_len, action, ray_name, xy, t, cls_id)

    for ray_name, action in plan:
        ray = r_by_name.get(ray_name)
        if ray is None:
            continue
        segs = segments_by_ray.get(ray_name, [])
        best_here = None
        best_len = -1.0

        for yy in range(SCAN_Y - band_px, SCAN_Y + band_px + 1):
            hit = _intersect_ray_with_y(ray["start"], float(ray["angle_deg"]), float(ray["length"]), yy)
            if not hit:
                continue
            t, (x, y) = hit
            seg = _segment_at_t(segs, t)
            if not seg:
                continue
            t0, t1, cls_id = seg
            if int(cls_id) not in TRAIN_CLASSES:
                continue
            seg_len = float(t1 - t0)
            if seg_len > best_len:
                best_len = seg_len
                best_here = (seg_len, action, ray_name, (int(x), int(y)), float(t), int(cls_id))

        if best_here is not None:
            print(f"[scan] candidate {ray_name}: seg_len={best_len:.1f} -> {action}")
            candidates.append(best_here)

    if not candidates:
        print(f"[scan] no valid train/ramp at y={SCAN_Y}±{band_px}")
        return None, {"reason": "no-train-or-ramp-on-scanline", "scan_y": SCAN_Y, "band": band_px}

    candidates.sort(key=lambda k: k[0], reverse=True)
    seg_len, action, rname, xy, tval, cid = candidates[0]
    print(f"[scan] choose {action} on {rname} len={seg_len:.0f} class={LABELS.get(cid, cid)} @xy={xy}")


    if do_overlays and out_img is not None:
        # draw scan band + chosen intersection
        y0 = SCAN_Y
        cv2.line(out_img, (0, y0), (out_img.shape[1]-1, y0), (255, 255, 255), 1, cv2.LINE_AA)
        if band_px > 0:
            cv2.line(out_img, (0, y0-band_px), (out_img.shape[1]-1, y0-band_px), (128, 128, 128), 1, cv2.LINE_AA)
            cv2.line(out_img, (0, y0+band_px), (out_img.shape[1]-1, y0+band_px), (128, 128, 128), 1, cv2.LINE_AA)
        cv2.circle(out_img, xy, 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out_img, xy, 4, (0, 255, 0), -1, cv2.LINE_AA)
        lbl = f"{rname}:{LABELS.get(cid, cid)} len={seg_len:.0f} → {action}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = xy
        cv2.rectangle(out_img, (x+8, y-8-th-6), (x+8+tw+6, y-8), (0,0,0), -1, cv2.LINE_AA)
        cv2.putText(out_img, lbl, (x+12, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    return action, {
        "reason": "train-or-ramp-on-scanline",
        "chosen_ray": rname,
        "scan_y": SCAN_Y,
        "band": band_px,
        "seg_len": seg_len,
        "t": tval,
        "class": cid
    }

def _rgba_dot_match(img: np.ndarray, xy: Tuple[int,int],
                    target_rgba: Tuple[int,int,int,int],
                    tol_frac: float) -> bool:
    """
    True if the pixel at xy matches target RGBA within ±tol on each channel.
    Works with BGR (3ch) or BGRA (4ch) frames. Alpha is ignored for 3ch.
    """
    H, W = img.shape[:2]
    x = _clampi(int(xy[0]), 0, W-1)
    y = _clampi(int(xy[1]), 0, H-1)

    tol = int(round(255 * tol_frac))
    # target in BGR(A) order
    tb, tg, tr, ta = target_rgba[2], target_rgba[1], target_rgba[0], target_rgba[3]

    px = img[y, x]
    if px.ndim == 0:  # safety: unexpected shape
        return False

    if img.shape[2] == 3:  # BGR
        b, g, r = int(px[0]), int(px[1]), int(px[2])
        return (abs(b - tb) <= tol and abs(g - tg) <= tol and abs(r - tr) <= tol)
    elif img.shape[2] == 4:  # BGRA
        b, g, r, a = int(px[0]), int(px[1]), int(px[2]), int(px[3])
        return (abs(b - tb) <= tol and abs(g - tg) <= tol and abs(r - tr) <= tol and abs(a - ta) <= tol)
    else:
        return False


def _lane_dot_color_gate(img: np.ndarray, lane_idx: int,
                         target_rgba: Tuple[int,int,int,int] = LANE_DOT_TARGET_RGBA,
                         tol_frac: float = LANE_DOT_TOL_FRAC) -> bool:
    """Return True iff at least one of the current lane's two dots matches the target color."""
    pts = LANE_DOTS.get(int(lane_idx), [])
    for p in pts:
        if p is None:
            continue
        if _rgba_dot_match(img, p, target_rgba, tol_frac):
            return True
    return False


def _to_gray_bgr(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def _resize_mask_to_frame(mask_f: np.ndarray, W: int, H: int) -> np.ndarray:
    m8 = (mask_f > 0.5).astype(np.uint8)
    if m8.shape != (H, W):
        m8 = cv2.resize(m8, (W, H), interpolation=cv2.INTER_NEAREST)
    return m8.astype(bool)

def _draw_neon_outline(img: np.ndarray, mask_bool: np.ndarray):
    if not mask_bool.any(): return
    m8 = (mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(img, contours, -1, NEON_GREEN, OUTLINE_THICKNESS, lineType=cv2.LINE_AA)

def bridge_small_gaps(segments: List[Tuple[float, float, int]],
                      gap_px: float = 20.0) -> List[Tuple[float, float, int]]:
    if not segments:
        return segments
    merged = [segments[0]]
    for t0, t1, c in segments[1:]:
        pt0, pt1, pc = merged[-1]
        gap = t0 - pt1
        if c == pc and 0 < gap <= gap_px:
            merged[-1] = (pt0, t1, pc)
        else:
            merged.append((t0, t1, c))
    return merged

def _unit(vx: float, vy: float) -> Tuple[float, float]:
    n = math.hypot(vx, vy)
    return (vx / n, vy / n) if n > 0 else (0.0, 0.0)

def _point_on_ray(start_xy: Tuple[int,int], angle_deg: float, t: float) -> Tuple[int,int]:
    x0, y0 = map(int, start_xy)
    r = math.radians(angle_deg)
    dx, dy = math.sin(r), -math.cos(r)  # 0° is straight up
    return (int(round(x0 + dx * t)), int(round(y0 + dy * t)))

def _draw_triangle_marker(img: np.ndarray,
                          center_xy: Tuple[int,int],
                          dir_vec: Tuple[float,float],
                          size_px: int,
                          fill_color: Tuple[int,int,int],
                          outline_color: Tuple[int,int,int],
                          outline_thick: int = 2) -> None:
    ux, uy = _unit(*dir_vec)
    px, py = -uy, ux
    tip = (int(round(center_xy[0] + ux * size_px)),
           int(round(center_xy[1] + uy * size_px)))
    bx  = center_xy[0] - ux * (0.6 * size_px)
    by  = center_xy[1] - uy * (0.6 * size_px)
    p1 = (int(round(bx + px * (0.6 * size_px))),
          int(round(by + py * (0.6 * size_px))))
    p2 = (int(round(bx - px * (0.6 * size_px))),
          int(round(by - py * (0.6 * size_px))))
    pts = np.array([tip, p1, p2], dtype=np.int32)
    cv2.fillConvexPoly(img, pts, fill_color, lineType=cv2.LINE_AA)
    cv2.polylines(img, [pts], isClosed=True, color=outline_color,
                  thickness=outline_thick, lineType=cv2.LINE_AA)

def shade_and_outline_trains(base_gray_bgr: np.ndarray,
                             masks_np: np.ndarray,
                             classes_np: np.ndarray) -> np.ndarray:
    H, W = base_gray_bgr.shape[:2]
    out = base_gray_bgr.copy()
    if masks_np is None or masks_np.size == 0: return out
    n = min(masks_np.shape[0], classes_np.shape[0])
    for m, c in zip(masks_np[:n], classes_np[:n]):
        cls_id = int(c)
        color = CLASS_COLOURS.get(cls_id, (255, 255, 255))
        mask = _resize_mask_to_frame(m, W, H)
        if mask.any():
            region = out[mask].astype(np.float32)
            blended = (1.0 - ALPHA) * region + ALPHA * np.array(color, np.float32)
            out[mask] = blended.astype(np.uint8)
            if cls_id in TRAIN_CLASSES:
                _draw_neon_outline(out, mask)
    return out

def _lane_from_x(x: int) -> int:
    xL = LANE_LEFT[0]; xM = LANE_MID[0]; xR = LANE_RIGHT[0]
    m01 = 0.5 * (xL + xM)
    m12 = 0.5 * (xM + xR)
    if x <= m01: return 0
    if x <  m12: return 1
    return 2

def _draw_anchor_vector(out: np.ndarray, anchor_xy: Tuple[int,int], tri_xy: Tuple[int,int],
                        color=(0,255,255)):
    ax, ay = anchor_xy
    tx, ty = tri_xy
    cv2.arrowedLine(out, (ax, ay), (tx, ty), color, 2, cv2.LINE_AA, tipLength=0.04)
    midx = int(round(0.5*(ax + tx)))
    midy = int(round(0.5*(ay + ty)))
    dy   = ay - ty
    label = f"Δy={dy:d}px"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(out, (midx+6, midy-8-th-6), (midx+6+tw+6, midy-8), (0,0,0), -1, cv2.LINE_AA)
    cv2.putText(out, label, (midx+10, midy-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.circle(out, (ax, ay), 4, (0,0,0), -1, cv2.LINE_AA)
    cv2.circle(out, (ax, ay), 2, (255,255,255), -1, cv2.LINE_AA)
    cv2.circle(out, (tx, ty), 4, (0,0,0), -1, cv2.LINE_AA)
    cv2.circle(out, (tx, ty), 2, (255,255,255), -1, cv2.LINE_AA)

def _draw_badge(img: np.ndarray, text: str, x: int = 10, y: int = 34):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x, y-16), (x+tw+12, y+6), (0,0,0), -1, cv2.LINE_AA)
    cv2.putText(img, text, (x+6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def ray_train_class_segments(start_xy: Tuple[int,int], angle_deg: float, length_px: float,
                             step_px: int, class_map: np.ndarray, train_map: np.ndarray,
                             band: int = 0) -> List[Tuple[float, float, int]]:
    H, W = class_map.shape
    x0, y0 = map(int, start_xy)
    r = math.radians(angle_deg)
    dx, dy = math.sin(r), -math.cos(r)
    steps = max(1, int(length_px // max(1, step_px)))
    t = np.arange(1, steps+1, dtype=np.int32) * step_px
    xs = np.clip(np.round(x0 + dx * t).astype(np.int32), 0, W-1)
    ys = np.clip(np.round(y0 + dy * t).astype(np.int32), 0, H-1)
    if band <= 0:
        cls_line = class_map[ys, xs]
    else:
        votes = [class_map[ys, xs]]
        for dxb in range(1, band+1):
            votes.append(class_map[ys, np.clip(xs + dxb, 0, W-1)])
            votes.append(class_map[ys, np.clip(xs - dxb, 0, W-1)])
        V = np.stack(votes, axis=0)
        cls_line = np.empty(steps, dtype=np.int32)
        for j in range(steps):
            col = V[:, j]
            col = col[col >= 0]
            if col.size:
                vals, counts = np.unique(col, return_counts=True)
                cls_line[j] = int(vals[np.argmax(counts)])
            else:
                cls_line[j] = -1
    segments: List[Tuple[float, float, int]] = []
    prev_cls = -1
    run_start = None
    for i in range(steps):
        c = int(cls_line[i])
        if c >= 0:
            if prev_cls == -1:
                prev_cls = c
                run_start = i
            elif c != prev_cls:
                segments.append((float(t[run_start]), float(t[i-1]), prev_cls))
                prev_cls = c
                run_start = i
        else:
            if prev_cls != -1 and run_start is not None:
                segments.append((float(t[run_start]), float(t[i-1]), prev_cls))
                prev_cls = -1
                run_start = None
    if prev_cls != -1 and run_start is not None:
        segments.append((float(t[run_start]), float(t[-1]), prev_cls))
    return segments

def _sx_sy(masks_np, H, W):
    mh, mw = masks_np.shape[1], masks_np.shape[2]
    sx = (mw - 1) / max(1, (W - 1))
    sy = (mh - 1) / max(1, (H - 1))
    return sx, sy

def _clampi(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def _first_hit_on_ray(masks_np, classes_np, H, W, start_xy, angle_deg, length_px,
                      step_px=2, exclude=EXCLUDE_CLASSES, band=PROBE_BAND):
    if masks_np is None or masks_np.size == 0 or classes_np is None:
        return (None, None, None)
    sx, sy = _sx_sy(masks_np, H, W)
    test_idxs = [i for i, c in enumerate(classes_np) if int(c) not in exclude]
    if not test_idxs:
        return (None, None, None)
    x0, y0 = map(int, start_xy)
    x0 = _clampi(x0, 0, W-1); y0 = _clampi(y0, 0, H-1)
    r = math.radians(angle_deg)
    dx, dy = math.sin(r), -math.cos(r)
    steps = max(1, int(length_px // max(1, step_px)))
    for k in range(1, steps+1):
        t = k * step_px
        xs = _clampi(int(round(x0 + dx * t)), 0, W-1)
        ys = _clampi(int(round(y0 + dy * t)), 0, H-1)
        if band <= 0:
            mx = _clampi(int(round(xs * sx)), 0, masks_np.shape[2]-1)
            my = _clampi(int(round(ys * sy)), 0, masks_np.shape[1]-1)
            for i in test_idxs:
                if masks_np[i][my, mx] > 0.5:
                    return (int(classes_np[i]), (xs, ys), float(t))
        else:
            for dxb in range(-band, band+1):
                xsb = _clampi(xs + dxb, 0, W-1)
                mx  = _clampi(int(round(xsb * sx)), 0, masks_np.shape[2]-1)
                my  = _clampi(int(round(ys  * sy)), 0, masks_np.shape[1]-1)
                for i in test_idxs:
                    if masks_np[i][my, mx] > 0.5:
                        return (int(classes_np[i]), (xs, ys), float(t))
    return (None, None, None)

def build_class_maps(H:int, W:int, masks_np:np.ndarray, classes_np:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    class_map = np.full((H, W), -1, dtype=np.int32)
    train_map = np.zeros((H, W), dtype=bool)
    if masks_np is None or masks_np.size == 0 or classes_np is None:
        return class_map, train_map
    n = min(masks_np.shape[0], classes_np.shape[0])
    for m, c in zip(masks_np[:n], classes_np[:n]):
        cls = int(c)
        if cls not in TRAIN_CLASSES:
            continue
        mask = _resize_mask_to_frame(m, W, H)
        if mask.any():
            class_map[mask] = cls
            train_map[mask] = True
    return class_map, train_map

def _draw_equations_overlay(img: np.ndarray, lane_2: int, rays: List[Dict]):
    header = f"lane_2 = {lane_2}  ({ {0:'LEFT',1:'MID',2:'RIGHT'}.get(lane_2,'?') })"
    cv2.rectangle(img, (10, 10), (10+420, 10+22), (0,0,0), -1, cv2.LINE_AA)
    cv2.putText(img, header, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    y = 54
    for ray in rays:
        x0, y0 = ray["start"]; L = ray["length"]; th = float(ray["angle_deg"])
        r = math.radians(th); dx, dy = math.sin(r), -math.cos(r)
        eq = f"{ray['name']}: P(t)=({x0},{y0}) + t*({dx:+.3f},{dy:+.3f}), t∈[0,{L}]"
        (tw, thh), _ = cv2.getTextSize(eq, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (10, y-16), (10+max(420, tw+12), y+6), (0,0,0), -1, cv2.LINE_AA)
        cv2.putText(img, eq, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y += 22

def _draw_labeled_line(img, start_xy, angle_deg, length_px, label_text,
                       color=(255,255,255), thick=2):
    x0, y0 = map(int, start_xy)
    r = math.radians(angle_deg); dx, dy = math.sin(r), -math.cos(r)
    x1 = int(round(x0 + dx * length_px)); y1 = int(round(y0 + dy * length_px))
    H, W = img.shape[:2]
    x1 = _clampi(x1, 0, W-1); y1 = _clampi(y1, 0, H-1)
    cv2.line(img, (x0,y0), (x1,y1), color, thick, cv2.LINE_AA)
    if label_text:
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x0+6, y0-8-th-6), (x0+6+tw+6, y0-8), (0,0,0), -1, cv2.LINE_AA)
        cv2.putText(img, label_text, (x0+10, y0-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def _draw_interval_segment(img, start_xy, angle_deg, t0, t1, color, thick=4):
    x0, y0 = map(int, start_xy)
    r = math.radians(angle_deg); dx, dy = math.sin(r), -math.cos(r)
    p0 = (int(round(x0 + dx * t0)), int(round(y0 + dy * t0)))
    p1 = (int(round(x0 + dx * t1)), int(round(y0 + dy * t1)))
    cv2.line(img, p0, p1, color, thick, cv2.LINE_AA)

# ----------------- Public API -----------------
def run_top_logic_on_frame(
    frame_bgr_2: np.ndarray,
    masks_np: Optional[np.ndarray],
    classes_np: Optional[np.ndarray],
    lane_2: int,
    *,
    save_frames: bool = False,
    save_path: Optional[str] = None,
    print_prefix: str = ""
) -> Dict[str, object]:
    """
    Executes the exact 'script2' logic on a single frame using precomputed masks/classes.
    Returns a dict with decision info; optionally saves the annotated frame.

    Guard behavior:
      - If SAVES_ON is False: skip ALL overlays/drawing and disk writes. Computation unchanged.
      - If SAVES_ON is True: behavior identical to original.
    """
    pfx = f"{print_prefix}[TL] " if print_prefix else "[TL] "
    t_total = time.perf_counter()
    H, W = frame_bgr_2.shape[:2]

    lat_move_active = False 

    # ---- LANE-DOT RGBA GATE (fast skip if not present) ----
    if not _lane_dot_color_gate(frame_bgr_2, lane_2,
                                target_rgba=LANE_DOT_TARGET_RGBA,
                                tol_frac=LANE_DOT_TOL_FRAC):
        _p(pfx, "[skip] lane-dot RGBA gate failed (no target-coloured dot in current lane)")
        # instant return; no overlays or disk I/O
        return {
            "out_img": frame_bgr_2,   # unchanged
            "target": None,
            "dy": None,
            "source_note": "skip: lane-dot rgba gate",
            "purple_tris": [],
            "segments_by_ray": {},
        }
    
    # Resolve guards
    do_overlays = bool(SAVES_ON)
    do_saves    = bool(SAVES_ON and save_frames)

    # [PROF] grayscale conversion (only needed for overlays)
    if do_overlays:
        t0 = time.perf_counter()
        base_gray = _to_gray_bgr(frame_bgr_2)
        _pmaybe(pfx, "to_gray_bgr", _ms(t0))

    if masks_np is None or classes_np is None or masks_np.size == 0:
        out = (base_gray.copy() if do_overlays else frame_bgr_2.copy())
        rays = LANE_RAYS.get(lane_2, [])
        if do_overlays:
            _draw_equations_overlay(out, lane_2, rays)
            for ray in rays:
                _draw_labeled_line(out, ray["start"], ray["angle_deg"], ray["length"], f'{ray["name"]}: none')
        if do_saves and save_path:
            t_save = time.perf_counter()
            cv2.imwrite(str(save_path), out)
            _pmaybe(pfx, "imwrite (no-masks)", _ms(t_save))
        _p(pfx, f"[ok] no masks/classes; lane_2={lane_2}")
        _pmaybe(pfx, "TOTAL", _ms(t_total))
        _mem(pfx)
        return {
            "out_img": out,
            "target": None,
            "dy": None,
            "source_note": "no-masks",
            "purple_tris": [],
            "segments_by_ray": {}
        }

    n = min(masks_np.shape[0], classes_np.shape[0])
    _p(pfx, f"[info] HxW={H}x{W} | masks={n} (input={masks_np.shape[0]}) | unique classes={np.unique(classes_np[:n]).tolist()} | lane_2={lane_2}")

    # [PROF] shading + outlines (only for overlays)
    if do_overlays:
        t0 = time.perf_counter()
        out = shade_and_outline_trains(base_gray, masks_np[:n], classes_np[:n])
        _pmaybe(pfx, "shade_and_outline_trains", _ms(t0))
    else:
        out = frame_bgr_2.copy()

    # [PROF] build class maps (pure compute; always)
    t0 = time.perf_counter()
    class_map, train_map = build_class_maps(H, W, masks_np[:n], classes_np[:n])

    # (NEW) rails map (class 9) for yellow-probe gating
    rails_map = _build_class_mask(H, W, masks_np[:n], classes_np[:n], class_ids={9})
    _p(pfx, f"[maps] rails_px={int(rails_map.sum())}")


    dt = _ms(t0)
    train_px = int(train_map.sum())
    _p(pfx, f"[info] train pixels={train_px}")
    _pmaybe(pfx, "build_class_maps", dt)

    rays = LANE_RAYS.get(lane_2, [])
    if do_overlays:
        _draw_equations_overlay(out, lane_2, rays)

    purple_tris: List[Dict[str, object]] = []
    segments_by_ray: Dict[str, List[Tuple[float,float,int]]] = {}

    # -------- per-ray work --------
    for ray in rays:
        rname = ray["name"]
        x0, y0 = ray["start"]
        L = int(ray["length"])
        th = float(ray["angle_deg"])
        steps = max(1, int(L // max(1, RAY_STEP_PX)))
        _p(pfx, f"[ray:{rname}] start=({x0},{y0}) angle={th:.2f} len={L} step={RAY_STEP_PX} steps={steps}")

        # draw probe line label (cheap) — overlays only
        if do_overlays:
            t0_drawlabel = time.perf_counter()
            _draw_labeled_line(out, ray["start"], th, L, f'{rname}')
            _pmaybe(pfx, f"draw_label[{rname}]", _ms(t0_drawlabel))

        # first-hit (optional) — compute always, draw marker only if overlays
        t0_hit = time.perf_counter()
        cls_id, hit_xy, dist_px = _first_hit_on_ray(
            masks_np, classes_np, H, W,
            start_xy=ray["start"],
            angle_deg=th,
            length_px=L,
            step_px=RAY_STEP_PX,
            exclude=EXCLUDE_CLASSES,
            band=PROBE_BAND,
        )
        _pmaybe(pfx, f"first_hit[{rname}]", _ms(t0_hit))
        if cls_id is not None:
            if do_overlays:
                cv2.circle(out, hit_xy, 5, (0,0,0), -1, cv2.LINE_AA)
                cv2.circle(out, hit_xy, 3, (255,255,255), -1, cv2.LINE_AA)
            _p(pfx, f"[ray:{rname}] first_hit cls={LABELS.get(int(cls_id),int(cls_id))} @t≈{dist_px:.0f}px xy={hit_xy}")
        else:
            _p(pfx, f"[ray:{rname}] first_hit none")

        # segments build — always
        t0_seg = time.perf_counter()
        segments = ray_train_class_segments(
            start_xy=ray["start"], angle_deg=th, length_px=L,
            step_px=RAY_STEP_PX, class_map=class_map, train_map=train_map,
            band=PROBE_BAND,
        )
        dt_seg = _ms(t0_seg)
        _pmaybe(pfx, f"segments[{rname}] build", dt_seg)
        _p(pfx, f"[ray:{rname}] segments_raw={len(segments)}")

        # gap bridging — always
        t0_br = time.perf_counter()
        segments = bridge_small_gaps(segments, gap_px=BRIDGE_GAP_PX)
        dt_br = _ms(t0_br)
        _pmaybe(pfx, f"segments[{rname}] bridge_small_gaps", dt_br)
        _p(pfx, f"[ray:{rname}] segments_merged={len(segments)} (gap≤{BRIDGE_GAP_PX}px)")

        segments_by_ray[rname] = segments

        # drawing + triangle math:
        # - math is ALWAYS computed to retain logic
        # - actual drawing calls are only executed if do_overlays
        t0_draw = time.perf_counter()
        rr = math.radians(th)
        dx, dy = math.sin(rr), -math.cos(rr)
        for (t0s, t1s, cls_id_) in segments:
            # interval line (overlay only)
            if do_overlays:
                col = CLASS_COLOURS.get(int(cls_id_), (255,255,255))
                _draw_interval_segment(out, ray["start"], th, t0s, t1s, col, thick=4)

            # label & triangle endpoints (math always, drawing guarded)
            tm = 0.5*(t0s+t1s)
            xm = int(round(ray["start"][0] + dx*tm))
            ym = int(round(ray["start"][1] + dy*tm))
            if do_overlays:
                label = f"{LABELS.get(int(cls_id_), int(cls_id_))} [{t0s:.0f}→{t1s:.0f}]"
                (tw, thh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(out, (xm+6, ym-8-thh-6), (xm+6+tw+6, ym-8), (0,0,0), -1, cv2.LINE_AA)
                cv2.putText(out, label, (xm+10, ym-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

            p0 = _point_on_ray(ray["start"], th, t0s)
            p1 = _point_on_ray(ray["start"], th, t1s)

            if p0[1] < p1[1]:
                upper_pt, upper_t = p0, t0s
                tail_pt,  tail_t  = p1, t1s
            else:
                upper_pt, upper_t = p1, t1s
                tail_pt,  tail_t  = p0, t0s

            forward_dir  = (dx, dy)
            backward_dir = (-dx, -dy)
            forward_pt   = p1
            backward_pt  = p0

            tail_dir = forward_dir if tail_pt == forward_pt else backward_dir
            upper_dir = forward_dir if upper_pt == forward_pt else backward_dir

            # draw triangles (overlay only)
            if do_overlays:
                _draw_triangle_marker(out, tail_pt, tail_dir,
                                      size_px=TRI_SIZE_PX,
                                      fill_color=TRI_BLUE,
                                      outline_color=TRI_OUTLINE,
                                      outline_thick=TRI_OUT_THICK)
                _draw_triangle_marker(out, upper_pt, upper_dir,
                                      size_px=TRI_SIZE_PX,
                                      fill_color=TRI_PURPLE,
                                      outline_color=TRI_OUTLINE,
                                      outline_thick=TRI_OUT_THICK)

            # RECORD the purple triangle (logic always)
            purple_tris.append({"pt": upper_pt, "t": float(upper_t), "ray_name": rname})

        _pmaybe(pfx, f"drawing[{rname}] segments+triangles", _ms(t0_draw))

    _p(pfx, f"[info] purple_triangles total={len(purple_tris)}")

    # choose target purple triangle (identical policy; compute always)
    anchor_xy = LANE_ANCHORS.get(lane_2, LANE_MID)
    anchor_y  = anchor_xy[1]
    primary_ray = PRIMARY_RAY_FOR_LANE.get(lane_2, "MID")

    primary_candidates = [d for d in purple_tris
                          if d["ray_name"] == primary_ray and d["pt"][1] < anchor_y]
    lane_candidates = [d for d in purple_tris
                       if d["pt"][1] < anchor_y and _lane_from_x(d["pt"][0]) == lane_2]
    any_candidates = [d for d in purple_tris if d["pt"][1] < anchor_y]
    _p(pfx, f"[choose] primary({primary_ray})={len(primary_candidates)} | same-lane={len(lane_candidates)} | any={len(any_candidates)}")

    target = None
    if primary_candidates:
        target = min(primary_candidates, key=lambda d: d["t"])["pt"]
        source_note = f"primary={primary_ray}"
    else:
        if lane_candidates:
            target = max(lane_candidates, key=lambda d: d["pt"][1])["pt"]
            source_note = "fallback=same-lane"
        else:
            if any_candidates:
                target = max(any_candidates, key=lambda d: d["pt"][1])["pt"]
                source_note = "fallback=any"
            else:
                source_note = "none-found"

    if target is not None:
        dy_val = anchor_y - target[1]
        lat_move_active = (dy_val is not None) and (dy_val < LAT_MOVE_Y_THRESH_PX)
        _p(pfx, f"[gate] LAT_Move={'ON' if lat_move_active else 'OFF'} (dy={dy_val}, thr={LAT_MOVE_Y_THRESH_PX})")

        if do_overlays:
            _draw_anchor_vector(out, anchor_xy, target, color=(0,255,255))
        _p(pfx, f"[choose] target={target} dy={dy_val} note={source_note}")
    else:
        dy_val = None
        _p(pfx, f"[choose] target=None note={source_note}")

        # ----------------- (NEW) SCANLINE DECISIONING -----------------
    scan_triggered = False
    scan_reason = None
    sideways_action = None

    rays_for_lane = LANE_RAYS.get(lane_2, [])

    # 4a) Gate: yellow-arrow vertical probe touching rails → turn on persistence
    global _scan_persist_active
    if target is not None:
        probe_hit = _rails_hit_vertical_probe(
            rails_map,
            tip_xy=target,
            length=YELLOW_PROBE_LEN,
            thick=YELLOW_PROBE_THICK,
            draw=do_overlays,
            out_img=out if do_overlays else None
        )
        _p(pfx, f"[probe] tip={target} len={YELLOW_PROBE_LEN} thick={YELLOW_PROBE_THICK} rails_hit={probe_hit}")

        if probe_hit:
            _scan_persist_active = True
            _p(pfx, "[scan] persistence ON (reason=yellow-probe-rails)")
            scan_reason = "yellow-probe-rails"
        elif not SCAN_PERSIST:
            # If persistence is disabled, only scan on frames that probe hits
            _scan_persist_active = False
            _p(pfx, "[scan] persistence OFF (probe miss & SCAN_PERSIST=False)")


    # 4b) If persistence active, run the scan every frame
    if _scan_persist_active:
        scan_triggered = True
        if scan_reason is None:
            scan_reason = "persisted"

        sideways_action, scan_dbg = _scan_and_decide_move(
            lane_idx=lane_2,
            rays_for_lane=rays_for_lane,
            segments_by_ray=segments_by_ray,
            band_px=SCAN_BAND_PX,
            do_overlays=do_overlays,
            out_img=out if do_overlays else None
        )
        _p(pfx, f"[scan] y={SCAN_Y} band=±{SCAN_BAND_PX} reason={scan_reason} -> action={sideways_action} dbg={scan_dbg}")

        # --- EXECUTE MOVEMENT KEYS: lateral_TL only (namespaced cooldown) ---
        if scan_triggered and sideways_action in ("MOVE_LEFT", "MOVE_RIGHT"):
            on_ramp_now = SUPPRESS_LAT_ON_RAMP and _is_on_ramp_now(
                lane_idx=lane_2,
                rays_for_lane=rays_for_lane,
                segments_by_ray=segments_by_ray,
                scan_y=SCAN_Y,
                band_px=SCAN_BAND_PX
            )

            if on_ramp_now:
                _p(pfx, "[INPUT/lateral_TL] suppressed: ON RAMP at scanline")
            elif lat_move_active:
                key = "left" if sideways_action == "MOVE_LEFT" else "right"
                if _ltl_try_tap(key):
                    _p(pfx, f"[INPUT/lateral_TL] tapped {key}")
            else:
                _p(pfx, f"[INPUT/lateral_TL] suppressed: LAT_Move OFF (dy={dy_val}, thr={LAT_MOVE_Y_THRESH_PX})")



    else:
        _p(pfx, "[scan] not triggered")

    # (optional) augment the badge
    if do_overlays:
        badge = f"Target {source_note}  Δy={dy_val:d}px" if target is not None \
                else f"No purple triangle above anchor ({source_note})"
        if sideways_action:
            badge += f" | SCAN→{sideways_action}"
        _draw_badge(out, badge, x=10, y=34)

    if do_saves and save_path:
        t0 = time.perf_counter()
        cv2.imwrite(str(save_path), out)
        _pmaybe(pfx, "imwrite", _ms(t0))

    _p(pfx, f"[ok] top-logic done (n={n}) lane_2={lane_2}")
    _pmaybe(pfx, "TOTAL", _ms(t_total))
    _mem(pfx)

    return {
        "out_img": out,
        "target": target,
        "dy": dy_val,
        "source_note": source_note,
        "purple_tris": purple_tris,
        "segments_by_ray": segments_by_ray,
        # NEW:
        "scan_triggered": scan_triggered,
        "sideways_action": sideways_action,
        "scan_y": SCAN_Y,
        "scan_band_px": SCAN_BAND_PX,
        "lat_move_active": lat_move_active,
    }

