#!/usr/bin/env python3
# Ultra-fast, single-image pipeline (no batching) with lane-aware curved sampling + ray-walk probes
# Optimized for being called repeatedly (~5 Hz). Keeps model warm; no overlay drawing or file I/O.

import os, sys, glob, time, math
import cv2, torch, numpy as np
from pathlib import Path
from ultralytics import YOLO

# =======================
# Config
# =======================
home       = os.path.expanduser("~")
weights    = f"{home}/models/jakes-loped/jakes-finder-mk1/1/weights.pt"
frames_dir = Path(home) / "Documents" / "GitHub" / "Ai-plays-SubwaySurfers" / "frames"

RAIL_ID    = 9
IMG_SIZE   = 512
CONF, IOU  = 0.30, 0.45
MAX_DET    = 30

# Color/region filter for "green rails"
TARGET_COLORS_RGB  = [(119,104,67), (81,42,45)]
TOLERANCE          = 20.0
MIN_REGION_SIZE    = 30
MIN_REGION_HEIGHT  = 150

# Heat/triangle params
HEAT_BLUR_KSIZE     = 51
RED_SCORE_THRESH    = 220
EXCLUDE_TOP_FRAC    = 0.40
EXCLUDE_BOTTOM_FRAC = 0.15
MIN_DARK_RED_AREA   = 1200
MIN_DARK_FRACTION   = 0.15

# Sampling ray
SAMPLE_UP_PX        = 180
RAY_STEP_PX         = 20       # walk the probe every 20 px

# ===== Bend degrees (tune here) =====
BEND_LEFT_STATE_RIGHT_DEG  = -20.0  # N1
BEND_MID_STATE_RIGHT_DEG   = -20.0  # N2
BEND_MID_STATE_LEFT_DEG    = +20.0  # N3
BEND_RIGHT_STATE_LEFT_DEG  = +20.0  # N4

# Runtime
SHOW_FIRST_N        = None     # None → all frames

# =======================
# Jake lane points
# =======================
LANE_LEFT   = (300, 1340)
LANE_MID    = (490, 1340)
LANE_RIGHT  = (680, 1340)
JAKE_POINT  = LANE_RIGHT  # pick: LANE_LEFT / LANE_MID / LANE_RIGHT

LANE_TARGET_DEG = {"left": -10.7, "mid": +1.5, "right": +15.0}

def lane_name_from_point(p):
    if p == LANE_LEFT:  return "left"
    if p == LANE_MID:   return "mid"
    if p == LANE_RIGHT: return "right"
    return "mid"

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
# Model (singleton, warmed)
# =======================
model = YOLO(weights)
try: model.fuse()
except Exception: pass

_dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
_ = model.predict(_dummy, task="segment", imgsz=IMG_SIZE, device=device,
                  conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET)

# =======================
# Precomputed & class buckets
# =======================
TARGETS_BGR_F32 = np.array([(r,g,b)[::-1] for (r,g,b) in TARGET_COLORS_RGB], dtype=np.float32)
TOL2            = TOLERANCE * TOLERANCE
MORPH_OPEN_SE   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))  # precompute

DANGER_RED   = {1, 6, 7, 11}
WARN_YELLOW  = {2, 3, 4, 5, 8}
BOOTS_PINK   = {0}

# ====== tiny helpers ======
def _clampi(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

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

    # precomputed kernel
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, MORPH_OPEN_SE, iterations=1)
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

# --------- Walk-the-ray classifier (20px steps, first-hit wins) ----------
def classify_triangles_at_sample_curved(
    tri_positions, masks_np, classes_np, H, W,
    jake_point, x_ref, best_idx, sample_px=SAMPLE_UP_PX, step_px=RAY_STEP_PX
):
    if masks_np is None or classes_np is None or len(tri_positions) == 0:
        return []

    mh, mw = masks_np.shape[1], masks_np.shape[2]
    sx = (mw - 1) / max(1, (W - 1))
    sy = (mh - 1) / max(1, (H - 1))

    # Build index lists once per frame
    red_idx    = [i for i, c in enumerate(classes_np) if int(c) in DANGER_RED]
    yellow_idx = [i for i, c in enumerate(classes_np) if int(c) in WARN_YELLOW]
    boots_idx  = [i for i, c in enumerate(classes_np) if int(c) in BOOTS_PINK]

    colours = []
    max_k = max(1, sample_px // max(1, step_px))

    for idx, (x0, y0) in enumerate(tri_positions):
        theta = pick_bend_angle(jake_point, x0, x_ref, idx, best_idx)
        dx1, dy1 = TRIG_TABLE[theta]

        hit_colour = None
        for k in range(1, max_k + 1):
            t  = k * step_px
            xs = _clampi(int(round(x0 + dx1 * t)), 0, W-1)
            ys = _clampi(int(round(y0 + dy1 * t)), 0, H-1)
            mx = _clampi(int(round(xs * sx)), 0, mw-1)
            my = _clampi(int(round(ys * sy)), 0, mh-1)

            # RED first
            for i in red_idx:
                if masks_np[i][my, mx] > 0.5:
                    hit_colour = 3  # code for RED
                    break
            if hit_colour is not None: break
            # then YELLOW
            for i in yellow_idx:
                if masks_np[i][my, mx] > 0.5:
                    hit_colour = 2  # YELLOW
                    break
            if hit_colour is not None: break
            # then BOOTS (pink)
            for i in boots_idx:
                if masks_np[i][my, mx] > 0.5:
                    hit_colour = 1  # PINK
                    break
            if hit_colour is not None: break

        # map code to color (None → GREEN)
        if hit_colour == 3: colours.append((0,0,255))
        elif hit_colour == 2: colours.append((0,255,255))
        elif hit_colour == 1: colours.append((203,192,255))
        else: colours.append((0,255,0))

    return colours
# -----------------------------------------------------------------------

# =======================
# Frame post-processing
# =======================
def process_frame_post(frame_bgr, yolo_res):
    H, W = frame_bgr.shape[:2]
    if yolo_res.masks is None:
        return 0, 0, 0.0, 0.0  # triangles, masks, to_cpu_ms, post_ms

    t0 = time.perf_counter()
    masks_np = yolo_res.masks.data.detach().cpu().numpy()  # [n,h,w]
    if hasattr(yolo_res.masks, "cls") and yolo_res.masks.cls is not None:
        classes_np = yolo_res.masks.cls.detach().cpu().numpy().astype(int)
    else:
        classes_np = yolo_res.boxes.cls.detach().cpu().numpy().astype(int)
    to_cpu_ms = (time.perf_counter() - t0) * 1000.0
    mask_count = int(masks_np.shape[0])
    if mask_count == 0 or classes_np.size == 0:
        return 0, mask_count, to_cpu_ms, 0.0

    rail_sel = (classes_np == RAIL_ID)
    if not np.any(rail_sel):
        return 0, mask_count, to_cpu_ms, 0.0

    t1 = time.perf_counter()
    rail_masks = masks_np[rail_sel].astype(bool, copy=False)
    union = np.any(rail_masks, axis=0).astype(np.uint8, copy=False)
    rail_mask = cv2.resize(union, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool, copy=False)

    # rails: split into green vs red
    green = highlight_rails_mask_only_fast(frame_bgr, rail_mask)
    red   = np.logical_and(rail_mask, np.logical_not(green))
    score = red_vs_green_score(red, green)
    tri_positions, _ = purple_triangles(score, H)

    # choose Jake triangle and set x_ref
    lane_name = lane_name_from_point(JAKE_POINT)
    target_deg = LANE_TARGET_DEG[lane_name]
    xj, yj = JAKE_POINT
    best_idx, _, _ = select_triangle_by_bearing(tri_positions, xj, yj, target_deg, min_dy=6)
    x_ref = tri_positions[best_idx][0] if (lane_name == "mid" and 0 <= best_idx < len(tri_positions)) else xj

    # run probe classification (not used further here, but keeps same behavior)
    _ = classify_triangles_at_sample_curved(
        tri_positions, masks_np, classes_np, H, W, JAKE_POINT, x_ref, best_idx,
        SAMPLE_UP_PX, RAY_STEP_PX
    )
    post_ms = (time.perf_counter() - t1) * 1000.0

    return len(tri_positions), mask_count, to_cpu_ms, post_ms

# =======================
# Core single-image API (no batching)
# =======================
def process_image_bgr(img_bgr, name="frame"):
    """Fast path: process a BGR numpy image already in memory. Prints timing line."""
    if img_bgr is None:
        raise ValueError("img_bgr is None")

    predict = model.predict  # bind to local (slightly faster lookups)

    t0_read = time.perf_counter()
    # (No disk I/O here; keep consistent field for print)
    read_ms = (time.perf_counter() - t0_read) * 1000.0

    t0_inf = time.perf_counter()
    yres_list = predict(
        [img_bgr], task="segment", imgsz=IMG_SIZE, device=device,
        conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET, batch=1
    )
    # Optional sync only if you need very accurate timing; otherwise skip for speed.
    # try:
    #     if device == 0 and torch.cuda.is_available():
    #         torch.cuda.synchronize()
    #     elif device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    #         torch.mps.synchronize()
    # except Exception:
    #     pass
    infer_ms = (time.perf_counter() - t0_inf) * 1000.0

    yres = yres_list[0]
    tri_count, mask_count, to_cpu_ms, post_ms = process_frame_post(img_bgr, yres)
    proc_ms = infer_ms + to_cpu_ms + post_ms

    print(f"[1/1] {name}  "
          f"read {read_ms:.1f} | infer {infer_ms:.1f} | "
          f"to_cpu {to_cpu_ms:.1f} | post {post_ms:.1f} | "
          f"masks {mask_count} | triangles {tri_count} "
          f"=> proc {proc_ms:.1f} ms")

def process_image_path(image_path: str):
    """Convenience path: reads image from disk then calls process_image_bgr."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    t0 = time.perf_counter()
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    read_ms = (time.perf_counter() - t0) * 1000.0
    if img is None:
        raise ValueError(f"cv2.imread failed for {image_path}")

    # Run inference and post (print uses read_ms consistent with disk I/O)
    predict = model.predict

    t0_inf = time.perf_counter()
    yres_list = predict(
        [img], task="segment", imgsz=IMG_SIZE, device=device,
        conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET, batch=1
    )
    # (Skip device synchronize for speed; uncomment if you want exact timings.)
    # try:
    #     if device == 0 and torch.cuda.is_available():
    #         torch.cuda.synchronize()
    #     elif device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    #         torch.mps.synchronize()
    # except Exception:
    #     pass
    infer_ms = (time.perf_counter() - t0_inf) * 1000.0

    yres = yres_list[0]
    tri_count, mask_count, to_cpu_ms, post_ms = process_frame_post(img, yres)
    proc_ms = infer_ms + to_cpu_ms + post_ms

    fname = os.path.basename(image_path)
    print(f"[1/1] {fname}  "
          f"read {read_ms:.1f} | infer {infer_ms:.1f} | "
          f"to_cpu {to_cpu_ms:.1f} | post {post_ms:.1f} | "
          f"masks {mask_count} | triangles {tri_count} "
          f"=> proc {proc_ms:.1f} ms")

# =======================
# Entry / quick benchmark
# =======================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single-image kick-off
        process_image_path(sys.argv[1])
    else:
        # Quick speed run over frames_dir (no batching, 1-by-1)
        paths = (
            glob.glob(str(frames_dir/"frame_*.jpg")) +
            glob.glob(str(frames_dir/"frame_*.png")) +
            glob.glob(str(frames_dir/"*.jpg")) +
            glob.glob(str(frames_dir/"*.png"))
        )
        paths = sorted(set(p for p in paths if os.path.isfile(p)))
        if SHOW_FIRST_N is not None:
            paths = paths[:SHOW_FIRST_N]
        if not paths:
            raise FileNotFoundError(f"No images in: {frames_dir}")

        N = len(paths)
        for idx, p in enumerate(paths, 1):
            # Inline version of process_image_path but with index for consistent print format
            t0 = time.perf_counter()
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            read_ms = (time.perf_counter() - t0) * 1000.0
            if img is None:
                continue

            t0_inf = time.perf_counter()
            yres_list = model.predict(
                [img], task="segment", imgsz=IMG_SIZE, device=device,
                conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET, batch=1
            )
            infer_ms = (time.perf_counter() - t0_inf) * 1000.0

            yres = yres_list[0]
            tri_count, mask_count, to_cpu_ms, post_ms = process_frame_post(img, yres)
            proc_ms = infer_ms + to_cpu_ms + post_ms

            fname = os.path.basename(p)
            print(f"[{idx}/{N}] {fname}  "
                  f"read {read_ms:.1f} | infer {infer_ms:.1f} | "
                  f"to_cpu {to_cpu_ms:.1f} | post {post_ms:.1f} | "
                  f"masks {mask_count} | triangles {tri_count} "
                  f"=> proc {proc_ms:.1f} ms")
