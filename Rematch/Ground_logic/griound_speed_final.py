#!/usr/bin/env python3
# Ultra-fast, lean pipeline with lane-aware curved sampling + ray-walk probes
# • Accepts a single image path (fast single-run) OR loops your frames dir for benchmarking
# • Prints per-frame timings; no overlay drawing or file I/O.

import os, sys, glob, time, math
import cv2, torch, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
import sys

all_proc_times: list[float] = []

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
THREADS_IO          = max(2, (os.cpu_count() or 4) // 2)
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
# Precomputed & class buckets
# =======================
TARGETS_BGR_F32 = np.array([(r,g,b)[::-1] for (r,g,b) in TARGET_COLORS_RGB], dtype=np.float32)
TOL2            = TOLERANCE * TOLERANCE

# Class buckets for probe classification (first-hit wins)
DANGER_RED   = {1, 6, 7, 11}
WARN_YELLOW  = {2, 3, 4, 5, 8}
BOOTS_PINK   = {0}

# ====== tiny helpers ======
def _clampi(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def load_image_with_time(path: str):
    t0 = time.perf_counter()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img, (time.perf_counter() - t0) * 1000.0

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

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
# Core runner (accepts explicit list of image paths)
# =======================
def run_fast(paths=None):
    predict = model.predict  # bind hot names to locals
    synchronize_cuda = torch.cuda.synchronize if torch.cuda.is_available() else None

    if paths is None:
        paths = (
            glob.glob(str(frames_dir/"frame_*.jpg")) +
            glob.glob(str(frames_dir/"frame_*.png")) +
            glob.glob(str(frames_dir/"*.jpg")) +
            glob.glob(str(frames_dir/"*.png"))
        )
    paths = sorted(set(p for p in (paths or []) if os.path.isfile(p)))
    if not paths:
        raise FileNotFoundError(f"No valid images. Checked explicit paths and: {frames_dir}")
    if SHOW_FIRST_N is not None:
        paths = paths[:SHOW_FIRST_N]

    N = len(paths)

    def load_batch(batch_paths):
        # Threaded I/O even for single image (overhead is tiny; keeps code uniform)
        imgs, read_ms = [None]*len(batch_paths), [0.0]*len(batch_paths)
        with ThreadPoolExecutor(max_workers=THREADS_IO) as ex:
            fut2i = {ex.submit(load_image_with_time, p): i for i, p in enumerate(batch_paths)}
            for fut in as_completed(fut2i):
                i = fut2i[fut]
                im, r = fut.result()
                imgs[i] = im; read_ms[i] = r
        ok = [(p, im, rm) for p, im, rm in zip(batch_paths, imgs, read_ms) if im is not None]
        if not ok: return [], [], []
        b_paths, b_imgs, b_read = zip(*ok)
        return list(b_paths), list(b_imgs), list(b_read)

    idx_global = 0
    for batch_paths in chunked(paths, 1):  # keep batch=1; fastest with current postproc
        batch_paths, imgs_bgr, read_ms_list = load_batch(batch_paths)
        if not imgs_bgr:
            idx_global += len(batch_paths); continue

        t0_inf = time.perf_counter()
        res_list = predict(
            imgs_bgr, task="segment", imgsz=IMG_SIZE, device=device,
            conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET, batch=1
        )
        try:
            if device == 0 and synchronize_cuda is not None:
                synchronize_cuda()
            elif device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.synchronize()
        except Exception:
            pass
        infer_ms_share = (time.perf_counter() - t0_inf) * 1000.0

        for j, (img, yres, read_ms) in enumerate(zip(imgs_bgr, res_list, read_ms_list)):
            tri_count, mask_count, to_cpu_ms, post_ms = process_frame_post(img, yres)

            proc_ms = infer_ms_share + to_cpu_ms + post_ms
            fname = os.path.basename(batch_paths[j])
            frame_idx = idx_global + j + 1

            all_proc_times.append(proc_ms)
            # Keep this single useful print:
            print(f"[{frame_idx}/{N}] {fname}  "
                  f"read {read_ms:.1f} | infer {infer_ms_share:.1f} | "
                  f"to_cpu {to_cpu_ms:.1f} | post {post_ms:.1f} | "
                  f"masks {mask_count} | triangles {tri_count} "
                  f"=> proc {proc_ms:.1f} ms")

        idx_global += 1

# =======================
# Convenience: single-image kick-off
# =======================
def run_on_image(image_path: str):
    """Kick the whole process off for a single image path (keeps the same prints)."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    run_fast(paths=[image_path])

# =======================
# Entry
# =======================
if __name__ == "__main__":
    # Usage:
    #   python script.py /path/to/image.jpg     -> runs on that one image
    #   python script.py                         -> runs on all frames in frames_dir (benchmark)
    if len(sys.argv) > 1:
        run_on_image(sys.argv[1])
    else:
        run_fast()  # iterate frames_dir for full-speed benchmarking


    if all_proc_times:
        # Ignore first 10 and last 10 frames
        if len(all_proc_times) > 20:
            proc_times = all_proc_times[10:-10]
        else:
            proc_times = all_proc_times[:]  # not enough frames to trim

        arr = np.array(proc_times)
        mean = arr.mean()
        std = arr.std(ddof=1)
        median = np.median(arr)
        p_min, p_max = arr.min(), arr.max()

        print("\n=== Frame Time Distribution (excluding first/last 10 frames) ===")
        print(f"Frames: {len(arr)}")
        print(f"Mean   : {mean:.2f} ms")
        print(f"Median : {median:.2f} ms")
        print(f"StdDev : {std:.2f} ms")
        print(f"Min    : {p_min:.2f} ms")
        print(f"Max    : {p_max:.2f} ms")

        # Optional ASCII histogram
        hist, bins = np.histogram(arr, bins=10)
        for h, b1, b2 in zip(hist, bins[:-1], bins[1:]):
            print(f"{b1:6.1f}–{b2:6.1f} ms | {'█' * h}")

