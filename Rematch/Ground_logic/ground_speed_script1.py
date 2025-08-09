#!/usr/bin/env python3
# Headless, optimized single-frame pipeline with threaded prefetch I/O and tuned threading.
# Prints per-frame timing like:
# [141/144] frame_00140.png  read 17.3 | infer 22.1 | to_cpu 0.0 | post 0.0 | masks 0 | triangles 0 => proc 22.1 ms

# --------------------------
# Set threading env *before* importing numpy/opencv/torch
# --------------------------
import os
cpu_cnt = os.cpu_count() or 4
threads = max(1, cpu_cnt - 1)
os.environ.setdefault("OMP_NUM_THREADS", str(threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
os.environ.setdefault("MKL_NUM_THREADS", str(threads))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(threads))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # safer on Apple

import sys, time, math, glob, queue, threading
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# --------------------------
# Config
# --------------------------
home       = os.path.expanduser("~")
weights    = f"{home}/models/jakes-loped/jakes-finder-mk1/1/weights.pt"
frames_dir = Path(home) / "Documents" / "GitHub" / "Ai-plays-SubwaySurfers" / "frames"

RAIL_ID    = 9
IMG_SIZE   = 512
CONF, IOU  = 0.30, 0.45
MAX_DET    = 30

# rails “green” color region filter
TARGET_COLORS_RGB  = [(119,104,67), (81,42,45)]
TOLERANCE          = 20.0
MIN_REGION_SIZE    = 30
MIN_REGION_HEIGHT  = 150

# heat/triangles
HEAT_BLUR_KSIZE     = 51
RED_SCORE_THRESH    = 220
EXCLUDE_TOP_FRAC    = 0.40
EXCLUDE_BOTTOM_FRAC = 0.15
MIN_DARK_RED_AREA   = 1200
MIN_DARK_FRACTION   = 0.15

# Jake lane target bearings (degrees from vertical; left +, right - per our convention)
LANE_LEFT   = (300, 1340)
LANE_MID    = (490, 1340)
LANE_RIGHT  = (680, 1340)
JAKE_POINT  = LANE_RIGHT  # pick one
LANE_TARGET_DEG = {"left": -10.7, "mid": 1.5, "right": 15.0}

def lane_name_from_point(p):
    if p == LANE_LEFT: return "left"
    if p == LANE_MID:  return "mid"
    if p == LANE_RIGHT:return "right"
    return "mid"

# --------------------------
# Backends / Threading
# --------------------------
cv2.setUseOptimized(True)
try: cv2.setNumThreads(threads)
except Exception: pass

if torch.cuda.is_available():
    device = 0
    half = True
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
    half = False  # MPS half can be flaky
else:
    device = "cpu"
    half = False

try:
    torch.set_num_threads(threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
except Exception:
    pass

# --------------------------
# Model (persisted)
# --------------------------
model = YOLO(weights)
try: model.fuse()
except Exception: pass

# small warmup to stabilize kernels
_dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
for _ in range(2):
    _ = model.predict(_dummy, task="segment", imgsz=IMG_SIZE, device=device,
                      conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET)

# --------------------------
# Precomputed constants
# --------------------------
TARGETS_BGR_F32 = np.array([(r,g,b)[::-1] for (r,g,b) in TARGET_COLORS_RGB], dtype=np.float32)
TOL2            = TOLERANCE * TOLERANCE

# --------------------------
# Helpers
# --------------------------
def highlight_rails_mask_only_fast(img_bgr, rail_mask):
    H, W = img_bgr.shape[:2]
    if not rail_mask.any():
        return np.zeros((H, W), dtype=bool)

    ys, xs = np.where(rail_mask)
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1

    img_roi  = img_bgr[y0:y1, x0:x1]
    mask_roi = rail_mask[y0:y1, x0:x1]

    img_f = img_roi.astype(np.float32)
    diff  = img_f[:, :, None, :] - TARGETS_BGR_F32[None, None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    colour_hit = np.any(dist2 <= TOL2, axis=-1)

    comp = np.logical_and(colour_hit, mask_roi).astype(np.uint8)
    n, lbls, stats, _ = cv2.connectedComponentsWithStats(comp, 8)
    if n <= 1:
        return np.zeros((H, W), dtype=bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    hs    = stats[1:, cv2.CC_STAT_HEIGHT]
    keep  = np.where((areas >= MIN_REGION_SIZE) & (hs >= MIN_REGION_HEIGHT))[0] + 1

    good = np.zeros_like(comp, dtype=bool)
    for k in keep:
        good[lbls == k] = True

    full = np.zeros((H, W), dtype=bool)
    full[y0:y1, x0:x1] = good
    return full

def red_vs_green_score(red_mask, green_mask):
    # use uint8 for blur (faster), then convert to float for normalization
    k = (HEAT_BLUR_KSIZE, HEAT_BLUR_KSIZE)
    r = cv2.blur((red_mask.astype(np.uint8) * 255), k)
    g = cv2.blur((green_mask.astype(np.uint8) * 255), k)
    r = r.astype(np.float32); g = g.astype(np.float32)
    diff = r - g
    amax = float(np.max(np.abs(diff))) + 1e-6
    norm = (diff / (2.0 * amax) + 0.5) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)

def purple_triangles(score, H):
    top_ex = int(H * EXCLUDE_TOP_FRAC)
    bot_ex = int(H * EXCLUDE_BOTTOM_FRAC)

    dark = (score >= RED_SCORE_THRESH).astype(np.uint8)
    if top_ex: dark[:top_ex, :] = 0
    if bot_ex: dark[-bot_ex:, :] = 0

    dark = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9)),
        iterations=1
    )
    total_dark = int(dark.sum())
    if total_dark == 0:
        return [], None

    frac_thresh = int(np.ceil(MIN_DARK_FRACTION * total_dark))
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    if n_lbl <= 1:
        return [], None

    tris = []
    for lbl in range(1, n_lbl):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= MIN_DARK_RED_AREA and area >= frac_thresh:
            ys, xs = np.where(lbls == lbl)
            if ys.size == 0: continue
            y_top = ys.min()
            x_mid = int(xs[ys == y_top].mean())
            tris.append((int(x_mid), int(y_top)))

    if not tris:
        return [], None

    best = min(tris, key=lambda xy: xy[1])
    return tris, best

def signed_degrees_from_vertical(dx, dy):
    if dx == 0 and dy == 0:
        return 0.0
    return -math.degrees(math.atan2(dx, -dy))

def select_triangle_by_bearing(tri_positions, jx, jy, target_deg, min_dy=6):
    best_i, best_deg, best_err = -1, None, None
    for i, (xt, yt) in enumerate(tri_positions):
        dx = xt - jx
        dy = yt - jy
        if dy >= -min_dy:
            continue
        deg = signed_degrees_from_vertical(dx, dy)
        err = abs(deg - target_deg)
        if (best_err is None) or (err < best_err):
            best_i, best_deg, best_err = i, deg, err
    return best_i, best_deg, best_err

# --------------------------
# Post-processing (NO drawing)
# --------------------------
def process_frame_post(frame_bgr, yolo_res):
    H, W = frame_bgr.shape[:2]
    if yolo_res.masks is None:
        return 0, 0.0, 0.0

    t0_to_cpu = time.perf_counter()
    masks_np = yolo_res.masks.data.cpu().numpy()  # [n,h,w]
    mask_count = int(masks_np.shape[0])
    if mask_count == 0:
        to_cpu_ms = (time.perf_counter() - t0_to_cpu) * 1000.0
        return 0, to_cpu_ms, 0.0

    if hasattr(yolo_res.masks, "cls") and yolo_res.masks.cls is not None:
        classes_np = yolo_res.masks.cls.cpu().numpy().astype(int)
    else:
        classes_np = yolo_res.boxes.cls.cpu().numpy().astype(int)
    to_cpu_ms = (time.perf_counter() - t0_to_cpu) * 1000.0

    rail_sel = (classes_np == RAIL_ID)
    if not np.any(rail_sel):
        return 0, to_cpu_ms, 0.0

    t0_post = time.perf_counter()

    rail_masks = masks_np[rail_sel].astype(bool)        # [k,h,w]
    union = np.any(rail_masks, axis=0).astype(np.uint8) # [h,w]
    rail_mask = cv2.resize(union, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    green = highlight_rails_mask_only_fast(frame_bgr, rail_mask)
    red   = np.logical_and(rail_mask, np.logical_not(green))
    score = red_vs_green_score(red, green)
    tri_positions, _ = purple_triangles(score, H)

    # Bearing selection (kept; result not printed for speed)
    jx, jy = JAKE_POINT
    lane_name = lane_name_from_point(JAKE_POINT)
    target_deg = LANE_TARGET_DEG[lane_name]
    _best_idx, _best_deg, _ = select_triangle_by_bearing(tri_positions, jx, jy, target_deg, min_dy=6)

    post_ms = (time.perf_counter() - t0_post) * 1000.0
    return len(tri_positions), to_cpu_ms, post_ms

# --------------------------
# I/O Prefetcher
# --------------------------
def list_frames(frames_dir: Path):
    paths = (
        glob.glob(str(frames_dir/"frame_*.jpg")) +
        glob.glob(str(frames_dir/"frame_*.png")) +
        glob.glob(str(frames_dir/"*.jpg")) +
        glob.glob(str(frames_dir/"*.png"))
    )
    return sorted(set(paths))

def reader_worker(paths, out_q: queue.Queue):
    for p in paths:
        t0 = time.perf_counter()
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        dt = (time.perf_counter() - t0) * 1000.0
        if img is not None:
            out_q.put((p, img, dt))
        else:
            out_q.put((p, None, dt))
    out_q.put(None)  # sentinel

# --------------------------
# Main loop (single-frame inference, overlapped with prefetch)
# --------------------------
def run_pipeline_headless_single():
    paths = list_frames(frames_dir)
    if not paths:
        raise FileNotFoundError(f"No images in: {frames_dir}")

    # Optional: nicer priority (macOS may ignore without privileges)
    try: os.nice(-5)
    except Exception: pass

    q_in = queue.Queue(maxsize=8)  # small buffer keeps memory low
    t_reader = threading.Thread(target=reader_worker, args=(paths, q_in), daemon=True)
    t_reader.start()

    N = len(paths)
    idx = 0

    try:
        while True:
            item = q_in.get()
            if item is None:
                break
            p, img, read_ms = item
            idx += 1
            if img is None:
                print(f"[{idx}/{N}] {os.path.basename(p)}  read {read_ms:.1f} | infer 0.0 | to_cpu 0.0 | post 0.0 | masks 0 | triangles 0 => proc 0.0 ms")
                continue

            # Inference (single image)
            t0_inf = time.perf_counter()
            res_list = model.predict(
                img, task="segment", imgsz=IMG_SIZE, device=device,
                conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET
            )
            # sync for accurate time
            try:
                if device == 0 and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    torch.mps.synchronize()
            except Exception:
                pass
            infer_ms = (time.perf_counter() - t0_inf) * 1000.0

            # Post
            tri_count, to_cpu_ms, post_ms = process_frame_post(img, res_list[0])
            proc_ms = infer_ms + to_cpu_ms + post_ms

            print(f"[{idx}/{N}] {os.path.basename(p)}  "
                  f"read {read_ms:.1f} | infer {infer_ms:.1f} | "
                  f"to_cpu {to_cpu_ms:.1f} | post {post_ms:.1f} | "
                  f"masks {int(res_list[0].masks.data.shape[0]) if res_list[0].masks is not None else 0} | "
                  f"triangles {tri_count} "
                  f"=> proc {proc_ms:.1f} ms")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            while q_in.get_nowait() is not None:
                pass
        except Exception:
            pass

# --------------------------
# Entry
# --------------------------
if __name__ == "__main__":
    run_pipeline_headless_single()
