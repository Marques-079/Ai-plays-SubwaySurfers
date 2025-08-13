#!/usr/bin/env python3
# Live capture + ultra-fast single-image pipeline (no batching)
# ESC to stop. Captures a cropped region, processes each frame immediately, prints per-frame timings.

import os, time, math, subprocess
import cv2, numpy as np
from mss import mss
import pyautogui
from pynput import keyboard
import torch
from ultralytics import YOLO
from pathlib import Path

# Lane state
lane = 1
MIN_LANE = 0
MAX_LANE = 2
running = True

def on_press(key):
    """Single handler: arrow keys change lanes; ESC stops."""
    global lane, running
    try:
        if key == keyboard.Key.left:
            lane = max(MIN_LANE, lane - 1)
            print(f"Moved Left → Lane {lane}")
        elif key == keyboard.Key.right:
            lane = min(MAX_LANE, lane + 1)
            print(f"Moved Right → Lane {lane}")
        elif key == keyboard.Key.esc:
            print("ESC pressed — exiting")
            running = False
            return False
    except Exception as e:
        print(f"Error: {e}")

# =======================
# Capture / UI bootstrap
# =======================
# Start the keyboard listener once
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Parsec to front (macOS)
try:
    subprocess.run(["osascript", "-e", 'tell application "Parsec" to activate'], check=False)
    time.sleep(0.4)
except Exception:
    pass

# Choose crop + click based on ad layout
advertisement = True
if advertisement:
    snap_coords = (644, 77, (1149-644), (981-75))  # (left, top, width, height)
    start_click = (1030, 900)
else:
    snap_coords = (483, 75, (988-483), (981-75))
    start_click = (870, 895)

# Click "Start"
try:
    pyautogui.click(start_click)
except Exception:
    pass

# Initialize capture
sct = mss()

# =======================
# Model / Pipeline config
# =======================
home       = os.path.expanduser("~")
weights    = f"{home}/models/jakes-loped/jakes-finder-mk1/1/weights.pt"

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
RAY_STEP_PX         = 20  # probe step

# Jake lane points + bearing targets
LANE_LEFT   = (300, 1340)
LANE_MID    = (490, 1340)
LANE_RIGHT  = (680, 1340)
LANE_POINTS = (LANE_LEFT, LANE_MID, LANE_RIGHT)  # index by lane (0/1/2)

LANE_TARGET_DEG = {"left": -10.7, "mid": +1.5, "right": +15.0}

# Bend degrees
BEND_LEFT_STATE_RIGHT_DEG  = -20.0  # N1
BEND_MID_STATE_RIGHT_DEG   = -20.0  # N2
BEND_MID_STATE_LEFT_DEG    = +20.0  # N3
BEND_RIGHT_STATE_LEFT_DEG  = +20.0  # N4

# =======================
# System / backends
# =======================
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(max(1, (os.cpu_count() or 1) - 1))
except Exception:
    pass

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
# Precomputed tables
# =======================
TARGETS_BGR_F32 = np.array([(r, g, b)[::-1] for (r, g, b) in TARGET_COLORS_RGB], dtype=np.float32)
TOL2            = TOLERANCE * TOLERANCE
MORPH_OPEN_SE   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))

DANGER_RED   = {1, 6, 7, 11}
WARN_YELLOW  = {2, 3, 4, 5, 8}
BOOTS_PINK   = {0}

def lane_name_from_point(p):
    if p == LANE_LEFT:  return "left"
    if p == LANE_MID:   return "mid"
    if p == LANE_RIGHT: return "right"
    return "mid"

def _clampi(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

# =======================
# Pipeline helpers
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
        table[a] = (math.sin(r), -math.cos(r))  # (dx, dy) unit ray (up = -y)
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

def classify_triangles_at_sample_curved(
    tri_positions, masks_np, classes_np, H, W,
    jake_point, x_ref, best_idx, sample_px=SAMPLE_UP_PX, step_px=RAY_STEP_PX
):
    if masks_np is None or classes_np is None or len(tri_positions) == 0:
        return []

    mh, mw = masks_np.shape[1], masks_np.shape[2]
    sx = (mw - 1) / max(1, (W - 1))
    sy = (mh - 1) / max(1, (H - 1))

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

            for i in red_idx:
                if masks_np[i][my, mx] > 0.5:
                    hit_colour = 3; break
            if hit_colour is not None: break
            for i in yellow_idx:
                if masks_np[i][my, mx] > 0.5:
                    hit_colour = 2; break
            if hit_colour is not None: break
            for i in boots_idx:
                if masks_np[i][my, mx] > 0.5:
                    hit_colour = 1; break
            if hit_colour is not None: break

        if hit_colour == 3: colours.append((0,0,255))
        elif hit_colour == 2: colours.append((0,255,255))
        elif hit_colour == 1: colours.append((203,192,255))
        else: colours.append((0,255,0))
    return colours

def process_frame_post(frame_bgr, yolo_res, jake_point):
    H, W = frame_bgr.shape[:2]
    if yolo_res.masks is None:
        return 0, 0, 0.0, 0.0

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

    green = highlight_rails_mask_only_fast(frame_bgr, rail_mask)
    red   = np.logical_and(rail_mask, np.logical_not(green))
    score = red_vs_green_score(red, green)
    tri_positions, _ = purple_triangles(score, H)

    # choose Jake triangle and set x_ref
    lane_name = lane_name_from_point(jake_point)
    target_deg = LANE_TARGET_DEG[lane_name]
    xj, yj = jake_point
    best_idx, _, _ = select_triangle_by_bearing(tri_positions, xj, yj, target_deg, min_dy=6)
    x_ref = tri_positions[best_idx][0] if (lane_name == "mid" and 0 <= best_idx < len(tri_positions)) else xj

    # run probe classification (not used further here, but triggers same work)
    _ = classify_triangles_at_sample_curved(
        tri_positions, masks_np, classes_np, H, W, jake_point, x_ref, best_idx,
        SAMPLE_UP_PX, RAY_STEP_PX
    )
    post_ms = (time.perf_counter() - t1) * 1000.0

    return len(tri_positions), mask_count, to_cpu_ms, post_ms

def process_image_bgr(img_bgr, name, jake_point):
    """Process one BGR frame already in memory and print timing line."""
    if img_bgr is None:
        return
    predict = model.predict  # local binding

    # In live mode there's no disk read; keep field for consistency
    read_ms = 0.0

    t0_inf = time.perf_counter()
    yres_list = predict(
        [img_bgr], task="segment", imgsz=IMG_SIZE, device=device,
        conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET, batch=1
    )
    infer_ms = (time.perf_counter() - t0_inf) * 1000.0

    yres = yres_list[0]
    tri_count, mask_count, to_cpu_ms, post_ms = process_frame_post(img_bgr, yres, jake_point)
    proc_ms = infer_ms + to_cpu_ms + post_ms

    print(f"[live] {name}  "
          f"read {read_ms:.1f} | infer {infer_ms:.1f} | "
          f"to_cpu {to_cpu_ms:.1f} | post {post_ms:.1f} | "
          f"masks {mask_count} | triangles {tri_count} "
          f"=> proc {proc_ms:.1f} ms")

# =======================
# Live loop
# =======================

# Output folder for saved frames
out_dir = Path.home() / "Documents" / "GitHub" / "Ai-plays-SubwaySurfers" / "live_run"
out_dir.mkdir(parents=True, exist_ok=True)

prev_ts = time.time()
frame_idx = 0

while running:
    # Grab screen region
    left, top, width, height = snap_coords
    raw = sct.grab({"left": left, "top": top, "width": width, "height": height})
    frame_bgr = np.array(raw)[:, :, :3]  # BGRA -> BGR

    # Determine JAKE_POINT for this frame from current lane (0/1/2)
    jake_point = LANE_POINTS[lane]

    # Process immediately (no batching, no saving)
    frame_idx += 1
    process_image_bgr(frame_bgr, name=f"frame_{frame_idx:05d}", jake_point=jake_point)

    # Save a copy with JAKE_POINT text at top-left (does not affect inference)
    annotated = frame_bgr.copy()
    jp_name = lane_name_from_point(jake_point).upper()
    cv2.putText(annotated, f"JAKE_POINT: {jp_name}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    out_path = out_dir / f"live_{frame_idx:05d}.jpg"
    cv2.imwrite(str(out_path), annotated)

    # (Optional) inter-frame delta print — comment out if noisy
    now = time.time()
    # print(f"Δ between frames: {now - prev_ts:.3f}s")
    prev_ts = now

# Cleanup
listener.join()
print("Script halted.")
