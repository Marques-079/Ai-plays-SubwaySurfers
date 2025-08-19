#!/usr/bin/env python3
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import math
import argparse
from typing import List, Dict, Tuple, Optional
import time

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

# Primary ray to use when selecting the triangle for a lane
PRIMARY_RAY_FOR_LANE = {0: "LEFT", 1: "MID", 2: "RIGHT"}

# ----------------- CONFIG -----------------
HOME = os.path.expanduser("~")
WEIGHTS = f"{HOME}/models/jakes-loped/jakes-finder-mk1/1/weights.pt"

IN_DIR  = Path("frames2")
OUT_DIR = Path("frames_plus2")
OUT_DIR.mkdir(parents=True, exist_ok =True)

IMG_SIZE = 512
CONF, IOU = 0.30, 0.45
MAX_DET = 60

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

# ------------------------------------------
# Device/precision
if torch.cuda.is_available():
    device, half = 0, True
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device, half = "mps", False
else:
    device, half = "cpu", False

# Load model
model = YOLO(WEIGHTS)
try: model.fuse()
except Exception: pass

# Warmup
_dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
_ = model.predict(_dummy, task="segment", imgsz=IMG_SIZE, device=device,
                  conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET)

# ---------- Helpers ----------
def _iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

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
    """Draw vector and Δy label between anchor and purple-triangle center in given color."""
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

    # small markers at endpoints (optional)
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

# === NEW: build per-pixel class maps for trains (fast) ========================
def build_class_maps(H:int, W:int, masks_np:np.ndarray, classes_np:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    class_map = np.full((H, W), -1, dtype=np.int32)
    train_map = np.zeros((H, W), dtype=bool)
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

# === draw helpers for intervals ==============================================
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

# ---------- main per-image ----------
def run_on_image(img_path: Path, save_path: Path, lane_2: int):
    frame_bgr_2 = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if frame_bgr_2 is None:
        print(f"[skip] failed to read {img_path}")
        return

    H, W = frame_bgr_2.shape[:2]
    base_gray = _to_gray_bgr(frame_bgr_2)

    res = model.predict([frame_bgr_2], task="segment", imgsz=IMG_SIZE, device=device,
                        conf=CONF, iou=IOU, verbose=False, half=half,
                        max_det=MAX_DET, batch=1)[0]

    if res.masks is None or getattr(res.masks, "data", None) is None:
        out = base_gray
        rays = LANE_RAYS.get(lane_2, [])
        _draw_equations_overlay(out, lane_2, rays)
        for ray in rays:
            _draw_labeled_line(out, ray["start"], ray["angle_deg"], ray["length"], f'{ray["name"]}: none')
        cv2.imwrite(str(save_path), out)
        print(f"[ok] {img_path.name}: no masks")
        return

    masks_np = res.masks.data.detach().cpu().numpy()
    if getattr(res, "boxes", None) is not None and getattr(res.boxes, "cls", None) is not None:
        classes_np = res.boxes.cls.detach().cpu().numpy().astype(int)
    else:
        classes_np = np.full((masks_np.shape[0],), -1, dtype=int)

    n = min(masks_np.shape[0], classes_np.shape[0])
    masks_np, classes_np = masks_np[:n], classes_np[:n]

    # shaded + neon for trains
    out = shade_and_outline_trains(base_gray, masks_np, classes_np)

    # === build per-pixel train maps once, reuse for all rays
    class_map, train_map = build_class_maps(H, W, masks_np, classes_np)

    rays = LANE_RAYS.get(lane_2, [])
    _draw_equations_overlay(out, lane_2, rays)

    # store purple triangle candidates with their ray and t of the purple endpoint
    purple_tris = []  # dicts: { "pt":(x,y), "t":float, "ray_name":str }

    for ray in rays:
        # draw full probe (white)
        _draw_labeled_line(out, ray["start"], ray["angle_deg"], ray["length"], f'{ray["name"]}')

        # optional: first-hit dot
        cls_id, hit_xy, dist_px = _first_hit_on_ray(
            masks_np, classes_np, H, W,
            start_xy=ray["start"],
            angle_deg=float(ray["angle_deg"]),
            length_px=float(ray["length"]),
            step_px=RAY_STEP_PX,
            exclude=EXCLUDE_CLASSES,
            band=PROBE_BAND,
        )
        if cls_id is not None:
            cv2.circle(out, hit_xy, 5, (0,0,0), -1, cv2.LINE_AA)
            cv2.circle(out, hit_xy, 3, (255,255,255), -1, cv2.LINE_AA)

        start_time = time.time()

        segments = ray_train_class_segments(
            start_xy=ray["start"],
            angle_deg=float(ray["angle_deg"]),
            length_px=float(ray["length"]),
            step_px=RAY_STEP_PX,
            class_map=class_map,
            train_map=train_map,
            band=PROBE_BAND,
        )

        segments = bridge_small_gaps(segments, gap_px=BRIDGE_GAP_PX)

        # draw merged segments + endpoint triangles
        rr = math.radians(float(ray["angle_deg"]))
        dx, dy = math.sin(rr), -math.cos(rr)
        for (t0, t1, cls_id_) in segments:
            col = CLASS_COLOURS.get(int(cls_id_), (255,255,255))
            _draw_interval_segment(out, ray["start"], ray["angle_deg"], t0, t1, col, thick=4)

            # label midpoint
            tm = 0.5*(t0+t1)
            xm = int(round(ray["start"][0] + dx*tm))
            ym = int(round(ray["start"][1] + dy*tm))
            label = f"{LABELS.get(int(cls_id_), int(cls_id_))} [{t0:.0f}→{t1:.0f}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (xm+6, ym-8-th-6), (xm+6+tw+6, ym-8), (0,0,0), -1, cv2.LINE_AA)
            cv2.putText(out, label, (xm+10, ym-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

            # endpoints
            p0 = _point_on_ray(ray["start"], ray["angle_deg"], t0)
            p1 = _point_on_ray(ray["start"], ray["angle_deg"], t1)

            # upper (smaller y) vs tail (larger y)
            if p0[1] < p1[1]:
                upper_pt, upper_t = p0, t0
                tail_pt,  tail_t  = p1, t1
            else:
                upper_pt, upper_t = p1, t1
                tail_pt,  tail_t  = p0, t0

            forward_dir  = (dx, dy)
            backward_dir = (-dx, -dy)
            forward_pt   = p1   # larger t
            backward_pt  = p0   # smaller t

            # blue at tail (lower end)
            tail_dir = forward_dir if tail_pt == forward_pt else backward_dir
            _draw_triangle_marker(out, tail_pt, tail_dir,
                                  size_px=TRI_SIZE_PX,
                                  fill_color=TRI_BLUE,
                                  outline_color=TRI_OUTLINE,
                                  outline_thick=TRI_OUT_THICK)

            # purple at upper (smaller y)
            upper_dir = forward_dir if upper_pt == forward_pt else backward_dir
            _draw_triangle_marker(out, upper_pt, upper_dir,
                                  size_px=TRI_SIZE_PX,
                                  fill_color=TRI_PURPLE,
                                  outline_color=TRI_OUTLINE,
                                  outline_thick=TRI_OUT_THICK)

            purple_tris.append({"pt": upper_pt, "t": float(upper_t), "ray_name": ray["name"]})

        end_time = time.time()
        print(f"[PROF] ray {ray['name']}: {end_time - start_time:.2f} s")

    # --- choose target purple triangle ---
    anchor_xy = LANE_ANCHORS.get(lane_2, LANE_MID)
    anchor_y  = anchor_xy[1]
    primary_ray = PRIMARY_RAY_FOR_LANE.get(lane_2, "MID")

    # 1) Prefer triangles from the lane's PRIMARY ray with y < anchor_y, choose FIRST along ray (min t)
    primary_candidates = [d for d in purple_tris
                          if d["ray_name"] == primary_ray and d["pt"][1] < anchor_y]
    target = None
    if primary_candidates:
        target = min(primary_candidates, key=lambda d: d["t"])["pt"]
        source_note = f"primary={primary_ray}"
    else:
        # 2) Fallback: triangles in the same lane (by x) with y < anchor_y; choose closest above (max y)
        lane_candidates = [d for d in purple_tris
                           if d["pt"][1] < anchor_y and _lane_from_x(d["pt"][0]) == lane_2]
        if lane_candidates:
            target = max(lane_candidates, key=lambda d: d["pt"][1])["pt"]
            source_note = "fallback=same-lane"
        else:
            # 3) Last resort: any triangle with y < anchor_y; choose closest above
            any_candidates = [d for d in purple_tris if d["pt"][1] < anchor_y]
            if any_candidates:
                target = max(any_candidates, key=lambda d: d["pt"][1])["pt"]
                source_note = "fallback=any"
            else:
                source_note = "none-found"

    if target is not None:
        _draw_anchor_vector(out, anchor_xy, target, color=(0,255,255))  # yellow
        dy_val = anchor_y - target[1]
        _draw_badge(out, f"Target {source_note}  Δy={dy_val:d}px", x=10, y=34)
    else:
        _draw_badge(out, f"No purple triangle above anchor ({source_note})", x=10, y=34)

    cv2.imwrite(str(save_path), out)
    print(f"[ok] {img_path.name} -> {save_path.name} (n={len(classes_np)}) lane_2={lane_2}")

def main():
    global RAY_STEP_PX, PROBE_BAND
    ap = argparse.ArgumentParser(description="Lane-aware overlay + train neon outlines + interval detection")
    ap.add_argument("--in_dir", type=str, default=str(IN_DIR), help="Input folder of images")
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR), help="Output folder for annotated images")
    ap.add_argument("--step", type=int, default=RAY_STEP_PX, help="Sampling step (px) along rays")
    ap.add_argument("--band", type=int, default=PROBE_BAND, help="Horizontal band half-width for intersection")
    ap.add_argument("--lane", type=int, choices=[0,1,2], default=0, help="Active lane/frame_type: 0=LEFT, 1=MID, 2=RIGHT")
    args = ap.parse_args()

    RAY_STEP_PX = max(1, int(args.step))
    PROBE_BAND  = max(0, int(args.band))

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[error] input folder not found: {in_dir.resolve()}")
        return

    lane_2 = 1

    for p in _iter_images(in_dir):
        save_path = out_dir / p.name
        print(save_path)
        run_on_image(p, save_path, lane_2)

if __name__ == "__main__":
    main()
