#!/usr/bin/env python3
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import math

# ----------------- CONFIG -----------------
HOME = os.path.expanduser("~")
WEIGHTS = f"{HOME}/models/jakes-loped/jakes-finder-mk1/1/weights.pt"

IN_DIR  = Path("frames")
OUT_DIR = Path("frames_plus2")   # <-- per your request
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 512
CONF, IOU = 0.30, 0.45
MAX_DET = 60

ALPHA = 0.60                # shading strength (0..1)
NEON_GREEN = (57, 255, 20)  # outline colour (BGR)
OUTLINE_THICKNESS = 2

# --------- Probe-line (Jake center) defaults ----------
# You can edit these 3 lines at runtime:
#   start: (x, y)
#   length: pixels upward along the ray
#   angle_deg: bearing from vertical (neg=CCW/left, pos=CW/right)
RAYS = [
    {"name": "LEFT",  "start": (0, 1600), "length": 1050, "angle_deg": +19.0},
    {"name": "MID",   "start": (490, 1800), "length": 1200, "angle_deg":  -1.5},
    {"name": "RIGHT", "start": (1010, 1600), "length": 1050, "angle_deg": -23.0},
]
RAY_STEP_PX = 2      # sampling stride along each line
PROBE_BAND  = 0      # 0 = single-pixel ray; >0 = +/- band in x for robustness
# Classes to ignore for "obstacle in lane" checks (default: rails only)
EXCLUDE_CLASSES = {9}   # {9}=RAILS
# -----------------------------------------------------

# BGR colours per class id (your map), with GREYTRAIN deepened
CLASS_COLOURS = {
    0:(255,255,0),    # BOOTS
    1:(96,96,96),     # GREYTRAIN (deeper grey)
    2:(0,128,255),    # HIGHBARRIER1
    3:(0,255,0),      # JUMP
    4:(255,0,255),    # LOWBARRIER1
    5:(0,255,255),    # LOWBARRIER2
    6:(255,128,0),    # ORANGETRAIN
    7:(128,0,255),    # PILLAR
    8:(0,0,128),      # RAMP
    9:(0,0,255),      # RAILS
    10:(128,128,0),   # SIDEWALK
    11:(255,255,102), # YELLOWTRAIN
}
LABELS = {
    0:"BOOTS",1:"GREYTRAIN",2:"HIGHBARRIER1",3:"JUMP",4:"LOWBARRIER1",
    5:"LOWBARRIER2",6:"ORANGETRAIN",7:"PILLAR",8:"RAMP",9:"RAILS",
    10:"SIDEWALK",11:"YELLOWTRAIN"
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
    """mask_f: float [0..1] (h,w) -> bool (H,W)"""
    m8 = (mask_f > 0.5).astype(np.uint8)
    if m8.shape != (H, W):
        m8 = cv2.resize(m8, (W, H), interpolation=cv2.INTER_NEAREST)
    return m8.astype(bool)

def _draw_neon_outline(img: np.ndarray, mask_bool: np.ndarray):
    """Draw neon-green outline around an instance mask."""
    if not mask_bool.any():
        return
    m8 = (mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(img, contours, -1, NEON_GREEN, OUTLINE_THICKNESS, lineType=cv2.LINE_AA)

def shade_and_outline(base_gray_bgr: np.ndarray,
                      masks_np: np.ndarray,
                      classes_np: np.ndarray) -> np.ndarray:
    """
    Start from grayscale; for each instance:
      - alpha-shade region with its class colour
      - draw neon-green outline around the mask
    Later instances overwrite earlier ones where they overlap.
    """
    H, W = base_gray_bgr.shape[:2]
    out = base_gray_bgr.copy()

    if masks_np is None or masks_np.size == 0:
        return out

    # Ensure lengths match (rare export mismatch guard)
    n = min(masks_np.shape[0], classes_np.shape[0])
    for m, c in zip(masks_np[:n], classes_np[:n]):
        cls_id = int(c)
        color = CLASS_COLOURS.get(cls_id, (255, 255, 255))
        mask = _resize_mask_to_frame(m, W, H)  # bool (H,W)

        if mask.any():
            # Shade
            region = out[mask].astype(np.float32)
            colvec = np.array(color, dtype=np.float32)
            blended = (1.0 - ALPHA) * region + ALPHA * colvec
            out[mask] = blended.astype(np.uint8)

            # Outline on top
            _draw_neon_outline(out, mask)

    return out

def _sx_sy(masks_np, H, W):
    mh, mw = masks_np.shape[1], masks_np.shape[2]
    sx = (mw - 1) / max(1, (W - 1))
    sy = (mh - 1) / max(1, (H - 1))
    return sx, sy

def _clampi(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def _first_hit_on_ray(masks_np, classes_np, H, W, start_xy, angle_deg, length_px,
                      step_px=2, exclude=EXCLUDE_CLASSES, band=PROBE_BAND):
    """
    March from start along the bearing; return (cls_id, (x,y), dist_px) for the first
    instance whose class ∉ exclude. If none, return (None, None, None).
    Angle measured from vertical: (dx,dy) = (sinθ, -cosθ).
    """
    if masks_np is None or masks_np.size == 0 or classes_np is None:
        return (None, None, None)

    sx, sy = _sx_sy(masks_np, H, W)
    # preselect indices to test
    test_idxs = [i for i, c in enumerate(classes_np) if int(c) not in exclude]
    if not test_idxs:
        return (None, None, None)

    x0, y0 = map(int, start_xy)
    x0 = _clampi(x0, 0, W-1); y0 = _clampi(y0, 0, H-1)

    r = math.radians(angle_deg)
    dx, dy = math.sin(r), -math.cos(r)   # 0° -> (0,-1) straight up

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
            # vote across a small horizontal band
            for dxb in range(-band, band+1):
                xsb = _clampi(xs + dxb, 0, W-1)
                mx  = _clampi(int(round(xsb * sx)), 0, masks_np.shape[2]-1)
                my  = _clampi(int(round(ys  * sy)), 0, masks_np.shape[1]-1)
                for i in test_idxs:
                    if masks_np[i][my, mx] > 0.5:
                        return (int(classes_np[i]), (xs, ys), float(t))
    return (None, None, None)

def _draw_labeled_line(img, start_xy, angle_deg, length_px, label_text,
                       color=(255,255,255), thick=2):
    x0, y0 = map(int, start_xy)
    r = math.radians(angle_deg)
    dx, dy = math.sin(r), -math.cos(r)
    x1 = int(round(x0 + dx * length_px))
    y1 = int(round(y0 + dy * length_px))
    H, W = img.shape[:2]
    x1 = _clampi(x1, 0, W-1); y1 = _clampi(y1, 0, H-1)
    cv2.line(img, (x0,y0), (x1,y1), color, thick, cv2.LINE_AA)
    # put label near the start
    if label_text:
        # simple text box
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x0+6, y0-8-th-6), (x0+6+tw+6, y0-8), (0,0,0), -1, cv2.LINE_AA)
        cv2.putText(img, label_text, (x0+10, y0-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

# ---------- main per-image ----------
def run_on_image(img_path: Path, save_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[skip] failed to read {img_path}")
        return

    H, W = img.shape[:2]
    base_gray = _to_gray_bgr(img)

    # Run segmentation
    res = model.predict([img], task="segment", imgsz=IMG_SIZE, device=device,
                        conf=CONF, iou=IOU, verbose=False, half=half,
                        max_det=MAX_DET, batch=1)[0]

    if res.masks is None or getattr(res.masks, "data", None) is None:
        out = base_gray
        # still draw lines with "none"
        for ray in RAYS:
            _draw_labeled_line(out, ray["start"], ray["angle_deg"], ray["length"],
                               f'{ray["name"]}: none')
        cv2.imwrite(str(save_path), out)
        print(f"[ok] {img_path.name}: no masks")
        return

    # --- extract masks + classes (classes only from boxes; fixes your error) ---
    masks_np = res.masks.data.detach().cpu().numpy()  # [n, h, w]
    if getattr(res, "boxes", None) is not None and getattr(res.boxes, "cls", None) is not None:
        classes_np = res.boxes.cls.detach().cpu().numpy().astype(int)
    else:
        classes_np = np.full((masks_np.shape[0],), -1, dtype=int)

    # Guard against weird count mismatch
    n = min(masks_np.shape[0], classes_np.shape[0])
    masks_np, classes_np = masks_np[:n], classes_np[:n]

    # Background greyscale + class shading + neon outlines
    out = shade_and_outline(base_gray, masks_np, classes_np)

    # For each user-defined ray, find first obstacle hit and draw
    for ray in RAYS:
        cls_id, hit_xy, dist_px = _first_hit_on_ray(
            masks_np, classes_np, H, W,
            start_xy=ray["start"],
            angle_deg=float(ray["angle_deg"]),
            length_px=float(ray["length"]),
            step_px=RAY_STEP_PX,
            exclude=EXCLUDE_CLASSES,
            band=PROBE_BAND,
        )
        if cls_id is None:
            text = f'{ray["name"]}: none'
        else:
            text = f'{ray["name"]}: {LABELS.get(cls_id, cls_id)} @ {dist_px:.0f}px'
            # mark hit point
            cv2.circle(out, hit_xy, 5, (0,0,0), -1, cv2.LINE_AA)
            cv2.circle(out, hit_xy, 3, (255,255,255), -1, cv2.LINE_AA)

        _draw_labeled_line(out, ray["start"], ray["angle_deg"], ray["length"], text)

    cv2.imwrite(str(save_path), out)
    print(f"[ok] {img_path.name} -> {save_path.name} (n={len(classes_np)})")

def main():
    if not IN_DIR.exists():
        print(f"[error] input folder not found: {IN_DIR.resolve()}")
        return
    for p in _iter_images(IN_DIR):
        run_on_image(p, OUT_DIR / p.name)

if __name__ == "__main__":
    main()
