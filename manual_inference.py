#!/usr/bin/env python3
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ----------------- CONFIG -----------------
HOME = os.path.expanduser("~")
WEIGHTS = f"{HOME}/models/jakes-loped/jakes-finder-mk1/1/weights.pt"

IN_DIR  = Path("frames")
OUT_DIR = Path("frames_plus")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 512
CONF, IOU = 0.30, 0.45
MAX_DET = 60

ALPHA = 0.60                # shading strength (0..1)
NEON_GREEN = (57, 255, 20)  # outline colour (BGR)
OUTLINE_THICKNESS = 2

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

    for m, c in zip(masks_np, classes_np):
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

def run_on_image(img_path: Path, save_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[skip] failed to read {img_path}")
        return

    base_gray = _to_gray_bgr(img)

    res_list = model.predict(
        [img], task="segment", imgsz=IMG_SIZE, device=device,
        conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET, batch=1
    )
    yres = res_list[0]

    if yres.masks is None:
        cv2.imwrite(str(save_path), base_gray)
        print(f"[ok] {img_path.name}: no masks -> greyscale only")
        return

    masks_np  = yres.masks.data.detach().cpu().numpy()  # [n, h, w]
    if hasattr(yres.masks, "cls") and yres.masks.cls is not None:
        classes_np = yres.masks.cls.detach().cpu().numpy().astype(int)
    else:
        classes_np = yres.boxes.cls.detach().cpu().numpy().astype(int)

    out = shade_and_outline(base_gray, masks_np, classes_np)
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
