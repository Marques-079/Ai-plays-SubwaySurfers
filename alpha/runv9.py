#!/usr/bin/env python3
import cv2
from pathlib import Path

# ---- base lane dots (screen coords) ----
LANE_LEFT  = (300, 1200)
LANE_MID   = (490, 1200)
LANE_RIGHT = (680, 1200)

# ---- add up to 6 more points here (use None to skip a slot) ----
EXTRA_DOTS = [
    (300, 1300),  # LEFT 2
    (490, 1300),  # MID 2
    (680, 1300),  # RIGHT 2
]

# Build the final list (skip Nones)
DOTS = [LANE_LEFT, LANE_MID, LANE_RIGHT] + [p for p in EXTRA_DOTS if p is not None]

# ---- I/O ----
INPUT_DIR  = Path("frames")      # put your input frames here (jpg/png…)
OUTPUT_DIR = Path("frames_out")  # overlaid frames will be written here
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File extensions to process (add more if needed)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---- drawing helpers ----
DOT_RADIUS = 10
FILLED     = -1
DOT_COLOR  = (57, 255, 20)  # neon green in BGR
OUTLINE    = (0, 0, 0)      # black outline for visibility
OUTLINE_TH = 3

DRAW_LABELS   = True
LABEL_COLOR_I = (0, 0, 0)      # black outline
LABEL_COLOR_O = (255, 255, 255)  # white fill
LABEL_SCALE   = 0.6
LABEL_THICK   = 2

def clamp_point(x, y, w, h):
    """Clamp a point into the image bounds."""
    x = max(0, min(w - 1, int(x)))
    y = max(0, min(h - 1, int(y)))
    return x, y

# ---- main loop ----
images = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in EXTS])

if not images:
    raise SystemExit(f"No images found in {INPUT_DIR.resolve()} with extensions {sorted(EXTS)}")

for i, path in enumerate(images, 1):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[skip] failed to read: {path.name}")
        continue

    h, w = img.shape[:2]

    # draw each dot (outline then fill for contrast)
    for idx, (x, y) in enumerate(DOTS, start=1):
        x, y = clamp_point(x, y, w, h)
        cv2.circle(img, (x, y), DOT_RADIUS + 2, OUTLINE, OUTLINE_TH, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), DOT_RADIUS, DOT_COLOR, FILLED, lineType=cv2.LINE_AA)

        # optional labels 1..N just above the dot
        if DRAW_LABELS:
            label = str(idx)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_THICK)
            pos = (x - tw // 2, max(0, y - DOT_RADIUS - 8))
            cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_COLOR_I, LABEL_THICK + 2, cv2.LINE_AA)
            cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_COLOR_O, LABEL_THICK, cv2.LINE_AA)

    out_path = OUTPUT_DIR / path.name
    cv2.imwrite(str(out_path), img)
    print(f"[{i:04d}/{len(images):04d}] saved -> {out_path}")

print("Done.")
