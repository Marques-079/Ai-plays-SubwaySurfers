#!/usr/bin/env python3
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse
import time
import math

# ----------------- CONFIG -----------------
HOME = os.path.expanduser("~")
WEIGHTS = f"{HOME}/models/jakes-loped/jakes-finder-mk1/1/weights.pt"

IMG_SIZE = 512
CONF, IOU = 0.30, 0.45
MAX_DET = 60
MASK_THRESH = 0.5

# Lane anchors (updated as requested)
LANE_LEFT  = (300, 1440)
LANE_MID   = (490, 1440)
LANE_RIGHT = (680, 1440)
LANE_ANCHORS = {0: LANE_LEFT, 1: LANE_MID, 2: LANE_RIGHT}

# Class IDs
ID_GREYTRAIN   = 1
ID_ORANGETRAIN = 6
ID_RAMP        = 8
ID_RAILS       = 9
ID_YELLOWTRAIN = 11

TRAIN_TOPS = {ID_GREYTRAIN, ID_ORANGETRAIN, ID_YELLOWTRAIN}
INTEREST   = {ID_RAMP, ID_RAILS} | TRAIN_TOPS  # only check these masks

# ------------------------------------------
# Device/precision
if torch.cuda.is_available():
    device, half = 0, True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device, half = "mps", False
else:
    device, half = "cpu", False

# Load model
model = YOLO(WEIGHTS)
try:
    model.fuse()
except Exception:
    pass

# Warmup
_dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
_ = model.predict(_dummy, task="segment", imgsz=IMG_SIZE, device=device,
                  conf=CONF, iou=IOU, verbose=False, half=half, max_det=MAX_DET)

# ------------------------------------------
def _iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def _sx_sy(masks_np, H, W):
    mh, mw = masks_np.shape[1], masks_np.shape[2]
    # scale: frame(x,y) -> mask(mx,my)
    sx = (mw - 1) / max(1, (W - 1))
    sy = (mh - 1) / max(1, (H - 1))
    return sx, sy

def _clampi(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def _check_classes_in_window(
    masks_np: np.ndarray,
    classes_np: np.ndarray,
    H: int, W: int,
    center_xy: tuple[int,int],
    window_radius: int = 1
):
    """
    Returns dict flags: has_train, has_ramp, has_rails, and the set of matched class ids
    by scanning a (2r+1)x(2r+1) window around center_xy in FRAME coordinates.
    """
    if masks_np is None or masks_np.size == 0 or classes_np is None or classes_np.size == 0:
        return False, False, False, set()

    # Build list of mask indices that we care about
    interest_idx = [i for i, c in enumerate(classes_np) if int(c) in INTEREST]
    if not interest_idx:
        return False, False, False, set()

    sx, sy = _sx_sy(masks_np, H, W)
    cx, cy = center_xy
    cx = _clampi(cx, 0, W - 1)
    cy = _clampi(cy, 0, H - 1)

    seen = set()
    # Iterate tiny 3x3 window
    for dy in range(-window_radius, window_radius + 1):
        y = _clampi(cy + dy, 0, H - 1)
        my = _clampi(int(round(y * sy)), 0, masks_np.shape[1] - 1)
        for dx in range(-window_radius, window_radius + 1):
            x = _clampi(cx + dx, 0, W - 1)
            mx = _clampi(int(round(x * sx)), 0, masks_np.shape[2] - 1)

            # Check only interest masks and early-exit fast when possible
            for i in interest_idx:
                cls = int(classes_np[i])
                if masks_np[i, my, mx] > MASK_THRESH:
                    seen.add(cls)
                    # tiny window so no heavy early-exit needed
    has_train = any(c in TRAIN_TOPS for c in seen)
    has_ramp  = (ID_RAMP  in seen)
    has_rails = (ID_RAILS in seen)
    return has_train, has_ramp, has_rails, seen

class OnTopTracker:
    def __init__(self, start_on_top: bool = False):
        self.on_top = start_on_top
        self.ramp_streak = 0
        self.rails_streak = 0

    def update(self, has_train: bool, has_ramp: bool, has_rails: bool, airtime: bool):
        # Instant True if on any train top
        if has_train:
            self.on_top = True
            self.ramp_streak = 0
            self.rails_streak = 0
            return self.on_top

        # Ramp requires 2 consecutive frames -> True
        if has_ramp:
            self.ramp_streak += 1
            if self.ramp_streak >= 2:
                self.on_top = True
        else:
            self.ramp_streak = 0

        # Rails requires 2 consecutive frames (and no airtime) -> False
        if (not airtime) and has_rails:
            self.rails_streak += 1
            if self.rails_streak >= 2:
                self.on_top = False
        else:
            self.rails_streak = 0

        return self.on_top

def _draw_window_marker(img, center_xy, r=1, color=(255, 255, 255)):
    x, y = center_xy
    x0, y0 = x - r, y - r
    x1, y1 = x + r, y + r
    H, W = img.shape[:2]
    x0 = _clampi(x0, 0, W - 1); y0 = _clampi(y0, 0, H - 1)
    x1 = _clampi(x1, 0, W - 1); y1 = _clampi(y1, 0, H - 1)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

def _put_badge(img, text: str, x: int = 10, y: int = 28):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x, y - 18), (x + tw + 12, y + 6), (0, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(img, text, (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser(description="3x3 ON_TOP state tracker using YOLO seg masks")
    ap.add_argument("--lane", type=int, choices=[0, 1, 2], default=1,
                    help="Active lane: 0=LEFT, 1=MID, 2=RIGHT")
    ap.add_argument("--airtime", type=int, default=0,
                    help="0/1 flag for airtime (if 0, rails can force ON_TOP=False)")
    ap.add_argument("--source", type=str, choices=["folder", "ring"], default="folder",
                    help="Read frames from a folder or via ring grabber")
    ap.add_argument("--in_dir", type=str, default="frames2",
                    help="Folder of input images if source=folder")
    ap.add_argument("--out_dir", type=str, default="top_bot",
                    help="Output folder for annotated frames and state log")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max frames to process (0 = no limit)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "state_log.txt"

    # Initialize tracker
    tracker = OnTopTracker(start_on_top=False)

    # Frame source
    use_ring = (args.source == "ring")
    if use_ring:
        try:
            from ring_grab import get_frame_bgr_from_ring
        except Exception as e:
            print(f"[error] Unable to import ring_grab.get_frame_bgr_from_ring: {e}")
            return
        frame_iter = None   # pulled on demand
    else:
        in_dir = Path(args.in_dir)
        if not in_dir.exists():
            print(f"[error] input folder not found: {in_dir.resolve()}")
            return
        frame_iter = _iter_images(in_dir)

    # ----------------- TIMING ACCUMULATORS -----------------
    bucket_infer_sum = 0.0
    bucket_post_sum  = 0.0
    bucket_n         = 0

    total_infer_sum  = 0.0
    total_post_sum   = 0.0
    total_n          = 0
    # -------------------------------------------------------

    # Process loop
    frame_idx = 0
    t0 = time.time()

    while True:
        # Acquire frame
        if use_ring:
            frame_bgr = get_frame_bgr_from_ring()
            if frame_bgr is None:
                print("[warn] ring returned no frame; stopping.")
                break
            save_name = f"ring_{frame_idx:06d}.jpg"
        else:
            try:
                p = next(frame_iter)
            except StopIteration:
                break
            frame_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                print(f"[skip] failed to read {p}")
                continue
            save_name = p.name

        H, W = frame_bgr.shape[:2]

        # ---------- Inference timing ----------
        t_inf0 = time.perf_counter()
        res = model.predict([frame_bgr], task="segment", imgsz=IMG_SIZE, device=device,
                            conf=CONF, iou=IOU, verbose=False, half=half,
                            max_det=MAX_DET, batch=1)[0]
        t_inf1 = time.perf_counter()
        t_infer = t_inf1 - t_inf0
        # -------------------------------------

        # ---------- Postproc timing ----------
        t_pp0 = time.perf_counter()

        has_train = has_ramp = has_rails = False
        seen = set()

        if res.masks is not None and getattr(res.masks, "data", None) is not None \
           and getattr(res, "boxes", None) is not None and getattr(res.boxes, "cls", None) is not None:

            masks_np   = res.masks.data.detach().cpu().numpy()
            classes_np = res.boxes.cls.detach().cpu().numpy().astype(int)

            # Active lane anchor (center of 3x3 window)
            center_xy = LANE_ANCHORS.get(args.lane, LANE_MID)

            has_train, has_ramp, has_rails, seen = _check_classes_in_window(
                masks_np, classes_np, H, W, center_xy, window_radius=1
            )

        # Update state
        on_top = tracker.update(
            has_train=has_train,
            has_ramp=has_ramp,
            has_rails=has_rails,
            airtime=bool(args.airtime),
        )

        '''
        # Minimal annotate (exclude disk I/O from timing)
        out = frame_bgr.copy()
        _draw_window_marker(out, LANE_ANCHORS.get(args.lane, LANE_MID), r=1, color=(255,255,255))
        state_txt = f"ON_TOP={int(on_top)}  seen={sorted(list(seen))}"
        _put_badge(out, state_txt, x=10, y=28)
        '''

        t_pp1 = time.perf_counter()
        t_post = t_pp1 - t_pp0
        # -------------------------------------

        # ---------- Update timing buckets ----------
        bucket_infer_sum += t_infer
        bucket_post_sum  += t_post
        bucket_n         += 1

        total_infer_sum  += t_infer
        total_post_sum   += t_post
        total_n          += 1

        # Print every 10 frames
        if bucket_n == 1:
            start_idx = frame_idx - (bucket_n - 1)
            end_idx   = frame_idx
            avg_inf_ms = (bucket_infer_sum / bucket_n) * 1000.0
            avg_pp_ms  = (bucket_post_sum  / bucket_n) * 1000.0
            print(f"[timing] frames {start_idx:06d}..{end_idx:06d}  "
                  f"infer={avg_inf_ms:.2f} ms/frame  postproc={avg_pp_ms:.2f} ms/frame")
            bucket_infer_sum = bucket_post_sum = 0.0
            bucket_n = 0
        # -------------------------------------------

        # Save outputs (excluded from timing)
        #cv2.imwrite(str(out_dir / save_name), out)

        # Append to log (excluded from timing)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{frame_idx:06d},ON_TOP={int(on_top)},seen={sorted(list(seen))}\n")

        frame_idx += 1
        if args.limit and frame_idx >= args.limit:
            break

    # Print trailing partial bucket, if any
    if bucket_n > 0:
        start_idx = frame_idx - bucket_n
        end_idx   = frame_idx - 1
        avg_inf_ms = (bucket_infer_sum / bucket_n) * 1000.0
        avg_pp_ms  = (bucket_post_sum  / bucket_n) * 1000.0
        print(f"[timing] frames {start_idx:06d}..{end_idx:06d}  "
              f"infer={avg_inf_ms:.2f} ms/frame  postproc={avg_pp_ms:.2f} ms/frame")

    dt = time.time() - t0
    if frame_idx > 0:
        print(f"[done] {frame_idx} frames in {dt:.2f}s  ({frame_idx/dt:.2f} FPS)")
        print(f"[out] frames -> {out_dir}/,  log -> {log_path}")
        # Optional: overall averages
        avg_inf_ms = (total_infer_sum / total_n) * 1000.0
        avg_pp_ms  = (total_post_sum  / total_n) * 1000.0
        print(f"[overall] infer={avg_inf_ms:.2f} ms/frame  postproc={avg_pp_ms:.2f} ms/frame")
    else:
        print("[done] no frames processed.")

if __name__ == "__main__":
    main()
