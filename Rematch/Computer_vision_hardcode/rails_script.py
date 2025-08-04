#!/usr/bin/env python3
import argparse, os, cv2, torch, numpy as np
from ultralytics import YOLO
from scipy.signal import find_peaks

# ---- CONFIG ----
home     = os.path.expanduser("~")
weights  = f"{home}/models/jakes-loped/jakes-finder-mk1/1/weights.pt"
RAIL_ID  = 9
IMG_SIZE = 512
CONF     = 0.30
IOU      = 0.45

# ---- DEVICE ----
if torch.cuda.is_available():
    device, half = 0, True
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device, half = "mps", False
else:
    device, half = "cpu", False

# ---- MODEL WARMUP ----
model = YOLO(weights)
try: model.fuse()
except: pass
_ = model.predict(
    np.zeros((IMG_SIZE,IMG_SIZE,3), np.uint8),
    task="segment", imgsz=IMG_SIZE,
    device=device, conf=CONF, iou=IOU,
    classes=[RAIL_ID], verbose=False, half=half
)

def get_rail_mask(img):
    res = model.predict(
        img, task="segment", imgsz=IMG_SIZE,
        device=device, conf=CONF, iou=IOU,
        classes=[RAIL_ID], max_det=20,
        verbose=False, half=half
    )[0]
    if res.masks is None:
        return np.zeros(img.shape[:2], dtype=bool)
    m = res.masks.data.sum(dim=0).cpu().numpy()
    uni = (m>0).astype(np.uint8)
    if uni.shape != img.shape[:2]:
        uni = cv2.resize(uni, (img.shape[1],img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    return uni.astype(bool)

def detect_lanes_projection(mask):
    colsum = mask.sum(axis=0)
    peaks, _ = find_peaks(
        colsum,
        distance=mask.shape[1]//8,
        prominence=0.1 * colsum.max()
    )
    return peaks

def visualize_peaks(img, peaks):
    vis = img.copy()
    H, _ = img.shape[:2]
    for x in peaks:
        cv2.line(vis, (int(x),0), (int(x),H-1), (0,255,0), 2)
    return vis

if __name__=="__main__":
    p = argparse.ArgumentParser(
        description="Rails → lane centers via projection & peak-finding"
    )
    p.add_argument("image_path", help="path to input screenshot")
    args, _ = p.parse_known_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"ERROR: could not load image '{args.image_path}'")
        exit(1)

    mask    = get_rail_mask(img)
    centers = detect_lanes_projection(mask)
    print("Detected rail X-centers:", centers)

    vis   = visualize_peaks(img, centers)
    combo = np.hstack((img, vis))
    cv2.imshow("Projection + Peak-Finding", combo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
