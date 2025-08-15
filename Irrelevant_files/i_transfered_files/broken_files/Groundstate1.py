import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
from IPython.display import display

def cycle(j, i):
    """
    Your original segmenter call; returns:
      - polygons: list of (N×2) float arrays of the mask outlines
      - class_ids: 1D int array, same length as polygons
      - names:     dict mapping class_id → class_name
    """
    home     = os.path.expanduser("~")
    weights  = f"{home}/models/jakes-loped/jakes-finder-mk1/1/weights.pt"
    img_path = f"{home}/SubwaySurfers/train_screenshots/frame_03{j}{i}.jpg"

    img = Image.open(img_path)
    img = img.resize((600, 400))
    display(img)

    model   = YOLO(weights)
    results = model.predict(
        source=img_path, task="segment", conf=0.30, iou=0.45
    )[0]

    polygons1   = results.masks.xy
    class_ids1  = results.boxes.cls.cpu().numpy().astype(int)
    names1      = results.names
    
    return polygons1, class_ids1, names1, img_path

def highlight_rails_mask_only(
    img_path: str,
    rail_mask: np.ndarray,
    target_colors_rgb: list,
    tolerance: float = 30.0,
    min_region_size: int = 50,
    min_region_height: int = 1000  # ⬅️ NEW height threshold
):
    # --- 1) Load original image ---
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f"[DEBUG] Loaded image of shape: {img.shape}")

    assert rail_mask.shape == (h, w), \
        f"rail_mask shape {rail_mask.shape} does not match image shape {(h, w)}"
    print(f"[DEBUG] rail_mask sum: {rail_mask.sum()}")

    plt.figure(); plt.imshow(rail_mask, cmap="gray")
    plt.title("Rail Mask"); plt.axis("off"); plt.show()

    # --- 2) Colour‐match inside the rail mask ---
    targets_bgr = [(c[2], c[1], c[0]) for c in target_colors_rgb]
    img_f       = img.astype(np.float32)
    color_mask  = np.zeros((h, w), dtype=bool)

    for i, tb in enumerate(targets_bgr):
        tb_arr = np.array(tb, dtype=np.float32).reshape((1, 1, 3))
        dist   = np.linalg.norm(img_f - tb_arr, axis=2)
        mask_this_color = dist <= tolerance
        color_mask |= mask_this_color
        print(f"[DEBUG] Color match {i}: BGR={tb}, matched pixels={mask_this_color.sum()}")

    plt.figure(); plt.imshow(color_mask, cmap="gray")
    plt.title("Color Match Mask"); plt.axis("off"); plt.show()

    # --- 3) Combine masks ---
    combined = rail_mask & color_mask
    raw_sum  = combined.sum()
    print(f"[DEBUG] Raw combined mask sum: {raw_sum}")

    # --- 4) Drop small and short neon‐green regions ---
    comp_uint8 = combined.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(comp_uint8, connectivity=8)
    filtered = np.zeros_like(combined)

    for lbl in range(1, n_labels):
        area   = stats[lbl, cv2.CC_STAT_AREA]
        height = stats[lbl, cv2.CC_STAT_HEIGHT]
        if area >= min_region_size and height >= min_region_height:
            filtered[labels == lbl] = True

    combined = filtered
    filt_sum = combined.sum()
    print(f"[DEBUG] Filtered combined mask sum (area ≥{min_region_size}, height ≥{min_region_height}): {filt_sum}")

    plt.figure(); plt.imshow(combined, cmap="gray")
    plt.title("Filtered Combined Mask"); plt.axis("off"); plt.show()

    # --- 5) Recolor matched pixels neon‐green in a copy ---
    result = img.copy()
    result[combined] = (0, 255, 0)
    print(f"[DEBUG] Recolored {combined.sum()} pixels to neon green")

    # --- 6) Show final result ---
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Final Result with Rails Highlighted")
    plt.axis("off"); plt.show()

    return result





# 1) Run your cycle() to get polygons, class IDs, and path
polygons, cls_ids, _, img_path = cycle(8, 4)

# 2) Filter for rail polygons
rail_polys = [p for p, c in zip(polygons, cls_ids) if c == 9]

# 3) Rasterize the first rail polygon into a boolean mask
img = cv2.imread(img_path)
rail_mask = polygon_to_mask(rail_polys[0], img.shape)  # now rail_mask is (H, W) bool array

highlight_rails_mask_only(
    img_path,
    rail_mask=rail_mask,
    target_colors_rgb=[(119,104,67),(81,42,45)],
    tolerance=20,
    min_region_size=50,
    min_region_height=150
)

