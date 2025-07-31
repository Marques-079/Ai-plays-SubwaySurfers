import cv2
import numpy as np
import matplotlib.pyplot as plt

def build_3d_lut(img1, tol=0.10):

    cols = np.unique(img1.reshape(-1, 3), axis=0).astype(np.int16)
    lut  = np.zeros((256, 256, 256), dtype=bool)
    for r, g, b in cols:
        lo_r = max(0,   int(r * (1 - tol))); hi_r = min(255, int(r * (1 + tol)))
        lo_g = max(0,   int(g * (1 - tol))); hi_g = min(255, int(g * (1 + tol)))
        lo_b = max(0,   int(b * (1 - tol))); hi_b = min(255, int(b * (1 + tol)))
        lut[lo_r:hi_r+1, lo_g:hi_g+1, lo_b:hi_b+1] = True
    return lut

def highlight_with_lut_keep_largest(img2, lut, highlight=(0,255,0)):

    mask = lut[img2[...,0], img2[...,1], img2[...,2]].astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if n_labels <= 1:
        filtered = np.zeros_like(mask, dtype=bool)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = 1 + int(np.argmax(areas))
        filtered = (labels == max_idx)

    out = img2.copy()
    out[filtered] = highlight
    return out

def run_analysis():
    img1 = cv2.cvtColor(cv2.imread("rails1.png"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("test1.png"), cv2.COLOR_BGR2RGB)

    lut    = build_3d_lut(img1, tol=0.04) #This is very important - its a confidence interval for colour assigment
    result = highlight_with_lut_keep_largest(img2, lut, highlight=(0,255,0))

    plt.figure(figsize=(8,5))
    plt.imshow(result)
    plt.axis('off')
    plt.title("Only the Largest Matching Region Highlighted")
    plt.show()


