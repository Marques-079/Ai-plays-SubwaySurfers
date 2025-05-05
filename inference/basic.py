import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from ultralytics import YOLO

home    = os.path.expanduser("~")
weights = f"{home}/models/jakes-loped/jakes-finder-mk1/1/weights.pt"

# ─── 0) MODEL INSTANTIATION & WARM-UP ───────────────────────────────────────────
model = YOLO(weights)
# warm-up on one sample so kernels are JIT-compiled
sample_path = f"{home}/SubwaySurfers/train_screenshots/frame_0200.jpg"
_ = model.predict(source=sample_path, task="segment", conf=0.30, iou=0.45)
# ────────────────────────────────────────────────────────────────────────────────

def cycle(j, i):
    img_path = f"{home}/SubwaySurfers/train_screenshots/frame_02{j}{i}.jpg"
    results  = model.predict(source=img_path, task="segment", conf=0.30, iou=0.45)[0]

    # get masks, classes, boxes & confidences
    polygons    = results.masks.xy
    class_ids   = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()
    boxes       = results.boxes.xyxy.cpu().numpy().astype(int)
    names       = results.names

    # build gray background
    img_color = cv2.imread(img_path)
    img_gray  = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    bg_gray3  = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # assign colors per class
    unique_ids    = sorted(set(class_ids))
    custom_colors = {1:(255,128,0), 6:(0,255,0), 9:(0,0,255), 11:(255,100,100)}
    color_map     = {cid: custom_colors.get(cid,(255,255,0)) for cid in unique_ids}

    # render masks
    overlay = bg_gray3.copy()
    for poly, cid in zip(polygons, class_ids):
        pts = poly.astype(np.int32).reshape(-1,1,2)
        cv2.fillPoly(overlay, [pts], color_map[cid])
    blended = cv2.addWeighted(overlay, 0.4, bg_gray3, 0.6, 0)



#######CONFIDENCE INTERVAL DISPLAY ##################################################
    # draw each confidence just above its mask's bbox
    for conf, box in zip(confidences, boxes):
        x1, y1 = box[0], box[1]
        label  = f"{conf:.2f}"
        cv2.putText(
            blended, label, (int(x1), int(y1)-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA
        )
#################################################################
    # display


    fig, ax = plt.subplots(figsize=(12,8))
    ax.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    ax.axis(False)

    # legend
    patches = [Patch(color=np.array(color_map[cid])/255.0, label=names[cid])
               for cid in unique_ids]
    fig.legend(handles=patches, loc="center right", bbox_to_anchor=(1.2,0.5))
    plt.tight_layout()
    plt.show()


# now this loop will be much faster AND annotate confidences
for i in range(1):
    for j in range(1):
        cycle(j, i)

'''



'''

#First call may be very slow due to positional embeddings rezising, nn.interpolate will run on the CPU
#Sfter first run (of cycle()) it can switch to MPS for low latency 
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from rfdetr import RFDETRBase
from PIL import Image
import supervision as sv
import time
import torch

weights_path = os.path.expanduser("~/downloads/weightsjake.pt")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = RFDETRBase(pretrain_weights=weights_path, num_classes=3)

def cycle2(j, i):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    image_path = os.path.expanduser(f"~/SubwaySurfers/train_screenshots/frame_00{j}{i}.jpg")
    image = Image.open(image_path)#.resize((512, 288))  # optional downscale <------ OPTIONAL DOWNSCALE FOR PERFORMANCE
    

    start = time.time()
    detections = model.predict(image, threshold=0.5, device=device)
    end = time.time()

    annot = sv.BoxAnnotator().annotate(image, detections)
    labels = [f"{cid} {conf:.2f}" for cid, conf in zip(detections.class_id, detections.confidence)]
    annot = sv.LabelAnnotator().annotate(annot, detections, labels)
    sv.plot_image(annot)

    print(f"Inference time: {(end - start) * 1000:.2f} ms")

#Test for ranges
for i in range(10):
    for j in range(9):
        cycle2(j, i)

'''
Basic inference for Jake finder
'''

import os
from rfdetr import RFDETRBase
from PIL import Image
import supervision as sv
import time
import torch


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Using device: {device}")

# point 'weightsjake.pt' at your locally-saved checkpoint
weights_path = os.path.expanduser("~/downloads/weightsjake.pt")

# instantiate with your custom weights
model = RFDETRBase(
  pretrain_weights=weights_path,
  num_classes=3,            # ← match your checkpoint
)

def cycle2(j,i):
    start = time.time()
    # now do your usual inference…
    image = Image.open(os.path.expanduser(f"~/SubwaySurfers/train_screenshots/frame_00{j}{i}.jpg"))
    detections = model.predict(image, threshold=0.5)
    
    # visualize
    annot = image.copy()
    annot = sv.BoxAnnotator().annotate(annot, detections)
    annot = sv.LabelAnnotator().annotate(annot, detections,
                                         [f"{cid} {conf:.2f}" for cid, conf in zip(detections.class_id, detections.confidence)])
    sv.plot_image(annot)
    end = time.time()
    
    print(f"Inference time: {(end - start) * 1000:.2f} ms")


#Test for ranges
for i in range(1):
    for j in range(1):
        cycle2(j, i)

