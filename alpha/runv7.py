import os
import time 
import pyautogui
import subprocess
from ring_grab import get_frame_bgr_from_ring 


# Crop + click (set by ad layout)
advertisement = True
if advertisement:
    snap_coords = (644, 77, (1149-644), (981-75))  # (left, top, width, height)
    start_click = (1030, 900)
else:
    snap_coords = (483, 75, (988-483), (981-75))
    start_click = (870, 895)

# =======================
# Parsec to front + click Start (non-blocking failures)
# =======================
try:
    subprocess.run(["osascript", "-e", 'tell application "Parsec" to activate'], check=False)
    time.sleep(0.4)
except Exception:
    pass

try:
    pyautogui.click(start_click)
except Exception:
    pass

# =======================
# Live loop
# =======================
running = True
frame_idx = 0

# =======================

while running:
    frame_start_time = time.perf_counter()

    frame_idx += 1
    print()
    print(f'===================================== Operating on frame {frame_idx} =====================================')

    # --- Screen grab ---
    t0_grab = time.perf_counter()
    left, top, width, height = snap_coords
   
   # NEW (ring)
    frame_bgr, meta = get_frame_bgr_from_ring(path="/tmp/scap.ring", wait_new=True, timeout_s=0.5)  # HxWx3, uint8, contiguous
    print('frame saved')
