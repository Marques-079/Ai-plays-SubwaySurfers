import os, time, threading, queue
import cv2, numpy as np
from mss import mss
import pyautogui
from pynput import keyboard
import subprocess

'''
jake_classes = {
    1: "JAKE",
    2: "JAKE+BOOTS",
    3: "JETPACK"
}

obstacle_classes = {
    0: "BOOTS",
    1: "GREYTRAIN",
    2: "HIGHBARRIER1",
    3: "JUMP",
    4: "LOWBARRIER1",
    5: "LOWBARRIER2",
    6: "ORANGETRAIN",
    7: "PILLAR",
    8: "RAMP",
    9: "RAILS",
    10: "SIDEWALK",
    11: "YELLOWTRAIN"
}
'''
# ─── Shutdown listener ─────────────────
running = True
def on_press(key):
    global running
    if key == keyboard.Key.esc:
        running = False
        return False

listener = keyboard.Listener(on_press=on_press)
listener.start()

# ─── Async saver ───────────────────────
SAVE_DIR = "frames"
os.makedirs(SAVE_DIR, exist_ok=True)
save_q = queue.Queue()

def disk_saver():
    while True:
        item = save_q.get()
        if item is None:
            break
        path, frame = item
        cv2.imwrite(path, frame)
        save_q.task_done()

threading.Thread(target=disk_saver, daemon=True).start()

# ─── Fast capture helper ───────────────
sct = mss()
frame_counter = 0

def capture_frame(region, save=False):
    global frame_counter
    img = sct.grab({
      "left": region[0],
      "top":  region[1],
      "width":region[2],
      "height":region[3],
    })
    frame = np.array(img)[:, :, :3]   # BGRA → BGR
    if save:
        path = os.path.join(SAVE_DIR, f"frame_{frame_counter:05d}.png")
        save_q.put((path, frame))
        frame_counter += 1
    return frame

# ─── Main loop ─────────────────────────
frame_save = True
advertisement = True

# Bring Parsec forward
subprocess.run([
  "osascript", "-e",
  'tell application "Parsec" to activate'
])
time.sleep(0.5)                  

prev_ts = time.time()

if advertisement:
    snap_coords = (644, 77, (1149-644), (981-75))  # (left, top, width, height)
    start_click =(1030, 900)
else:
    snap_coords = (483, 75, (988-483), (981-75))
    start_click = (870, 895)  

pyautogui.click(start_click)   
# Main Logic loop here
while running:
    #pyautogui.press('up')

    #Cropped version (642,136,508,845)
    #time.sleep(0.2)
    frame = capture_frame(snap_coords, save=frame_save)

    now = time.time()
    print(f"Δ between frames: {now - prev_ts:.3f}s")
    prev_ts = now

# ─── Clean up ──────────────────────────
save_q.put(None)        
listener.join()
print("Script halted.")






