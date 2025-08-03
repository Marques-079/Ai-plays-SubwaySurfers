import os, time, threading, queue
import cv2, numpy as np
from mss import mss
import pyautogui
from pynput import keyboard
import subprocess

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

# Bring Parsec forward
subprocess.run([
  "osascript", "-e",
  'tell application "Parsec" to activate'
])
time.sleep(0.5)                  

pyautogui.click(1030, 890)   

prev_ts = time.time()
frame_save = True  

while running:
    pyautogui.press('up')

    frame = capture_frame((642,136,508,845), save=frame_save)

    now = time.time()
    print(f"Δ between frames: {now - prev_ts:.3f}s")
    prev_ts = now

# ─── Clean up ──────────────────────────
save_q.put(None)        
listener.join()
print("Script halted.")






