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
    pyautogui.press('up')




    

# ─── Clean up ──────────────────────────
listener.join()
print("Script halted.")






