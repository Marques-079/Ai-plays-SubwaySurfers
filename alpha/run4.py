import time
import numpy as np
from mss import mss
import pyautogui



import pyautogui

# Move instantly to coordinates (X=500, Y=300)
pyautogui.moveTo(1030, 900)


# Setup MSS
sct = mss()

print("Hover your mouse over a pixel to see its (B, G, R) value. Press Ctrl+C to stop.")

try:
    while True:
        # Get current mouse position
        x, y = pyautogui.position()

        # Grab 1x1 pixel at cursor position
        pixel = np.array(sct.grab({"left": x, "top": y, "width": 1, "height": 1}))

        # Extract BGRA
        b, g, r, a = pixel[0, 0]

        print(f"Mouse: ({x}, {y})  ->  (B, G, R) = ({b}, {g}, {r})")

        time.sleep(0.05)  # Slight delay to prevent spamming output

except KeyboardInterrupt:
    print("\nStopped.")
