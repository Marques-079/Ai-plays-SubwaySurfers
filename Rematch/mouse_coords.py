import pyautogui
import time

#----------------------Mouse position tracker tool---------------------#
print("Press Ctrl+C to stop.\n")
try:
    while True:
        x, y = pyautogui.position()
        print(f"Mouse at ({x}, {y})", end="\r")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nDone.")


'''
(800, 720) left lane
890, 720) middle lane
980, 720    right lane

'''
