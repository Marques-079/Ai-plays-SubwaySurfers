#!/usr/bin/env python3
# Ultra-fast screen grab timing (no saving). Aim: <10 ms per frame on small ROI.

import time, numpy as np
from mss import mss

# Tight ROI around the game (left, top, width, height)
REGION = (644, 77, 640, 900)   # <- CHANGE THIS to your game window crop

def main():
    with mss() as sct:
        mon = {"left": REGION[0], "top": REGION[1], "width": REGION[2], "height": REGION[3]}
        # Warmup
        for _ in range(5): sct.grab(mon)

        count = 0
        t0 = time.perf_counter()
        prev = t0
        try:
            while True:
                t_start = time.perf_counter()
                shot = sct.grab(mon)             # BGRA bytes; **no conversion, no copy**
                # Optional: touch the buffer once so it’s actually realized
                _ = shot.size                    # or len(shot.bgra)
                t_end = time.perf_counter()

                dt_ms = (t_start - prev) * 1000.0
                grab_ms = (t_end - t_start) * 1000.0
                prev = t_start
                count += 1
                if count % 50 == 0:
                    elapsed = t_end - t0
                    fps = count / elapsed
                    print(f"[{count:6d}] Δ={dt_ms:5.1f} ms | grab={grab_ms:4.1f} ms | {fps:.1f} fps")
        except KeyboardInterrupt:
            elapsed = time.perf_counter() - t0
            print(f"\nStopped. {count} frames in {elapsed:.3f}s ({count/elapsed:.1f} fps).")

if __name__ == "__main__":
    main()
