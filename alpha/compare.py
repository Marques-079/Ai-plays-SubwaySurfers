from dataclasses import dataclass
import numpy as np
import time
from mss import mss
from ring_grab import get_frame_bgr_from_ring 

@dataclass(slots=True)
class OldProc:
    frame_bgr: np.ndarray   # HxWx3, uint8 (contiguous)
    grab_ms: float
    width: int
    height: int


def grab_frame_bgr_mss(sct, snap_coords, verbose=True) -> OldProc:
    """
    Grabs a BGRA frame via mss, prints the BGRA view shape (if verbose),
    returns a contiguous BGR copy and the timing in ms.
    """
    t0 = time.perf_counter()
    left, top, width, height = snap_coords
    raw = sct.grab({"left": left, "top": top, "width": width, "height": height})

    # BGRA view (zero-copy)
    buf = np.frombuffer(raw.raw, dtype=np.uint8).reshape(raw.height, raw.width, 4)
    if verbose:
        print(f"[view] shape: {buf.shape[1]}x{buf.shape[0]} px   (BGRA)")

    # BGR copy (matches np.array(raw)[:, :, :3] semantics)
    frame_bgr = buf[:, :, :3]

    grab_ms = (time.perf_counter() - t0) * 1000.0
    return OldProc(frame_bgr=frame_bgr, grab_ms=grab_ms, width=raw.width, height=raw.height)



sct = mss()
snap_coords = (644, 77, (1149-644), (981-75))
# --- Screen grab ---
old_proc = grab_frame_bgr_mss(sct, snap_coords, verbose=True)
frame_bgr_1 = old_proc.frame_bgr
grab_ms   = old_proc.grab_ms
#print(frame_bgr_1)
#print(grab_ms)
frame_bgr_2, meta = get_frame_bgr_from_ring(path="/tmp/scap.ring", wait_new=True, timeout_s=0.5)  # HxWx3, uint8, contiguous
print('For NEW PIPELINE')
print(f"[view] shape: {frame_bgr_2.shape[1]}x{frame_bgr_2.shape[0]} px   (BGR)   seq={meta['seq']}")


# if frame_bgr_1 == frame_bgr_2:
#     print("Frames match!")
# else:
#     print("Frames differ!")