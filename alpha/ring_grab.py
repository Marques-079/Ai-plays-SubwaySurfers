# ---- ring_grab.py (you can paste into your script, above main loop) ----
import os, mmap, struct, time
import numpy as np
import cv2
from functools import lru_cache

# Ring header formats (must match your producer)
HDR_FMT       = "<I I I I I I I I Q Q"   # magic,ver,slots,pixfmt,W,H,strideY,strideUV,slotBytes,headSeq
SLOT_HDR_FMT  = "<Q Q I I Q"             # seqStart,tNanos,ready,pad,seqEnd
HDR_SIZE      = struct.calcsize(HDR_FMT)
SLOT_HDR_SIZE = struct.calcsize(SLOT_HDR_FMT)

class RingGrabber:
    """
    Minimal ring reader that returns one BGR frame per .grab() call.
    - Returns: (frame_bgr, meta) where frame_bgr is HxWx3 uint8, contiguous.
    - meta: {'seq', 'W', 'H', 'pixfmt', 't_ns'}
    """
    def __init__(self, path="/tmp/scap.ring", sleep_idle=0.0003):
        self.path = path
        self.sleep_idle = sleep_idle
        fd = os.open(path, os.O_RDONLY)
        size = os.path.getsize(path)
        self.mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        os.close(fd)

        (magic, ver, slots, pixfmt, W, H,
         strideY, strideUV, slot_bytes, head_seq) = struct.unpack_from(HDR_FMT, self.mm, 0)

        assert magic == 0x534E5247 and ver == 1, "bad ring header"
        assert pixfmt in (0, 1), f"unsupported pixfmt {pixfmt} (0=NV12, 1=BGRA)"

        # basic stride checks (your producer writes tight rows)
        if pixfmt == 0:  # NV12
            assert strideY == W and strideUV == W, f"unexpected NV12 strides ({strideY},{strideUV})"
        else:            # BGRA
            assert strideY == W * 4, f"unexpected BGRA stride ({strideY} vs {W*4})"

        self.slots = slots
        self.pixfmt = pixfmt
        self.W, self.H = W, H
        self.strideY, self.strideUV = strideY, strideUV
        self.slot_bytes = slot_bytes
        self.last_seq_seen = -1  # for wait_new semantics

    def _slot_offsets(self, i):
        base = HDR_SIZE + i * (SLOT_HDR_SIZE + self.slot_bytes)
        return base, base + SLOT_HDR_SIZE

    def grab(self, wait_new=True, timeout_s=None, copy=True):
        """
        Get one BGR frame from the latest ready slot.
        - wait_new=True: waits for a newer seq than last call.
        - timeout_s: None for infinite wait, else seconds.
        - copy=True: return a C-contiguous copy (mss-like). Set False for zero-copy where possible.
        """
        deadline = None if timeout_s is None else (time.perf_counter() + timeout_s)
        while True:
            # read fresh head seq
            _, _, _, _, _, _, _, _, _, head_seq = struct.unpack_from(HDR_FMT, self.mm, 0)
            if (not wait_new) or (head_seq > self.last_seq_seen):
                slot_idx = int(head_seq % self.slots)
                off_hdr, off_data = self._slot_offsets(slot_idx)
                seq_start, t_ns, ready, _, seq_end = struct.unpack_from(SLOT_HDR_FMT, self.mm, off_hdr)

                if ready != 0 and seq_start == seq_end:
                    if self.pixfmt == 0:
                        # NV12: [Y plane HxW] + [UV plane (H/2)xW interleaved]
                        y  = np.ndarray((self.H,    self.W), dtype=np.uint8, buffer=self.mm, offset=off_data)
                        uv = np.ndarray((self.H//2, self.W), dtype=np.uint8, buffer=self.mm, offset=off_data + self.H*self.W)
                        yuv = np.vstack([y, uv])  # (H*3//2, W)
                        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)  # makes a new array
                    else:
                        # BGRA → drop A; this is a view into the mmap
                        bgra = np.ndarray((self.H, self.W, 4), dtype=np.uint8, buffer=self.mm, offset=off_data)
                        bgr  = bgra[..., :3]  # view

                    self.last_seq_seen = head_seq
                    if copy or (self.pixfmt == 0):
                        # NV12 path already created a new array; ascontiguousarray is cheap no-op if already contiguous
                        bgr = np.ascontiguousarray(bgr)

                    meta = {'seq': int(head_seq), 'W': self.W, 'H': self.H, 'pixfmt': self.pixfmt, 't_ns': int(t_ns)}
                    return bgr, meta

            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError("No new ready frame before timeout")
            time.sleep(self.sleep_idle)

    def close(self):
        try:
            self.mm.close()
        except Exception:
            pass

# Cache a single RingGrabber so simple function calls don't reopen the file
@lru_cache(maxsize=1)
def _get_cached_ring(path="/tmp/scap.ring", sleep_idle=0.0003):
    return RingGrabber(path, sleep_idle)

def get_frame_bgr_from_ring(path="/tmp/scap.ring", wait_new=True, timeout_s=0.5, copy=True):
    """
    One-call convenience: returns (frame_bgr, meta).
    Reuses a cached RingGrabber under the hood.
    """
    ring = _get_cached_ring(path)
    return ring.grab(wait_new=wait_new, timeout_s=timeout_s, copy=copy)

def close_ring_cache():
    try:
        ring = _get_cached_ring()
        ring.close()
    finally:
        _get_cached_ring.cache_clear()
