#!/usr/bin/env python3
# sc_grab10.py — grab 10 frames from /tmp/scap.ring as BGR (mss-compatible arrays)
# Usage: python3 sc_grab10.py [/tmp/scap.ring] [out_dir]
# Notes:
# - Works with pixfmt 0 (NV12) and 1 (BGRA) rings produced by your scgrab.swift.
# - Timing excludes disk I/O (saving). It measures ring read + conversion only.

import sys, os, time, mmap, struct, statistics
import numpy as np

try:
    import cv2
except Exception:
    raise SystemExit("This script requires OpenCV. Please: pip install opencv-python")

# ----------------- Config -----------------
RING_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scap.ring"
OUT_DIR   = sys.argv[2] if len(sys.argv) > 2 else "./sc_frames"
NUM_FRAMES = 10
SLEEP_IDLE = 0.0003  # light spin sleep when waiting on a fresh/ready frame

# ------------ Ring layout (must match producer) ------------
HDR_FMT       = "<I I I I I I I I Q Q"   # magic,ver,slots,pixfmt,w,h,strY,strUV,slotBytes,headSeq
SLOT_HDR_FMT  = "<Q Q I I Q"             # seqStart,tNanos,ready,pad,seqEnd
HDR_SIZE      = struct.calcsize(HDR_FMT)
SLOT_HDR_SIZE = struct.calcsize(SLOT_HDR_FMT)

def open_ring(path):
    fd = os.open(path, os.O_RDONLY)
    size = os.path.getsize(path)
    mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
    os.close(fd)
    return mm

def read_header(mm):
    return struct.unpack_from(HDR_FMT, mm, 0)

def slot_offsets(i, slot_bytes):
    base = HDR_SIZE + i * (SLOT_HDR_SIZE + slot_bytes)
    return base, base + SLOT_HDR_SIZE

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    mm = open_ring(RING_PATH)
    try:
        magic, ver, slots, pixfmt, W, H, strideY, strideUV, slot_bytes, head_seq = read_header(mm)
        assert magic == 0x534E5247 and ver == 1, "bad ring header"
        assert pixfmt in (0, 1), f"unsupported pixfmt {pixfmt} (0=NV12, 1=BGRA)"

        if pixfmt == 0:  # NV12
            assert strideY == W and strideUV == W, f"unexpected strideY/UV for NV12 ({strideY},{strideUV})"
            print(f"[info] Ring: NV12  {W}x{H}, slots={slots}, file={RING_PATH}")
        else:            # BGRA
            assert strideY == W * 4, f"unexpected stride for BGRA ({strideY} vs {W*4})"
            print(f"[info] Ring: BGRA  {W}x{H}, slots={slots}, file={RING_PATH}")

        grabbed = 0
        last_seq_seen = -1
        per_frame_ms = []

        print(f"[info] Grabbing {NUM_FRAMES} frames (timing excludes file saving)…")

        # ----------- Capture loop (read + convert only is timed) -----------
        while grabbed < NUM_FRAMES:
            # hop to freshest head
            _,_,_,_,_,_,_,_,_, head_seq = read_header(mm)
            if head_seq <= last_seq_seen:
                time.sleep(SLEEP_IDLE)
                continue

            slot_index = int(head_seq % slots)
            off_hdr, off_data = slot_offsets(slot_index, slot_bytes)
            seq_start, t_ns, ready, _, seq_end = struct.unpack_from(SLOT_HDR_FMT, mm, off_hdr)

            if ready == 0 or seq_start != seq_end:
                time.sleep(SLEEP_IDLE)
                continue

            t0 = time.perf_counter()

            # ---- Build BGR like mss() → numpy ----
            if pixfmt == 0:
                # NV12: stacked Y + interleaved UV, convert to BGR
                y  = np.ndarray((H,    W), dtype=np.uint8, buffer=mm, offset=off_data)
                uv = np.ndarray((H//2, W), dtype=np.uint8, buffer=mm, offset=off_data + H*W)
                yuv = np.vstack([y, uv])  # (H*3//2, W)
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)  # (H,W,3)
            else:
                # BGRA: single plane; drop alpha — mss-like BGR ndarray
                bgra = np.ndarray((H, W, 4), dtype=np.uint8, buffer=mm, offset=off_data)
                bgr = bgra[..., :3]  # view (zero copy)

            t1 = time.perf_counter()
            per_frame_ms.append((t1 - t0) * 1000.0)

            # Save AFTER timing (ensure contiguous for imwrite)
            bgr_to_save = np.ascontiguousarray(bgr)
            out_path = os.path.join(OUT_DIR, f"frame_{grabbed:02d}.png")
            cv2.imwrite(out_path, bgr_to_save)

            grabbed += 1
            last_seq_seen = head_seq

        # ----------- Report -----------
        med = statistics.median(per_frame_ms)
        p95 = sorted(per_frame_ms)[int(0.95 * (len(per_frame_ms) - 1))]
        total_ms = sum(per_frame_ms)
        print(f"[done] Captured {NUM_FRAMES} frames.")
        print(f"[timing] read+convert per-frame: median {med:.2f} ms   p95 {p95:.2f} ms   mean {total_ms/NUM_FRAMES:.2f} ms")
        print(f"[timing] total (read+convert only): {total_ms:.2f} ms")
        print(f"[out] Saved frames to: {OUT_DIR}")

    finally:
        try: mm.close()
        except Exception: pass

if __name__ == "__main__":
    time.sleep(5)
    main()
