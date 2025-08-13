#!/usr/bin/env python3
# sc_reader.py — ScreenCaptureKit shared-memory reader
# Reads NV12 frames from /tmp/scap.ring, prints CAP FPS + turnaround stats.
# Optional NV12→BGR conversion and preview.

import mmap, os, struct, time, statistics
import numpy as np
try:
    import cv2  # only needed if RETURN_BGR or PREVIEW is True
except Exception:
    cv2 = None

# ---- Settings ----
RING_PATH   = "/tmp/scap.ring"
RETURN_BGR  = False   # set True to get BGR frames (adds ~1–2 ms)
PREVIEW     = False   # set True to show a window (Esc to close)
PRINT_EVERY = 1.0     # seconds

# ---- Ring layout (must match producer) ----
HDR_FMT        = "<I I I I I I I I Q Q"   # magic,ver,slots,pixfmt,w,h,strY,strUV,slotSize,headSeq
SLOT_HDR_FMT   = "<Q Q I I Q"             # seqStart,tNanos,ready,pad,seqEnd
HDR_SIZE       = struct.calcsize(HDR_FMT)
SLOT_HDR_SIZE  = struct.calcsize(SLOT_HDR_FMT)

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
    mm = open_ring(RING_PATH)
    magic, ver, slots, pixfmt, W, H, strideY, strideUV, slot_bytes, head_seq = read_header(mm)
    assert magic == 0x534E5247 and ver == 1, "bad ring header"
    assert pixfmt == 0, "expected NV12 (pixfmt=0)"
    assert strideY == W and strideUV == W, f"unexpected stride (got {strideY},{strideUV}, expected {W},{W})"

    print(f"Reading NV12 from {RING_PATH} ({W}x{H}), slots={slots}")

    last_seq     = -1
    ages_ms      = []
    last_print   = time.perf_counter()

    # CAP FPS (producer rate) over the print window
    cap_last_seq = head_seq
    cap_last_t   = last_print

    try:
        while True:
            # Always jump to freshest frame
            _,_,_,_,_,_,_,_,_, head_seq = read_header(mm)
            if head_seq <= last_seq:
                time.sleep(0.0005)
                continue

            i = int(head_seq % slots)
            off_hdr, off_data = slot_offsets(i, slot_bytes)
            seq_start, t_ns, ready, _, seq_end = struct.unpack_from(SLOT_HDR_FMT, mm, off_hdr)
            if ready == 0 or seq_start != seq_end:
                # not ready or torn; spin lightly
                time.sleep(0.0002)
                continue

            # Zero-copy NV12 views (Y then UV), each tightly packed to W columns
            y  = np.ndarray((H,    W), dtype=np.uint8, buffer=mm, offset=off_data)
            uv = np.ndarray((H//2, W), dtype=np.uint8, buffer=mm, offset=off_data + H*W)
            yuv = np.vstack([y, uv])  # (H*3//2, W)

            if RETURN_BGR:
                if cv2 is None:
                    raise RuntimeError("RETURN_BGR=True requires `pip install opencv-python`.")
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                if PREVIEW:
                    cv2.imshow("SCStream ROI (BGR)", bgr)
                    if cv2.waitKey(1) == 27:  # Esc
                        break
            elif PREVIEW and cv2 is not None:
                # Cheap luma preview without conversion
                cv2.imshow("SCStream ROI (Y)", y)
                if cv2.waitKey(1) == 27:
                    break

            # Turnaround: capture→now age using producer timestamp
            age_ms = (time.perf_counter_ns() - t_ns) / 1e6
            ages_ms.append(age_ms)
            last_seq = head_seq

            # Once per PRINT_EVERY seconds: print CAP fps + latency stats
            now = time.perf_counter()
            if now - last_print >= PRINT_EVERY:
                cap_fps = (head_seq - cap_last_seq) / (now - cap_last_t) if now > cap_last_t else float("nan")
                cap_last_seq, cap_last_t = head_seq, now

                arr = sorted(ages_ms)
                if arr:
                    med  = statistics.median(arr)
                    p95  = arr[int(0.95 * (len(arr) - 1))]
                    worst = arr[-1]
                else:
                    med = p95 = worst = float('nan')

                print(f"[SCStream] CAP {cap_fps:6.1f} fps | "
                      f"TURNAROUND median {med:6.2f} ms  p95 {p95:6.2f} ms  max {worst:6.2f} ms   (ROI {W}x{H})")

                ages_ms.clear()
                last_print = now

    except KeyboardInterrupt:
        pass
    finally:
        if PREVIEW and cv2 is not None:
            try: cv2.destroyAllWindows()
            except Exception: pass
        try: mm.close()
        except Exception: pass

if __name__ == "__main__":
    main()


'''
After you activate the SCStream producer, run this script to read frames in a DIFFERENT terminal:
'''
#python3 sc_reader.py