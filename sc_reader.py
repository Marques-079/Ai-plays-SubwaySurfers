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
    assert pixfmt in (0, 1), f"unsupported pixfmt {pixfmt} (0=NV12, 1=BGRA)"

    if pixfmt == 0:
        # NV12 (Y + interleaved UV), tightly packed to width
        assert strideY == W and strideUV == W, f"unexpected stride (got {strideY},{strideUV}, expected {W},{W})"
        print(f"Reading NV12 from {RING_PATH} ({W}x{H}), slots={slots}")
    else:
        # BGRA single-plane; strideY carries row-bytes (w*4) written by producer
        assert strideY == W * 4, f"unexpected BGRA stride (got {strideY}, expected {W*4})"
        print(f"Reading BGRA from {RING_PATH} ({W}x{H}), slots={slots}")

    if cv2 is None:
        raise RuntimeError("This test path requires OpenCV. Please `pip install opencv-python`.")

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

            # -----------------------------
            # Build BGR like mss() → numpy
            # -----------------------------
            if pixfmt == 0:
                # NV12 path: construct stacked NV12 view then convert to BGR
                y  = np.ndarray((H,    W), dtype=np.uint8, buffer=mm, offset=off_data)
                uv = np.ndarray((H//2, W), dtype=np.uint8, buffer=mm, offset=off_data + H*W)
                yuv = np.vstack([y, uv])  # (H*3//2, W)

                t0c = time.perf_counter()
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)  # (H,W,3), uint8
                convert_ms = (time.perf_counter() - t0c) * 1000.0

            else:
                # BGRA path: take zero-copy view and drop alpha
                bgra = np.ndarray((H, W, 4), dtype=np.uint8, buffer=mm, offset=off_data)
                # Mimic "mss -> np.array(shot)[:, :, :3]" (BGR) without copy
                t0c = time.perf_counter()
                bgr = bgra[..., :3]  # view, (H,W,3)
                convert_ms = (time.perf_counter() - t0c) * 1000.0  # ~0.00 ms

            # Optional preview of the true BGR frame
            if PREVIEW:
                cv2.imshow("SCStream ROI (BGR)", bgr)
                if cv2.waitKey(1) == 27:  # Esc
                    break

            # Turnaround: capture→now age using producer timestamp
            age_ms = (time.perf_counter_ns() - t_ns) / 1e6
            ages_ms.append(age_ms)
            last_seq = head_seq

            # Once per PRINT_EVERY seconds: print CAP fps + latency stats + conversion cost
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

                kind = "NV12->BGR (cv2)" if pixfmt == 0 else "BGRA view → BGR slice"
                print(f"[SCStream] CAP {cap_fps:6.1f} fps | "
                      f"TURNAROUND median {med:6.2f} ms  p95 {p95:6.2f} ms  max {worst:6.2f} ms   "
                      f"(ROI {W}x{H}) | [convert] {kind}: +{convert_ms:5.2f} ms")

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

