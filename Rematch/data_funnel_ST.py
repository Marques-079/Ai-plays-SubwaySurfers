#!/usr/bin/env python3
"""
Measure *disk write* latency by repeatedly writing the same JPEG bytes
(ROI-sized 505x906 by default) to your Transcend flash drive.

Default target path (from your screenshot): /Volumes/Transcend

Usage:
  python disk_write_latency_test.py
  python disk_write_latency_test.py --duration 3 --quality 85
  python disk_write_latency_test.py --same-file                # overwrite same file each time
  python disk_write_latency_test.py --drive /Volumes/Transcend --w 505 --h 906
  python disk_write_latency_test.py --usefile /path/to/sample.jpg  # use an existing image (cropped to w×h)
"""

import os, io, time, argparse, statistics
from datetime import datetime

DEFAULT_DRIVE = "/Volumes/Transcend"

def build_payload(w: int, h: int, quality: int, usefile: str | None) -> bytes:
    """Create (or load) a JPEG once, outside the timed loop."""
    from PIL import Image
    if usefile:
        im = Image.open(usefile).convert("RGB")
        # center-crop or pad to w×h
        iw, ih = im.size
        left = max(0, (iw - w)//2)
        top  = max(0, (ih - h)//2)
        im = im.crop((left, top, left + min(w, iw), top + min(h, ih)))
        im = im.resize((w, h), Image.BILINEAR)
    else:
        # Make a randomized image so size is realistic (not ultra-compressible).
        rnd = os.urandom(w*h*3)
        im = Image.frombytes("RGB", (w, h), rnd)
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=quality, subsampling=2, optimize=False)
    return b.getvalue()

def run(drive: str, duration: float, w: int, h: int, quality: int,
        same_file: bool, usefile: str | None):
    if not os.path.isdir(drive):
        raise SystemExit(f"[error] Drive not found: {drive}")

    # Build payload once (not timed)
    payload = build_payload(w, h, quality, usefile)
    print(f"[info] JPEG payload: {w}x{h}, quality={quality}, {len(payload)/1024:.1f} KiB")

    out_dir = os.path.join(drive, f"io_latency_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[info] Writing to: {out_dir}")

    timings_ms = []
    bytes_written = 0
    i = 0
    end = time.monotonic() + duration

    while time.monotonic() < end:
        fname = "sample.jpg" if same_file else f"frame_{i:06d}.jpg"
        path = os.path.join(out_dir, fname)

        t0 = time.perf_counter_ns()
        with open(path, "wb", buffering=0) as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())   # ensure it hits the device
        t1 = time.perf_counter_ns()

        timings_ms.append((t1 - t0) / 1e6)
        bytes_written += len(payload)
        i += 1

    # Summary
    n = len(timings_ms)
    elapsed = duration  # loop bounded by monotonic+duration
    fps = n / elapsed if elapsed > 0 else 0.0
    mb = bytes_written / (1024*1024)
    p50 = statistics.median(timings_ms) if timings_ms else 0.0
    p95 = statistics.quantiles(timings_ms, n=100)[94] if n >= 20 else max(timings_ms) if timings_ms else 0.0

    print("\n[summary]")
    print(f"  Files written : {n}  ({'same-file overwrite' if same_file else 'new file each time'})")
    print(f"  Directory     : {out_dir}")
    print(f"  Write latency : avg {sum(timings_ms)/n:.2f} ms | p50 {p50:.2f} ms | p95 {p95:.2f} ms")
    print(f"  Throughput    : {fps:.1f} FPS, {mb/elapsed:.1f} MiB/s, total {mb:.1f} MiB")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default=DEFAULT_DRIVE, help="Target mount (default: /Volumes/Transcend)")
    ap.add_argument("--duration", type=float, default=3.0, help="Seconds to run (default 3)")
    ap.add_argument("--x", type=int, default=644, help="(Unused for timing; for record only)")
    ap.add_argument("--y", type=int, default=77, help="(Unused for timing; for record only)")
    ap.add_argument("--w", type=int, default=505, help="ROI width (default 505)")
    ap.add_argument("--h", type=int, default=906, help="ROI height (default 906)")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality 1–95 (default 85)")
    ap.add_argument("--same-file", action="store_true",
                    help="Overwrite the same filename each iteration to remove file-create overhead")
    ap.add_argument("--usefile", help="Use an existing image file as the payload (cropped/resized to w×h)")
    args = ap.parse_args()
    run(args.drive, args.duration, args.w, args.h, args.quality, args.same_file, args.usefile)
