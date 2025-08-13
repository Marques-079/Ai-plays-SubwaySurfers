# capture_auto.py
# Auto-managed ScreenCaptureKit (fast) with fallback to MSS (portable).
# Returns BGR uint8 frames shaped (H, W, 3) so it can replace np.array(raw)[:,:,:3].

from __future__ import annotations
import os, sys, time, mmap, struct, shutil, subprocess, platform, threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2  # required for NV12->BGR
except Exception as e:
    cv2 = None

# ---------- Config ----------
RING_PATH_DEFAULT = "/tmp/scap.ring"
SCGRAB_BIN        = os.path.abspath("./scgrab")     # where we'll build/launch scgrab
SCGRAB_SRC        = os.path.abspath("./scgrab.swift")

HDR_FMT       = "<I I I I I I I I Q Q"  # magic,ver,slots,pixfmt,w,h,strY,strUV,slotSize,headSeq
SLOT_HDR_FMT  = "<Q Q I I Q"            # seqStart,tNanos,ready,pad,seqEnd
HDR_SIZE      = struct.calcsize(HDR_FMT)
SLOT_HDR_SIZE = struct.calcsize(SLOT_HDR_FMT)

MAGIC = 0x534E5247  # 'GRNS'
PIXFMT_NV12 = 0

# ---------- Exceptions ----------
class CaptureError(RuntimeError): ...
class BuildError(RuntimeError): ...

# ---------- Helpers ----------
def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def _read_header(mm: mmap.mmap):
    return struct.unpack_from(HDR_FMT, mm, 0)

def _slot_offsets(i: int, slot_bytes: int):
    base = HDR_SIZE + i * (SLOT_HDR_SIZE + slot_bytes)
    return base, base + SLOT_HDR_SIZE

# ---------- Fast SC (ScreenCaptureKit) reader ----------
class _SCRingReader:
    """Reads NV12 frames from a shared-memory ring and converts to BGR when asked."""
    def __init__(self, ring_path: str):
        self.ring_path = ring_path
        self.fd = os.open(ring_path, os.O_RDONLY)
        self.mm = mmap.mmap(self.fd, os.path.getsize(ring_path), access=mmap.ACCESS_READ)
        os.close(self.fd)

        (magic, ver, self.slots, self.pixfmt,
         self.W, self.H, self.strY, self.strUV,
         self.slot_bytes, self.head_seq) = _read_header(self.mm)
        assert magic == MAGIC and ver == 1, "bad ring header"
        assert self.pixfmt == PIXFMT_NV12, "expected NV12"
        # packed to width
        assert self.strY == self.W and self.strUV == self.W, f"unexpected stride {self.strY}/{self.strUV}"

        self._last_seq = -1
        self._last_ts = 0.0
        self._last_age = float("nan")
        self._last_bgr = None  # cache to avoid double conversion per seq

    def latest_with_age(self) -> Optional[Tuple[float, np.ndarray, float]]:
        # always jump to freshest written slot
        _,_,_,_,_,_,_,_,_, head_seq = _read_header(self.mm)
        if head_seq <= self._last_seq:
            return (self._last_ts, self._last_bgr, self._last_age) if self._last_bgr is not None else None

        i = int(head_seq % self.slots)
        off_hdr, off_data = _slot_offsets(i, self.slot_bytes)
        seq_start, t_ns, ready, _, seq_end = struct.unpack_from(SLOT_HDR_FMT, self.mm, off_hdr)
        if ready == 0 or seq_start != seq_end:
            # torn or not yet ready; return previous if any
            return (self._last_ts, self._last_bgr, self._last_age) if self._last_bgr is not None else None

        H, W = self.H, self.W
        # zero-copy NV12 view: Y then UV
        y  = np.ndarray((H,    W), dtype=np.uint8, buffer=self.mm, offset=off_data)
        uv = np.ndarray((H//2, W), dtype=np.uint8, buffer=self.mm, offset=off_data + H*W)
        yuv = np.vstack([y, uv])  # shape (H*3//2, W)

        if cv2 is None:
            raise CaptureError("OpenCV (opencv-python) required for NV12→BGR. pip install opencv-python")

        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)  # ~1–2 ms at ~2MP
        age_ms = (time.perf_counter_ns() - t_ns) / 1e6

        self._last_seq = head_seq
        self._last_ts = time.perf_counter()
        self._last_bgr = bgr
        self._last_age = age_ms
        return self._last_ts, bgr, age_ms

    def stop(self):
        try: self.mm.close()
        except Exception: pass

# ---------- Manage scgrab process ----------
@dataclass
class _SCManaged:
    left: int; top: int; width: int; height: int
    fps: int = 120
    out: str = RING_PATH_DEFAULT
    slots: int = 3
    scbin: str = SCGRAB_BIN
    scsrc: str = SCGRAB_SRC
    proc: Optional[subprocess.Popen] = None
    reader: Optional[_SCRingReader] = None

    def _compile_if_needed(self):
        if os.path.isfile(self.scbin):
            return
        if not os.path.isfile(self.scsrc):
            raise BuildError(f"scgrab binary not found and source missing at {self.scsrc}")
        swiftc = _which("swiftc")
        if not swiftc:
            raise BuildError("swiftc not found. Install Xcode command line tools.")
        cmd = [swiftc, "-O", "-parse-as-library",
               "-framework", "ScreenCaptureKit", "-framework", "AVFoundation",
               "-o", self.scbin, self.scsrc]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            raise BuildError("swiftc build failed:\n" + r.stderr.decode("utf-8", "ignore"))

    def _start_proc(self):
        self._compile_if_needed()
        # kill old ring if exists (fresh header)
        try: os.remove(self.out)
        except FileNotFoundError: pass

        cmd = [self.scbin, "--x", str(self.left), "--y", str(self.top),
               "--w", str(self.width), "--h", str(self.height),
               "--fps", str(self.fps), "--out", self.out, "--slots", str(self.slots)]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _wait_ring_ready(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(self.out) and os.path.getsize(self.out) > HDR_SIZE + SLOT_HDR_SIZE:
                # map and ensure headSeq advances
                try:
                    self.reader = _SCRingReader(self.out)
                    # wait for at least one frame
                    t1 = time.time()
                    last_head = _read_header(self.reader.mm)[-1]
                    while time.time() - t1 < timeout:
                        head = _read_header(self.reader.mm)[-1]
                        if head > last_head:
                            return
                        time.sleep(0.01)
                except Exception:
                    time.sleep(0.05)
            time.sleep(0.05)
        raise CaptureError("Timed out waiting for ScreenCaptureKit ring to become ready. "
                           "Check Screen Recording permission for your terminal/VSCode.")

    def start(self):
        self._start_proc()
        self._wait_ring_ready()

    def latest_with_age(self) -> Optional[Tuple[float, np.ndarray, float]]:
        if not self.reader:
            return None
        return self.reader.latest_with_age()

    def stop(self):
        try:
            if self.reader: self.reader.stop()
        finally:
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=1.0)
                except Exception:
                    try: self.proc.kill()
                    except Exception: pass

# ---------- MSS fallback ----------
class _MSSGrabber:
    def __init__(self, left:int, top:int, width:int, height:int):
        try:
            import mss  # lazy import
        except Exception as e:
            raise CaptureError("MSS not installed; pip install mss") from e
        self.sct = mss.mss()
        self.region = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}

    def latest_with_age(self) -> Optional[Tuple[float, np.ndarray, float]]:
        ts = time.perf_counter()
        import numpy as _np
        raw = self.sct.grab(self.region)
        frame_bgr = _np.array(raw)[:, :, :3]  # BGRA->BGR
        age_ms = (time.perf_counter() - ts) * 1000.0  # not exact source age, but consistent
        return ts, frame_bgr, age_ms

    def stop(self): 
        try: self.sct.close()
        except Exception: pass

# ---------- Public factory ----------
class Capture:
    """Unified capture that prefers fast SCStream, falls back to MSS. Returns BGR (H,W,3)."""
    def __init__(self, left:int, top:int, width:int, height:int,
                 fps:int=120, ring_path:str=RING_PATH_DEFAULT, slots:int=3,
                 prefer_sc:bool=True):
        self.backend = None
        self.kind = None
        if prefer_sc and platform.system() == "Darwin":
            try:
                self.backend = _SCManaged(left, top, width, height, fps=fps, out=ring_path, slots=slots)
                self.backend.start()
                self.kind = "scstream"
            except Exception as e:
                sys.stderr.write(f"[Capture] SCStream unavailable ({e}); falling back to MSS.\n")
        if self.backend is None:
            self.backend = _MSSGrabber(left, top, width, height)
            self.kind = "mss"

    def latest_with_age(self) -> Optional[Tuple[float, np.ndarray, float]]:
        return self.backend.latest_with_age()

    def stop(self):
        self.backend.stop()
