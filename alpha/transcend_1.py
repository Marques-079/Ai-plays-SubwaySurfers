#!/usr/bin/env python3
# arrow_save_to_transcend.py — start/shutdown service that saves a frame to Transcend on ANY arrow keypress.
# Start:   python arrow_save_to_transcend.py start
# Stop:    python arrow_save_to_transcend.py shutdown
# Programmatic: from arrow_save_to_transcend import run; run(["start"]); run(["shutdown"])

import os, time, json, signal, argparse, threading, queue, subprocess
import cv2, numpy as np
from pynput import keyboard
from ring_grab import get_frame_bgr_from_ring  # returns (frame_bgr, meta)
from collections import deque

time.sleep(7.0)
# ------------ Config (tweak if needed) --------------
RING_PATH   = "/tmp/scap.ring"
VOL_NAME    = "transcend"
ENC_PARAMS  = [cv2.IMWRITE_JPEG_QUALITY, 92]  # speed/size trade-off
STATE_PATH  = "/tmp/arrow_saver_state.json"   # pid, dest_root, recent paths/records
PID_PATH    = "/tmp/arrow_saver.pid"          # running pid
RECENT_KEEP = 256                              # remember last N saved

# NM (no-move) capture settings
NM_PERIOD_S = 1.5                              # capture every NM_PERIOD_S when idle

# If an NM frame was saved within this window before an arrow keypress, delete it
NM_DELETE_WINDOW_S = 0.5

# Idle auto-press: if no arrow pressed for this long, purge last window and press RIGHT once
IDLE_AUTOPRESS_S   = 30.0
IDLE_AUTOPRESS_KEY = "right"
# ----------------------------------------------------

STOP = threading.Event()
Q = queue.Queue(maxsize=64)

# Monotonic clock bookkeeping
LAST_PRESS_MONO_NS = time.monotonic_ns()
NEXT_NM_DUE_MONO_NS = LAST_PRESS_MONO_NS + int(NM_PERIOD_S * 1e9)
NEXT_IDLE_AUTOPRESS_DUE_MONO_NS = LAST_PRESS_MONO_NS + int(IDLE_AUTOPRESS_S * 1e9)

# Recent-saves tracking (in-memory + persisted)
REC_LOCK = threading.Lock()
RECENT_LOCAL = deque(maxlen=RECENT_KEEP)

# Transcend root (resolved on start)
DEST_ROOT = None

def _find_transcend(name="transcend"):
    for v in os.listdir("/Volumes"):
        if v.lower().startswith(name.lower()):
            return os.path.join("/Volumes", v)
    raise SystemExit("Transcend drive not found under /Volumes")

def _day_from_ns(t_ns):
    # YYYYMMDD in local time from a nanosecond timestamp
    return time.strftime("%Y%m%d", time.localtime(t_ns / 1e9))

def _dest_dir_for_ns(t_ns, root=None):
    # Return (and create) the per-day folder for the given timestamp
    base = root or DEST_ROOT
    day  = _day_from_ns(t_ns)
    dest = os.path.join(base, f"arrow_frames_{day}")
    os.makedirs(dest, exist_ok=True)
    return dest

def _prepare_dest_root():
    root = _find_transcend(VOL_NAME)
    subprocess.run(["mdutil", "-i", "off", root], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    # Ensure today's folder exists at startup (we still choose per-save by timestamp)
    _dest_dir_for_ns(time.time_ns(), root)
    return root

def _fast_write(path, data):
    tmp = path + ".part"
    with open(tmp, "wb", buffering=0) as f:
        fd = f.fileno()
        try:
            import fcntl
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)  # don't thrash page cache
        except Exception:
            pass
        f.write(data); f.flush(); os.fsync(fd)
    os.replace(tmp, path)

def _b36(n):
    if n == 0: return "0"
    a = "0123456789abcdefghijklmnopqrstuvwxyz"; s = []
    while n:
        n, r = divmod(n, 36); s.append(a[r])
    return "".join(reversed(s))

def _write_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, separators=(",",":"))
    os.replace(tmp, STATE_PATH)

def _read_state():
    if not os.path.exists(STATE_PATH): return None
    with open(STATE_PATH, "r") as f:
        return json.load(f)

def _save_pid(pid, dest_root):
    state = _read_state() or {}
    recent = state.get("recent", [])
    if len(recent) > RECENT_KEEP:
        recent = recent[-RECENT_KEEP:]
        state["recent"] = recent
    state.update({"pid": pid, "dest_root": dest_root})
    _write_state(state)
    with open(PID_PATH, "w") as f:
        f.write(str(pid))

def _append_recent(rec_or_path):
    # Store as {'path','key','t_ns'}; accept legacy string too
    if isinstance(rec_or_path, str):
        record = {"path": rec_or_path, "key": None, "t_ns": 0}
    else:
        record = rec_or_path

    with REC_LOCK:
        RECENT_LOCAL.append(record)

    state = _read_state() or {}
    lst = state.get("recent", [])
    lst.append(record)
    if len(lst) > RECENT_KEEP:
        lst = lst[-RECENT_KEEP:]
    state["recent"] = lst
    _write_state(state)

def _get_pid():
    try:
        with open(PID_PATH, "r") as f:
            return int(f.read().strip())
    except Exception:
        s = _read_state() or {}
        return int(s.get("pid") or 0)

def _alive(pid):
    if not pid: return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False

def _saver():
    # Writes always into the folder for the day of t_ns (handles midnight rollover + cross-run consistency)
    while not STOP.is_set():
        try:
            frame, keyname, t_ns = Q.get(timeout=0.1)
        except queue.Empty:
            continue
        ok, buf = cv2.imencode(".jpg", frame, ENC_PARAMS)
        if ok:
            dest_dir = _dest_dir_for_ns(t_ns)  # <-- choose day folder at save-time
            out = os.path.join(dest_dir, f"{keyname}_{_b36(t_ns)}.jpg")
            try:
                _fast_write(out, buf.tobytes())
                _append_recent({"path": out, "key": keyname, "t_ns": t_ns})
            except Exception:
                pass
        Q.task_done()

def _enqueue_capture(keyname):
    # Grab latest frame and enqueue for save, labeled with keyname
    try:
        frame_bgr, _ = get_frame_bgr_from_ring(path=RING_PATH, wait_new=False, timeout_s=0.0)
        frame = frame_bgr.copy(order="C")  # detach immediately
        t_ns  = time.time_ns()
        try:
            Q.put_nowait((frame, keyname, t_ns))
        except queue.Full:
            try: Q.get_nowait()
            except queue.Empty: pass
            Q.put_nowait((frame, keyname, t_ns))
    except Exception:
        pass

def _on_term(signum, frame):
    STOP.set()

def _maybe_delete_recent_nm(now_ns, window_s=NM_DELETE_WINDOW_S):
    # If an NM image was saved within the last window_s seconds, delete the most recent one
    window_ns = int(window_s * 1e9)
    with REC_LOCK:
        for rec in reversed(RECENT_LOCAL):
            path = rec.get("path") if isinstance(rec, dict) else rec
            key  = rec.get("key")  if isinstance(rec, dict) else None
            t_ns = rec.get("t_ns", 0) if isinstance(rec, dict) else 0
            if key == "nm" and path and now_ns - t_ns <= window_ns:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        print(f"[arrow-saver] Deleted recent NM ({path}) within {window_s:.2f}s window")
                except Exception:
                    pass
                # remove from memory
                try:
                    RECENT_LOCAL.remove(rec)
                except ValueError:
                    pass
                # remove from state
                state = _read_state() or {}
                lst = state.get("recent", [])
                for i in range(len(lst) - 1, -1, -1):
                    item = lst[i]
                    item_path = item.get("path") if isinstance(item, dict) else item
                    if item_path == path:
                        del lst[i]
                        break
                state["recent"] = lst
                _write_state(state)
                break

def _delete_recent_images(window_s, now_ns=None):
    # Delete ALL images (any key) saved within the last window_s seconds
    if now_ns is None:
        now_ns = time.time_ns()
    cutoff = now_ns - int(window_s * 1e9)
    to_delete = []

    with REC_LOCK:
        for rec in list(RECENT_LOCAL):
            if isinstance(rec, dict):
                t_ns = rec.get("t_ns", 0)
                if t_ns >= cutoff:
                    to_delete.append(rec)

        paths_set = set()
        for rec in to_delete:
            try:
                RECENT_LOCAL.remove(rec)
            except ValueError:
                pass
            p = rec.get("path")
            if p:
                paths_set.add(p)

        state = _read_state() or {}
        lst = state.get("recent", [])
        filtered = []
        for item in lst:
            item_path = item.get("path") if isinstance(item, dict) else item
            item_t = item.get("t_ns", 0) if isinstance(item, dict) else 0
            if item_t >= cutoff and item_path in paths_set:
                continue
            filtered.append(item)
        state["recent"] = filtered
        _write_state(state)

    for rec in to_delete:
        p = rec.get("path")
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    if to_delete:
        print(f"[arrow-saver] Deleted {len(to_delete)} file(s) from the last {window_s:.1f}s due to idle auto-press")

ARROWS = {
    keyboard.Key.up: "up",
    keyboard.Key.down: "down",
    keyboard.Key.left: "left",
    keyboard.Key.right: "right",
}

def _nm_ticker():
    # When idle (no arrow) for NM_PERIOD_S, capture 'nm' every NM_PERIOD_S until an arrow arrives
    global NEXT_NM_DUE_MONO_NS
    period_ns = int(NM_PERIOD_S * 1e9)
    while not STOP.is_set():
        now = time.monotonic_ns()
        if now >= NEXT_NM_DUE_MONO_NS:
            _enqueue_capture("nm")
            NEXT_NM_DUE_MONO_NS = now + period_ns
        time.sleep(0.02)  # ~50Hz check

def _idle_watchdog():
    # If no arrow pressed for IDLE_AUTOPRESS_S, purge last window and auto-press RIGHT once
    global LAST_PRESS_MONO_NS, NEXT_NM_DUE_MONO_NS, NEXT_IDLE_AUTOPRESS_DUE_MONO_NS
    period_nm_ns = int(NM_PERIOD_S * 1e9)
    autopress_ns = int(IDLE_AUTOPRESS_S * 1e9)
    while not STOP.is_set():
        now_mono = time.monotonic_ns()
        if now_mono >= NEXT_IDLE_AUTOPRESS_DUE_MONO_NS:
            _delete_recent_images(IDLE_AUTOPRESS_S, now_ns=time.time_ns())
            _enqueue_capture(IDLE_AUTOPRESS_KEY)
            LAST_PRESS_MONO_NS = now_mono
            NEXT_NM_DUE_MONO_NS = now_mono + period_nm_ns
            NEXT_IDLE_AUTOPRESS_DUE_MONO_NS = now_mono + autopress_ns
        time.sleep(0.05)

def _listener_loop():
    # Warm JPEG encoder so first keypress is instant.
    _ = cv2.imencode(".jpg", np.zeros((8,8,3), np.uint8), ENC_PARAMS)

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT,  _on_term)

    threading.Thread(target=_saver, daemon=True).start()
    threading.Thread(target=_nm_ticker, daemon=True).start()
    threading.Thread(target=_idle_watchdog, daemon=True).start()

    period_ns = int(NM_PERIOD_S * 1e9)
    autopress_ns = int(IDLE_AUTOPRESS_S * 1e9)

    def on_press(key):
        global LAST_PRESS_MONO_NS, NEXT_NM_DUE_MONO_NS, NEXT_IDLE_AUTOPRESS_DUE_MONO_NS
        keyname = ARROWS.get(key)
        if keyname:
            _maybe_delete_recent_nm(time.time_ns(), NM_DELETE_WINDOW_S)
            now_mono = time.monotonic_ns()
            LAST_PRESS_MONO_NS = now_mono
            NEXT_NM_DUE_MONO_NS = now_mono + period_ns
            NEXT_IDLE_AUTOPRESS_DUE_MONO_NS = now_mono + autopress_ns
            _enqueue_capture(keyname)
        elif key == keyboard.Key.esc:
            STOP.set(); return False

    today_dir = _dest_dir_for_ns(time.time_ns())
    print(f"[arrow-saver] Listening… writing into daily folder: {today_dir} (ESC to quit)")
    with keyboard.Listener(on_press=on_press, suppress=False) as L:
        while not STOP.is_set():
            time.sleep(0.05)
        try:
            L.stop()
        except Exception:
            pass

def cmd_start():
    global DEST_ROOT
    pid = _get_pid()
    if _alive(pid):
        print(f"[arrow-saver] Already running (pid={pid}).")
        return
    DEST_ROOT = _prepare_dest_root()
    _save_pid(os.getpid(), DEST_ROOT)
    _listener_loop()

def cmd_shutdown():
    state = _read_state() or {}
    recent = list(state.get("recent", []))
    for item in list(reversed(recent))[:6]:
        path = item.get("path") if isinstance(item, dict) else item
        try:
            if path and os.path.exists(path):
                os.remove(path)
                print(f"[arrow-saver] Deleted {path}")
        except Exception as e:
            print(f"[arrow-saver] Could not delete {path}: {e}")

    pid = _get_pid()
    if _alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[arrow-saver] Sent SIGTERM to pid={pid}")
        except Exception as e:
            print(f"[arrow-saver] Kill failed: {e}")
    try: os.remove(PID_PATH)
    except Exception: pass

def run(argv=None):
    ap = argparse.ArgumentParser(prog="arrow_save_to_transcend", add_help=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start", help="Start the arrow key frame saver")
    sub.add_parser("shutdown", help="Delete last 6 images saved and stop the saver")
    args = ap.parse_args(argv)

    if args.cmd == "start":
        cmd_start()
    elif args.cmd == "shutdown":
        cmd_shutdown()

if __name__ == "__main__":
    run()
