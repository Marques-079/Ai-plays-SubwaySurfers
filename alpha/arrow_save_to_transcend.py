#!/usr/bin/env python3
# arrow_save_to_transcend.py — per-day folder, NM ticker, NM purge-on-press,
# idle auto-press + purge, and "short-run scrub" (delete all files from this run if runtime < threshold)
# Start:   python arrow_save_to_transcend.py start
# Stop:    python arrow_save_to_transcend.py shutdown
# Programmatic: from arrow_save_to_transcend import run; run(["start"]); run(["shutdown"])

import os, time, json, signal, argparse, threading, queue, subprocess
import cv2, numpy as np
from pynput import keyboard
from collections import deque
from ring_grab import get_frame_bgr_from_ring  # returns (frame_bgr, meta)

time.sleep(7.0)  # wait for boot processes

# ------------ Config (tweak if needed) --------------
RING_PATH   = "/tmp/scap.ring"
VOL_NAME    = "transcend"
ENC_PARAMS  = [cv2.IMWRITE_JPEG_QUALITY, 92]     # speed/size trade-off
STATE_PATH  = "/tmp/arrow_saver_state.json"      # pid, dest_root, session, recent records
PID_PATH    = "/tmp/arrow_saver.pid"             # running pid
RECENT_KEEP = 1024                               # remember last N saved (bumped a bit)

# NM (no-move) capture settings
NM_PERIOD_S = 1.5                                 # capture every NM_PERIOD_S when idle
NM_DELETE_WINDOW_S = 0.5                          # if an NM was saved within this window before an arrow, delete that NM

# Idle auto-press: if no arrow pressed for this long, purge last window and press RIGHT once
IDLE_AUTOPRESS_S   = 30.0
IDLE_AUTOPRESS_KEY = "right"

# Short-run scrub: if total runtime (start -> end) is below this, delete ALL files from this run
BAD_RUN_SCRUB_S    = 20.0
# ----------------------------------------------------

STOP = threading.Event()
Q = queue.Queue(maxsize=128)

# Monotonic clock bookkeeping
START_WALL_NS = None          # set at start
START_MONO_NS = None          # set at start
LAST_PRESS_MONO_NS = time.monotonic_ns()
NEXT_NM_DUE_MONO_NS = LAST_PRESS_MONO_NS + int(NM_PERIOD_S * 1e9)
NEXT_IDLE_AUTOPRESS_DUE_MONO_NS = LAST_PRESS_MONO_NS + int(IDLE_AUTOPRESS_S * 1e9)

# Recent-saves tracking (in-memory + persisted)
REC_LOCK = threading.Lock()
RECENT_LOCAL = deque(maxlen=RECENT_KEEP)

# Per-run identifiers
SESSION_ID = None
DEST_ROOT = None  # Transcend root

# ---------- Helpers ----------
def _b36(n):
    if n == 0: return "0"
    a = "0123456789abcdefghijklmnopqrstuvwxyz"; s = []
    while n:
        n, r = divmod(n, 36); s.append(a[r])
    return "".join(reversed(s))

def _make_session_id():
    return f"{_b36(time.time_ns())}-{os.getpid()}"

def _find_transcend(name="transcend"):
    for v in os.listdir("/Volumes"):
        if v.lower().startswith(name.lower()):
            return os.path.join("/Volumes", v)
    raise SystemExit("Transcend drive not found under /Volumes")

def _day_from_ns(t_ns):
    return time.strftime("%Y%m%d", time.localtime(t_ns / 1e9))

def _dest_dir_for_ns(t_ns, root=None):
    base = root or DEST_ROOT
    day  = _day_from_ns(t_ns)
    dest = os.path.join(base, f"arrow_frames_{day}")
    os.makedirs(dest, exist_ok=True)
    return dest

def _prepare_dest_root():
    root = _find_transcend(VOL_NAME)
    subprocess.run(["mdutil", "-i", "off", root], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    # Ensure today's folder exists at startup (we still choose per-save by t_ns)
    _dest_dir_for_ns(time.time_ns(), root)
    return root

def _fast_write(path, data):
    tmp = path + ".part"
    with open(tmp, "wb", buffering=0) as f:
        fd = f.fileno()
        try:
            import fcntl
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        except Exception:
            pass
        f.write(data); f.flush(); os.fsync(fd)
    os.replace(tmp, path)

def _write_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, separators=(",",":"))
    os.replace(tmp, STATE_PATH)

def _read_state():
    if not os.path.exists(STATE_PATH): return None
    with open(STATE_PATH, "r") as f:
        return json.load(f)

def _save_pid_and_session(pid, dest_root, session_id, start_wall_ns):
    state = _read_state() or {}
    recent = state.get("recent", [])
    if len(recent) > RECENT_KEEP:
        recent = recent[-RECENT_KEEP:]
        state["recent"] = recent
    state.update({
        "pid": pid,
        "dest_root": dest_root,
        "session_id": session_id,
        "start_wall_ns": start_wall_ns
    })
    _write_state(state)
    with open(PID_PATH, "w") as f:
        f.write(str(pid))

def _append_recent(record):
    """record: {'path','key','t_ns','sid'}"""
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

# ---------- Core saving ----------
def _saver():
    while not STOP.is_set():
        try:
            frame, keyname, t_ns = Q.get(timeout=0.1)
        except queue.Empty:
            continue
        ok, buf = cv2.imencode(".jpg", frame, ENC_PARAMS)
        if ok:
            dest_dir = _dest_dir_for_ns(t_ns)
            out = os.path.join(dest_dir, f"{keyname}_{_b36(t_ns)}.jpg")
            try:
                _fast_write(out, buf.tobytes())
                _append_recent({"path": out, "key": keyname, "t_ns": t_ns, "sid": SESSION_ID})
            except Exception:
                pass
        Q.task_done()

def _enqueue_capture(keyname):
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

# ---------- NM / purge helpers ----------
def _maybe_delete_recent_nm(now_ns, window_s=NM_DELETE_WINDOW_S):
    window_ns = int(window_s * 1e9)
    with REC_LOCK:
        for rec in reversed(RECENT_LOCAL):
            path = rec.get("path")
            key  = rec.get("key")
            t_ns = rec.get("t_ns", 0)
            if key == "nm" and path and now_ns - t_ns <= window_ns:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        print(f"[arrow-saver] Deleted recent NM ({path}) within {window_s:.2f}s window")
                except Exception:
                    pass
                try:
                    RECENT_LOCAL.remove(rec)
                except ValueError:
                    pass
                state = _read_state() or {}
                lst = state.get("recent", [])
                for i in range(len(lst) - 1, -1, -1):
                    item = lst[i]
                    if isinstance(item, dict) and item.get("path") == path:
                        del lst[i]
                        break
                state["recent"] = lst
                _write_state(state)
                break

def _delete_recent_images(window_s, now_ns=None):
    if now_ns is None:
        now_ns = time.time_ns()
    cutoff = now_ns - int(window_s * 1e9)
    to_delete = []

    with REC_LOCK:
        for rec in list(RECENT_LOCAL):
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
            if isinstance(item, dict) and item.get("path") in paths_set and item.get("t_ns", 0) >= cutoff:
                continue
            filtered.append(item)
        state["recent"] = filtered
        _write_state(state)

    for rec in to_delete:
        p = rec.get("path")
        if p and os.path.exists(p):
            try: os.remove(p)
            except Exception: pass
    if to_delete:
        print(f"[arrow-saver] Deleted {len(to_delete)} file(s) from the last {window_s:.1f}s due to idle auto-press")

# ---------- Short-run scrub ----------
def _scrub_entire_session(session_id):
    """Delete every file created by this session (by 'sid')."""
    state = _read_state() or {}
    lst = state.get("recent", [])
    targets = [item for item in lst if isinstance(item, dict) and item.get("sid") == session_id]
    paths = [it.get("path") for it in targets if it.get("path")]
    # Remove from disk
    n = 0
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
                n += 1
            except Exception:
                pass
    # Remove from in-memory recent and state
    with REC_LOCK:
        for rec in list(RECENT_LOCAL):
            if rec.get("sid") == session_id:
                try:
                    RECENT_LOCAL.remove(rec)
                except ValueError:
                    pass
    state["recent"] = [it for it in lst if not (isinstance(it, dict) and it.get("sid") == session_id)]
    _write_state(state)
    print(f"[arrow-saver] Short-run scrub: deleted {n} file(s) for session {session_id}")

def _maybe_scrub_short_run_at_exit():
    """Called when the 'start' process is exiting (ESC/CTRL-C)."""
    try:
        runtime_s = (time.monotonic_ns() - START_MONO_NS) / 1e9
    except Exception:
        return
    if runtime_s < BAD_RUN_SCRUB_S:
        _scrub_entire_session(SESSION_ID)

# ---------- Threads ----------
def _nm_ticker():
    global NEXT_NM_DUE_MONO_NS
    period_ns = int(NM_PERIOD_S * 1e9)
    while not STOP.is_set():
        now = time.monotonic_ns()
        if now >= NEXT_NM_DUE_MONO_NS:
            _enqueue_capture("nm")
            NEXT_NM_DUE_MONO_NS = now + period_ns
        time.sleep(0.02)

def _idle_watchdog():
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

# ---------- Listener / lifecycle ----------
def _on_term(signum, frame):
    STOP.set()

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
        keymap = {
            keyboard.Key.up: "up",
            keyboard.Key.down: "down",
            keyboard.Key.left: "left",
            keyboard.Key.right: "right",
        }
        keyname = keymap.get(key)
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
    # if we’re exiting (ESC/CTRL-C), check short-run scrub here too
    _maybe_scrub_short_run_at_exit()

# ---------- Commands ----------
def cmd_start():
    global DEST_ROOT, SESSION_ID, START_WALL_NS, START_MONO_NS
    pid = _get_pid()
    if _alive(pid):
        print(f"[arrow-saver] Already running (pid={pid}).")
        return
    DEST_ROOT = _prepare_dest_root()
    SESSION_ID = _make_session_id()
    START_WALL_NS = time.time_ns()
    START_MONO_NS = time.monotonic_ns()
    _save_pid_and_session(os.getpid(), DEST_ROOT, SESSION_ID, START_WALL_NS)
    _listener_loop()

def cmd_shutdown():
    state = _read_state() or {}
    start_wall_ns = state.get("start_wall_ns")
    session_id = state.get("session_id")
    short_run_scrubbed = False

    # If we have a start time, evaluate total runtime and scrub the session if too short
    if start_wall_ns:
        runtime_s = (time.time_ns() - int(start_wall_ns)) / 1e9
        if runtime_s < BAD_RUN_SCRUB_S and session_id:
            _scrub_entire_session(session_id)
            short_run_scrubbed = True

    # If not a short run (or no start time), do the usual "delete last 6" cleanup
    if not short_run_scrubbed:
        recent = list(state.get("recent", []))
        for item in list(reversed(recent))[:6]:
            path = item.get("path") if isinstance(item, dict) else item
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    print(f"[arrow-saver] Deleted {path}")
            except Exception as e:
                print(f"[arrow-saver] Could not delete {path}: {e}")

    # Signal the running listener to stop
    pid = state.get("pid") or _get_pid()
    if pid and _alive(int(pid)):
        try:
            os.kill(int(pid), signal.SIGTERM)
            print(f"[arrow-saver] Sent SIGTERM to pid={pid}")
        except Exception as e:
            print(f"[arrow-saver] Kill failed: {e}")
    try: os.remove(PID_PATH)
    except Exception: pass

def run(argv=None):
    ap = argparse.ArgumentParser(prog="arrow_save_to_transcend", add_help=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start", help="Start the arrow key frame saver")
    sub.add_parser("shutdown", help="Delete last 6 images or scrub session if runtime < threshold, then stop")
    args = ap.parse_args(argv)

    if args.cmd == "start":
        cmd_start()
    elif args.cmd == "shutdown":
        cmd_shutdown()

if __name__ == "__main__":
    run()
