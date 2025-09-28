#!/usr/bin/env python3
# announcer.py — instant arrow-key callouts (macOS-safe; no simpleaudio)
# Deps: pynput, sounddevice, numpy, (pyttsx3 optional). macOS: grant Accessibility perms.

import os, sys, argparse, shutil, tempfile, subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

from pynput import keyboard

try:
    import numpy as np
except Exception:
    print("ERROR: numpy is required: pip install numpy")
    raise

# Stable playback backend
try:
    import sounddevice as sd
except Exception:
    print("ERROR: sounddevice is required: pip install sounddevice")
    raise

# Optional TTS for first-run synthesis
try:
    import pyttsx3
    _HAS_PYTTSX3 = True
except Exception:
    _HAS_PYTTSX3 = False


WORDS = {
    keyboard.Key.left:  "Left",
    keyboard.Key.right: "Right",
    keyboard.Key.up:    "Up",
    keyboard.Key.down:  "Down",
}

# ---------- cache dir ----------
def _cache_dir() -> Path:
    if sys.platform.startswith("win"):
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "announcer" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "announcer"

# ---------- file format helpers ----------
def _is_riff_wav(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"RIFF"
    except Exception:
        return False

def _aiff_to_wav_py(aiff_path: Path, wav_path: Path):
    """Pure-Python AIFF→WAV for 16-bit PCM using NumPy (big→little endian)."""
    import aifc, wave
    with aifc.open(str(aiff_path), "rb") as ai:
        nchan = ai.getnchannels()
        sw    = ai.getsampwidth()
        sr    = ai.getframerate()
        nfrm  = ai.getnframes()
        data  = ai.readframes(nfrm)
    if sw != 2:
        raise RuntimeError("AIFF sample width not 16-bit; install ffmpeg or use afconvert.")
    arr = np.frombuffer(data, dtype=">i2").astype("<i2", copy=False)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(nchan)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(arr.tobytes())

def _coerce_to_wav16le(in_path: Path, out_wav: Path):
    """Ensure 16-bit LE WAV at out_wav. Prefer afconvert/ffmpeg; fallback to pure-Python AIFF."""
    if shutil.which("afconvert"):
        subprocess.run(
            ["afconvert", str(in_path), str(out_wav), "-f", "WAVE", "-d", "LEI16"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(in_path), "-acodec", "pcm_s16le", "-f", "wav", str(out_wav)],
            check=True
        )
        return
    _aiff_to_wav_py(in_path, out_wav)

# ---------- trimming (NumPy, no audioop) ----------
def _trim_wav_bytes(wav_path: Path, threshold: int = 200, chunk_ms: int = 8) -> Tuple[bytes, int, int, int]:
    """
    Remove leading/trailing silence from a PCM WAV using NumPy RMS.
    Returns (pcm_bytes, channels, sample_width_bytes, sample_rate).
    If not 16-bit, returns untrimmed (still plays).
    """
    import wave
    with wave.open(str(wav_path), "rb") as w:
        nchan = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        nframes = w.getnframes()
        frames = w.readframes(nframes)

    if sw != 2:
        return frames, nchan, sw, sr

    bytes_per_frame = nchan * sw
    if len(frames) < bytes_per_frame:
        return frames, nchan, sw, sr

    pcm = np.frombuffer(frames, dtype=np.int16).reshape(-1, nchan)

    frames_per_chunk = max(1, int(sr * (chunk_ms / 1000.0)))
    total_chunks = int(np.ceil(pcm.shape[0] / frames_per_chunk))
    pad = total_chunks * frames_per_chunk - pcm.shape[0]
    if pad:
        pcm = np.vstack([pcm, np.zeros((pad, nchan), dtype=pcm.dtype)])

    chunks = pcm.reshape(total_chunks, frames_per_chunk, nchan).astype(np.float64)
    rms = np.sqrt(np.mean(chunks ** 2, axis=(1, 2)))

    idx = np.where(rms >= float(threshold))[0]
    if idx.size == 0:
        return frames, nchan, sw, sr

    first = max(0, int(idx[0]) - 1)
    last  = min(total_chunks - 1, int(idx[-1]) + 1)

    start_f = first * frames_per_chunk
    end_f   = min(pcm.shape[0], (last + 1) * frames_per_chunk)

    trimmed = pcm[start_f:end_f].astype(np.int16).tobytes()
    return trimmed or frames, nchan, sw, sr

# ---------- synthesis ----------
def _synthesize_with_pyttsx3(text: str, out_path: Path, rate: int, volume: float, voice_id: Optional[str]):
    eng = pyttsx3.init()
    eng.setProperty("rate", int(rate))
    eng.setProperty("volume", float(max(0.0, min(1.0, volume))))
    if voice_id:
        try: eng.setProperty("voice", voice_id)
        except Exception: pass
    eng.save_to_file(text, str(out_path))
    eng.runAndWait()

def _synthesize_with_say(text: str, aiff_out: Path, rate: int, voice_id: Optional[str]):
    cmd = ["say", text, "-o", str(aiff_out), "-r", str(rate)]
    if voice_id:
        cmd += ["-v", voice_id]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def _ensure_wav(word: str, cache: Path, rate: int, volume: float, voice_id: Optional[str]) -> Path:
    """
    Guarantee a 16-bit LE WAV at the cached path, regardless of TTS output format.
    """
    cache.mkdir(parents=True, exist_ok=True)
    out_wav = cache / f"{word.lower()}_{rate}.wav"
    if out_wav.exists() and out_wav.stat().st_size > 1024 and _is_riff_wav(out_wav):
        return out_wav

    tmp_src = Path(tempfile.mkstemp(prefix=f"ann_{word}_", suffix=(".aiff" if sys.platform == "darwin" else ".wav"))[1])
    try:
        if sys.platform == "darwin" and shutil.which("say"):
            _synthesize_with_say(word, tmp_src, rate, voice_id)
        elif _HAS_PYTTSX3:
            _synthesize_with_pyttsx3(word, tmp_src, rate, volume, voice_id)
        else:
            raise RuntimeError("No TTS available. Install pyttsx3 or use macOS 'say'.")

        if _is_riff_wav(tmp_src):
            # Might already be WAV; still ensure 16-bit LE by copying as-is first.
            tmp_src.replace(out_wav)
        else:
            _coerce_to_wav16le(tmp_src, out_wav)
    finally:
        try:
            if tmp_src.exists():
                tmp_src.unlink()
        except Exception:
            pass

    return out_wav

# ---------- core announcer (sounddevice) ----------
class Announcer:
    def __init__(self, rate: int = 210, volume: float = 1.0, voice: Optional[str] = None,
                 silence_threshold: int = 200, chunk_ms: int = 8, preempt: bool = True):
        self.preempt = preempt
        self._buffers: Dict[str, Tuple[np.ndarray, int]] = {}  # word -> (int16 array [N, C], sr)

        # Hint to PortAudio: go for low latency
        sd.default.latency = ("low", "low")
        sd.default.dtype = "int16"

        cache = _cache_dir()
        for word in {"Left", "Right", "Up", "Down"}:
            wav_path = _ensure_wav(word, cache, rate, volume, voice)
            if not _is_riff_wav(wav_path):
                coerced = wav_path.with_suffix(".coerced.wav")
                _coerce_to_wav16le(wav_path, coerced)
                wav_path = coerced
            pcm_bytes, ch, sw, sr = _trim_wav_bytes(wav_path, threshold=silence_threshold, chunk_ms=chunk_ms)
            if sw != 2:
                # Convert to int16 if needed (shouldn't happen after coercion)
                pcm = np.frombuffer(pcm_bytes, dtype=np.int8).astype(np.int16)
                ch = 1
            else:
                pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
            try:
                pcm = pcm.reshape(-1, ch)
            except ValueError:
                pcm = pcm.reshape(-1, 1)
                ch = 1
            self._buffers[word] = (pcm, sr)

    def say(self, word: str):
        buf = self._buffers.get(word)
        if buf is None:
            return
        pcm, sr = buf
        # preempt by stopping any ongoing global stream
        if self.preempt:
            try: sd.stop()
            except Exception: pass
        sd.play(pcm, sr, blocking=False)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Instant arrow-key voice announcer (ESC to quit)")
    ap.add_argument("--rate", type=int, default=210, help="Synthesis speaking rate for cached samples")
    ap.add_argument("--volume", type=float, default=1.0, help="TTS volume for cached samples (0..1)")
    ap.add_argument("--voice", type=str, default=None, help="Voice id/name (pyttsx3 or macOS 'say')")
    ap.add_argument("--silence-threshold", type=int, default=200, help="Trim threshold (RMS)")
    ap.add_argument("--chunk-ms", type=int, default=8, help="Trim analysis chunk size (ms)")
    ap.add_argument("--no-preempt", action="store_true", help="Do not cut off a word when another key is pressed")
    ap.add_argument("--quiet", action="store_true", help="Do not print to console")
    args = ap.parse_args()

    announcer = Announcer(rate=args.rate, volume=args.volume, voice=args.voice,
                          silence_threshold=args.silence_threshold, chunk_ms=args.chunk_ms,
                          preempt=(not args.no_preempt))

    pressed = set()

    def on_press(key):
        if key == keyboard.Key.esc:
            return False
        if key in WORDS and key not in pressed:
            pressed.add(key)
            word = WORDS[key]
            if not args.quiet:
                from time import strftime
                print(f"[{strftime('%H:%M:%S')}] {word}")
            announcer.say(word)

    def on_release(key):
        if key in pressed:
            pressed.discard(key)

    print("Listening for arrow keys… (ESC to quit)")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass
    # Ensure audio stops on exit
    try: sd.stop()
    except Exception: pass

if __name__ == "__main__":
    main()
