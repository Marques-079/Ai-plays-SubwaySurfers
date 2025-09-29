import os, sys, time, subprocess

ALPHA_DIR = os.path.dirname(os.path.abspath(__file__))
CMD = [sys.executable, os.path.join(ALPHA_DIR, "runv11.py")]
SLEEP_SECONDS = 2.0  

def main():
    os.chdir(ALPHA_DIR)
    run_idx = 0
    try:
        while True:
            run_idx += 1
            print(f"[loop] starting run {run_idx} …")
            rc = subprocess.run(CMD, cwd=ALPHA_DIR, check=False).returncode
            print(f"[loop] run {run_idx} ended with rc={rc} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(SLEEP_SECONDS)
    except KeyboardInterrupt:
        print("\n[loop] stopped by user")

if __name__ == "__main__":
    main()
