import os, sys, time, subprocess

ALPHA_DIR   = os.path.dirname(os.path.abspath(__file__)) 
REPO_ROOT   = os.path.dirname(ALPHA_DIR)                    
CMD         = [sys.executable, "alpha/runv13.py"]          
SLEEP_SECONDS = 2.0

def main():
    os.chdir(REPO_ROOT)

    run_idx = 0
    try:
        while True:
            run_idx += 1
            print(f"[loop] starting run {run_idx} … (cwd={REPO_ROOT})")
            rc = subprocess.run(CMD, cwd=REPO_ROOT, check=False).returncode
            print(f"[loop] run {run_idx} ended with rc={rc} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("\n[loop] stopped")

if __name__ == "__main__":
    main()
