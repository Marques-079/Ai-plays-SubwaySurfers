import os
import time
import cv2
import jwt
import numpy as np
import requests

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# ────────────────────────────────────────────────
#  0) grab your LS root URL and personal token from env
# ────────────────────────────────────────────────
LS_ROOT    = os.environ.get("LABEL_STUDIO_URL")
LS_API_KEY = os.environ.get("LABEL_STUDIO_API_KEY")  # this is your Personal Access Token

if not LS_ROOT:
    raise RuntimeError(
        "Please set LABEL_STUDIO_URL to your Label Studio server root (e.g. http://localhost:8080)"
    )
if not LS_API_KEY:
    raise RuntimeError(
        "Please set LABEL_STUDIO_API_KEY to your Label Studio Personal Access Token"
    )

print(f"▶︎ Label Studio URL: {LS_ROOT!r}")
print(f"▶︎ Personal Access Token: {'<set>' if LS_API_KEY else '<MISSING>'}")

# ────────────────────────────────────────────────
#  Helper to refresh & cache a short‐lived access token
# ────────────────────────────────────────────────
_cached_access = None
_expires_at    = 0

def get_access_token():
    global _cached_access, _expires_at

    # still valid?
    if _cached_access and time.time() < (_expires_at - 30):
        return _cached_access

    # fetch a new one
    url = f"{LS_ROOT.rstrip('/')}/api/token/refresh"
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"refresh": LS_API_KEY},
        timeout=5,
    )
    resp.raise_for_status()
    data = resp.json()
    access = data["access"]

    # decode expiry
    decoded = jwt.decode(access, options={"verify_signature": False})
    _expires_at = decoded.get("exp", time.time() + 300)
    _cached_access = access
    print(f"🔑 Fetched new access token, expires at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_expires_at))}")
    return access

# ────────────────────────────────────────────────
#  FastAPI setup
# ────────────────────────────────────────────────
app = FastAPI()

@app.get("/health")
@app.get("/predict/health")
async def health():
    return JSONResponse({"status": "ok"})

@app.post("/setup")
@app.post("/predict/setup")
async def setup():
    return JSONResponse({"status": "ok"})

@app.post("/webhook")
@app.post("/predict/webhook")
async def webhook(payload: dict):
    print("🔔 webhook payload:", payload)
    return JSONResponse({"status": "ok"})

# ────────────────────────────────────────────────
#  Load YOLO once on startup
# ────────────────────────────────────────────────
model = YOLO("weights.pt")
print("▶︎ YOLO model loaded.")

# ────────────────────────────────────────────────
#  Main predict endpoint
# ────────────────────────────────────────────────
@app.post("/")
@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(None),
):
    print("=== New /predict call ===")

    # A) direct file upload
    if file is not None:
        print("💾 Received file upload:", file.filename)
        raw = await file.read()

    # B) JSON pre-annotation branch
    else:
        body = await request.json()
        print("🔎 Incoming JSON payload:", body)

        # find the inner data dict
        d = body.get("data", {})
        if "tasks" in body and isinstance(body["tasks"], list):
            d = body["tasks"][0].get("data", {})
        print("→ Resolved data dict:", d)

        # extract the image URL
        img_url = None
        if isinstance(d.get("image"), str):
            img_url = d["image"]
        elif isinstance(d.get("image"), dict):
            img_url = d["image"].get("url")
        elif isinstance(d.get("url"), str):
            img_url = d["url"]
        print("→ Raw img_url:", img_url)

        if not img_url:
            raise HTTPException(422, "no file and no image URL in JSON")

        # prefix LS_ROOT if it’s a leading-slash path
        if img_url.startswith("/"):
            img_url = LS_ROOT.rstrip("/") + img_url
        print("🕵️  Fetching image from URL:", img_url)

        # get a fresh access token
        access = get_access_token()
        headers = {"Authorization": f"Bearer {access}"}
        print("🕵️  Using headers:", headers)

        # fetch the image bytes
        try:
            resp = requests.get(img_url, headers=headers, timeout=10)
        except Exception as e:
            print("❗ Error during requests.get:", repr(e))
            raise HTTPException(500, f"error fetching image: {e}")

        print("🕵️  ➜ status_code =", resp.status_code)
        print("🕵️  ➜ resp.headers =", resp.headers)

        if resp.status_code != 200:
            snippet = resp.text[:200].replace("\n", " ")
            print("🕵️  ➜ body snippet:", repr(snippet))
            raise HTTPException(400, f"image fetch failed: {resp.status_code}")

        raw = resp.content
        print("✅ Image downloaded, bytes:", len(raw))

    # C) decode & run YOLO
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("❗ cv2.imdecode failed, raw length:", len(raw))
        raise HTTPException(400, "cannot decode image bytes")
    print("▶︎ Image decoded:", img.shape)

    results = model(img)[0]
    ls_results = []

    # masks → polygonlabels
    if results.masks is not None:
        print(f"▶︎ {len(results.masks.xy)} masks found")
        for mask, cls in zip(results.masks.xy, results.boxes.cls):
            polygon = [[float(x), float(y)] for x, y in mask]
            ls_results.append({
                "type": "polygonlabels",
                "value": {
                    "points":       polygon,
                    "polygonlabels":[ model.names[int(cls)] ],
                },
                "from_name":    "label",
                "to_name":      "image",
                "original_width":  img.shape[1],
                "original_height": img.shape[0],
            })

    # fallback → rectanglelabels
    else:
        print(f"▶︎ {len(results.boxes.cls)} boxes found")
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1
            ls_results.append({
                "type": "rectanglelabels",
                "value": {
                    "x":      (x1 / img.shape[1]) * 100,
                    "y":      (y1 / img.shape[0]) * 100,
                    "width":  (w  / img.shape[1]) * 100,
                    "height": (h  / img.shape[0]) * 100,
                    "rectanglelabels":[ model.names[int(cls)] ],
                },
                "from_name":    "label",
                "to_name":      "image",
                "original_width":  img.shape[1],
                "original_height": img.shape[0],
            })

    # D) return to LS
    print("✅ Returning", len(ls_results), "labels back to LS")
    return JSONResponse({"predictions":[{"result": ls_results}]})