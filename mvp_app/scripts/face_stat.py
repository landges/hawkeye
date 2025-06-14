import os, csv, logging
from collections import defaultdict
from pathlib import Path
import cv2, requests
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# === settings ===
FRAME_STEP = 15
MIN_CROP = 20
MIN_LEN = 1.0
API = "http://localhost:8000/search"
OUT = "output_faces"
TIMEOUT = 5

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

def search_frame(frame, k=5):
    _, enc = cv2.imencode(".jpg", frame)
    files = {"file": ("f.jpg", enc.tobytes(), "image/jpeg")}
    data = {"k": str(k), "auto_add": "true"}
    r = requests.post(API, files=files, data=data, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def merge_ts(ts, m=0.5, g=0.8):
    if not ts: return []
    ts = sorted(ts); segs=[]; s=ts[0]-m; e=ts[0]+m
    for t in ts[1:]:
        if t-e<=g: e=t+m
        else: segs.append((max(0,s),e)); s=t-m; e=t+m
    segs.append((max(0,s),e)); return segs

def analyze(video_path: Path):
    os.makedirs(OUT, exist_ok=True)
    stats = defaultdict(list)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError("Can't open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total, desc="frames")

    idx=0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % FRAME_STEP == 0:
            t = idx/fps
            try:
                matches = search_frame(frame)
            except Exception as e:
                logging.error("API failed: %s", e)
                matches=[]
            h,w = frame.shape[:2]
            for m in matches:
                fid = m["face_id"]
                qb = m.get("query_bbox", [])
                # статистика до фильтров
                stats[fid].append(t)
                # clamped crop
                if len(qb)==4:
                    x1,y1,x2,y2 = map(int, qb)
                    x1, x2 = sorted((x1, x2))
                    y1, y2 = sorted((y1, y2))
                    x1c,y1c = max(0,x1), max(0,y1)
                    x2c,y2c = min(w,x2), min(h,y2)
                    cw,ch = x2c-x1c, y2c-y1c
                else:
                    x1c,y1c,x2c,y2c = 0,0,w,h
                    cw,ch = w,h
                # fallback full frame if too small
                if cw<MIN_CROP or ch<MIN_CROP:
                    crop = frame; suf="_full"
                else:
                    crop = frame[y1c:y2c, x1c:x2c]; suf=""
                # save
                d = os.path.join(OUT, str(fid)); os.makedirs(d, exist_ok=True)
                fn = f"{fid}_f{idx}_t{t:.2f}{suf}.jpg"
                p = os.path.join(d, fn)
                if crop.size and cv2.imwrite(p, crop):
                    logging.debug("Saved %s", p)
        idx+=1; pbar.update(1)
    cap.release(); pbar.close()

    # CSV
    with open(os.path.join(OUT,f"face_stats_{video_path.stem}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["face_id","count","timestamps"])
        for fid,ts in stats.items():
            w.writerow([fid, len(ts), ";".join(f"{x:.2f}" for x in ts)])
    # clips
    for fid, ts in stats.items():
        for i, (s, e) in enumerate(merge_ts(ts), 1):
            if e - s >= MIN_LEN:
                fn = os.path.join(OUT, str(fid), f"{fid}_seg{i}.mp4")
                try:
                    # имя выходного файла — 4-й позиционный аргумент
                    ffmpeg_extract_subclip(video_path, s, e, fn)
                except Exception as err:
                    logging.error("Clip extraction failed: %s", err)

    logging.info("Done, %d unique faces.", len(stats))
    return stats

if __name__=="__main__":
    filename = Path("scripts/3413952367205.mp4")
    analyze(filename)
