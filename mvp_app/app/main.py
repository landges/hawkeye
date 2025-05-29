from pathlib import Path
from typing import List, Union
import os, uuid, io, threading, pickle, cv2, numpy as np, faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from deepface import DeepFace
from retinaface import RetinaFace
from pydantic import BaseModel
from app.utils import _read_image
from app.face_pipeline import detect_align_embed

# ---------- глобальные объекты ----------
# ────────────────────────────── persistence ─────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = DATA_DIR / "faces.idx"
META_PATH  = DATA_DIR / "faces_meta.pkl"

_index_lock = threading.Lock()

if INDEX_PATH.exists():
    _index: faiss.Index = faiss.read_index(str(INDEX_PATH))
else:
    # cosine similarity ≈ inner‑product if vectors are L2‑normalised
    base = faiss.IndexFlatIP(512)
    _index = faiss.IndexIDMap(base)

if META_PATH.exists():
    _meta: Dict[int, Dict[str, Any]] = np.load(META_PATH, allow_pickle=True).item()  # type: ignore
else:
    _meta = {}                                    # {face_id:int: {...}}

# Facenet512 модель загружаем один раз
_embedder = DeepFace.build_model("Facenet512")

app = FastAPI(title="Face-Search API")

# ---------- pydantic ----------
class FaceMeta(BaseModel):
    face_id: int
    bbox: List[int]
    photo_id: str

class FaceAddResponse(BaseModel):
    photo_id: str
    faces: List[FaceMeta]

class Match(BaseModel):
    face_id: int
    score: float
    photo_id: str
    bbox: List[int]

# ---------- endpoints ----------
@app.get("/healthz")
def health():
    return {"status": "ok"}

def gen_face_id() -> int:
    """Return a *signed* 63‑bit positive integer.

    We mask off the high bit so the value fits in int64 without overflow
    when converted to numpy dtype('int64') (Faiss expects signed ids).
    """
    return uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF  # keep lower 63 bits

@app.post("/faces")
async def add_faces(file: UploadFile = File(...)):
    bgr = _read_image(file.file.read())
    photo_id = uuid.uuid4().hex

    faces_data = detect_align_embed(bgr)
    if not faces_data:
        raise HTTPException(400, "No faces found")

    face_metas: List[FaceMeta] = []
    with _index_lock:
        for item in faces_data:
            face_id = gen_face_id() # 64‑бит из UUID
            _index.add_with_ids(item["vec"].reshape(1, -1), np.asarray([face_id], dtype="int64"))
            _meta[face_id] = {"photo_id": photo_id, "bbox": item["bbox"]}
            face_metas.append(FaceMeta(face_id=face_id, bbox=item["bbox"], photo_id=photo_id))
        # sync to disk (for MVP synchronous is fine)
        faiss.write_index(_index, str(INDEX_PATH))
        np.save(str(META_PATH), _meta, allow_pickle=True)
    return FaceAddResponse(photo_id=photo_id, faces=face_metas)

@app.post("/search", response_model=List[Match])
async def search_faces(
    file: UploadFile = File(...),
    # vector: Union[List[float], None] = None,
    k: int = 5,
):
    vector = None
    if (file is None and vector is None) or (file and vector):
        raise HTTPException(422, "provide either image file OR vector")

    if file:
        bgr = _read_image(file.file.read())
        faces_data = detect_align_embed(bgr)
        if not faces_data:
            raise HTTPException(400, "No face detected in query image")
        query_vec = faces_data[0]["vec"].reshape(1, -1)  # возьмём первое лицо
    else:
        query_vec = np.asarray(vector, dtype="float32").reshape(1, -1)

    # Faiss expects normalised vectors for cosine (IP) search
    faiss.normalize_L2(query_vec)
    with _index_lock:
        D, I = _index.search(query_vec, k)
    matches: List[Match] = []
    for score, fid in zip(D[0], I[0]):
        if int(fid) in _meta:
            meta = _meta[int(fid)]
            matches.append(
                Match(face_id=int(fid), score=float(score), photo_id=meta["photo_id"], bbox=meta["bbox"])
            )
    return matches