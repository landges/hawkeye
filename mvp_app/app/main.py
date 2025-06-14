import os
import uuid
import threading
import pickle
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from app.utils import _read_image
from app.face_pipeline import detect_align_embed
from deepface import DeepFace

# ——— Логирование ———
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ——— Константы и инициализация ———
DATA_DIR   = Path(os.getenv("DATA_DIR", "data"))
INDEX_PATH = DATA_DIR / "faces.index"
META_PATH  = DATA_DIR / "faces_meta.pkl"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_index_lock = threading.Lock()

# Загружаем или создаём FAISS индекс
if INDEX_PATH.exists():
    _index = faiss.read_index(str(INDEX_PATH))
    logging.info("Loaded FAISS index from disk")
else:
    base   = faiss.IndexFlatIP(512)
    _index = faiss.IndexIDMap2(base)
    logging.info("Initialized new FAISS index")

# Загружаем или создаём метаданные
if META_PATH.exists():
    with open(META_PATH, "rb") as f:
        _meta: Dict[int, Dict[str, Any]] = pickle.load(f)
    logging.info("Loaded metadata store from disk")
else:
    _meta = {}
    logging.info("Initialized new metadata store")

# Предзагружаем модель эмбеддингов
_embedder = DeepFace.build_model("Facenet512")
app = FastAPI(title="Face-Search API")

# ——— Pydantic модели ———
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
    bbox: List[int]         # bbox из индексированных метаданных
    query_bbox: List[int]   # bbox, где лицо найдено в запросе

# ——— Утилиты ———
def gen_face_id() -> int:
    """63-bit positive integer"""
    return uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF

def save_state():
    """Сохраняем индекс и метаданные атомарно"""
    with _index_lock:
        faiss.write_index(_index, str(INDEX_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump(_meta, f)
    logging.debug("State saved to disk")

# ——— Эндпоинты ———

@app.post("/detect", response_model=FaceAddResponse)
async def detect_and_add(file: UploadFile = File(...)):
    """Детектим на изображении все лица и добавляем их в индекс."""
    data = await file.read()
    img  = _read_image(data)
    faces = detect_align_embed(img)
    if not faces:
        raise HTTPException(400, "No faces found in image")

    photo_id = uuid.uuid4().hex
    added: List[FaceMeta] = []

    with _index_lock:
        for item in faces:
            vec = item["vec"].astype("float32").reshape(1, -1)
            faiss.normalize_L2(vec)
            face_id = gen_face_id()
            _index.add_with_ids(vec, np.array([face_id], dtype="int64"))
            _meta[face_id] = {
                "photo_id": photo_id,
                "bbox": item["bbox"]
            }
            added.append(FaceMeta(
                face_id=face_id,
                bbox=item["bbox"],
                photo_id=photo_id
            ))
        save_state()

    logging.info(f"Added {len(added)} faces from photo {photo_id}")
    return FaceAddResponse(photo_id=photo_id, faces=added)


@app.post("/search", response_model=List[Match])
async def search_faces(
    file: UploadFile = File(None),
    vector: Optional[str] = Form(None),
    k: int = Form(5),
    auto_add: bool = Form(True),
):
    """
    Ищем по изображению или по вектору.
    Если auto_add=True и нет совпадений с score>=0.6, добавляем как новый face_id.
    """
    if bool(file) == bool(vector):
        raise HTTPException(422, "Provide exactly one of file or vector")

    # Подготавливаем запросы (может быть несколько лиц на картинке)
    results: List[Match] = []

    if file:
        data = await file.read()
        img  = _read_image(data)
        faces = detect_align_embed(img)
        if not faces:
            raise HTTPException(400, "No face detected in query image")

        for item in faces:                       # <--- цикл по всем лицам
            vec  = item["vec"].astype("float32").reshape(1, -1)
            qbox = item["bbox"]

            faiss.normalize_L2(vec)
            D, I = _index.search(vec, k)
            valid = (I[0][0] != -1) and (D[0][0] >= 0.75)
            if auto_add and not valid:
                new_id = gen_face_id()
                with _index_lock:
                    _index.add_with_ids(vec, np.array([new_id], dtype="int64"))
                    _meta[new_id] = {"photo_id": uuid.uuid4().hex, "bbox": qbox}
                save_state()
                results.append(Match(face_id=new_id, score=1.0,
                                    photo_id=_meta[new_id]["photo_id"],
                                    bbox=qbox, query_bbox=qbox))
                continue

            for score, fid in zip(D[0], I[0]):
                fid = int(fid)
                if fid in _meta:
                    m = _meta[fid]
                    results.append(Match(face_id=fid, score=float(score),
                                        photo_id=m["photo_id"],
                                        bbox=m["bbox"], query_bbox=qbox))

    return results