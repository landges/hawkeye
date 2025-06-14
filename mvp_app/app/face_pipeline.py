# from retinaface import RetinaFace
# from deepface   import DeepFace
# import cv2, numpy as np, math, matplotlib.pyplot as plt

# IMG_PATH  = "./mvp_app/test_files/photo_2025-02-02_19-14-49.jpg"
# THRESHOLD = 0.9
# FACENET_IN = 160                       # Facenet512 ждёт ≥160×160

# # ──────────── вспомогательная функция ────────────
# def align_by_eyes(bgr_img, landmarks, output_size=FACENET_IN, scale=1.4):
#     """
#     Выравниваем лицо по двум глазам, возвращаем квадрат output_size × output_size (BGR).
#     scale > 1  – сколько «запаса» брать вокруг глаз (1.4 ≈ чётко по подбородку).
#     """
#     le = np.array(landmarks["left_eye"])   # (x,y)
#     re = np.array(landmarks["right_eye"])

#     # середина глаз и угол
#     eye_center = (le + re) / 2
#     dx, dy = re - le
#     angle = math.degrees(math.atan2(dy, dx))

#     # расстояние между глазами → сторона квадрата
#     dist = np.sqrt(dx*dx + dy*dy)
#     side = int(dist * scale)

#     # матрица вращения вокруг eye_center
#     rot_M = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.0)

#     # сдвигаем так, чтобы глазной центр стал серединой квадрата
#     rot_M[0, 2] += output_size/2 - eye_center[0]
#     rot_M[1, 2] += output_size/2 - eye_center[1]

#     # обрезаем / дополняем до квадратного output_size
#     aligned = cv2.warpAffine(
#         bgr_img, rot_M, (output_size, output_size),
#         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
#     )
#     return aligned              # BGR, готов к DeepFace

# # ──────────── основной код ────────────
# bgr = cv2.imread(IMG_PATH)
# det = RetinaFace.detect_faces(IMG_PATH, threshold=THRESHOLD)

# embeddings, boxes, aligned_faces = [], [], []

# embedder = DeepFace.build_model("Facenet512")

# for idx, info in det.items():
#     x1,y1,x2,y2 = info["facial_area"]
#     lm = info["landmarks"]                 # dict: left_eye/right_eye/nose/mouth_l/mouth_r

#     # 1) выравнивание
#     aligned = align_by_eyes(bgr, lm)
#     aligned_faces.append(aligned)
#     # 2) эмбеддинг без детектора
#     vec = DeepFace.represent(
#         img_path=aligned,
#         model_name="Facenet512",
#         detector_backend="skip",
#         enforce_detection=False
#     )[0]["embedding"]

#     embeddings.append(vec)
#     boxes.append([x1,y1,x2,y2])

# # numpy-масивы для дальнейшей обработки или сохранения
# embeddings = np.asarray(embeddings, dtype="float32")   # (n_faces, 512)
# boxes      = np.asarray(boxes,      dtype="int32")     # (n_faces, 4)

# print(f"Лиц найдено: {len(boxes)}, shape эмбеддингов: {embeddings.shape}")

# # ──────────── визуальный контроль ────────────
# for (x1,y1,x2,y2) in boxes:
#     cv2.rectangle(bgr, (x1,y1), (x2,y2), (0,255,0), 2)
# plt.figure(figsize=(7,5)); plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); plt.axis("off")
# plt.show()
# detect_align_embed.py  (файл с помощью/детекцией)

from typing import Any, Dict, List
from retinaface import RetinaFace
from deepface import DeepFace
import cv2, numpy as np, math

THR            = 0.9      # RetinaFace score threshold
MIN_FACE       = 20       # px: игнорируем лица мельче
MARGIN         = 0.25     # доля bbox добавляется по всем сторонам
TARGET_EYE_DIST = 100
CROP_SIZE       = 224     # итоговый квадрат для Facenet512

# ---------- helpers ----------
def angle_between(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def align_face(face_bgr: np.ndarray, landmarks):
    """Выравнивает один кроп лица → квадрат CROP_SIZE×CROP_SIZE (BGR)."""
    l_eye = np.array(landmarks["left_eye"])
    r_eye = np.array(landmarks["right_eye"])
    center_eye = (l_eye + r_eye) / 2

    # угол
    ang = angle_between(r_eye, l_eye)
    # масштаб
    cur_dist = np.linalg.norm(r_eye - l_eye)
    scale = TARGET_EYE_DIST / cur_dist

    # аффин-матрица
    M = cv2.getRotationMatrix2D(tuple(center_eye), ang, scale)

    # сдвигаем, чтобы глаза оказались ~35 % сверху
    M[0, 2] += CROP_SIZE * 0.5 - center_eye[0]
    M[1, 2] += CROP_SIZE * 0.35 - center_eye[1]

    return cv2.warpAffine(
        face_bgr, M, (CROP_SIZE, CROP_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

# ---------- main ----------
def detect_align_embed(bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Возвращает [{'bbox':[x1,y1,x2,y2], 'vec':512f32}, ...]"""
    H, W = bgr.shape[:2]
    dets = RetinaFace.detect_faces(bgr, threshold=THR) or {}
    out: List[Dict[str, Any]] = []

    for info in dets.values():
        x1, y1, x2, y2 = map(int, info["facial_area"])
        w, h = x2 - x1, y2 - y1
        if w < MIN_FACE or h < MIN_FACE:
            continue                               # слишком маленькое лицо

        # расширяем bbox на MARGIN
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w2, h2 = int(w * (1 + MARGIN) / 2), int(h * (1 + MARGIN) / 2)
        x1e, y1e = max(0, cx - w2), max(0, cy - h2)
        x2e, y2e = min(W, cx + w2), min(H, cy + h2)

        # вырезаем кроп и сдвигаем landmarks
        face_crop = bgr[y1e:y2e, x1e:x2e].copy()
        lm_shifted = {
            k: (np.array(v) - np.array([x1e, y1e]))
            for k, v in info["landmarks"].items()
        }

        aligned = align_face(face_crop, lm_shifted)

        rep = DeepFace.represent(
            img_path=aligned,
            model_name="Facenet512",
            detector_backend="skip",
            enforce_detection=False,
        )[0]["embedding"]

        out.append({
            "bbox": [x1, y1, x2, y2],
            "vec":  np.asarray(rep, dtype="float32")
        })

    return out


# ----- в удобные массивы -----
# embeddings = np.asarray(embeddings, dtype="float32")  # (n,512)
# boxes      = np.asarray(boxes, dtype="int32")         # (n,4)

# print(f"Лиц: {len(boxes)};   Embeddings: {embeddings.shape}")

# # ---------- визуальная проверка ----------
# vis = bgr.copy()
# for (x1,y1,x2,y2) in boxes:
#     cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
# plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.show()
# n = len(aligned_faces)
# cols = min(5, n)                         # макс 5 кол-онок
# rows = math.ceil(n / cols)

# fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
# if rows == 1: axes = np.asarray(axes).reshape(1, -1)    # всегда 2-D

# for i, face in enumerate(aligned_faces):
#     r, c = divmod(i, cols)
#     ax = axes[r, c]
#     ax.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
#     ax.set_title(f"face #{i}", fontsize=10)
#     ax.axis("off")

# # выключаем пустые ячейки, если лиц < rows*cols
# for j in range(n, rows*cols):
#     r, c = divmod(j, cols)
#     axes[r, c].axis("off")

# plt.tight_layout()
# plt.show()

