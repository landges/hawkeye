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
from typing import Any, Dict, List, Union
from retinaface import RetinaFace
from deepface import DeepFace
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

IMG = "./mvp_app/test_files/photo_2025-02-02_19-14-49.jpg"
THR = 0.9                  # RetinaFace threshold
TARGET_EYE_DIST = 100      # ширина между глазами в итоговом кропе
CROP_SIZE = 424            # итоговый квадрат для DeepFace

# ------------------ helpers ------------------
def angle_between(p1, p2):
    """угол (°) наклона линии p1-p2 относительно горизонтали"""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def align_face(img, landmarks):
    """
    landmarks: dict('left_eye':(x,y), 'right_eye':(x,y), …)
    ➜ возвращает уже выровненный квадрат 224×224 BGR
    """
    l_eye = np.array(landmarks["left_eye"])
    r_eye = np.array(landmarks["right_eye"])
    center_eye = (l_eye + r_eye) / 2

    # 1) угол поворота
    ang = angle_between(r_eye, l_eye)

    # 2) масштаб: хотим фиксированное расстояние между глазами
    cur_dist = np.linalg.norm(r_eye - l_eye)
    scale = TARGET_EYE_DIST / cur_dist

    # 3) построить аффинную матрицу (вращение + масштаб + сдвиг)
    M = cv2.getRotationMatrix2D(tuple(center_eye), ang, scale)

    # 4) сдвинем так, чтобы глаза оказались примерно посередине кропа
    tx = CROP_SIZE * 0.5 - center_eye[0]
    ty = CROP_SIZE * 0.35 - center_eye[1]  # глаза чуть выше центра
    M[0, 2] += tx
    M[1, 2] += ty

    aligned = cv2.warpAffine(img, M, (CROP_SIZE, CROP_SIZE),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    return aligned

# ------------------ main ------------------
def detect_align_embed(bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Detect all faces, align, embed. Returns list of dicts with bbox, vec."""
    result = []
    detections = RetinaFace.detect_faces(bgr, threshold=THR,)
    for info in detections.values():
        x1, y1, x2, y2 = map(int, info["facial_area"])
        lm = info["landmarks"]
        aligned = align_face(bgr, lm)
        rep = DeepFace.represent(
            img_path=aligned,
            model_name="Facenet512",
            detector_backend="skip",
            enforce_detection=False,
        )[0]["embedding"]
        vec = np.asarray(rep, dtype="float32")
        result.append({"bbox": [x1, y1, x2, y2], "vec": vec})
    return result

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

