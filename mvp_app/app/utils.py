import cv2
import numpy as np


def _read_image(bytes_):
    file_bytes = np.frombuffer(bytes_, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not an image")
    return img