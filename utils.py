# utils.py
# ✅ 本程式負責：提供公用函式，如人體關節角度計算

import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_landmark_vector(landmarks, index):
    lm = landmarks[index]
    return [lm.x, lm.y, lm.z]