"""
Demo uchun sintetik ma'lumot generatori.
Haqiqiy loyihada 1_data_collection.py dan foydalaning.
Bu fayl — kamerasiz test qilish uchun.
"""
import numpy as np
import pandas as pd
import os

GESTURES = {
    "salom":    {"wrist_angle": 0.0,   "spread": 0.85, "curl": 0.1},
    "rahmat":   {"wrist_angle": 0.3,   "spread": 0.70, "curl": 0.2},
    "yordam":   {"wrist_angle": -0.2,  "spread": 0.90, "curl": 0.05},
    "ha":       {"wrist_angle": 0.5,   "spread": 0.40, "curl": 0.7},
    "yoq":      {"wrist_angle": -0.4,  "spread": 0.50, "curl": 0.65},
    "suv":      {"wrist_angle": 0.1,   "spread": 0.30, "curl": 0.8},
    "ovqat":    {"wrist_angle": 0.2,   "spread": 0.60, "curl": 0.45},
    "doktor":   {"wrist_angle": -0.1,  "spread": 0.75, "curl": 0.3},
    "uy":       {"wrist_angle": 0.6,   "spread": 0.55, "curl": 0.55},
    "yaxshi":   {"wrist_angle": -0.3,  "spread": 0.45, "curl": 0.6},
}

# MediaPipe qo'l skeleti: 21 ta nuqta
HAND_SKELETON = np.array([
    [0.0,  0.0,  0.0],   # 0  wrist
    [0.1,  0.05, 0.0],   # 1  thumb_cmc
    [0.15, 0.10, 0.0],   # 2  thumb_mcp
    [0.18, 0.15, 0.0],   # 3  thumb_ip
    [0.20, 0.20, 0.0],   # 4  thumb_tip
    [0.05, 0.25, 0.0],   # 5  index_mcp
    [0.05, 0.38, 0.0],   # 6  index_pip
    [0.05, 0.48, 0.0],   # 7  index_dip
    [0.05, 0.55, 0.0],   # 8  index_tip
    [0.00, 0.26, 0.0],   # 9  middle_mcp
    [0.00, 0.40, 0.0],   # 10 middle_pip
    [0.00, 0.50, 0.0],   # 11 middle_dip
    [0.00, 0.57, 0.0],   # 12 middle_tip
    [-0.05,0.25, 0.0],   # 13 ring_mcp
    [-0.05,0.38, 0.0],   # 14 ring_pip
    [-0.05,0.48, 0.0],   # 15 ring_dip
    [-0.05,0.54, 0.0],   # 16 ring_tip
    [-0.10,0.23, 0.0],   # 17 pinky_mcp
    [-0.10,0.34, 0.0],   # 18 pinky_pip
    [-0.10,0.42, 0.0],   # 19 pinky_dip
    [-0.10,0.48, 0.0],   # 20 pinky_tip
], dtype=np.float32)


def simulate_gesture(params, n_samples=350, noise=0.018):
    """
    Har bir ishora uchun biomexanik jihatdan ishonchli qo'l pozitsiyalarini
    simulyatsiya qiladi.
    """
    spread   = params["spread"]
    curl     = params["curl"]
    angle    = params["wrist_angle"]

    samples = []
    for _ in range(n_samples):
        pts = HAND_SKELETON.copy()

        # Rotation matrix (bilak atrofida)
        c, s  = np.cos(angle), np.sin(angle)
        rot   = np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]])
        pts   = pts @ rot.T

        # Barmoq yoyilishi
        finger_groups = [[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
        for i, group in enumerate(finger_groups):
            x_offset = (i - 1.5) * 0.04 * spread
            for j in group:
                pts[j, 0] += x_offset

        # Barmoq bukilishi (curl)
        fingertips = [4, 8, 12, 16, 20]
        for tip in fingertips:
            pts[tip, 1] -= curl * 0.25
            pts[tip, 0] += curl * 0.08

        # Normalizatsiya
        pts -= pts[0]
        d    = np.max(np.linalg.norm(pts, axis=1))
        if d > 0:
            pts /= d

        # Shovqin qo'shish (real kamera effekti)
        pts += np.random.normal(0, noise, pts.shape)

        samples.append(pts.flatten())

    return np.array(samples)


def generate_dataset(output_path, samples_per_class=350):
    print("🔄 Sintetik ma'lumot generatsiya qilinmoqda...")
    print(f"   Sinflar: {len(GESTURES)} ta")
    print(f"   Har biri: {samples_per_class} ta namuna\n")

    header = ["label"] + [
        f"{c}_{i}" for i in range(21) for c in ["x","y","z"]
    ]

    rows = []
    for gesture, params in GESTURES.items():
        data = simulate_gesture(params, n_samples=samples_per_class)
        for row in data:
            rows.append([gesture] + row.tolist())
        print(f"   ✓ '{gesture}' → {samples_per_class} ta namuna")

    df = pd.DataFrame(rows, columns=header)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    total = len(df)
    print(f"\n✅ CSV saqlandi: {output_path}")
    print(f"   Jami: {total} ta namuna  |  {len(GESTURES)} ta ishora")
    return df


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out  = os.path.join(base, "data", "imo_ishoralar.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    generate_dataset(out)
