"""
╔══════════════════════════════════════════════════════╗
║   IMO-ISHORA MA'LUMOT YIG'ISH DASTURI               ║
║   Aqlli Imo-ishora Tili Tarjimoni — v1.0            ║
╚══════════════════════════════════════════════════════╝

Boshqaruv tugmalari:
  [S] → Yozishni BOSHLASH
  [P] → PAUZA
  [N] → Yangi imo-ishora (nom kiritish)
  [Q] → CHIQish
"""

import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from datetime import datetime

# ── Yo'llar ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_FILE = os.path.join(DATA_DIR, "imo_ishoralar.csv")
os.makedirs(DATA_DIR, exist_ok=True)

HEADER = ["label"] + [
    f"{coord}_{i}" for i in range(21) for coord in ["x", "y", "z"]
]

MIN_SAMPLES_PER_CLASS = 300

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles


def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)
        print(f"[OK] Yangi CSV: {CSV_FILE}")
    else:
        count = sum(1 for _ in open(CSV_FILE)) - 1
        print(f"[OK] Mavjud CSV: {count} ta namuna")


def normalize(landmarks):
    """Bilagni markazga olib, o'lchamni 0-1 ga keltiradi."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    pts -= pts[0]
    d = np.max(np.linalg.norm(pts, axis=1))
    if d > 0:
        pts /= d
    return pts.flatten().tolist()


def get_stats():
    stats = {}
    if not os.path.exists(CSV_FILE):
        return stats
    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                stats[row[0]] = stats.get(row[0], 0) + 1
    return stats


def draw_panel(frame, label, recording, hand_ok, sample_count, total, stats):
    h, w = frame.shape[:2]

    # Yuqori panel
    cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 30), -1)
    cv2.line(frame, (0, 100), (w, 100), (60, 60, 80), 2)

    title_color = (0, 200, 255)
    cv2.putText(frame, "IMO-ISHORA MA'LUMOT YIG'ISH",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, title_color, 2)

    if recording and hand_ok:
        st, sc = "● YOZILMOQDA", (0, 255, 120)
    elif recording:
        st, sc = "⚠  QO'L KO'RINMAYDI", (0, 140, 255)
    else:
        st, sc = "⏸  PAUZA", (120, 120, 120)

    cv2.putText(frame, st, (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2)
    cv2.putText(frame, f"Ishora: {label.upper()}",
                (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Sessiya: {sample_count}  |  Jami: {total}",
                (w - 300, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Progress bar
    progress  = min(total / MIN_SAMPLES_PER_CLASS, 1.0)
    bar_w     = int((w - 30) * progress)
    cv2.rectangle(frame, (15, 78), (w - 15, 90), (50, 50, 60), -1)
    cv2.rectangle(frame, (15, 78), (15 + bar_w, 90), sc, -1)
    cv2.putText(frame, f"{int(progress*100)}%",
                (w - 50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

    # O'ng panel — statistika
    ox = w - 210
    cv2.rectangle(frame, (ox - 5, 105), (w, h - 40), (15, 15, 25), -1)
    cv2.putText(frame, "STATISTIKA", (ox, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 180, 255), 1)
    for i, (lbl, cnt) in enumerate(list(stats.items())[-10:]):
        done = "✓" if cnt >= MIN_SAMPLES_PER_CLASS else " "
        col  = (0, 220, 120) if cnt >= MIN_SAMPLES_PER_CLASS else (180, 180, 180)
        cv2.putText(frame, f"{done} {lbl:<10} {cnt:>4}",
                    (ox, 148 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)

    # Pastki panel
    cv2.rectangle(frame, (0, h - 38), (w, h), (20, 20, 30), -1)
    cv2.putText(frame, "[S] Boshlash   [P] Pauza   [N] Yangi ishora   [Q] Chiqish",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (120, 120, 140), 1)

    return frame


def main():
    init_csv()

    print("\n" + "="*52)
    stats = get_stats()
    print("Mavjud ishoralar:")
    for lbl, cnt in stats.items():
        mark = "✓" if cnt >= MIN_SAMPLES_PER_CLASS else "○"
        print(f"  {mark} {lbl:<15} {cnt:>4} ta namuna")

    print("\nMisol: salom, rahmat, yordam, ha, yoq")
    label = input("Birinchi imo-ishora nomi: ").strip().lower()
    if not label:
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording    = False
    sample_count = 0
    total        = stats.get(label, 0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame      = cv2.flip(frame, 1)
            rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results    = hands.process(rgb)
            rgb.flags.writeable = True

            hand_ok = False
            if results.multi_hand_landmarks:
                hand_ok = True
                for hl in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hl,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                    if recording:
                        row = [label] + normalize(hl.landmark)
                        with open(CSV_FILE, "a", newline="") as f:
                            csv.writer(f).writerow(row)
                        sample_count += 1
                        total        += 1

            frame = draw_panel(
                frame, label, recording,
                hand_ok, sample_count, total, get_stats()
            )
            cv2.imshow("Imo-ishora Ma'lumot Yig'ish", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                recording = True
                print(f"▶ Yozish boshlandi: '{label}'")
            elif key == ord('p'):
                recording = False
                print(f"⏸ Pauza — {sample_count} ta yozildi")
            elif key == ord('n'):
                recording = False
                stats     = get_stats()
                print(f"\n'{label}' → {sample_count} ta yangi namuna")
                label        = input("Yangi ishora nomi: ").strip().lower()
                sample_count = 0
                total        = stats.get(label, 0)

    cap.release()
    cv2.destroyAllWindows()
    stats = get_stats()
    print(f"\n✅ Yakunlandi. CSV: {CSV_FILE}")
    for lbl, cnt in stats.items():
        print(f"   {lbl}: {cnt} ta")


if __name__ == "__main__":
    main()
