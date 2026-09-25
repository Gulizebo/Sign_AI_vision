"""
╔══════════════════════════════════════════════════════╗
║   REAL-VAQTDA IMO-ISHORA TARJIMONI + O'ZBEK OVOZI  ║
║   Aqlli Imo-ishora Tili Tarjimoni — v1.0            ║
╚══════════════════════════════════════════════════════╝

Boshqaruv:
  [Q] → Chiqish
  [M] → Ovozni yoq/o'chir
  [H] → Tarix tozalash
  [+] → Ishonch chegarasini oshirish
  [-] → Ishonch chegarasini kamaytirish
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import threading
import time
import collections
from datetime import datetime

# ── Yo'llar ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE   = os.path.join(BASE_DIR, "models", "imo_ishora_model.pkl")
ENCODER_FILE = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
CONFIG_FILE  = os.path.join(BASE_DIR, "models", "config.pkl")
AUDIO_DIR    = os.path.join(BASE_DIR, "audio_cache")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ── O'zbek tilidagi tarjimalar ────────────────────────────────────────────────
TRANSLATIONS = {
    "salom"  : "Salom",
    "rahmat" : "Rahmat",
    "yordam" : "Yordam bering",
    "ha"     : "Ha",
    "yoq"    : "Yo'q",
    "suv"    : "Suv bering",
    "ovqat"  : "Ovqat bering",
    "doktor" : "Doktor chaqiring",
    "uy"     : "Uyga ketaman",
    "yaxshi" : "Yaxshi",
    "yomon"  : "Yomon",
    "kutish" : "Bir oz kuting",
    "sevaman": "Sizni sevaman",
    "kerak"  : "Menga kerak",
}

# Rang palitra
CYAN   = (255, 220, 0)
GREEN  = (0, 255, 120)
ORANGE = (0, 165, 255)
GRAY   = (150, 150, 160)
WHITE  = (240, 240, 240)
DARK   = (18, 18, 28)


# ═══════════════════════════════════════════════════════════════════════════════
class TTSEngine:
    """Asinxron gTTS + pygame ovoz chiqarish."""

    def __init__(self, lang="uz"):
        self.lang       = lang
        self.muted      = False
        self.is_playing = False
        self._cache     = {}
        self._enabled   = True

        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self._pygame = pygame
            print("[OK] Audio tizimi tayyor.")
        except Exception as e:
            print(f"[OGOHLANTIRISH] Audio yuklanmadi: {e}")
            self._enabled = False

    def _path(self, text):
        safe = "".join(c if c.isalnum() else "_" for c in text)
        return os.path.join(AUDIO_DIR, f"{safe}_{self.lang}.mp3")

    def _synthesize(self, text):
        path = self._path(text)
        if not os.path.exists(path):
            try:
                from gtts import gTTS
                gTTS(text=text, lang=self.lang, slow=False).save(path)
            except Exception as e:
                print(f"[TTS xato] {e}")
                return None
        return path

    def speak(self, key):
        if not self._enabled or self.muted or self.is_playing:
            return
        text = TRANSLATIONS.get(key, key)

        def _run():
            self.is_playing = True
            try:
                path = self._synthesize(text)
                if path and os.path.exists(path):
                    self._pygame.mixer.music.load(path)
                    self._pygame.mixer.music.play()
                    while self._pygame.mixer.music.get_busy():
                        time.sleep(0.05)
            except Exception as e:
                print(f"[Audio xato] {e}")
            finally:
                self.is_playing = False

        threading.Thread(target=_run, daemon=True).start()

    def preload(self, labels):
        if not self._enabled:
            return
        print("🔄 Audio oldindan yuklanmoqda...")
        for lbl in labels:
            txt  = TRANSLATIONS.get(lbl, lbl)
            self._synthesize(txt)
            print(f"   ✓ '{lbl}' → '{txt}'")
        print("✅ Audio tayyor!\n")

    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted


# ═══════════════════════════════════════════════════════════════════════════════
class Predictor:
    """Sliding-window barqarorlashtiruvchi prediktor."""

    WINDOW      = 12
    CONF_THRESH = 0.72

    def __init__(self):
        self.model   = joblib.load(MODEL_FILE)
        self.encoder = joblib.load(ENCODER_FILE)
        self.config  = joblib.load(CONFIG_FILE)
        self.history = collections.deque(maxlen=self.WINDOW)
        print(f"[OK] Model yuklanди: {self.config.get('model_name', '—')}")
        print(f"     Aniqlik: {self.config.get('accuracy', 0)*100:.1f}%")
        print(f"     Sinflar: {self.config.get('classes', [])}")

    def _normalize(self, landmarks):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], np.float32)
        pts -= pts[0]
        d    = np.max(np.linalg.norm(pts, axis=1))
        if d > 0:
            pts /= d
        return pts.flatten()

    def _add_features(self, raw):
        pts  = raw.reshape(21, 3)
        tips = [4, 8, 12, 16, 20]
        extra = [np.linalg.norm(pts[t] - pts[0]) for t in tips]
        for i in range(4):
            diff = pts[tips[i+1], :2] - pts[tips[i], :2]
            extra.append(np.arctan2(diff[1], diff[0]))
        return np.concatenate([raw, extra])

    def predict(self, landmarks):
        raw       = self._normalize(landmarks)
        feat      = self._add_features(raw).reshape(1, -1)
        probas    = self.model.predict_proba(feat)[0]
        idx       = np.argmax(probas)
        conf      = probas[idx]
        label     = self.encoder.classes_[idx]

        stable = None
        if conf >= self.CONF_THRESH:
            self.history.append(label)
        if len(self.history) == self.WINDOW:
            top, top_cnt = collections.Counter(self.history).most_common(1)[0]
            if top_cnt >= int(self.WINDOW * 0.75):
                stable = top

        return label, float(conf), stable

    def reset(self):
        self.history.clear()


# ═══════════════════════════════════════════════════════════════════════════════
def rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=1.0):
    """Yumaloq burchakli to'rtburchak."""
    x1, y1 = pt1
    x2, y2 = pt2
    if alpha < 1.0:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, thickness)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(overlay, (cx, cy), radius, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    else:
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(img, (cx, cy), radius, color, thickness)


def draw_ui(frame, label, conf, stable, last_stable,
            muted, log, hand_ok, fps, conf_thr):
    h, w = frame.shape[:2]

    # ── Yuqori panel ──────────────────────────────────────────────────────────
    rounded_rect(frame, (0, 0), (w, 95), DARK, radius=0, alpha=0.88)

    cv2.putText(frame, "IMO-ISHORA TARJIMONI",
                (16, 32), cv2.FONT_HERSHEY_DUPLEX, 0.75, CYAN, 2)
    cv2.putText(frame, "Aqlli | O'zbek tili | Real vaqt",
                (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1)

    # FPS chip
    fps_col = GREEN if fps > 20 else ORANGE
    rounded_rect(frame, (w-120, 12), (w-10, 48), (40,40,55), radius=8, alpha=0.9)
    cv2.putText(frame, f"FPS {fps:.0f}", (w-108, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_col, 1)

    # Ovoz holati
    mute_txt = "MUTE" if muted else "OVOZ"
    mute_col = (80, 80, 100) if muted else GREEN
    rounded_rect(frame, (w-230, 12), (w-130, 48), (40,40,55), radius=8, alpha=0.9)
    cv2.putText(frame, mute_txt, (w-216, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, mute_col, 1)

    # ── Ishonch panel ─────────────────────────────────────────────────────────
    if hand_ok and label:
        pw    = 340
        bfill = int(pw * conf)
        col   = GREEN if conf > 0.85 else (ORANGE if conf > 0.6 else (80,80,200))

        cv2.putText(frame, label.upper(),
                    (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.putText(frame, f"{conf*100:.1f}%",
                    (16+len(label)*14, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        # Ishonch bar
        rounded_rect(frame, (200, 70), (200+pw, 86), (50,50,65), radius=4, alpha=0.9)
        rounded_rect(frame, (200, 70), (200+bfill, 86), col, radius=4, alpha=0.9)

        # Chegara chizig'i
        thr_x = 200 + int(pw * conf_thr)
        cv2.line(frame, (thr_x, 67), (thr_x, 89), WHITE, 1)
    else:
        cv2.putText(frame, "Qo'lingizni kamera oldiga ko'rsating...",
                    (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1)

    # ── Markaziy natija ───────────────────────────────────────────────────────
    if last_stable:
        disp     = TRANSLATIONS.get(last_stable, last_stable).upper()
        ts       = cv2.getTextSize(disp, cv2.FONT_HERSHEY_DUPLEX, 2.8, 3)[0]
        tx, ty   = (w - ts[0])//2, h//2 + 30

        # Ko'rmas fon
        pad = 20
        rounded_rect(frame,
                     (tx - pad, ty - ts[1] - pad),
                     (tx + ts[0] + pad, ty + pad),
                     (10, 10, 20), radius=16, alpha=0.7)

        # Soya
        cv2.putText(frame, disp, (tx+3, ty+3),
                    cv2.FONT_HERSHEY_DUPLEX, 2.8, (0,0,0), 4)
        # Matn
        cv2.putText(frame, disp, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 2.8, GREEN, 3)

        # Kichik ishora nomi
        sub = f"({last_stable})"
        cv2.putText(frame, sub, (tx + ts[0]//3, ty + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GRAY, 1)

    # ── O'ng: tarix ───────────────────────────────────────────────────────────
    ox = w - 215
    rounded_rect(frame, (ox-8, 100), (w, h-42), DARK, radius=0, alpha=0.75)
    cv2.putText(frame, "▸ TARIX", (ox+2, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
    cv2.line(frame, (ox, 128), (w, 128), (60,60,80), 1)

    for i, (ts, gest) in enumerate(reversed(list(log)[-9:])):
        al  = max(0.4, 1.0 - i*0.1)
        col = tuple(int(c*al) for c in WHITE)
        cv2.putText(frame, ts, (ox+4, 148 + i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1)
        cv2.putText(frame, gest, (ox+72, 148 + i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)

    # ── Pastki panel ──────────────────────────────────────────────────────────
    rounded_rect(frame, (0, h-40), (w, h), DARK, radius=0, alpha=0.88)
    cv2.putText(frame,
                "[Q] Chiqish   [M] Ovoz   [H] Tarix   [+/-] Chegara",
                (14, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, GRAY, 1)
    cv2.putText(frame, f"Chegara: {conf_thr:.2f}",
                (w-150, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, ORANGE, 1)

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*55)
    print("  🤟  AQLLI IMO-ISHORA TARJIMONI")
    print("="*55 + "\n")

    # Model va audio yuklanishi
    predictor  = Predictor()
    tts        = TTSEngine(lang="uz")
    tts.preload(predictor.encoder.classes_)

    # MediaPipe
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    # Holat
    last_stable      = None
    last_spoken      = None
    last_spoken_time = 0
    COOLDOWN         = 2.8    # Bir xil so'z orasidagi minimum vaqt (s)
    conf_thr         = 0.72

    log       = collections.deque(maxlen=20)
    prev_time = time.time()

    # Kamera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ Kamera ochilmadi! USB/webcam tekshiring.")
        return

    print("📹 Kamera yoqildi. Imo-ishora ko'rsating!\n")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.72,
        min_tracking_confidence=0.65
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            curr  = time.time()
            fps   = 1.0 / max(curr - prev_time, 1e-5)
            prev_time = curr

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            lbl = conf = stable = None
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
                    lbl, conf, stable = predictor.predict(hl.landmark)

                    if stable and stable != last_stable:
                        last_stable = stable
                        now = time.time()
                        if (stable != last_spoken or
                                now - last_spoken_time > COOLDOWN):
                            tts.speak(stable)
                            last_spoken      = stable
                            last_spoken_time = now
                            disp = TRANSLATIONS.get(stable, stable)
                            ts   = datetime.now().strftime("%H:%M:%S")
                            log.append((ts, disp))
                            print(f"🤟 [{ts}] {stable:<12} → {disp}")
            else:
                hand_ok     = False
                last_stable = None
                predictor.reset()

            frame = draw_ui(
                frame, lbl, conf or 0, stable, last_stable,
                tts.muted, log, hand_ok, fps, conf_thr
            )
            cv2.imshow("Aqlli Imo-ishora Tarjimoni", frame)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('m'):
                m = tts.toggle_mute()
                print(f"{'🔇 Mute' if m else '🔊 Ovoz yoqiq'}")
            elif key == ord('h'):
                log.clear()
                print("🗑️  Tarix tozalandi.")
            elif key == ord('+') or key == ord('='):
                conf_thr = min(conf_thr + 0.05, 0.99)
                print(f"⬆ Chegara: {conf_thr:.2f}")
            elif key == ord('-'):
                conf_thr = max(conf_thr - 0.05, 0.30)
                print(f"⬇ Chegara: {conf_thr:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Dastur yopildi.")


if __name__ == "__main__":
    main()
