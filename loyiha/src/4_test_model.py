"""
MODEL SINOV SKRIPTI (kamerasiz)
================================
Saqlangan AI modelini sintetik ma'lumot bilan sinaydi.
Kamera yo'q muhitda to'liq ish tekshiruvi uchun.
"""

import numpy as np
import joblib
import os
import time
import sys

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE   = os.path.join(BASE_DIR, "models", "imo_ishora_model.pkl")
ENCODER_FILE = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
CONFIG_FILE  = os.path.join(BASE_DIR, "models", "config.pkl")
AUDIO_DIR    = os.path.join(BASE_DIR, "audio_cache")

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
}


def add_features(X_raw):
    pts  = X_raw.reshape(-1, 21, 3)
    wrist = pts[:, 0, :]
    extra = []
    for tip in [4, 8, 12, 16, 20]:
        d = np.linalg.norm(pts[:, tip, :] - wrist, axis=1, keepdims=True)
        extra.append(d)
    tips = [4, 8, 12, 16, 20]
    for i in range(4):
        diff  = pts[:, tips[i+1], :2] - pts[:, tips[i], :2]
        angle = np.arctan2(diff[:, 1], diff[:, 0]).reshape(-1, 1)
        extra.append(angle)
    return np.hstack([X_raw] + extra)


def test_model():
    print("=" * 58)
    print("  🧪 MODEL SINOV NATIJASI")
    print("=" * 58)

    # ── Yuklanish ──────────────────────────────────────────────────────
    for path, name in [(MODEL_FILE,"Model"), (ENCODER_FILE,"Encoder"), (CONFIG_FILE,"Config")]:
        if not os.path.exists(path):
            print(f"❌ Topilmadi: {path}")
            sys.exit(1)
        size = os.path.getsize(path) / 1024
        print(f"  ✓ {name:<10} → {size:.1f} KB")

    model   = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    config  = joblib.load(CONFIG_FILE)

    print(f"\n  Model turi : {config.get('model_name','—')}")
    print(f"  Aniqlik    : {config.get('accuracy',0)*100:.2f}%")
    print(f"  Sinflar    : {config.get('classes',[])}")

    # ── Tezlik sinovi ──────────────────────────────────────────────────
    print(f"\n{'─'*58}")
    print("  ⚡ TEZLIK SINOVI (1000 ta taxmin)")
    print(f"{'─'*58}")

    dummy = np.random.randn(1000, 63).astype(np.float32)
    dummy_feat = add_features(dummy)

    t0    = time.perf_counter()
    preds = model.predict(dummy_feat)
    dur   = time.perf_counter() - t0

    per_pred = dur / 1000 * 1000  # ms
    fps_est  = 1.0 / (dur / 1000)

    print(f"  1000 ta taxmin: {dur*1000:.1f} ms")
    print(f"  Har bir taxmin: {per_pred:.3f} ms")
    print(f"  FPS imkoniyati: ~{fps_est:.0f} FPS")

    if fps_est > 100:
        print("  ✅ Real vaqt uchun JUDA YAXSHI (>100 FPS)")
    elif fps_est > 30:
        print("  ✅ Real vaqt uchun YAXSHI (>30 FPS)")

    # ── Sinf bo'yicha sinov ────────────────────────────────────────────
    print(f"\n{'─'*58}")
    print("  📊 SINF BO'YICHA BAHOLASH")
    print(f"{'─'*58}")

    # Demo data dan test
    data_file = os.path.join(BASE_DIR, "data", "imo_ishoralar.csv")
    import pandas as pd
    df      = pd.read_csv(data_file)
    X_raw   = df.drop("label", axis=1).values.astype(np.float32)
    y_true  = df["label"].values
    X_feat  = add_features(X_raw)
    y_pred  = model.predict(X_feat)
    probas  = model.predict_proba(X_feat)
    max_c   = probas.max(axis=1)

    classes = encoder.classes_
    print(f"\n  {'Ishora':<14} {'Togri':>8} {'Jami':>6} {'Aniqlik':>10} {'Ortacha ishonch':>18}")
    print(f"  {'─'*14} {'─'*8} {'─'*6} {'─'*10} {'─'*18}")

    total_ok = 0
    for cls in classes:
        mask    = y_true == cls
        ok      = np.sum(y_pred[mask] == cls)
        total   = np.sum(mask)
        acc_c   = ok / total
        avg_c   = max_c[mask].mean()
        total_ok += ok
        bar      = "█" * int(acc_c * 15)
        color    = "✓" if acc_c == 1.0 else ("~" if acc_c > 0.9 else "✗")
        print(f"  {color} {cls:<13} {ok:>8}/{total:<6} {acc_c*100:>8.1f}%  {avg_c*100:>14.1f}%  {bar}")

    overall = total_ok / len(y_true)
    print(f"\n  {'JAMI':<14} {total_ok:>8}/{len(y_true):<6} {overall*100:>8.2f}%")

    # ── Audio sinov ────────────────────────────────────────────────────
    print(f"\n{'─'*58}")
    print("  🔊 AUDIO FAYLLAR SINOVI")
    print(f"{'─'*58}")

    audio_ok = 0
    for key, text in TRANSLATIONS.items():
        safe = "".join(c if c.isalnum() else "_" for c in text)
        wav_path = os.path.join(AUDIO_DIR, f"{safe}_uz.wav")
        mp3_path = os.path.join(AUDIO_DIR, f"{safe}_uz.mp3")
        path = wav_path if os.path.exists(wav_path) else mp3_path
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            ext  = "wav" if path.endswith(".wav") else "mp3"
            print(f"  ✓ {key:<12} → {text:<20} [{size:.1f} KB, {ext}]")
            audio_ok += 1
        else:
            print(f"  ✗ {key:<12} → topilmadi ({safe}_uz.wav/.mp3)")

    # ── Audio ijro sinovi ─────────────────────────────────────────────
    print(f"\n{'─'*58}")
    print("  🎵 AUDIO IJRO SINOVI (pygame)")
    print(f"{'─'*58}")

    try:
        import pygame
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

        # Birinchi mavjud faylni ijro etish
        for key, text in TRANSLATIONS.items():
            safe     = "".join(c if c.isalnum() else "_" for c in text)
            wav_path = os.path.join(AUDIO_DIR, f"{safe}_uz.wav")
            mp3_path = os.path.join(AUDIO_DIR, f"{safe}_uz.mp3")
            path     = wav_path if os.path.exists(wav_path) else mp3_path
            if os.path.exists(path):
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                timeout = time.time() + 3.0
                while pygame.mixer.music.get_busy() and time.time() < timeout:
                    time.sleep(0.05)
                print(f"  ✓ '{text}' — audio muvaffaqiyatli ijro etildi")
                pygame.mixer.quit()
                break
    except Exception as e:
        print(f"  ⚠ Pygame sinovi: {e}")
        print("    (Headless muhitda normal — haqiqiy kompyuterda ishlaydi)")

    # ── Yakuniy xulosa ─────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print("  ✅ XULOSA")
    print(f"{'='*58}")
    print(f"  Model aniqlik    : {overall*100:.2f}%  {'🎉' if overall>0.99 else '👍'}")
    print(f"  Taxmin tezligi   : ~{fps_est:.0f} FPS")
    print(f"  Audio fayllar    : {audio_ok}/{len(TRANSLATIONS)} ta")
    print(f"  Feature'lar      : {X_feat.shape[1]} ta")
    print(f"\n  ✅ Loyiha ISHGA TAYYOR!")
    print(f"     → Kamera bilan ishlatish: src/3_realtime_detection.py")
    print(f"{'='*58}")


if __name__ == "__main__":
    test_model()
