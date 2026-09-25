"""
╔══════════════════════════════════════════════════════════════════╗
║   IMO-ISHORA TARJIMONI — Grafik Interfeys (GUI)                 ║
║   Python · CustomTkinter · OpenCV · MediaPipe · Scikit-learn    ║
║   v2.0  |  2025                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys, os, threading, time, collections, warnings
import tkinter as tk
from tkinter import ttk, messagebox
warnings.filterwarnings("ignore")

# ── Paket tekshiruvi ──────────────────────────────────────────────────────────
MISSING = []
try:
    import customtkinter as ctk
except ImportError:
    MISSING.append("customtkinter")
try:
    import cv2
except ImportError:
    MISSING.append("opencv-python")
try:
    import mediapipe as mp
except Exception:
    MISSING.append("mediapipe")
try:
    import numpy as np
except ImportError:
    MISSING.append("numpy")
try:
    import joblib
except ImportError:
    MISSING.append("joblib")
try:
    from PIL import Image, ImageTk
except ImportError:
    MISSING.append("Pillow")

if MISSING:
    root = tk.Tk()
    root.withdraw()
    msg = (
        "Quyidagi paketlar o'rnatilmagan:\n\n"
        + "\n".join(f"  • {p}" for p in MISSING)
        + "\n\nO'rnatish uchun:\n"
        + "pip install " + " ".join(MISSING)
    )
    messagebox.showerror("Paket xatosi", msg)
    root.destroy()
    sys.exit(1)

import customtkinter as ctk
import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
from PIL import Image, ImageTk
from datetime import datetime

# ── Yo'llar ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE   = os.path.join(BASE_DIR, "models", "imo_ishora_model.pkl")
ENCODER_FILE = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
CONFIG_FILE  = os.path.join(BASE_DIR, "models", "config.pkl")
AUDIO_DIR        = os.path.join(BASE_DIR, "audio_cache")
DATA_FILE        = os.path.join(BASE_DIR, "data", "imo_ishoralar.csv")
HAND_MODEL_PATH  = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ── Sozlamalar o'qish ─────────────────────────────────────────────────────────
_SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
_DEFAULT_SETTINGS = {
    "lang": "uz", "conf_thr": 0.72, "cooldown": 2.5,
    "window_size": 12, "mirror": True, "landmarks": True, "theme": "dark",
}

def _load_app_settings():
    if os.path.exists(_SETTINGS_FILE):
        try:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
            return {**_DEFAULT_SETTINGS, **s}
        except Exception:
            pass
    return dict(_DEFAULT_SETTINGS)

# ── Tarjimalar ────────────────────────────────────────────────────────────────
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

# ── Rang palitra ──────────────────────────────────────────────────────────────
THEME = {
    "bg"       : "#0d1117",
    "panel"    : "#161b22",
    "card"     : "#1c2128",
    "border"   : "#30363d",
    "accent"   : "#58a6ff",
    "green"    : "#3fb950",
    "orange"   : "#d29922",
    "red"      : "#f85149",
    "purple"   : "#bc8cff",
    "text"     : "#e6edf3",
    "subtext"  : "#8b949e",
    "cam_bg"   : "#010409",
}

# ── Ekran o'lcham sababi ──────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class AudioEngine:
    def __init__(self):
        self.muted      = False
        self.is_playing = False
        self._enabled   = False
        self._lang      = "uz"
        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self._pygame  = pygame
            self._enabled = True
        except Exception as e:
            print(f"[Audio] {e}")

    def _path(self, text):
        safe = "".join(c if c.isalnum() else "_" for c in text)
        return os.path.join(AUDIO_DIR, f"{safe}_{self._lang}.mp3")

    def _synthesize(self, text):
        path = self._path(text)
        if not os.path.exists(path):
            try:
                from gtts import gTTS
                gTTS(text=text, lang=self._lang, slow=False).save(path)
            except Exception as e:
                print(f"[TTS] {e}")
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
            except Exception:
                pass
            finally:
                self.is_playing = False
        threading.Thread(target=_run, daemon=True).start()

    def preload(self, labels):
        for lbl in labels:
            self._synthesize(TRANSLATIONS.get(lbl, lbl))

    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted

    def set_lang(self, lang):
        self._lang = lang


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
class Predictor:
    WINDOW = 12

    def __init__(self):
        self.model      = joblib.load(MODEL_FILE)
        self.encoder    = joblib.load(ENCODER_FILE)
        self.config     = joblib.load(CONFIG_FILE)
        self.history    = collections.deque(maxlen=self.WINDOW)
        self.conf_thr   = 0.72

    def _normalize(self, landmarks):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], np.float32)
        pts -= pts[0]
        d    = np.max(np.linalg.norm(pts, axis=1))
        if d > 0:
            pts /= d
        return pts.flatten()

    def _add_features(self, raw):
        pts   = raw.reshape(21, 3)
        tips  = [4, 8, 12, 16, 20]
        extra = [np.linalg.norm(pts[t] - pts[0]) for t in tips]
        for i in range(4):
            diff = pts[tips[i+1], :2] - pts[tips[i], :2]
            extra.append(np.arctan2(diff[1], diff[0]))
        return np.concatenate([raw, extra])

    def predict(self, landmarks):
        raw    = self._normalize(landmarks)
        feat   = self._add_features(raw).reshape(1, -1)
        probas = self.model.predict_proba(feat)[0]
        idx    = np.argmax(probas)
        conf   = float(probas[idx])
        label  = self.encoder.classes_[idx]
        stable = None
        if conf >= self.conf_thr:
            self.history.append(label)
        if len(self.history) == self.WINDOW:
            top, cnt = collections.Counter(self.history).most_common(1)[0]
            if cnt >= int(self.WINDOW * 0.75):
                stable = top
        return label, conf, stable

    def get_all_probs(self, landmarks):
        raw    = self._normalize(landmarks)
        feat   = self._add_features(raw).reshape(1, -1)
        probas = self.model.predict_proba(feat)[0]
        return {cls: float(p) for cls, p in zip(self.encoder.classes_, probas)}

    def reset(self):
        self.history.clear()


# ══════════════════════════════════════════════════════════════════════════════
# KAMERA THREAD
# ══════════════════════════════════════════════════════════════════════════════
class CameraThread(threading.Thread):
    def __init__(self, callback, error_cb):
        super().__init__(daemon=True)
        self.callback  = callback
        self.error_cb  = error_cb
        self.running   = False
        self.paused    = False
        self._cap      = None
        self.show_landmarks = True
        self.mirror    = True

    # MediaPipe qo'l skeleti ulanishlari (Tasks API uchun)
    _HAND_CONN = [
        (0,1),(1,2),(2,3),(3,4),
        (5,6),(6,7),(7,8),
        (9,10),(10,11),(11,12),
        (13,14),(14,15),(15,16),
        (17,18),(18,19),(19,20),
        (0,5),(5,9),(9,13),(13,17),(0,17),
    ]

    def run(self):
        self.running = True
        try:
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
            self._cap.set(cv2.CAP_PROP_FPS, 30)

            if not self._cap.isOpened():
                self.error_cb("Kamera ochilmadi!")
                return

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=HAND_MODEL_PATH
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.72,
                min_hand_presence_confidence=0.65,
                min_tracking_confidence=0.5,
            )

            with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
                while self.running:
                    if self.paused:
                        time.sleep(0.05)
                        continue
                    ret, frame = self._cap.read()
                    if not ret:
                        continue
                    if self.mirror:
                        frame = cv2.flip(frame, 1)

                    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect(mp_img)

                    landmarks = None
                    if result.hand_landmarks:
                        landmarks = result.hand_landmarks[0]
                        if self.show_landmarks:
                            h_fr, w_fr = frame.shape[:2]
                            for s, e in self._HAND_CONN:
                                p1 = (int(landmarks[s].x * w_fr),
                                      int(landmarks[s].y * h_fr))
                                p2 = (int(landmarks[e].x * w_fr),
                                      int(landmarks[e].y * h_fr))
                                cv2.line(frame, p1, p2, (121, 196, 255), 2)
                            for lm in landmarks:
                                cx = int(lm.x * w_fr)
                                cy = int(lm.y * h_fr)
                                cv2.circle(frame, (cx, cy), 5, (0, 255, 128), -1)

                    self.callback(frame, landmarks)

        except Exception as e:
            self.error_cb(str(e))
        finally:
            if self._cap:
                self._cap.release()

    def stop(self):
        self.running = False

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused


# ══════════════════════════════════════════════════════════════════════════════
# ASOSIY ILOVA
# ══════════════════════════════════════════════════════════════════════════════
class ImoIshoraApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ── Oyna sozlamalari ───────────────────────────────────────────────────
        self.title("🤟  Imo-Ishora Tarjimoni  v2.0")
        self.geometry("1400x820")
        self.minsize(1100, 700)
        self.configure(fg_color=THEME["bg"])

        # ── Sozlamalar va holat o'zgaruvchilari ───────────────────────────────
        self._cfg            = _load_app_settings()
        self.predictor       = None
        self.audio           = AudioEngine()
        self.audio.set_lang(self._cfg["lang"])
        self.cam_thread      = None
        self.fps_counter     = collections.deque(maxlen=30)
        self.last_stable     = None
        self.last_spoken     = None
        self.last_spoken_t   = 0
        self.cooldown        = self._cfg["cooldown"]
        self.history_log     = []
        self.session_count   = 0
        self.running         = False
        self.cam_paused      = False
        self._frame_pending  = False
        self._photo          = None

        # Mavzu
        if self._cfg.get("theme", "dark") == "light":
            ctk.set_appearance_mode("light")

        # ── Model yuklanishi ───────────────────────────────────────────────────
        self._load_model()

        # ── UI qurish ─────────────────────────────────────────────────────────
        self._build_ui()
        self._update_status("Tayyor", "green")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────────────────────
    def _load_model(self):
        try:
            self.predictor = Predictor()
            self.audio.preload(self.predictor.encoder.classes_)
        except Exception as e:
            self.predictor = None
            print(f"[Model] {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # UI QURISH
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color=THEME["panel"],
                              corner_radius=0, height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="🤟  IMO-ISHORA TARJIMONI",
            font=ctk.CTkFont("Helvetica", 22, "bold"),
            text_color=THEME["accent"]
        ).pack(side="left", padx=24, pady=14)

        ctk.CTkLabel(
            header,
            text="AI · O'zbek tili · Real vaqt · v2.0",
            font=ctk.CTkFont("Helvetica", 12),
            text_color=THEME["subtext"]
        ).pack(side="left", padx=4, pady=14)

        # Header o'ng tugmalar
        self._mute_btn = ctk.CTkButton(
            header, text="🔊  Ovoz", width=110, height=36,
            fg_color=THEME["card"], hover_color=THEME["border"],
            border_color=THEME["green"], border_width=1,
            text_color=THEME["green"],
            font=ctk.CTkFont("Helvetica", 13, "bold"),
            command=self._toggle_mute
        )
        self._mute_btn.pack(side="right", padx=8, pady=14)

        self._theme_btn = ctk.CTkButton(
            header, text="☀  Yorug'", width=100, height=36,
            fg_color=THEME["card"], hover_color=THEME["border"],
            border_color=THEME["border"], border_width=1,
            text_color=THEME["subtext"],
            font=ctk.CTkFont("Helvetica", 12),
            command=self._toggle_theme
        )
        self._theme_btn.pack(side="right", padx=4, pady=14)

        ctk.CTkButton(
            header, text="⚙  Sozlamalar", width=120, height=36,
            fg_color=THEME["card"], hover_color=THEME["border"],
            border_color=THEME["border"], border_width=1,
            text_color=THEME["subtext"],
            font=ctk.CTkFont("Helvetica", 12),
            command=self._open_settings
        ).pack(side="right", padx=4, pady=14)

        # ── Asosiy kontent ─────────────────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color=THEME["bg"], corner_radius=0)
        body.pack(fill="both", expand=True, padx=0, pady=0)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # ── SOL USTUN: Kamera + natija ────────────────────────────────────────
        left = ctk.CTkFrame(body, fg_color=THEME["bg"], corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(12,6), pady=12)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # Kamera boshqaruv paneli
        ctrl = ctk.CTkFrame(left, fg_color=THEME["panel"],
                            corner_radius=12, height=56)
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0,8))
        ctrl.pack_propagate(False)

        self._start_btn = ctk.CTkButton(
            ctrl, text="▶  Boshlash", width=130, height=36,
            fg_color=THEME["green"], hover_color="#2ea043",
            text_color="#ffffff",
            font=ctk.CTkFont("Helvetica", 13, "bold"),
            command=self._toggle_camera
        )
        self._start_btn.pack(side="left", padx=12, pady=10)

        self._pause_btn = ctk.CTkButton(
            ctrl, text="⏸  Pauza", width=110, height=36,
            fg_color=THEME["card"], hover_color=THEME["border"],
            border_color=THEME["orange"], border_width=1,
            text_color=THEME["orange"],
            font=ctk.CTkFont("Helvetica", 13),
            state="disabled",
            command=self._toggle_pause
        )
        self._pause_btn.pack(side="left", padx=4, pady=10)

        ctk.CTkButton(
            ctrl, text="🗑  Tarix", width=100, height=36,
            fg_color=THEME["card"], hover_color=THEME["border"],
            border_color=THEME["border"], border_width=1,
            text_color=THEME["subtext"],
            font=ctk.CTkFont("Helvetica", 12),
            command=self._clear_history
        ).pack(side="left", padx=4, pady=10)

        # Chegara slider
        ctk.CTkLabel(
            ctrl, text="Chegara:",
            font=ctk.CTkFont("Helvetica", 12),
            text_color=THEME["subtext"]
        ).pack(side="left", padx=(20, 4), pady=10)

        self._thr_var = tk.DoubleVar(value=0.72)
        self._thr_slider = ctk.CTkSlider(
            ctrl, from_=0.3, to=0.99, variable=self._thr_var,
            width=140, height=16,
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            command=self._on_thr_change
        )
        self._thr_slider.pack(side="left", pady=10)

        self._thr_lbl = ctk.CTkLabel(
            ctrl, text="0.72",
            font=ctk.CTkFont("Helvetica", 12, "bold"),
            text_color=THEME["accent"], width=40
        )
        self._thr_lbl.pack(side="left", pady=10)

        # FPS va status
        self._fps_lbl = ctk.CTkLabel(
            ctrl, text="FPS: --",
            font=ctk.CTkFont("Helvetica", 12),
            text_color=THEME["subtext"], width=80
        )
        self._fps_lbl.pack(side="right", padx=12, pady=10)

        self._status_dot = ctk.CTkLabel(
            ctrl, text="●",
            font=ctk.CTkFont("Helvetica", 14),
            text_color=THEME["green"], width=20
        )
        self._status_dot.pack(side="right", padx=4, pady=10)

        self._status_lbl = ctk.CTkLabel(
            ctrl, text="Tayyor",
            font=ctk.CTkFont("Helvetica", 12),
            text_color=THEME["subtext"]
        )
        self._status_lbl.pack(side="right", padx=0, pady=10)

        # ── Kamera tasvir ─────────────────────────────────────────────────────
        cam_frame = ctk.CTkFrame(left, fg_color=THEME["cam_bg"],
                                 corner_radius=12)
        cam_frame.grid(row=1, column=0, sticky="nsew")

        self._cam_lbl = tk.Label(
            cam_frame, bg=THEME["cam_bg"],
            cursor="none"
        )
        self._cam_lbl.pack(fill="both", expand=True, padx=2, pady=2)

        # Placeholder
        self._show_placeholder()

        # ── O'NG USTUN ─────────────────────────────────────────────────────────
        right = ctk.CTkFrame(body, fg_color=THEME["bg"], corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew", padx=(6,12), pady=12)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        # ── NATIJA KARTI ──────────────────────────────────────────────────────
        res_card = ctk.CTkFrame(right, fg_color=THEME["panel"],
                                corner_radius=14)
        res_card.grid(row=0, column=0, sticky="ew", pady=(0,8))

        ctk.CTkLabel(
            res_card, text="ANIQLANGAN ISHORA",
            font=ctk.CTkFont("Helvetica", 11),
            text_color=THEME["subtext"]
        ).pack(pady=(14,4))

        self._result_lbl = ctk.CTkLabel(
            res_card, text="—",
            font=ctk.CTkFont("Helvetica", 38, "bold"),
            text_color=THEME["text"]
        )
        self._result_lbl.pack()

        self._trans_lbl = ctk.CTkLabel(
            res_card, text="",
            font=ctk.CTkFont("Helvetica", 16),
            text_color=THEME["accent"]
        )
        self._trans_lbl.pack(pady=(0,4))

        # Ishonch progress bar
        conf_frame = ctk.CTkFrame(res_card, fg_color="transparent")
        conf_frame.pack(fill="x", padx=16, pady=(4,4))

        ctk.CTkLabel(
            conf_frame, text="Ishonch:",
            font=ctk.CTkFont("Helvetica", 11),
            text_color=THEME["subtext"]
        ).pack(side="left")

        self._conf_pct_lbl = ctk.CTkLabel(
            conf_frame, text="0%",
            font=ctk.CTkFont("Helvetica", 11, "bold"),
            text_color=THEME["green"]
        )
        self._conf_pct_lbl.pack(side="right")

        self._conf_bar = ctk.CTkProgressBar(
            res_card, height=12, corner_radius=6,
            progress_color=THEME["green"],
            fg_color=THEME["card"]
        )
        self._conf_bar.pack(fill="x", padx=16, pady=(0,8))
        self._conf_bar.set(0)

        # Sessiya statistikasi
        stats_row = ctk.CTkFrame(res_card, fg_color="transparent")
        stats_row.pack(fill="x", padx=16, pady=(0,12))
        stats_row.columnconfigure((0,1,2), weight=1)

        self._sess_lbl  = self._stat_chip(stats_row, "Sessiya", "0",  0)
        self._total_lbl = self._stat_chip(stats_row, "Jami",    "0",  1)
        self._acc_lbl   = self._stat_chip(stats_row, "Aniqlik",
                           f"{self.predictor.config.get('accuracy',0)*100:.0f}%"
                           if self.predictor else "--", 2)

        # ── EHTIMOLLIK PANELI ─────────────────────────────────────────────────
        prob_card = ctk.CTkFrame(right, fg_color=THEME["panel"],
                                 corner_radius=14)
        prob_card.grid(row=1, column=0, sticky="ew", pady=(0,8))

        ctk.CTkLabel(
            prob_card, text="BARCHA ISHORALAR EHTIMOLLIGI",
            font=ctk.CTkFont("Helvetica", 11),
            text_color=THEME["subtext"]
        ).pack(pady=(12,6), padx=16, anchor="w")

        self._prob_bars = {}
        classes = (
            list(self.predictor.encoder.classes_)
            if self.predictor else list(TRANSLATIONS.keys())
        )
        for cls in classes:
            row = ctk.CTkFrame(prob_card, fg_color="transparent", height=26)
            row.pack(fill="x", padx=12, pady=1)
            row.pack_propagate(False)
            row.columnconfigure(1, weight=1)

            lbl = ctk.CTkLabel(
                row, text=cls, width=68,
                font=ctk.CTkFont("Helvetica", 11),
                text_color=THEME["text"], anchor="w"
            )
            lbl.grid(row=0, column=0, sticky="w")

            bar = ctk.CTkProgressBar(
                row, height=10, corner_radius=5,
                progress_color=THEME["accent"],
                fg_color=THEME["card"]
            )
            bar.grid(row=0, column=1, sticky="ew", padx=4)
            bar.set(0)

            pct = ctk.CTkLabel(
                row, text="0%", width=40,
                font=ctk.CTkFont("Helvetica", 10),
                text_color=THEME["subtext"], anchor="e"
            )
            pct.grid(row=0, column=2, sticky="e")

            self._prob_bars[cls] = (bar, pct)

        ctk.CTkFrame(prob_card, fg_color="transparent", height=8).pack()

        # ── TARIX ─────────────────────────────────────────────────────────────
        hist_card = ctk.CTkFrame(right, fg_color=THEME["panel"],
                                 corner_radius=14)
        hist_card.grid(row=2, column=0, sticky="nsew")
        hist_card.rowconfigure(1, weight=1)
        hist_card.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hist_card, text="TARJIMA TARIXI",
            font=ctk.CTkFont("Helvetica", 11),
            text_color=THEME["subtext"]
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12,4))

        self._hist_box = ctk.CTkScrollableFrame(
            hist_card, fg_color=THEME["card"],
            corner_radius=8, scrollbar_button_color=THEME["border"]
        )
        self._hist_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,10))
        self._hist_box.columnconfigure(0, weight=1)

        # ── PASTKI STATUS BAR ──────────────────────────────────────────────────
        footer = ctk.CTkFrame(self, fg_color=THEME["panel"],
                              corner_radius=0, height=34)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        self._foot_lbl = ctk.CTkLabel(
            footer,
            text="Imo-Ishora Tarjimoni  ·  AI | O'zbek tili | MediaPipe | Random Forest",
            font=ctk.CTkFont("Helvetica", 11),
            text_color=THEME["subtext"]
        )
        self._foot_lbl.pack(side="left", padx=16, pady=8)

        ctk.CTkLabel(
            footer,
            text="v2.0  ·  2025",
            font=ctk.CTkFont("Helvetica", 11),
            text_color=THEME["subtext"]
        ).pack(side="right", padx=16, pady=8)

    # ─────────────────────────────────────────────────────────────────────────
    def _stat_chip(self, parent, title, value, col):
        f = ctk.CTkFrame(parent, fg_color=THEME["card"], corner_radius=8)
        f.grid(row=0, column=col, padx=3, sticky="ew")
        ctk.CTkLabel(
            f, text=title,
            font=ctk.CTkFont("Helvetica", 9),
            text_color=THEME["subtext"]
        ).pack(pady=(6,0))
        lbl = ctk.CTkLabel(
            f, text=value,
            font=ctk.CTkFont("Helvetica", 15, "bold"),
            text_color=THEME["text"]
        )
        lbl.pack(pady=(0,6))
        return lbl

    # ─────────────────────────────────────────────────────────────────────────
    def _show_placeholder(self):
        """Kamera yoqilmagan holda placeholder ko'rsatish."""
        w, h  = 860, 480
        img   = Image.new("RGB", (w, h), color="#010409")
        # Markazda matn chizish (PIL orqali)
        from PIL import ImageDraw, ImageFont
        draw  = ImageDraw.Draw(img)
        draw.text((w//2, h//2 - 20), "📷", anchor="mm",
                  fill="#30363d", font=None)
        draw.text((w//2, h//2 + 40),
                  "Kamerani yoqish uchun  ▶ Boshlash  tugmasini bosing",
                  anchor="mm", fill="#30363d", font=None)
        photo = ImageTk.PhotoImage(img)
        self._cam_lbl.configure(image=photo, text="")
        self._cam_lbl.image = photo

    # ─────────────────────────────────────────────────────────────────────────
    # KAMERA BOSHQARUV
    # ─────────────────────────────────────────────────────────────────────────
    def _toggle_camera(self):
        if not self.running:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self):
        if self.predictor is None:
            messagebox.showerror(
                "Xato", "Model topilmadi!\n\n"
                "Avval src/2_model_training.py ni ishga tushiring."
            )
            return
        self.running    = True
        self.fps_counter.clear()
        self._start_btn.configure(
            text="⏹  To'xtatish",
            fg_color=THEME["red"],
            hover_color="#b62324"
        )
        self._pause_btn.configure(state="normal")
        self._update_status("Jonli", "green")

        # Sozlamalardan mirror va landmarks olish
        cfg = _load_app_settings()
        self.cam_thread = CameraThread(self._on_frame, self._on_cam_error)
        self.cam_thread.mirror          = cfg.get("mirror", True)
        self.cam_thread.show_landmarks  = cfg.get("landmarks", True)
        self.predictor.conf_thr         = cfg.get("conf_thr", 0.72)
        self.predictor.WINDOW           = cfg.get("window_size", 12)
        self.cooldown                   = cfg.get("cooldown", 2.5)
        self.audio.set_lang(cfg.get("lang", "uz"))
        self.cam_thread.start()

    def _stop_camera(self):
        self.running = False
        if self.cam_thread:
            self.cam_thread.stop()
            self.cam_thread = None
        self.predictor.reset()
        self.last_stable = None
        self._start_btn.configure(
            text="▶  Boshlash",
            fg_color=THEME["green"],
            hover_color="#2ea043"
        )
        self._pause_btn.configure(state="disabled", text="⏸  Pauza")
        self.cam_paused = False
        self._update_status("Tayyor", "green")
        self._result_lbl.configure(text="—", text_color=THEME["text"])
        self._trans_lbl.configure(text="")
        self._conf_bar.set(0)
        self._conf_pct_lbl.configure(text="0%")
        for bar, pct in self._prob_bars.values():
            bar.set(0)
            pct.configure(text="0%")
        self._fps_lbl.configure(text="FPS: --")
        self._show_placeholder()

    def _toggle_pause(self):
        if self.cam_thread:
            paused = self.cam_thread.toggle_pause()
            self.cam_paused = paused
            if paused:
                self._pause_btn.configure(text="▶  Davom")
                self._update_status("Pauza", "orange")
            else:
                self._pause_btn.configure(text="⏸  Pauza")
                self._update_status("Jonli", "green")

    def _on_cam_error(self, msg):
        self.after(0, lambda: messagebox.showerror("Kamera xatosi", msg))
        self.after(0, self._stop_camera)

    # ─────────────────────────────────────────────────────────────────────────
    # KADR QAYTA ISHLASH
    # ─────────────────────────────────────────────────────────────────────────
    def _on_frame(self, frame, landmarks):
        if self._frame_pending:
            return
        self._frame_pending = True
        self.after(0, lambda: self._process_frame(frame, landmarks))

    def _process_frame(self, frame, landmarks):
        self._frame_pending = False

        # FPS hisoblash
        now = time.time()
        self.fps_counter.append(now)
        if len(self.fps_counter) >= 2:
            fps = (len(self.fps_counter) - 1) / (
                self.fps_counter[-1] - self.fps_counter[0] + 1e-5
            )
            self._fps_lbl.configure(text=f"FPS: {fps:.0f}")

        # Kamera tasvir
        h, w = frame.shape[:2]
        # Maqsad: to'ldirish bilan resize
        disp_w = self._cam_lbl.winfo_width()  or 860
        disp_h = self._cam_lbl.winfo_height() or 480
        if disp_w > 10 and disp_h > 10:
            scale = min(disp_w / w, disp_h / h)
            nw, nh = int(w * scale), int(h * scale)
            frame_r = cv2.resize(frame, (nw, nh))
        else:
            frame_r = cv2.resize(frame, (860, 480))

        # OpenCV BGR → PIL RGB
        img   = Image.fromarray(cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(img)
        self._cam_lbl.configure(image=photo)
        self._cam_lbl.image = photo

        # Ishora aniqlash
        if landmarks and self.predictor:
            label, conf, stable = self.predictor.predict(landmarks)
            probs = self.predictor.get_all_probs(landmarks)
            self._update_prediction(label, conf, stable, probs)
        else:
            if self.predictor:
                self.predictor.reset()
            self.last_stable = None
            self._result_lbl.configure(text="—", text_color=THEME["subtext"])
            self._trans_lbl.configure(text="Qo'lingizni ko'rsating...")
            self._conf_bar.set(0)
            self._conf_pct_lbl.configure(text="0%")
            for bar, pct in self._prob_bars.values():
                bar.set(0)
                pct.configure(text="0%")

    def _update_prediction(self, label, conf, stable, probs):
        # Ishonch rangi
        if conf > 0.85:
            col = THEME["green"]
        elif conf > 0.6:
            col = THEME["orange"]
        else:
            col = THEME["red"]

        self._result_lbl.configure(text=label.upper(), text_color=col)
        self._trans_lbl.configure(
            text=TRANSLATIONS.get(label, label),
            text_color=THEME["accent"]
        )
        self._conf_bar.set(conf)
        self._conf_bar.configure(progress_color=col)
        self._conf_pct_lbl.configure(
            text=f"{conf*100:.0f}%", text_color=col
        )

        # Ehtimollik paneli yangilash
        max_p = max(probs.values()) if probs else 1.0
        for cls, (bar, pct) in self._prob_bars.items():
            p = probs.get(cls, 0.0)
            bar.set(p)
            bar.configure(
                progress_color=THEME["green"] if cls == label else THEME["accent"]
            )
            pct.configure(text=f"{p*100:.0f}%")

        # Barqaror ishora aniqlandi
        if stable and stable != self.last_stable:
            self.last_stable = stable
            t = time.time()
            if (stable != self.last_spoken or
                    t - self.last_spoken_t > self.cooldown):
                self.audio.speak(stable)
                self.last_spoken   = stable
                self.last_spoken_t = t
                self.session_count += 1
                ts   = datetime.now().strftime("%H:%M:%S")
                disp = TRANSLATIONS.get(stable, stable)
                self._add_history(ts, stable, disp)
                self._sess_lbl.configure(text=str(self.session_count))

    # ─────────────────────────────────────────────────────────────────────────
    # TARIX
    # ─────────────────────────────────────────────────────────────────────────
    def _add_history(self, ts, key, disp):
        self.history_log.append((ts, key, disp))
        n = len(self.history_log)
        self._total_lbl.configure(text=str(n))

        row = ctk.CTkFrame(
            self._hist_box, fg_color=THEME["panel"],
            corner_radius=8, height=40
        )
        row.grid(row=n-1, column=0, sticky="ew", pady=2, padx=2)
        row.pack_propagate(False)
        row.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            row, text=ts,
            font=ctk.CTkFont("Helvetica", 10),
            text_color=THEME["subtext"], width=62
        ).grid(row=0, column=0, padx=8, pady=10)

        ctk.CTkLabel(
            row, text=f"  {disp}",
            font=ctk.CTkFont("Helvetica", 13, "bold"),
            text_color=THEME["text"], anchor="w"
        ).grid(row=0, column=1, sticky="ew", pady=10)

        ctk.CTkLabel(
            row, text=key,
            font=ctk.CTkFont("Helvetica", 10),
            text_color=THEME["subtext"], width=60
        ).grid(row=0, column=2, padx=8, pady=10)

        # Eng pastga scroll
        self.after(50, lambda: self._hist_box._parent_canvas.yview_moveto(1.0))

    def _clear_history(self):
        for w in self._hist_box.winfo_children():
            w.destroy()
        self.history_log.clear()
        self.session_count = 0
        self._sess_lbl.configure(text="0")
        self._total_lbl.configure(text="0")

    # ─────────────────────────────────────────────────────────────────────────
    # STATUS
    # ─────────────────────────────────────────────────────────────────────────
    def _update_status(self, text, color):
        colors = {
            "green"  : THEME["green"],
            "orange" : THEME["orange"],
            "red"    : THEME["red"],
        }
        c = colors.get(color, THEME["subtext"])
        self._status_dot.configure(text_color=c)
        self._status_lbl.configure(text=text, text_color=c)

    # ─────────────────────────────────────────────────────────────────────────
    # SOZLAMALAR
    # ─────────────────────────────────────────────────────────────────────────
    def _on_thr_change(self, val):
        v = float(val)
        self._thr_lbl.configure(text=f"{v:.2f}")
        if self.predictor:
            self.predictor.conf_thr = v

    def _toggle_mute(self):
        muted = self.audio.toggle_mute()
        if muted:
            self._mute_btn.configure(
                text="🔇  Mute",
                border_color=THEME["subtext"],
                text_color=THEME["subtext"]
            )
        else:
            self._mute_btn.configure(
                text="🔊  Ovoz",
                border_color=THEME["green"],
                text_color=THEME["green"]
            )

    _dark_mode = True

    def _toggle_theme(self):
        self._dark_mode = not self._dark_mode
        if self._dark_mode:
            ctk.set_appearance_mode("dark")
            self._theme_btn.configure(text="☀  Yorug'")
        else:
            ctk.set_appearance_mode("light")
            self._theme_btn.configure(text="🌙  Qorong'u")

    def _open_settings(self):
        import subprocess, sys
        script = os.path.join(BASE_DIR, "gui_settings.py")
        subprocess.Popen(
            [sys.executable, script],
            creationflags=subprocess.CREATE_NO_WINDOW
            if sys.platform == "win32" else 0
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _on_close(self):
        self._stop_camera()
        self.after(200, self.destroy)


# ══════════════════════════════════════════════════════════════════════════════
# ISHGA TUSHURISH
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = ImoIshoraApp()
    app.mainloop()
