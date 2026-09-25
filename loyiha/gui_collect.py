"""
╔══════════════════════════════════════════════════════╗
║   MA'LUMOT YIG'ISH — Grafik Interfeys               ║
║   Kamera bilan real vaqtda ishora yig'ish            ║
╚══════════════════════════════════════════════════════╝
"""

import sys, os, threading, time, csv, warnings
import tkinter as tk
from tkinter import messagebox, simpledialog
warnings.filterwarnings("ignore")

try:
    import customtkinter as ctk
    import cv2
    import mediapipe as mp
    import numpy as np
    from PIL import Image, ImageTk
except Exception as e:
    tk.Tk().withdraw()
    messagebox.showerror("Paket xatosi", f"Kerakli paket topilmadi:\n{e}\n\npip install customtkinter opencv-python mediapipe Pillow")
    sys.exit(1)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
CSV_FILE        = os.path.join(DATA_DIR, "imo_ishoralar.csv")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
os.makedirs(DATA_DIR, exist_ok=True)

HEADER = ["label"] + [f"{c}_{i}" for i in range(21) for c in ["x","y","z"]]
MIN_SAMPLES = 300

THEME = {
    "bg": "#0d1117", "panel": "#161b22", "card": "#1c2128",
    "border": "#30363d", "accent": "#58a6ff", "green": "#3fb950",
    "orange": "#d29922", "red": "#f85149", "text": "#e6edf3",
    "subtext": "#8b949e",
}

GESTURES_LIST = [
    "salom","rahmat","yordam","ha","yoq",
    "suv","ovqat","doktor","uy","yaxshi"
]


def normalize(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], np.float32)
    pts -= pts[0]
    d = np.max(np.linalg.norm(pts, axis=1))
    if d > 0:
        pts /= d
    return pts.flatten().tolist()


def get_stats():
    stats = {}
    if not os.path.exists(CSV_FILE):
        return stats
    try:
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    stats[row[0]] = stats.get(row[0], 0) + 1
    except Exception:
        pass
    return stats


def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)


class CollectApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("📊  Imo-Ishora Ma'lumot Yig'ish")
        self.geometry("1280x740")
        self.minsize(1000, 620)
        self.configure(fg_color=THEME["bg"])

        self.current_label  = tk.StringVar(value=GESTURES_LIST[0])
        self.recording      = False
        self.running        = False
        self.cam_thread     = None
        self._frame_pending = False
        self.session_count  = 0
        self._cap           = None
        self._landmarks_buf = None

        init_csv()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=0, height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="📊  MA'LUMOT YIG'ISH",
                     font=ctk.CTkFont("Helvetica", 20, "bold"),
                     text_color=THEME["accent"]).pack(side="left", padx=20, pady=14)
        ctk.CTkLabel(hdr, text="Kamera bilan qo'l ishoralarini CSV ga yozish",
                     font=ctk.CTkFont("Helvetica", 12),
                     text_color=THEME["subtext"]).pack(side="left", padx=4)

        # Body
        body = ctk.CTkFrame(self, fg_color=THEME["bg"], corner_radius=0)
        body.pack(fill="both", expand=True, padx=12, pady=8)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # ── SOL: Kamera ───────────────────────────────────────────────────────
        left = ctk.CTkFrame(body, fg_color=THEME["bg"], corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # Ctrl
        ctrl = ctk.CTkFrame(left, fg_color=THEME["panel"], corner_radius=10, height=52)
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0,6))
        ctrl.pack_propagate(False)

        self._cam_btn = ctk.CTkButton(
            ctrl, text="▶  Kamerani yoq", width=150, height=36,
            fg_color=THEME["green"], hover_color="#2ea043",
            font=ctk.CTkFont("Helvetica", 13, "bold"),
            command=self._toggle_cam
        )
        self._cam_btn.pack(side="left", padx=10, pady=8)

        self._rec_btn = ctk.CTkButton(
            ctrl, text="⏺  Yozishni boshlash", width=170, height=36,
            fg_color=THEME["card"], hover_color=THEME["border"],
            border_color=THEME["red"], border_width=1,
            text_color=THEME["red"],
            font=ctk.CTkFont("Helvetica", 13, "bold"),
            state="disabled",
            command=self._toggle_record
        )
        self._rec_btn.pack(side="left", padx=4, pady=8)

        ctk.CTkLabel(ctrl, text="Ishora:",
                     font=ctk.CTkFont("Helvetica", 12),
                     text_color=THEME["subtext"]).pack(side="left", padx=(20,4), pady=8)

        self._gest_menu = ctk.CTkOptionMenu(
            ctrl, values=GESTURES_LIST,
            variable=self.current_label,
            width=130, height=34,
            fg_color=THEME["card"],
            button_color=THEME["accent"],
            font=ctk.CTkFont("Helvetica", 12),
            command=self._on_label_change
        )
        self._gest_menu.pack(side="left", padx=4, pady=8)

        ctk.CTkButton(
            ctrl, text="＋ Yangi", width=90, height=34,
            fg_color=THEME["card"], hover_color=THEME["border"],
            border_color=THEME["accent"], border_width=1,
            text_color=THEME["accent"],
            font=ctk.CTkFont("Helvetica", 12),
            command=self._add_custom_label
        ).pack(side="left", padx=4, pady=8)

        self._sess_lbl = ctk.CTkLabel(
            ctrl, text="Sessiya: 0",
            font=ctk.CTkFont("Helvetica", 12),
            text_color=THEME["subtext"]
        )
        self._sess_lbl.pack(side="right", padx=14, pady=8)

        self._rec_dot = ctk.CTkLabel(
            ctrl, text="●",
            font=ctk.CTkFont("Helvetica", 16),
            text_color=THEME["border"]
        )
        self._rec_dot.pack(side="right", padx=4, pady=8)

        # Kamera
        cam_f = ctk.CTkFrame(left, fg_color="#010409", corner_radius=10)
        cam_f.grid(row=1, column=0, sticky="nsew")
        self._cam_lbl = tk.Label(cam_f, bg="#010409")
        self._cam_lbl.pack(fill="both", expand=True, padx=2, pady=2)
        self._placeholder()

        # ── O'NG: Statistika ──────────────────────────────────────────────────
        right = ctk.CTkFrame(body, fg_color=THEME["bg"], corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew", padx=(6,0))
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Progress card
        prog_card = ctk.CTkFrame(right, fg_color=THEME["panel"], corner_radius=12)
        prog_card.grid(row=0, column=0, sticky="ew", pady=(0,8))

        ctk.CTkLabel(prog_card, text="HOZIRGI ISHORA",
                     font=ctk.CTkFont("Helvetica", 11),
                     text_color=THEME["subtext"]).pack(pady=(12,2), padx=14, anchor="w")

        self._cur_lbl = ctk.CTkLabel(
            prog_card, text=GESTURES_LIST[0].upper(),
            font=ctk.CTkFont("Helvetica", 28, "bold"),
            text_color=THEME["accent"]
        )
        self._cur_lbl.pack(pady=(0,4))

        self._prog_bar = ctk.CTkProgressBar(
            prog_card, height=14, corner_radius=7,
            progress_color=THEME["green"], fg_color=THEME["card"]
        )
        self._prog_bar.pack(fill="x", padx=14, pady=(0,4))
        self._prog_bar.set(0)

        self._prog_lbl = ctk.CTkLabel(
            prog_card, text="0 / 300",
            font=ctk.CTkFont("Helvetica", 12),
            text_color=THEME["subtext"]
        )
        self._prog_lbl.pack(pady=(0,12))

        # Barcha ishoralar statistikasi
        stat_card = ctk.CTkFrame(right, fg_color=THEME["panel"], corner_radius=12)
        stat_card.grid(row=1, column=0, sticky="nsew")
        stat_card.rowconfigure(1, weight=1)
        stat_card.columnconfigure(0, weight=1)

        ctk.CTkLabel(stat_card, text="BARCHA ISHORALAR",
                     font=ctk.CTkFont("Helvetica", 11),
                     text_color=THEME["subtext"]).grid(
                         row=0, column=0, sticky="w", padx=14, pady=(12,4))

        self._stat_frame = ctk.CTkScrollableFrame(
            stat_card, fg_color=THEME["card"],
            corner_radius=8, scrollbar_button_color=THEME["border"]
        )
        self._stat_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0,8))
        self._stat_frame.columnconfigure(0, weight=1)

        self._stat_rows = {}
        self._refresh_stats()

        # Footer
        foot = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=0, height=34)
        foot.pack(fill="x", side="bottom")
        foot.pack_propagate(False)
        self._foot_lbl = ctk.CTkLabel(
            foot, text="CSV: " + CSV_FILE,
            font=ctk.CTkFont("Helvetica", 10),
            text_color=THEME["subtext"]
        )
        self._foot_lbl.pack(side="left", padx=12, pady=8)

    # ─────────────────────────────────────────────────────────────────────────
    def _placeholder(self):
        img   = Image.new("RGB", (800, 440), "#010409")
        photo = ImageTk.PhotoImage(img)
        self._cam_lbl.configure(image=photo)
        self._cam_lbl.image = photo

    def _toggle_cam(self):
        if not self.running:
            self._start_cam()
        else:
            self._stop_cam()

    def _start_cam(self):
        self.running = True
        self._cam_btn.configure(
            text="⏹  To'xtatish", fg_color=THEME["red"], hover_color="#b62324"
        )
        self._rec_btn.configure(state="normal")
        self.cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self.cam_thread.start()

    def _stop_cam(self):
        self.running   = False
        self.recording = False
        self._cam_btn.configure(
            text="▶  Kamerani yoq", fg_color=THEME["green"], hover_color="#2ea043"
        )
        self._rec_btn.configure(state="disabled", text="⏺  Yozishni boshlash",
                                border_color=THEME["red"], text_color=THEME["red"])
        self._rec_dot.configure(text_color=THEME["border"])
        self._placeholder()

    _HAND_CONN = [
        (0,1),(1,2),(2,3),(3,4),
        (5,6),(6,7),(7,8),
        (9,10),(10,11),(11,12),
        (13,14),(14,15),(15,16),
        (17,18),(18,19),(19,20),
        (0,5),(5,9),(9,13),(13,17),(0,17),
    ]

    def _cam_loop(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_img)

                landmarks = None
                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
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
                    if self.recording:
                        row = [self.current_label.get()] + normalize(landmarks)
                        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow(row)
                        self.session_count += 1
                        self.after(0, self._on_sample_written)

                if self.recording and landmarks:
                    cv2.circle(frame, (30, 30), 12, (0, 0, 220), -1)
                elif self.recording:
                    cv2.putText(frame, "QO'L KO'RINMAYDI", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,140,255), 2)

                if not self._frame_pending:
                    self._frame_pending = True
                    self.after(0, lambda f=frame: self._show_frame(f))

        cap.release()

    def _show_frame(self, frame):
        self._frame_pending = False
        dw = self._cam_lbl.winfo_width()  or 800
        dh = self._cam_lbl.winfo_height() or 440
        h, w = frame.shape[:2]
        scale = min(dw/w, dh/h)
        frame_r = cv2.resize(frame, (int(w*scale), int(h*scale)))
        img   = Image.fromarray(cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(img)
        self._cam_lbl.configure(image=photo)
        self._cam_lbl.image = photo

    def _on_sample_written(self):
        stats = get_stats()
        cnt   = stats.get(self.current_label.get(), 0)
        self._sess_lbl.configure(text=f"Sessiya: {self.session_count}")
        self._prog_bar.set(min(cnt / MIN_SAMPLES, 1.0))
        self._prog_lbl.configure(text=f"{cnt} / {MIN_SAMPLES}")
        if cnt % 20 == 0:
            self._refresh_stats()

    def _toggle_record(self):
        self.recording = not self.recording
        if self.recording:
            self.session_count = 0
            self._rec_btn.configure(
                text="⏸  Pauza",
                border_color=THEME["orange"], text_color=THEME["orange"]
            )
            self._rec_dot.configure(text_color=THEME["red"])
        else:
            self._rec_btn.configure(
                text="⏺  Yozishni boshlash",
                border_color=THEME["red"], text_color=THEME["red"]
            )
            self._rec_dot.configure(text_color=THEME["border"])

    def _on_label_change(self, val):
        self._cur_lbl.configure(text=val.upper())
        stats = get_stats()
        cnt   = stats.get(val, 0)
        self._prog_bar.set(min(cnt / MIN_SAMPLES, 1.0))
        self._prog_lbl.configure(text=f"{cnt} / {MIN_SAMPLES}")

    def _add_custom_label(self):
        name = simpledialog.askstring(
            "Yangi ishora", "Yangi ishora nomi (lotin, kichik):",
            parent=self
        )
        if name:
            name = name.strip().lower()
            vals = self._gest_menu.cget("values")
            if name not in vals:
                new_vals = list(vals) + [name]
                self._gest_menu.configure(values=new_vals)
            self.current_label.set(name)
            self._on_label_change(name)

    def _refresh_stats(self):
        stats = get_stats()
        for w in self._stat_frame.winfo_children():
            w.destroy()

        all_labels = list(set(list(stats.keys()) + GESTURES_LIST))
        all_labels.sort()

        for i, lbl in enumerate(all_labels):
            cnt  = stats.get(lbl, 0)
            done = cnt >= MIN_SAMPLES
            row  = ctk.CTkFrame(self._stat_frame, fg_color="transparent", height=28)
            row.grid(row=i, column=0, sticky="ew", pady=1)
            row.columnconfigure(1, weight=1)
            row.pack_propagate(False)

            ico = ctk.CTkLabel(row, text="✓" if done else "○", width=20,
                               font=ctk.CTkFont("Helvetica", 12),
                               text_color=THEME["green"] if done else THEME["subtext"])
            ico.grid(row=0, column=0, padx=6)

            nlbl = ctk.CTkLabel(row, text=lbl, anchor="w",
                                font=ctk.CTkFont("Helvetica", 12),
                                text_color=THEME["text"] if done else THEME["subtext"])
            nlbl.grid(row=0, column=1, sticky="ew")

            bar = ctk.CTkProgressBar(row, height=8, width=80,
                                     corner_radius=4,
                                     progress_color=THEME["green"] if done else THEME["accent"],
                                     fg_color=THEME["border"])
            bar.grid(row=0, column=2, padx=4)
            bar.set(min(cnt / MIN_SAMPLES, 1.0))

            ctk.CTkLabel(row, text=str(cnt), width=36,
                         font=ctk.CTkFont("Helvetica", 11),
                         text_color=THEME["subtext"], anchor="e").grid(
                             row=0, column=3, padx=6)

    def _on_close(self):
        self.running   = False
        self.recording = False
        time.sleep(0.15)
        self.destroy()


if __name__ == "__main__":
    CollectApp().mainloop()
