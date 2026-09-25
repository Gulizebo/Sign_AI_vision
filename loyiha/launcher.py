"""
╔══════════════════════════════════════════════════════════════════╗
║   IMO-ISHORA TARJIMONI — Bosh Launcher                         ║
║   Barcha modullarni bitta joydan boshqarish                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys, os, subprocess, threading, time
import tkinter as tk
from tkinter import messagebox

MISSING = []
try:
    import customtkinter as ctk
except ImportError:
    MISSING.append("customtkinter")
try:
    from PIL import Image, ImageTk, ImageDraw
except ImportError:
    MISSING.append("Pillow")

if MISSING:
    root = tk.Tk()
    root.withdraw()
    msg = ("Quyidagi paketlar o'rnatilmagan:\n\n"
           + "\n".join(f"  • {p}" for p in MISSING)
           + "\n\nO'rnatish:\n"
           + "pip install " + " ".join(MISSING))
    messagebox.showerror("Paket xatosi", msg)
    root.destroy()
    sys.exit(1)

import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

THEME = {
    "bg"      : "#0d1117",
    "panel"   : "#161b22",
    "card"    : "#1c2128",
    "border"  : "#30363d",
    "accent"  : "#58a6ff",
    "green"   : "#3fb950",
    "orange"  : "#d29922",
    "red"     : "#f85149",
    "purple"  : "#bc8cff",
    "text"    : "#e6edf3",
    "subtext" : "#8b949e",
}

# ── Paketlar ro'yxati ─────────────────────────────────────────────────────────
REQUIRED_PACKAGES = [
    ("customtkinter",  "customtkinter"),
    ("opencv-python",  "cv2"),
    ("mediapipe",      "mediapipe"),
    ("scikit-learn",   "sklearn"),
    ("numpy",          "numpy"),
    ("pandas",         "pandas"),
    ("Pillow",         "PIL"),
    ("joblib",         "joblib"),
    ("gtts",           "gtts"),
    ("pygame",         "pygame"),
    ("matplotlib",     "matplotlib"),
]

def check_package(import_name):
    """Paket o'rnatilganligini tekshiradi."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def hex_fade(hex_color, alpha=0.15):
    """Hex rangni alpha bilan aralashtirib yangi hex qaytaradi (dark bg uchun)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    br, bg_, bb = 13, 17, 23
    nr = int(r * alpha + br * (1 - alpha))
    ng = int(g * alpha + bg_ * (1 - alpha))
    nb = int(b * alpha + bb * (1 - alpha))
    return f"#{nr:02x}{ng:02x}{nb:02x}"


def hex_darken(hex_color, factor=0.75):
    """Hex rangni qoraytiradi."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


def install_packages(packages, log_cb, done_cb):
    """Background threadda paketlarni o'rnatish."""
    for pip_name, import_name in packages:
        if not check_package(import_name):
            log_cb(f"⬇  {pip_name} o'rnatilmoqda...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", pip_name,
                     "--quiet", "--no-warn-script-location"],
                    check=True, capture_output=True
                )
                log_cb(f"✅  {pip_name} o'rnatildi")
            except subprocess.CalledProcessError:
                log_cb(f"❌  {pip_name}: xato!")
        else:
            log_cb(f"✓   {pip_name} mavjud")
    done_cb()


# Ochiq oynalarni kuzatish
_open_processes: dict = {}


def run_script(script_path):
    """Skriptni alohida jarayonda ishga tushurish — duplicate ochilmaydi."""
    global _open_processes

    key = os.path.basename(script_path)

    # Agar avval ochilgan bo'lsa va hali ishlayotgan bo'lsa — yangi ochmaydi
    existing = _open_processes.get(key)
    if existing is not None:
        if existing.poll() is None:
            return   # Hali ishlayapti — jim qaytamiz (tugmada "● Ochiq" ko'rinadi)
        else:
            del _open_processes[key]

    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.Popen(
            [sys.executable, script_path],
            creationflags=flags
        )
        _open_processes[key] = proc
    except Exception as e:
        messagebox.showerror("Xato", f"Skript ishlamadi:\n{e}")


# ══════════════════════════════════════════════════════════════════════════════
class LauncherApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VisionAssist — Imo-Ishora Tarjimoni")
        self.geometry("860x640")
        self.resizable(False, False)
        self.configure(fg_color=THEME["bg"])
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after(300, self._refresh_status)
        self._schedule_refresh()

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── HERO BANNER ───────────────────────────────────────────────────────
        banner = ctk.CTkFrame(self, fg_color=THEME["panel"],
                              corner_radius=0, height=140)
        banner.pack(fill="x")
        banner.pack_propagate(False)

        # Logotip (PIL bilan generatsiya)
        logo_img = self._make_logo(80)
        self._logo_photo = ImageTk.PhotoImage(logo_img)
        tk.Label(banner, image=self._logo_photo,
                 bg=THEME["panel"]).pack(side="left", padx=24, pady=20)

        txt_f = ctk.CTkFrame(banner, fg_color="transparent")
        txt_f.pack(side="left", pady=20)

        ctk.CTkLabel(
            txt_f, text="Imo-Ishora Tarjimoni",
            font=ctk.CTkFont("Helvetica", 30, "bold"),
            text_color=THEME["text"]
        ).pack(anchor="w")

        ctk.CTkLabel(
            txt_f,
            text="AI · MediaPipe · Random Forest · O'zbek tili · Real vaqt",
            font=ctk.CTkFont("Helvetica", 13),
            text_color=THEME["subtext"]
        ).pack(anchor="w", pady=(2,0))

        ctk.CTkLabel(
            txt_f, text="v2.0  ·  2025",
            font=ctk.CTkFont("Helvetica", 11),
            text_color=THEME["border"]
        ).pack(anchor="w", pady=(4,0))

        # Status chip
        self._hero_status = ctk.CTkLabel(
            banner, text="● Tekshirilmoqda...",
            font=ctk.CTkFont("Helvetica", 12),
            text_color=THEME["orange"]
        )
        self._hero_status.pack(side="right", padx=24)

        # ── ASOSIY TARKIB ─────────────────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color=THEME["bg"], corner_radius=0)
        body.pack(fill="both", expand=True, padx=20, pady=16)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=0)

        # ── SOL: Modullar ─────────────────────────────────────────────────────
        left = ctk.CTkFrame(body, fg_color=THEME["bg"], corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        left.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left, text="MODULLAR",
            font=ctk.CTkFont("Helvetica", 11),
            text_color=THEME["subtext"]
        ).pack(anchor="w", pady=(0,8))

        modules = [
            {
                "icon"   : "🤟",
                "title"  : "Real Vaqt Tarjimon",
                "desc"   : "Kamera orqali ishoralarni aniqlash\nva O'zbek tilida ovoz chiqarish",
                "script" : "app.py",
                "color"  : THEME["green"],
                "key"    : "app",
            },
            {
                "icon"   : "📊",
                "title"  : "Ma'lumot Yig'ish",
                "desc"   : "Kamera bilan yangi ishoralar\nyozib CSV ga saqlash",
                "script" : "gui_collect.py",
                "color"  : THEME["accent"],
                "key"    : "collect",
            },
            {
                "icon"   : "🧠",
                "title"  : "Model O'qitish",
                "desc"   : "AI modelini o'qitish, grafiklar\nva confusion matrix ko'rish",
                "script" : "gui_train.py",
                "color"  : THEME["purple"],
                "key"    : "train",
            },
            {
                "icon"   : "🔧",
                "title"  : "Demo Data Yaratish",
                "desc"   : "Kamerasiz sintetik ma'lumot\ngeneratsiya qilish",
                "script" : os.path.join("src", "0_generate_demo_data.py"),
                "color"  : THEME["orange"],
                "key"    : "demo",
            },
            {
                "icon"   : "⚙",
                "title"  : "Sozlamalar",
                "desc"   : "Ishonch chegarasi, audio tili,\neksport va boshqa sozlamalar",
                "script" : "gui_settings.py",
                "color"  : THEME["subtext"],
                "key"    : "settings",
            },
        ]

        self._module_btns = {}
        for mod in modules:
            card = self._module_card(left, mod)
            card.pack(fill="x", pady=4)

        # ── O'NG: Holat paneli ────────────────────────────────────────────────
        right = ctk.CTkFrame(body, fg_color=THEME["bg"], corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ctk.CTkLabel(
            right, text="TIZIM HOLATI",
            font=ctk.CTkFont("Helvetica", 11),
            text_color=THEME["subtext"]
        ).pack(anchor="w", pady=(0,8))

        # Fayl holatlari
        files_card = ctk.CTkFrame(right, fg_color=THEME["panel"], corner_radius=12)
        files_card.pack(fill="x", pady=(0,8))

        ctk.CTkLabel(
            files_card, text="Fayllar",
            font=ctk.CTkFont("Helvetica", 11, "bold"),
            text_color=THEME["text"]
        ).pack(anchor="w", padx=14, pady=(10,4))

        files_to_check = [
            ("data/imo_ishoralar.csv",    "CSV Ma'lumot",    "data"),
            ("models/imo_ishora_model.pkl","AI Modeli",       "model"),
            ("models/label_encoder.pkl",   "Label Encoder",   "encoder"),
            ("models/config.pkl",          "Config",          "config"),
        ]

        self._file_labels = {}
        for rel_path, name, key in files_to_check:
            row = ctk.CTkFrame(files_card, fg_color="transparent", height=28)
            row.pack(fill="x", padx=12, pady=1)
            row.pack_propagate(False)
            row.columnconfigure(1, weight=1)

            dot = ctk.CTkLabel(row, text="●", width=20,
                               font=ctk.CTkFont("Helvetica", 12),
                               text_color=THEME["border"])
            dot.grid(row=0, column=0)

            ctk.CTkLabel(row, text=name, anchor="w",
                         font=ctk.CTkFont("Helvetica", 11),
                         text_color=THEME["text"]).grid(
                             row=0, column=1, sticky="ew", padx=4)

            size_lbl = ctk.CTkLabel(row, text="--", width=70,
                                    font=ctk.CTkFont("Helvetica", 10),
                                    text_color=THEME["subtext"], anchor="e")
            size_lbl.grid(row=0, column=2, padx=4)

            self._file_labels[key] = (dot, size_lbl, rel_path)

        ctk.CTkFrame(files_card, fg_color="transparent", height=6).pack()

        # Paketlar holati
        pkg_card = ctk.CTkFrame(right, fg_color=THEME["panel"], corner_radius=12)
        pkg_card.pack(fill="x", pady=(0,8))

        pkg_hdr = ctk.CTkFrame(pkg_card, fg_color="transparent")
        pkg_hdr.pack(fill="x", padx=14, pady=(10,4))
        ctk.CTkLabel(pkg_hdr, text="Paketlar",
                     font=ctk.CTkFont("Helvetica", 11, "bold"),
                     text_color=THEME["text"]).pack(side="left")

        ctk.CTkButton(
            pkg_hdr, text="⬇  O'rnatish", width=100, height=26,
            fg_color=THEME["card"], hover_color=THEME["border"],
            border_color=THEME["accent"], border_width=1,
            text_color=THEME["accent"],
            font=ctk.CTkFont("Helvetica", 10),
            command=self._install_packages
        ).pack(side="right")

        self._pkg_frame = ctk.CTkScrollableFrame(
            pkg_card, fg_color=THEME["card"],
            corner_radius=8, height=130,
            scrollbar_button_color=THEME["border"]
        )
        self._pkg_frame.pack(fill="x", padx=8, pady=(0,8))
        self._pkg_frame.columnconfigure(0, weight=1)

        self._pkg_labels = {}
        for i, (pip_name, import_name) in enumerate(REQUIRED_PACKAGES):
            row = ctk.CTkFrame(self._pkg_frame, fg_color="transparent", height=22)
            row.grid(row=i, column=0, sticky="ew", pady=1)
            row.pack_propagate(False)
            row.columnconfigure(1, weight=1)

            dot = ctk.CTkLabel(row, text="●", width=18,
                               font=ctk.CTkFont("Helvetica", 11),
                               text_color=THEME["border"])
            dot.grid(row=0, column=0, padx=4)

            ctk.CTkLabel(row, text=pip_name, anchor="w",
                         font=ctk.CTkFont("Helvetica", 10),
                         text_color=THEME["subtext"]).grid(
                             row=0, column=1, sticky="ew")

            self._pkg_labels[import_name] = dot

        # Install log
        self._inst_log = ctk.CTkTextbox(
            right, height=70,
            font=ctk.CTkFont("Courier", 10),
            fg_color=THEME["card"],
            text_color=THEME["subtext"],
            corner_radius=8, state="disabled"
        )
        self._inst_log.pack(fill="x", pady=(0,0))

        # ── PASTKI TUGMALAR ───────────────────────────────────────────────────
        foot_row = ctk.CTkFrame(body, fg_color="transparent")
        foot_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12,0))
        foot_row.columnconfigure(0, weight=1)
        foot_row.columnconfigure(1, weight=0)

        ctk.CTkButton(
            foot_row,
            text="🔄  Holat yangilash",
            width=160, height=36,
            fg_color=THEME["card"],
            hover_color=THEME["border"],
            border_color=THEME["border"], border_width=1,
            text_color=THEME["subtext"],
            font=ctk.CTkFont("Helvetica", 12),
            command=self._refresh_status
        ).grid(row=0, column=1, padx=4)

        # Footer
        foot = ctk.CTkFrame(self, fg_color=THEME["panel"],
                            corner_radius=0, height=32)
        foot.pack(fill="x", side="bottom")
        foot.pack_propagate(False)
        ctk.CTkLabel(
            foot,
            text="Imo-Ishora Tarjimoni  ·  Python · CustomTkinter · MediaPipe · Scikit-learn  ·  v2.0",
            font=ctk.CTkFont("Helvetica", 10),
            text_color=THEME["subtext"]
        ).pack(side="left", padx=14, pady=6)

        ctk.CTkLabel(
            foot, text="2025",
            font=ctk.CTkFont("Helvetica", 10),
            text_color=THEME["border"]
        ).pack(side="right", padx=14)

    def _schedule_refresh(self):
        """Har 3 soniyada modul tugmalar holatini yangilaydi."""
        self._update_btn_states()
        self.after(3000, self._schedule_refresh)

    def _update_btn_states(self):
        """Ochiq/yopiq modullarni tugmada ko'rsatadi."""
        colors = {
            "app": THEME["green"], "collect": THEME["accent"],
            "train": THEME["purple"], "demo": THEME["orange"],
            "settings": THEME["subtext"],
        }
        script_map = {
            "app": "app.py", "collect": "gui_collect.py",
            "train": "gui_train.py", "demo": "0_generate_demo_data.py",
            "settings": "gui_settings.py",
        }
        for key, btn in self._module_btns.items():
            script = script_map.get(key, "")
            proc   = _open_processes.get(script)
            if proc and proc.poll() is None:
                btn.configure(text="● Ochiq", fg_color=THEME["orange"])
            else:
                if proc:
                    _open_processes.pop(script, None)
                btn.configure(
                    text="▶  Ochish",
                    fg_color=colors.get(key, THEME["accent"])
                )

    # ─────────────────────────────────────────────────────────────────────────
    def _module_card(self, parent, mod):
        card = ctk.CTkFrame(parent, fg_color=THEME["panel"],
                            corner_radius=12)
        card.columnconfigure(1, weight=1)

        # Ikon
        icon_f = ctk.CTkFrame(card, fg_color=hex_fade(mod["color"], 0.20),
                               corner_radius=10, width=56, height=56)
        icon_f.grid(row=0, column=0, rowspan=2, padx=14, pady=12, sticky="ns")
        icon_f.pack_propagate(False)
        ctk.CTkLabel(icon_f, text=mod["icon"],
                     font=ctk.CTkFont("Helvetica", 24)).pack(
                         expand=True)

        # Sarlavha
        ctk.CTkLabel(
            card, text=mod["title"],
            font=ctk.CTkFont("Helvetica", 14, "bold"),
            text_color=THEME["text"], anchor="w"
        ).grid(row=0, column=1, sticky="ew", padx=(0,10), pady=(12,2))

        # Tavsif
        ctk.CTkLabel(
            card, text=mod["desc"],
            font=ctk.CTkFont("Helvetica", 11),
            text_color=THEME["subtext"], anchor="w",
            justify="left"
        ).grid(row=1, column=1, sticky="ew", padx=(0,10), pady=(0,12))

        # Tugma
        btn = ctk.CTkButton(
            card, text="▶  Ochish",
            width=100, height=34,
            fg_color=mod["color"],
            hover_color=hex_darken(mod["color"], 0.75),
            text_color="#ffffff",
            font=ctk.CTkFont("Helvetica", 12, "bold"),
            command=lambda s=mod["script"]: run_script(
                os.path.join(BASE_DIR, s))
        )
        btn.grid(row=0, column=2, rowspan=2, padx=14, pady=12)
        self._module_btns[mod["key"]] = btn

        return card

    # ─────────────────────────────────────────────────────────────────────────
    def _make_logo(self, size):
        img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Doira fon — RGBA formatda
        draw.ellipse([0, 0, size-1, size-1], fill=(88, 166, 255, 40))
        draw.ellipse([4, 4, size-5, size-5], outline="#58a6ff", width=3)
        cx, cy = size//2, size//2
        # Barmoq simulyatsiyasi
        for i, (dx, h) in enumerate([(-16,30),(-6,40),(4,42),(14,36)]):
            draw.rounded_rectangle(
                [cx+dx, cy-h, cx+dx+8, cy+10],
                radius=4, fill="#58a6ff"
            )
        # Bosh barmoq
        draw.rounded_rectangle([cx-24, cy-10, cx-10, cy+4],
                                 radius=4, fill="#3fb950")
        return img

    # ─────────────────────────────────────────────────────────────────────────
    def _refresh_status(self):
        """Paket va fayl holatlarini background threadda tekshirib, UI ni yangilaydi."""
        def _worker():
            # Paket holatlari
            results = {}
            for pip_name, import_name in REQUIRED_PACKAGES:
                results[import_name] = check_package(import_name)

            # Fayl holatlari
            file_results = {}
            for key, (dot, size_lbl, rel_path) in self._file_labels.items():
                full = os.path.join(BASE_DIR, rel_path)
                if os.path.exists(full):
                    size = os.path.getsize(full)
                    if size > 1024 * 1024:
                        sz_txt = f"{size/1024/1024:.1f} MB"
                    elif size > 1024:
                        sz_txt = f"{size/1024:.1f} KB"
                    else:
                        sz_txt = f"{size} B"
                    file_results[key] = (True, sz_txt)
                else:
                    file_results[key] = (False, "Yo'q")

            # UI ni main threadda yangilash
            self.after(0, lambda: self._apply_status(results, file_results))

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_status(self, pkg_results, file_results):
        """Main threadda UI ni yangilaydi."""
        # Fayl holatlari
        for key, (dot, size_lbl, rel_path) in self._file_labels.items():
            ok, txt = file_results.get(key, (False, "?"))
            dot.configure(text_color=THEME["green"] if ok else THEME["red"])
            size_lbl.configure(
                text=txt,
                text_color=THEME["subtext"] if ok else THEME["red"]
            )

        # Paket holatlari
        pkg_ok = 0
        for pip_name, import_name in REQUIRED_PACKAGES:
            dot = self._pkg_labels.get(import_name)
            ok  = pkg_results.get(import_name, False)
            if dot:
                dot.configure(text_color=THEME["green"] if ok else THEME["red"])
            if ok:
                pkg_ok += 1

        # Modul tugmalar — ochiq/yopiq holati
        for key, btn in self._module_btns.items():
            proc = _open_processes.get(key + ".py") or _open_processes.get(
                os.path.join("src", "0_generate_demo_data.py") if key == "demo" else key
            )
            if proc and proc.poll() is None:
                btn.configure(text="● Ochiq", fg_color=THEME["orange"])
            else:
                # Default rangni qayta tiklash
                colors = {
                    "app": THEME["green"], "collect": THEME["accent"],
                    "train": THEME["purple"], "demo": THEME["orange"],
                    "settings": THEME["subtext"],
                }
                btn.configure(text="▶  Ochish",
                              fg_color=colors.get(key, THEME["accent"]))

        # Hero status
        model_exists = os.path.exists(
            os.path.join(BASE_DIR, "models", "imo_ishora_model.pkl"))
        total = len(REQUIRED_PACKAGES)
        if model_exists and pkg_ok == total:
            self._hero_status.configure(
                text=f"● Ishga tayyor  ({pkg_ok}/{total} paket)",
                text_color=THEME["green"]
            )
        elif pkg_ok < total:
            missing = total - pkg_ok
            self._hero_status.configure(
                text=f"⚠  {missing} ta paket yo'q",
                text_color=THEME["orange"]
            )
        else:
            self._hero_status.configure(
                text="⚠  Model o'qitilmagan",
                text_color=THEME["orange"]
            )

    # ─────────────────────────────────────────────────────────────────────────
    def _install_packages(self):
        missing = [
            (pip, imp) for pip, imp in REQUIRED_PACKAGES
            if not check_package(imp)
        ]
        if not missing:
            self._inst_log.configure(state="normal")
            self._inst_log.insert("end", "✅  Barcha paketlar o'rnatilgan!\n")
            self._inst_log.see("end")
            self._inst_log.configure(state="disabled")
            return

        self._inst_log.configure(state="normal")
        self._inst_log.delete("1.0", "end")
        self._inst_log.insert("end",
            f"⬇  {len(missing)} ta paket o'rnatilmoqda...\n")
        self._inst_log.configure(state="disabled")

        def log_cb(text):
            self.after(0, lambda t=text: self._append_log(t))

        def done_cb():
            self.after(0, self._refresh_status)
            self.after(0, lambda: self._append_log("✅  Yakunlandi!"))

        threading.Thread(
            target=install_packages,
            args=(missing, log_cb, done_cb),
            daemon=True
        ).start()

    def _append_log(self, text):
        self._inst_log.configure(state="normal")
        self._inst_log.insert("end", text + "\n")
        self._inst_log.see("end")
        self._inst_log.configure(state="disabled")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # EXE rejimida: argument berilgan bo'lsa, o'sha skriptni ishga tushur
    if getattr(sys, 'frozen', False) and len(sys.argv) > 1:
        import runpy
        runpy.run_path(sys.argv[1], run_name='__main__')
    else:
        LauncherApp().mainloop()
