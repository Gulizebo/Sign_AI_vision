"""
╔══════════════════════════════════════════════════════╗
║   SOZLAMALAR PANELI                                  ║
║   Ishora qo'shish, audio boshqarish, eksport        ║
╚══════════════════════════════════════════════════════╝
"""

import sys, os, json, threading, time, csv
import tkinter as tk
from tkinter import messagebox, filedialog
import warnings
warnings.filterwarnings("ignore")

try:
    import customtkinter as ctk
    import numpy as np
    import pandas as pd
    import joblib
except ImportError as e:
    tk.Tk().withdraw()
    messagebox.showerror("Paket xatosi", str(e))
    sys.exit(1)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_FILE     = os.path.join(BASE_DIR, "data", "imo_ishoralar.csv")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
AUDIO_DIR     = os.path.join(BASE_DIR, "audio_cache")
CONFIG_FILE   = os.path.join(MODEL_DIR, "config.pkl")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
os.makedirs(AUDIO_DIR, exist_ok=True)

THEME = {
    "bg": "#0d1117", "panel": "#161b22", "card": "#1c2128",
    "border": "#30363d", "accent": "#58a6ff", "green": "#3fb950",
    "orange": "#d29922", "red": "#f85149", "purple": "#bc8cff",
    "text": "#e6edf3", "subtext": "#8b949e",
}

DEFAULT_SETTINGS = {
    "lang"       : "uz",
    "conf_thr"   : 0.72,
    "cooldown"   : 2.5,
    "window_size": 12,
    "mirror"     : True,
    "landmarks"  : True,
    "theme"      : "dark",
}


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
            return {**DEFAULT_SETTINGS, **s}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(s):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


class SettingsApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("⚙  Sozlamalar")
        self.geometry("860x680")
        self.minsize(720, 560)
        self.configure(fg_color=THEME["bg"])
        self.settings = load_settings()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        hdr = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=0, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="⚙  SOZLAMALAR",
                     font=ctk.CTkFont("Helvetica", 20, "bold"),
                     text_color=THEME["accent"]).pack(side="left", padx=20, pady=14)

        body = ctk.CTkFrame(self, fg_color=THEME["bg"], corner_radius=0)
        body.pack(fill="both", expand=True, padx=16, pady=12)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        # ── SOL ───────────────────────────────────────────────────────────────
        left = ctk.CTkFrame(body, fg_color=THEME["bg"], corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        left.columnconfigure(0, weight=1)

        # Aniqlash sozlamalari
        det_card = ctk.CTkFrame(left, fg_color=THEME["panel"], corner_radius=12)
        det_card.pack(fill="x", pady=(0,8))

        ctk.CTkLabel(det_card, text="ANIQLASH",
                     font=ctk.CTkFont("Helvetica", 12, "bold"),
                     text_color=THEME["text"]).pack(
                         anchor="w", padx=16, pady=(14,6))

        # Ishonch chegarasi
        self._conf_var = tk.DoubleVar(value=self.settings["conf_thr"])
        self._add_slider(det_card, "Ishonch chegarasi",
                         self._conf_var, 0.3, 0.99,
                         lambda v: f"{float(v):.2f}")

        # Sliding window
        self._win_var = tk.IntVar(value=self.settings["window_size"])
        self._add_slider(det_card, "Sliding window (kadr)",
                         self._win_var, 5, 30,
                         lambda v: str(int(float(v))))

        # Cooldown
        self._cool_var = tk.DoubleVar(value=self.settings["cooldown"])
        self._add_slider(det_card, "Cooldown (soniya)",
                         self._cool_var, 0.5, 10.0,
                         lambda v: f"{float(v):.1f}s")

        ctk.CTkFrame(det_card, fg_color="transparent", height=6).pack()

        # Kamera sozlamalari
        cam_card = ctk.CTkFrame(left, fg_color=THEME["panel"], corner_radius=12)
        cam_card.pack(fill="x", pady=(0,8))

        ctk.CTkLabel(cam_card, text="KAMERA",
                     font=ctk.CTkFont("Helvetica", 12, "bold"),
                     text_color=THEME["text"]).pack(
                         anchor="w", padx=16, pady=(14,6))

        self._mirror_var = tk.BooleanVar(value=self.settings["mirror"])
        self._add_toggle(cam_card, "Ko'zgu rejimi (mirror)", self._mirror_var)

        self._lm_var = tk.BooleanVar(value=self.settings["landmarks"])
        self._add_toggle(cam_card, "Qo'l nuqtalarini ko'rsatish", self._lm_var)

        ctk.CTkFrame(cam_card, fg_color="transparent", height=6).pack()

        # ── O'NG ──────────────────────────────────────────────────────────────
        right = ctk.CTkFrame(body, fg_color=THEME["bg"], corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        right.columnconfigure(0, weight=1)

        # Audio sozlamalari
        aud_card = ctk.CTkFrame(right, fg_color=THEME["panel"], corner_radius=12)
        aud_card.pack(fill="x", pady=(0,8))

        ctk.CTkLabel(aud_card, text="AUDIO",
                     font=ctk.CTkFont("Helvetica", 12, "bold"),
                     text_color=THEME["text"]).pack(
                         anchor="w", padx=16, pady=(14,6))

        lang_row = ctk.CTkFrame(aud_card, fg_color="transparent")
        lang_row.pack(fill="x", padx=16, pady=4)
        ctk.CTkLabel(lang_row, text="Til:", width=130,
                     font=ctk.CTkFont("Helvetica", 12),
                     text_color=THEME["text"]).pack(side="left")
        self._lang_var = tk.StringVar(value=self.settings["lang"])
        ctk.CTkSegmentedButton(
            lang_row,
            values=["uz", "ru", "en"],
            variable=self._lang_var,
            font=ctk.CTkFont("Helvetica", 12),
            fg_color=THEME["card"],
            selected_color=THEME["accent"],
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            aud_card,
            text="🔄  Audio Keshni Yangilash",
            height=34, corner_radius=8,
            fg_color=THEME["card"],
            hover_color=THEME["border"],
            border_color=THEME["accent"], border_width=1,
            text_color=THEME["accent"],
            font=ctk.CTkFont("Helvetica", 12),
            command=self._regenerate_audio
        ).pack(fill="x", padx=16, pady=(6,12))

        # Ma'lumot boshqaruvi
        data_card = ctk.CTkFrame(right, fg_color=THEME["panel"], corner_radius=12)
        data_card.pack(fill="x", pady=(0,8))

        ctk.CTkLabel(data_card, text="MA'LUMOT BOSHQARUVI",
                     font=ctk.CTkFont("Helvetica", 12, "bold"),
                     text_color=THEME["text"]).pack(
                         anchor="w", padx=16, pady=(14,6))

        # CSV statistikasi
        self._data_info = ctk.CTkLabel(
            data_card, text=self._get_data_info(),
            font=ctk.CTkFont("Courier", 11),
            text_color=THEME["subtext"], justify="left", anchor="w"
        )
        self._data_info.pack(fill="x", padx=16, pady=(0,6))

        btn_row = ctk.CTkFrame(data_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0,12))

        ctk.CTkButton(
            btn_row, text="📤  Eksport CSV",
            height=32, width=130, corner_radius=8,
            fg_color=THEME["card"],
            hover_color=THEME["border"],
            border_color=THEME["green"], border_width=1,
            text_color=THEME["green"],
            font=ctk.CTkFont("Helvetica", 11),
            command=self._export_csv
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btn_row, text="🗑  CSV Tozalash",
            height=32, width=130, corner_radius=8,
            fg_color=THEME["card"],
            hover_color=THEME["border"],
            border_color=THEME["red"], border_width=1,
            text_color=THEME["red"],
            font=ctk.CTkFont("Helvetica", 11),
            command=self._clear_csv
        ).pack(side="left", padx=4)

        # Model ma'lumoti
        mdl_card = ctk.CTkFrame(right, fg_color=THEME["panel"], corner_radius=12)
        mdl_card.pack(fill="x", pady=(0,8))

        ctk.CTkLabel(mdl_card, text="MODEL MA'LUMOTI",
                     font=ctk.CTkFont("Helvetica", 12, "bold"),
                     text_color=THEME["text"]).pack(
                         anchor="w", padx=16, pady=(14,6))

        ctk.CTkLabel(
            mdl_card,
            text=self._get_model_info(),
            font=ctk.CTkFont("Courier", 11),
            text_color=THEME["subtext"],
            justify="left", anchor="w"
        ).pack(fill="x", padx=16, pady=(0,12))

        # Log
        self._log = ctk.CTkTextbox(
            left, height=80,
            font=ctk.CTkFont("Courier", 10),
            fg_color=THEME["card"],
            text_color=THEME["subtext"],
            corner_radius=8, state="disabled"
        )
        self._log.pack(fill="x", pady=(0,0))

        # Saqlash tugmasi
        save_row = ctk.CTkFrame(self, fg_color=THEME["panel"],
                                corner_radius=0, height=54)
        save_row.pack(fill="x", side="bottom")
        save_row.pack_propagate(False)

        ctk.CTkButton(
            save_row, text="✅  Sozlamalarni Saqlash",
            height=38, width=220, corner_radius=10,
            fg_color=THEME["green"], hover_color="#2ea043",
            font=ctk.CTkFont("Helvetica", 13, "bold"),
            command=self._save
        ).pack(side="right", padx=16, pady=8)

        ctk.CTkButton(
            save_row, text="↩  Standart",
            height=38, width=140, corner_radius=10,
            fg_color=THEME["card"],
            hover_color=THEME["border"],
            border_color=THEME["border"], border_width=1,
            text_color=THEME["subtext"],
            font=ctk.CTkFont("Helvetica", 12),
            command=self._reset_defaults
        ).pack(side="right", padx=4, pady=8)

    # ─────────────────────────────────────────────────────────────────────────
    def _add_slider(self, parent, label, var, from_, to, fmt_fn):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=3)

        ctk.CTkLabel(row, text=label, width=180,
                     font=ctk.CTkFont("Helvetica", 12),
                     text_color=THEME["text"], anchor="w").pack(side="left")

        val_lbl = ctk.CTkLabel(row, text=fmt_fn(var.get()),
                               width=50,
                               font=ctk.CTkFont("Helvetica", 12, "bold"),
                               text_color=THEME["accent"])
        val_lbl.pack(side="right")

        slider = ctk.CTkSlider(
            row, from_=from_, to=to, variable=var,
            width=150, height=14,
            progress_color=THEME["accent"],
            button_color=THEME["accent"],
            command=lambda v, l=val_lbl, f=fmt_fn: l.configure(text=f(v))
        )
        slider.pack(side="right", padx=4)

    def _add_toggle(self, parent, label, var):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=3)
        ctk.CTkLabel(row, text=label,
                     font=ctk.CTkFont("Helvetica", 12),
                     text_color=THEME["text"]).pack(side="left")
        ctk.CTkSwitch(
            row, text="", variable=var,
            onvalue=True, offvalue=False,
            progress_color=THEME["accent"],
            button_color=THEME["text"]
        ).pack(side="right")

    # ─────────────────────────────────────────────────────────────────────────
    def _get_data_info(self):
        if not os.path.exists(DATA_FILE):
            return "  CSV topilmadi"
        try:
            df  = pd.read_csv(DATA_FILE)
            cnts = df["label"].value_counts()
            lines = [f"  Jami: {len(df):,} ta namuna"]
            for lbl, cnt in cnts.items():
                lines.append(f"  {lbl:<12}  {cnt}")
            return "\n".join(lines)
        except Exception:
            return "  CSV o'qib bo'lmadi"

    def _get_model_info(self):
        if not os.path.exists(os.path.join(MODEL_DIR, "config.pkl")):
            return "  Model topilmadi"
        try:
            cfg = joblib.load(os.path.join(MODEL_DIR, "config.pkl"))
            return (
                f"  Model    : {cfg.get('model_name','—')}\n"
                f"  Aniqlik  : {cfg.get('accuracy',0)*100:.1f}%\n"
                f"  Sinflar  : {len(cfg.get('classes',[]))} ta\n"
                f"  Feature  : {cfg.get('total_features','—')}"
            )
        except Exception:
            return "  Config o'qib bo'lmadi"

    # ─────────────────────────────────────────────────────────────────────────
    def _regenerate_audio(self):
        lang = self._lang_var.get()
        self._log_append(f"🔄  {lang} tilida audio yangilanmoqda...")

        TRANSLATIONS = {
            "salom":"Salom","rahmat":"Rahmat","yordam":"Yordam bering",
            "ha":"Ha","yoq":"Yo'q","suv":"Suv bering","ovqat":"Ovqat bering",
            "doktor":"Doktor chaqiring","uy":"Uyga ketaman","yaxshi":"Yaxshi",
        }

        def _run():
            try:
                from gtts import gTTS
                for key, text in TRANSLATIONS.items():
                    safe = "".join(c if c.isalnum() else "_" for c in text)
                    path = os.path.join(AUDIO_DIR, f"{safe}_{lang}.mp3")
                    gTTS(text=text, lang=lang, slow=False).save(path)
                    self.after(0, lambda k=key: self._log_append(f"  ✓ {k}"))
                self.after(0, lambda: self._log_append("✅  Audio yangilandi!"))
            except Exception as e:
                self.after(0, lambda: self._log_append(f"❌  {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _export_csv(self):
        if not os.path.exists(DATA_FILE):
            messagebox.showwarning("Ogohlantirish", "CSV fayl topilmadi.")
            return
        dst = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="imo_ishoralar_export.csv"
        )
        if dst:
            import shutil
            shutil.copy2(DATA_FILE, dst)
            self._log_append(f"✅  Eksport: {dst}")

    def _clear_csv(self):
        if not messagebox.askyesno(
            "CSV Tozalash",
            "Barcha yig'ilgan ma'lumotlar o'chadi!\n\nDavom etilsinmi?"
        ):
            return
        HEADER = ["label"] + [f"{c}_{i}" for i in range(21) for c in ["x","y","z"]]
        with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)
        self._data_info.configure(text="  CSV tozalandi (0 ta namuna)")
        self._log_append("🗑  CSV tozalandi")

    def _log_append(self, text):
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _save(self):
        self.settings.update({
            "lang"       : self._lang_var.get(),
            "conf_thr"   : round(float(self._conf_var.get()), 2),
            "cooldown"   : round(float(self._cool_var.get()), 1),
            "window_size": int(self._win_var.get()),
            "mirror"     : self._mirror_var.get(),
            "landmarks"  : self._lm_var.get(),
        })
        save_settings(self.settings)
        self._log_append("✅  Sozlamalar saqlandi!")
        messagebox.showinfo("Saqlandi", "Sozlamalar muvaffaqiyatli saqlandi.")

    def _reset_defaults(self):
        self.settings = dict(DEFAULT_SETTINGS)
        self._conf_var.set(DEFAULT_SETTINGS["conf_thr"])
        self._cool_var.set(DEFAULT_SETTINGS["cooldown"])
        self._win_var.set(DEFAULT_SETTINGS["window_size"])
        self._mirror_var.set(DEFAULT_SETTINGS["mirror"])
        self._lm_var.set(DEFAULT_SETTINGS["landmarks"])
        self._lang_var.set(DEFAULT_SETTINGS["lang"])
        self._log_append("↩  Standart sozlamalar tiklandi")


if __name__ == "__main__":
    SettingsApp().mainloop()
