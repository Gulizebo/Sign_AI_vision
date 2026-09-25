"""
╔══════════════════════════════════════════════════════╗
║   MODEL O'QITISH — Grafik Interfeys                 ║
║   Real vaqtda progress, grafik, natijalar           ║
╚══════════════════════════════════════════════════════╝
"""

import sys, os, threading, time, warnings
import tkinter as tk
from tkinter import messagebox
warnings.filterwarnings("ignore")

try:
    import customtkinter as ctk
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except Exception as e:
    tk.Tk().withdraw()
    messagebox.showerror("Paket xatosi",
        f"Kerakli paket:\n{e}\n\npip install customtkinter numpy pandas scikit-learn matplotlib joblib")
    sys.exit(1)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_FILE    = os.path.join(BASE_DIR, "data", "imo_ishoralar.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
MODEL_FILE   = os.path.join(MODEL_DIR, "imo_ishora_model.pkl")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")
CONFIG_FILE  = os.path.join(MODEL_DIR, "config.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

THEME = {
    "bg": "#0d1117", "panel": "#161b22", "card": "#1c2128",
    "border": "#30363d", "accent": "#58a6ff", "green": "#3fb950",
    "orange": "#d29922", "red": "#f85149", "purple": "#bc8cff",
    "text": "#e6edf3", "subtext": "#8b949e",
}

PLOT_STYLE = {
    "figure.facecolor"   : "#161b22",
    "axes.facecolor"     : "#1c2128",
    "axes.edgecolor"     : "#30363d",
    "axes.labelcolor"    : "#8b949e",
    "text.color"         : "#e6edf3",
    "xtick.color"        : "#8b949e",
    "ytick.color"        : "#8b949e",
    "grid.color"         : "#30363d",
    "grid.linestyle"     : "--",
    "grid.alpha"         : 0.5,
}
plt.rcParams.update(PLOT_STYLE)


def add_features(X_raw):
    pts   = X_raw.reshape(-1, 21, 3)
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


class TrainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🧠  Model O'qitish")
        self.geometry("1280x800")
        self.minsize(1000, 660)
        self.configure(fg_color=THEME["bg"])

        self._training = False
        self._log_lines = []
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._check_data()

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=0, height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="🧠  MODEL O'QITISH",
                     font=ctk.CTkFont("Helvetica", 20, "bold"),
                     text_color=THEME["accent"]).pack(side="left", padx=20, pady=14)
        ctk.CTkLabel(hdr, text="Random Forest · SVM · Gradient Boosting · 5-Fold CV",
                     font=ctk.CTkFont("Helvetica", 12),
                     text_color=THEME["subtext"]).pack(side="left", padx=4)

        # Body
        body = ctk.CTkFrame(self, fg_color=THEME["bg"], corner_radius=0)
        body.pack(fill="both", expand=True, padx=12, pady=8)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        # ── SOL: Sozlamalar + Log ─────────────────────────────────────────────
        left = ctk.CTkFrame(body, fg_color=THEME["bg"], corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        # Ma'lumot holati
        data_card = ctk.CTkFrame(left, fg_color=THEME["panel"], corner_radius=12)
        data_card.grid(row=0, column=0, sticky="ew", pady=(0,8))

        ctk.CTkLabel(data_card, text="MA'LUMOT HOLATI",
                     font=ctk.CTkFont("Helvetica", 11),
                     text_color=THEME["subtext"]).pack(pady=(12,6), padx=14, anchor="w")

        self._data_frame = ctk.CTkFrame(data_card, fg_color=THEME["card"], corner_radius=8)
        self._data_frame.pack(fill="x", padx=10, pady=(0,10))

        self._data_lbl = ctk.CTkLabel(
            self._data_frame, text="CSV tekshirilmoqda...",
            font=ctk.CTkFont("Helvetica", 12),
            text_color=THEME["subtext"]
        )
        self._data_lbl.pack(pady=10, padx=12)

        # Sozlamalar
        cfg_card = ctk.CTkFrame(left, fg_color=THEME["panel"], corner_radius=12)
        cfg_card.grid(row=1, column=0, sticky="ew", pady=(0,8))

        ctk.CTkLabel(cfg_card, text="SOZLAMALAR",
                     font=ctk.CTkFont("Helvetica", 11),
                     text_color=THEME["subtext"]).pack(pady=(12,6), padx=14, anchor="w")

        # Test ulushi
        row1 = ctk.CTkFrame(cfg_card, fg_color="transparent")
        row1.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(row1, text="Test ulushi:", width=120,
                     font=ctk.CTkFont("Helvetica", 12),
                     text_color=THEME["text"]).pack(side="left")
        self._test_var = tk.DoubleVar(value=0.20)
        ctk.CTkSlider(row1, from_=0.1, to=0.4, variable=self._test_var,
                      width=120, progress_color=THEME["accent"],
                      button_color=THEME["accent"]).pack(side="left", padx=4)
        self._test_lbl = ctk.CTkLabel(row1, text="20%", width=40,
                                       font=ctk.CTkFont("Helvetica", 12, "bold"),
                                       text_color=THEME["accent"])
        self._test_lbl.pack(side="left")
        self._test_var.trace_add("write", lambda *a: self._test_lbl.configure(
            text=f"{self._test_var.get()*100:.0f}%"))

        # CV folds
        row2 = ctk.CTkFrame(cfg_card, fg_color="transparent")
        row2.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(row2, text="CV Folds:", width=120,
                     font=ctk.CTkFont("Helvetica", 12),
                     text_color=THEME["text"]).pack(side="left")
        self._cv_var = tk.IntVar(value=5)
        ctk.CTkSegmentedButton(
            row2, values=["3", "5", "10"],
            font=ctk.CTkFont("Helvetica", 12),
            fg_color=THEME["card"],
            selected_color=THEME["accent"],
            command=lambda v: self._cv_var.set(int(v))
        ).pack(side="left", padx=4)

        # Model tanlash
        row3 = ctk.CTkFrame(cfg_card, fg_color="transparent")
        row3.pack(fill="x", padx=14, pady=(4,12))
        ctk.CTkLabel(row3, text="Modellar:", width=120,
                     font=ctk.CTkFont("Helvetica", 12),
                     text_color=THEME["text"]).pack(side="left")
        self._rf_var  = tk.BooleanVar(value=True)
        self._svm_var = tk.BooleanVar(value=True)
        self._gb_var  = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(row3, text="RF", variable=self._rf_var,
                        checkbox_width=18, checkbox_height=18,
                        font=ctk.CTkFont("Helvetica", 12)).pack(side="left", padx=4)
        ctk.CTkCheckBox(row3, text="SVM", variable=self._svm_var,
                        checkbox_width=18, checkbox_height=18,
                        font=ctk.CTkFont("Helvetica", 12)).pack(side="left", padx=4)
        ctk.CTkCheckBox(row3, text="GB", variable=self._gb_var,
                        checkbox_width=18, checkbox_height=18,
                        font=ctk.CTkFont("Helvetica", 12)).pack(side="left", padx=4)

        # Log
        log_card = ctk.CTkFrame(left, fg_color=THEME["panel"], corner_radius=12)
        log_card.grid(row=2, column=0, sticky="nsew")
        log_card.rowconfigure(1, weight=1)
        log_card.columnconfigure(0, weight=1)

        ctk.CTkLabel(log_card, text="LOG",
                     font=ctk.CTkFont("Helvetica", 11),
                     text_color=THEME["subtext"]).grid(
                         row=0, column=0, sticky="w", padx=14, pady=(10,4))

        self._log_box = ctk.CTkTextbox(
            log_card, font=ctk.CTkFont("Courier", 11),
            fg_color=THEME["card"], text_color=THEME["text"],
            corner_radius=8, state="disabled"
        )
        self._log_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0,8))

        # Train tugma
        self._train_btn = ctk.CTkButton(
            left, text="🚀  O'qitishni Boshlash",
            height=46, corner_radius=10,
            fg_color=THEME["green"], hover_color="#2ea043",
            font=ctk.CTkFont("Helvetica", 15, "bold"),
            command=self._start_training
        )
        self._train_btn.grid(row=3, column=0, sticky="ew", pady=8)

        self._progress = ctk.CTkProgressBar(
            left, height=10, corner_radius=5,
            progress_color=THEME["accent"], fg_color=THEME["card"]
        )
        self._progress.grid(row=4, column=0, sticky="ew", pady=(0,4))
        self._progress.set(0)

        self._prog_lbl = ctk.CTkLabel(
            left, text="",
            font=ctk.CTkFont("Helvetica", 11),
            text_color=THEME["subtext"]
        )
        self._prog_lbl.grid(row=5, column=0, pady=(0,4))

        # ── O'NG: Grafik panel ────────────────────────────────────────────────
        right = ctk.CTkFrame(body, fg_color=THEME["panel"], corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(6,0))
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="NATIJALAR VA GRAFIKLAR",
                     font=ctk.CTkFont("Helvetica", 11),
                     text_color=THEME["subtext"]).grid(
                         row=0, column=0, sticky="w", padx=14, pady=(12,4))

        # Tablar
        self._tab_view = ctk.CTkTabview(
            right, fg_color=THEME["card"],
            segmented_button_fg_color=THEME["panel"],
            segmented_button_selected_color=THEME["accent"],
            corner_radius=10
        )
        self._tab_view.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0,8))

        t1 = self._tab_view.add("📊 CV Natijalari")
        t2 = self._tab_view.add("🔲 Confusion Matrix")
        t3 = self._tab_view.add("⭐ Feature Importance")
        t4 = self._tab_view.add("📈 Sinf Aniqliği")

        for tab in [t1, t2, t3, t4]:
            tab.rowconfigure(0, weight=1)
            tab.columnconfigure(0, weight=1)

        self._fig1, self._ax1 = plt.subplots(figsize=(6, 4))
        self._canvas1 = FigureCanvasTkAgg(self._fig1, master=t1)
        self._canvas1.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._ax1.set_title("O'qitish kutilmoqda...", color="#8b949e")

        self._fig2, self._ax2 = plt.subplots(figsize=(6, 4))
        self._canvas2 = FigureCanvasTkAgg(self._fig2, master=t2)
        self._canvas2.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._ax2.set_title("Confusion Matrix", color="#8b949e")

        self._fig3, self._ax3 = plt.subplots(figsize=(6, 4))
        self._canvas3 = FigureCanvasTkAgg(self._fig3, master=t3)
        self._canvas3.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._ax3.set_title("Feature Importance", color="#8b949e")

        self._fig4, self._ax4 = plt.subplots(figsize=(6, 4))
        self._canvas4 = FigureCanvasTkAgg(self._fig4, master=t4)
        self._canvas4.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._ax4.set_title("Sinf Aniqliği", color="#8b949e")

        for fig in [self._fig1, self._fig2, self._fig3, self._fig4]:
            fig.tight_layout()

        # Footer
        foot = ctk.CTkFrame(self, fg_color=THEME["panel"], corner_radius=0, height=34)
        foot.pack(fill="x", side="bottom")
        foot.pack_propagate(False)
        self._foot_lbl = ctk.CTkLabel(
            foot, text="Model: " + MODEL_FILE,
            font=ctk.CTkFont("Helvetica", 10),
            text_color=THEME["subtext"]
        )
        self._foot_lbl.pack(side="left", padx=12, pady=8)

    # ─────────────────────────────────────────────────────────────────────────
    def _check_data(self):
        if not os.path.exists(DATA_FILE):
            self._data_lbl.configure(
                text="❌  CSV topilmadi!\n   0_generate_demo_data.py ni ishga tushiring",
                text_color=THEME["red"]
            )
            self._train_btn.configure(state="disabled")
            return

        df     = pd.read_csv(DATA_FILE)
        counts = df["label"].value_counts()
        total  = len(df)
        n_cls  = len(counts)

        self._data_lbl.configure(
            text=f"✅  {total:,} ta namuna  ·  {n_cls} ta sinf",
            text_color=THEME["green"]
        )

        # Mini statistika
        for w in self._data_frame.winfo_children():
            w.destroy()
        self._data_lbl = ctk.CTkLabel(
            self._data_frame,
            text=f"✅  {total:,} ta namuna  ·  {n_cls} ta sinf",
            font=ctk.CTkFont("Helvetica", 12, "bold"),
            text_color=THEME["green"]
        )
        self._data_lbl.pack(pady=(8,4), padx=12)
        for lbl, cnt in counts.items():
            ok = "✓" if cnt >= 300 else "~"
            col = THEME["green"] if cnt >= 300 else THEME["orange"]
            ctk.CTkLabel(
                self._data_frame,
                text=f"  {ok}  {lbl:<12}  {cnt}",
                font=ctk.CTkFont("Courier", 11),
                text_color=col, anchor="w"
            ).pack(fill="x", padx=12, pady=1)
        ctk.CTkFrame(self._data_frame, fg_color="transparent", height=6).pack()

    # ─────────────────────────────────────────────────────────────────────────
    def _log(self, text, color=None):
        self._log_lines.append(text)
        self._log_box.configure(state="normal")
        self._log_box.insert("end", text + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _start_training(self):
        if self._training:
            return
        self._training = True
        self._train_btn.configure(
            state="disabled", text="⏳  O'qitilmoqda...",
            fg_color=THEME["card"]
        )
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _set_progress(self, val, text=""):
        self.after(0, lambda: self._progress.set(val))
        self.after(0, lambda: self._prog_lbl.configure(text=text))

    def _train_worker(self):
        try:
            self.after(0, lambda: self._log("📂  Ma'lumot yuklanmoqda..."))
            df    = pd.read_csv(DATA_FILE)
            X_raw = df.drop("label", axis=1).values.astype(np.float32)
            y_str = df["label"].values
            le    = LabelEncoder()
            y     = le.fit_transform(y_str)
            self.after(0, lambda: self._log(
                f"   {len(df):,} ta namuna  ·  {len(le.classes_)} ta sinf"))
            self._set_progress(0.05, "Ma'lumot yuklandi")

            # Feature engineering
            self.after(0, lambda: self._log("🔧  Feature engineering..."))
            X    = add_features(X_raw)
            self.after(0, lambda: self._log(f"   {X_raw.shape[1]} → {X.shape[1]} feature"))
            self._set_progress(0.10, "Feature engineering")

            # Split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=self._test_var.get(),
                random_state=42, stratify=y
            )
            self.after(0, lambda: self._log(
                f"   Train: {len(X_tr):,}  Test: {len(X_te):,}"))
            self._set_progress(0.15, "Ma'lumot bo'lindi")

            # Modellar
            candidates = {}
            if self._rf_var.get():
                candidates["Random Forest"] = Pipeline([
                    ("sc",  StandardScaler()),
                    ("clf", RandomForestClassifier(
                        n_estimators=200, max_features="sqrt",
                        class_weight="balanced", random_state=42, n_jobs=-1
                    ))
                ])
            if self._svm_var.get():
                candidates["SVM (RBF)"] = Pipeline([
                    ("sc",  StandardScaler()),
                    ("clf", SVC(
                        kernel="rbf", C=10, gamma="scale",
                        probability=True, class_weight="balanced", random_state=42
                    ))
                ])
            if self._gb_var.get():
                candidates["Gradient Boost"] = Pipeline([
                    ("sc",  StandardScaler()),
                    ("clf", GradientBoostingClassifier(
                        n_estimators=150, learning_rate=0.1,
                        max_depth=5, random_state=42
                    ))
                ])

            if not candidates:
                self.after(0, lambda: self._log("❌  Hech bo'lmaganda 1 ta model tanlang!"))
                self._finish_training()
                return

            cv       = StratifiedKFold(n_splits=self._cv_var.get(),
                                        shuffle=True, random_state=42)
            scores   = {}
            cv_data  = {}
            n_models = len(candidates)

            self.after(0, lambda: self._log(
                f"\n📊  {self._cv_var.get()}-Fold CV boshlanmoqda..."))

            for i, (name, mdl) in enumerate(candidates.items()):
                t0  = time.time()
                cvs = cross_val_score(mdl, X_tr, y_tr, cv=cv,
                                      scoring="accuracy", n_jobs=-1)
                dur = time.time() - t0
                scores[name]  = cvs.mean()
                cv_data[name] = cvs
                prog = 0.15 + 0.50 * (i+1) / n_models
                self._set_progress(prog, f"CV: {name}")
                msg = (f"   {name}:  {cvs.mean()*100:.2f}%"
                       f"  ±{cvs.std()*100:.2f}%  [{dur:.1f}s]")
                self.after(0, lambda m=msg: self._log(m))

            # CV grafigi
            self.after(0, lambda: self._plot_cv(cv_data))

            # Eng yaxshi
            best_name  = max(scores, key=scores.get)
            best_model = candidates[best_name]
            self.after(0, lambda: self._log(
                f"\n🏆  Tanlandi: {best_name}  ({scores[best_name]*100:.2f}%)"))
            self._set_progress(0.70, "Eng yaxshi model tanlandi")

            # To'liq o'qitish
            self.after(0, lambda: self._log("\n⚙  To'liq o'qitish..."))
            t0 = time.time()
            best_model.fit(X_tr, y_tr)
            dur = time.time() - t0
            self.after(0, lambda: self._log(f"   O'qitish: {dur:.1f}s"))
            self._set_progress(0.85, "O'qitish yakunlandi")

            # Baholash
            y_pred = best_model.predict(X_te)
            acc    = accuracy_score(y_te, y_pred)
            self.after(0, lambda: self._log(f"\n✅  Test aniqlik: {acc*100:.2f}%"))

            # Confusion matrix
            cm = confusion_matrix(y_te, y_pred)
            self.after(0, lambda: self._plot_cm(cm, le.classes_))

            # Sinf aniqliği
            self.after(0, lambda: self._plot_class_acc(y_te, y_pred, le.classes_))

            # Feature importance
            if hasattr(best_model.named_steps["clf"], "feature_importances_"):
                imps = best_model.named_steps["clf"].feature_importances_
                self.after(0, lambda: self._plot_importance(imps))

            # Saqlash
            joblib.dump(best_model, MODEL_FILE)
            joblib.dump(le, ENCODER_FILE)
            joblib.dump({
                "raw_features"   : int(X_raw.shape[1]),
                "total_features" : int(X.shape[1]),
                "classes"        : list(le.classes_),
                "accuracy"       : float(acc),
                "model_name"     : best_name,
            }, CONFIG_FILE)

            self.after(0, lambda: self._log(f"\n💾  Model saqlandi: models/"))
            self._set_progress(1.0, f"✅ Tayyor! Aniqlik: {acc*100:.1f}%")

        except Exception as e:
            self.after(0, lambda: self._log(f"❌  Xato: {e}"))
        finally:
            self.after(0, self._finish_training)

    def _finish_training(self):
        self._training = False
        self._train_btn.configure(
            state="normal", text="🚀  O'qitishni Boshlash",
            fg_color=THEME["green"]
        )

    # ─────────────────────────────────────────────────────────────────────────
    # GRAFIKLAR
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_cv(self, cv_data):
        self._ax1.clear()
        names  = list(cv_data.keys())
        colors = [THEME["green"], THEME["accent"], THEME["orange"]]

        x = np.arange(len(names))
        for i, (name, cvs) in enumerate(cv_data.items()):
            self._ax1.bar(
                i, cvs.mean() * 100,
                color=colors[i % len(colors)],
                alpha=0.85, width=0.5,
                label=f"{name}: {cvs.mean()*100:.1f}%"
            )
            self._ax1.errorbar(
                i, cvs.mean() * 100,
                yerr=cvs.std() * 100,
                color="white", capsize=5, linewidth=2
            )

        self._ax1.set_xticks(x)
        self._ax1.set_xticklabels(names, fontsize=10)
        self._ax1.set_ylabel("Aniqlik (%)")
        self._ax1.set_title("Cross-Validation Natijalari", color=THEME["text"])
        self._ax1.set_ylim(
            max(0, min(v.mean() for v in cv_data.values()) * 100 - 5), 102
        )
        self._ax1.legend(fontsize=9)
        self._ax1.grid(axis="y", alpha=0.4)
        self._fig1.tight_layout()
        self._canvas1.draw()

    def _plot_cm(self, cm, labels):
        self._ax2.clear()
        im = self._ax2.imshow(cm, interpolation="nearest", cmap="Blues")
        self._fig2.colorbar(im, ax=self._ax2, fraction=0.046, pad=0.04)

        n = len(labels)
        self._ax2.set_xticks(range(n))
        self._ax2.set_yticks(range(n))
        self._ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        self._ax2.set_yticklabels(labels, fontsize=9)
        self._ax2.set_title("Confusion Matrix", color=THEME["text"])
        self._ax2.set_ylabel("Haqiqiy")
        self._ax2.set_xlabel("Bashorat")

        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                self._ax2.text(j, i, str(cm[i, j]),
                               ha="center", va="center", fontsize=9,
                               color="white" if cm[i, j] > thresh else "black")

        self._fig2.tight_layout()
        self._canvas2.draw()

    def _plot_importance(self, imps):
        self._ax3.clear()
        feat_names = (
            [f"{c}{i}" for i in range(21) for c in ["x","y","z"]]
            + [f"tip{i}" for i in range(5)]
            + [f"ang{i}" for i in range(4)]
        )
        top_n = 15
        idx   = np.argsort(imps)[::-1][:top_n]
        names = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in idx]
        vals  = imps[idx]

        colors = [THEME["green"] if i < 3 else THEME["accent"] for i in range(top_n)]
        self._ax3.barh(range(top_n), vals[::-1], color=colors[::-1], alpha=0.85)
        self._ax3.set_yticks(range(top_n))
        self._ax3.set_yticklabels(names[::-1], fontsize=9)
        self._ax3.set_xlabel("Muhimlik")
        self._ax3.set_title(f"Top-{top_n} Muhim Feature", color=THEME["text"])
        self._ax3.grid(axis="x", alpha=0.4)
        self._fig3.tight_layout()
        self._canvas3.draw()

    def _plot_class_acc(self, y_true, y_pred, classes):
        self._ax4.clear()
        le_temp = LabelEncoder()
        le_temp.classes_ = classes

        accs   = []
        counts = []
        for i, cls in enumerate(classes):
            mask = y_true == i
            if mask.sum() == 0:
                accs.append(0)
                counts.append(0)
            else:
                a = (y_pred[mask] == i).mean()
                accs.append(a * 100)
                counts.append(mask.sum())

        colors = [THEME["green"] if a >= 95 else
                  THEME["orange"] if a >= 85 else
                  THEME["red"] for a in accs]

        bars = self._ax4.barh(range(len(classes)), accs,
                               color=colors, alpha=0.85)

        for bar, cnt in zip(bars, counts):
            self._ax4.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"n={cnt}", va="center", fontsize=9,
                color=THEME["subtext"]
            )

        self._ax4.set_yticks(range(len(classes)))
        self._ax4.set_yticklabels(classes, fontsize=10)
        self._ax4.set_xlabel("Aniqlik (%)")
        self._ax4.set_title("Sinf bo'yicha aniqlik", color=THEME["text"])
        self._ax4.set_xlim(0, 110)
        self._ax4.axvline(x=95, color=THEME["green"], linestyle="--",
                          alpha=0.5, linewidth=1)
        self._ax4.grid(axis="x", alpha=0.4)
        self._fig4.tight_layout()
        self._canvas4.draw()


if __name__ == "__main__":
    TrainApp().mainloop()
