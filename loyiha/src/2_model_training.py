"""
╔══════════════════════════════════════════════════════╗
║   AI MODELINI O'QITISH — Random Forest + SVM        ║
║   Aqlli Imo-ishora Tili Tarjimoni — v1.0            ║
╚══════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score
)
from sklearn.pipeline import Pipeline

# ── Yo'llar ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE    = os.path.join(BASE_DIR, "data", "imo_ishoralar.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
MODEL_FILE   = os.path.join(MODEL_DIR, "imo_ishora_model.pkl")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")
CONFIG_FILE  = os.path.join(MODEL_DIR, "config.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)


def separator(char="─", width=54):
    print(char * width)


def load_data():
    separator("═")
    print("  BOSQICH 1: MA'LUMOT YUKLASH VA TEKSHIRISH")
    separator("═")

    if not os.path.exists(DATA_FILE):
        print(f"❌ CSV topilmadi: {DATA_FILE}")
        print("   Avval 1_data_collection.py yoki 0_generate_demo_data.py ni ishga tushiring!")
        exit(1)

    df = pd.read_csv(DATA_FILE)
    print(f"\n📂 Fayl: {DATA_FILE}")
    print(f"   Qatorlar: {len(df):,}   |   Ustunlar: {len(df.columns)}")

    # Sifat tekshiruvi
    nan_count = df.isnull().sum().sum()
    dup_count = df.duplicated().sum()
    print(f"   NaN: {nan_count}   |   Duplikat: {dup_count}")

    if nan_count:
        df.dropna(inplace=True)
    if dup_count:
        df.drop_duplicates(inplace=True)
        print(f"   🧹 {dup_count} ta duplikat o'chirildi.")

    print(f"\n📊 Sinflar bo'yicha taqsimot:")
    counts = df["label"].value_counts()
    max_c  = counts.max()
    for lbl, cnt in counts.items():
        bar  = "█" * int(cnt / max_c * 30)
        warn = " ⚠ kam!" if cnt < 100 else ""
        print(f"   {lbl:<14} {cnt:>5} ta  {bar}{warn}")

    return df


def add_features(X_raw):
    """
    Qo'l geometriyasidan qo'shimcha feature'lar:
      • 5 barmoq uchibning bilagdan masofasi
      • 4 qo'shni barmoq uchi orasidagi burchak
    """
    n      = X_raw.shape[0]
    pts    = X_raw.reshape(n, 21, 3)
    wrist  = pts[:, 0, :]

    extra = []

    # Barmoq uchi → bilag masofasi
    for tip in [4, 8, 12, 16, 20]:
        d = np.linalg.norm(pts[:, tip, :] - wrist, axis=1, keepdims=True)
        extra.append(d)

    # Qo'shni barmoq uchi orasidagi 2D burchak
    tips = [4, 8, 12, 16, 20]
    for i in range(len(tips) - 1):
        diff  = pts[:, tips[i+1], :2] - pts[:, tips[i], :2]
        angle = np.arctan2(diff[:, 1], diff[:, 0]).reshape(-1, 1)
        extra.append(angle)

    return np.hstack([X_raw] + extra)


def train():
    df      = load_data()
    X_raw   = df.drop("label", axis=1).values.astype(np.float32)
    y_str   = df["label"].values

    le = LabelEncoder()
    y  = le.fit_transform(y_str)

    print(f"\n🏷️  {len(le.classes_)} ta sinf: {list(le.classes_)}")

    # Feature engineering
    separator()
    print("\n  BOSQICH 2: FEATURE ENGINEERING")
    separator()
    X       = add_features(X_raw)
    print(f"   Feature soni: {X_raw.shape[1]} → {X.shape[1]}")

    # Train / Test bo'lish
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_tr):,}   Test: {len(X_te):,}")

    # ── Modellar ro'yxati ──────────────────────────────────────────────────────
    separator("═")
    print("\n  BOSQICH 3: MODELLARNI SOLISHTIRISH (5-Fold CV)")
    separator("═")

    candidates = {
        "Random Forest (200 daraxt)": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_features="sqrt",
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ]),
        "SVM (RBF kernel)": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, class_weight="balanced", random_state=42
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1,
                max_depth=5, random_state=42
            ))
        ]),
    }

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}

    for name, mdl in candidates.items():
        t0   = time.time()
        cvs  = cross_val_score(mdl, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
        dur  = time.time() - t0
        scores[name] = cvs.mean()
        print(f"\n   {name}")
        print(f"   Aniqlik : {cvs.mean()*100:6.2f}%  ±{cvs.std()*100:.2f}%   [{dur:.1f}s]")
        print(f"   Folds   : {' | '.join(f'{v*100:.1f}%' for v in cvs)}")

    # Eng yaxshi model
    best_name  = max(scores, key=scores.get)
    best_model = candidates[best_name]

    separator("═")
    print(f"\n  🏆 ENG YAXSHI: {best_name}")
    print(f"     CV aniqlik: {scores[best_name]*100:.2f}%")
    separator("═")

    # ── To'liq o'qitish ────────────────────────────────────────────────────────
    print("\n  BOSQICH 4: TO'LIQ O'QITISH VA BAHOLASH")
    separator()

    t0 = time.time()
    best_model.fit(X_tr, y_tr)
    print(f"   O'qitish vaqti: {time.time()-t0:.1f}s")

    y_pred = best_model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    bal    = balanced_accuracy_score(y_te, y_pred)

    print(f"\n   Test aniqlik         : {acc*100:.2f}%")
    print(f"   Balanced aniqlik     : {bal*100:.2f}%")

    print(f"\n{'─'*54}")
    print("  SINF BO'YICHA HISOBOT:")
    print(f"{'─'*54}")
    print(classification_report(
        y_te, y_pred,
        target_names=le.classes_, digits=3
    ))

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    labels = le.classes_
    col_w  = 10
    print("  CONFUSION MATRIX:")
    print(" " * 16 + "".join(f"{l:>{col_w}}" for l in labels))
    for i, row_lbl in enumerate(labels):
        row_str = f"  {row_lbl:<14}" + "".join(
            f"\033[92m{cm[i,j]:>{col_w}}\033[0m" if i == j
            else f"{cm[i,j]:>{col_w}}"
            for j in range(len(labels))
        )
        print(row_str)

    # Feature importance
    if hasattr(best_model.named_steps["clf"], "feature_importances_"):
        imps = best_model.named_steps["clf"].feature_importances_
        top  = np.argsort(imps)[::-1][:5]
        feat_names = (
            [f"{c}_{i}" for i in range(21) for c in ["x","y","z"]]
            + [f"tip_dist_{i}" for i in range(5)]
            + [f"angle_{i}"   for i in range(4)]
        )
        print("\n  TOP-5 MUHIM FEATURE:")
        for rank, idx in enumerate(top, 1):
            nm = feat_names[idx] if idx < len(feat_names) else f"f{idx}"
            bar = "█" * int(imps[idx] * 500)
            print(f"   {rank}. {nm:<20} {imps[idx]:.4f}  {bar}")

    # ── Saqlash ────────────────────────────────────────────────────────────────
    separator("═")
    print("\n  BOSQICH 5: MODELNI SAQLASH")
    separator("═")

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(le, ENCODER_FILE)
    joblib.dump({
        "raw_features"   : int(X_raw.shape[1]),
        "total_features" : int(X.shape[1]),
        "classes"        : list(le.classes_),
        "accuracy"       : float(acc),
        "model_name"     : best_name,
    }, CONFIG_FILE)

    print(f"\n   Model   → {MODEL_FILE}")
    print(f"   Encoder → {ENCODER_FILE}")
    print(f"   Config  → {CONFIG_FILE}")

    separator("═")
    if acc >= 0.95:
        emoji = "🎉"
        msg   = "AJOYIB! Ishlab chiqarishga tayyor."
    elif acc >= 0.85:
        emoji = "👍"
        msg   = "Yaxshi. Ko'proq ma'lumot yig'ish tavsiya etiladi."
    else:
        emoji = "⚠️ "
        msg   = "Har bir ishora uchun 300+ namuna yig'ing."

    print(f"\n  {emoji} {acc*100:.1f}% aniqlik — {msg}")
    separator("═")
    return acc


if __name__ == "__main__":
    train()
