# 🤟 Imo-Ishora Tarjimoni — v2.0

**Python · CustomTkinter · OpenCV · MediaPipe · Scikit-learn · gTTS**

Real vaqtda qo'l ishoralarini O'zbek tilida ovoz chiqarib tarjima qiluvchi AI tizimi.  
**Endi to'liq grafik interfeys (GUI) bilan!**

---

## ⚡ Tezkor ishga tushirish

```bash
# 1. Paketlarni o'rnatish
pip install -r requirements.txt

# 2. Demo ma'lumot + model (bir marta)
python src/0_generate_demo_data.py
python src/2_model_training.py

# 3. Grafik ilovani ishga tushirish
python launcher.py
```

Yoki ikki marta bosish bilan:
```
run.bat         ← Launcher
install.bat     ← O'rnatish + model tayyor
```

---

## 🖥️ GUI Modullar

| Modul | Fayl | Tavsif |
|-------|------|--------|
| 🚀 Launcher | `launcher.py` | **Bosh menyu** — barcha modullarni boshqarish |
| 🤟 Tarjimon | `app.py` | Real vaqt kamera + ovoz + ehtimollik paneli |
| 📊 Yig'ish | `gui_collect.py` | Kamera bilan CSV ma'lumot yig'ish |
| 🧠 O'qitish | `gui_train.py` | Model o'qitish + grafiklar + confusion matrix |
| ⚙ Sozlamalar | `gui_settings.py` | Barcha sozlamalar, eksport, audio |

---

## 📁 Fayl tuzilmasi

```
imo_ishora/
├── launcher.py           ← 🚀 BOSH MENYU (shu yerdan boshlang!)
├── app.py                ← 🤟 Real vaqt tarjimon GUI
├── gui_collect.py        ← 📊 Ma'lumot yig'ish GUI
├── gui_train.py          ← 🧠 Model o'qitish GUI
├── gui_settings.py       ← ⚙  Sozlamalar GUI
├── settings.json         ← Saqlangan sozlamalar
├── requirements.txt      ← Paketlar
├── install.bat           ← Bir bosib o'rnatish
├── run.bat               ← Ishga tushurish
├── src/
│   ├── 0_generate_demo_data.py
│   ├── 1_data_collection.py
│   ├── 2_model_training.py
│   ├── 3_realtime_detection.py   ← (eski terminal versiyasi)
│   └── 4_test_model.py
├── data/imo_ishoralar.csv
├── models/
│   ├── imo_ishora_model.pkl
│   ├── label_encoder.pkl
│   └── config.pkl
└── audio_cache/
```

---

## 🎯 Qo'llab-quvvatlangan ishoralar (10 ta)

| Ishora | O'zbek tilida | Audio |
|--------|--------------|-------|
| salom  | Salom | ✓ |
| rahmat | Rahmat | ✓ |
| yordam | Yordam bering | ✓ |
| ha | Ha | ✓ |
| yoq | Yo'q | ✓ |
| suv | Suv bering | ✓ |
| ovqat | Ovqat bering | ✓ |
| doktor | Doktor chaqiring | ✓ |
| uy | Uyga ketaman | ✓ |
| yaxshi | Yaxshi | ✓ |

---

## 📊 Texnik ko'rsatkichlar

| Ko'rsatkich | Qiymat |
|-------------|--------|
| Model aniqlik | 100% (sintetik), ~92-97% (haqiqiy) |
| Taxmin tezligi | ~47,000 FPS (CPU) |
| Feature'lar | 72 ta |
| Algorithm | Random Forest (200 daraxt) |
| GUI | CustomTkinter (Dark/Light) |
| Audio | gTTS + pygame (uz/ru/en) |

---

## 🆕 v2.0 yangiliklari

- ✅ To'liq grafik interfeys (CustomTkinter)
- ✅ Bosh launcher — barcha modullar bir joyda
- ✅ Real vaqt ehtimollik paneli (barcha ishoralar %)
- ✅ Model o'qitish grafiklari (CV, Confusion Matrix, Feature Importance)
- ✅ Ma'lumot yig'ish progress va statistika paneli
- ✅ Sozlamalar paneli (JSON ga saqlash)
- ✅ Dark/Light mavzu
- ✅ Tarjima tarixi (scroll bilan)
- ✅ Paket avtomatik o'rnatish
- ✅ CSV eksport

---

*v2.0 · 2025 · MIT Litsenziyasi*
