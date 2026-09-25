@echo off
chcp 65001 >nul
title VisionAssist — Ishga tushirilmoqda...

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║       VisionAssist  —  Imo-Ishora        ║
echo  ╚══════════════════════════════════════════╝
echo.

:: Python borligini tekshirish
python --version >nul 2>&1
if errorlevel 1 (
    echo  [XATO] Python o'rnatilmagan!
    echo  https://python.org dan yuklab o'rnating.
    pause
    exit /b 1
)

:: Kerakli paketlarni o'rnatish (yo'q bo'lsa)
echo  [1/2] Paketlar tekshirilmoqda...
python -c "import customtkinter, cv2, mediapipe, sklearn, pygame, gtts, PIL, pandas, joblib" >nul 2>&1
if errorlevel 1 (
    echo  [!] Ba'zi paketlar yo'q. O'rnatilmoqda...
    python -m pip install customtkinter opencv-python mediapipe scikit-learn pygame gtts Pillow pandas joblib matplotlib --quiet --no-warn-script-location
    echo  [OK] Paketlar o'rnatildi.
) else (
    echo  [OK] Barcha paketlar tayyor.
)

:: Dasturni ishga tushirish
echo  [2/2] VisionAssist ishga tushirilmoqda...
echo.
python launcher.py

if errorlevel 1 (
    echo.
    echo  [XATO] Dastur xato bilan yopildi.
    pause
)
