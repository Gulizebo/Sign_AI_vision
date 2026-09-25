@echo off
chcp 65001 >nul
echo ============================================================
echo   IMO-ISHORA TARJIMONI — EXE Yaratish
echo ============================================================
echo.

:: PyInstaller borligini tekshirish
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [!] PyInstaller o'rnatilmoqda...
    python -m pip install pyinstaller --quiet
)

:: Eski build papkalarini tozalash
echo [1/3] Eski build tozalanmoqda...
if exist "dist\ImoIshora" rmdir /s /q "dist\ImoIshora"
if exist "build\ImoIshora" rmdir /s /q "build\ImoIshora"

:: Build
echo [2/3] EXE yaratilmoqda (5-10 daqiqa ketishi mumkin)...
echo.
python -m PyInstaller imo_ishora.spec --noconfirm

:: Natija tekshirish
echo.
if exist "dist\ImoIshora\ImoIshora.exe" (
    echo ============================================================
    echo   [OK] EXE muvaffaqiyatli yaratildi!
    echo.
    echo   Joylashuv: dist\ImoIshora\ImoIshora.exe
    echo.
    echo   ESLATMA: Butun dist\ImoIshora\ papkasini ko'chiring,
    echo            faqat .exe ni emas!
    echo ============================================================
    explorer dist\ImoIshora
) else (
    echo ============================================================
    echo   [XATO] EXE yaratilmadi. Yuqoridagi xatolarni ko'ring.
    echo ============================================================
)
echo.
pause
