@echo off
chcp 65001 >nul
echo ============================================
echo   IMO-ISHORA TARJIMONI — O'rnatish
echo ============================================
echo.
echo [1/3] pip yangilanmoqda...
python -m pip install --upgrade pip --quiet

echo [2/3] Paketlar o'rnatilmoqda...
pip install -r requirements.txt

echo [3/3] Demo ma'lumot yaratilmoqda...
python src/0_generate_demo_data.py

echo [4/4] Model o'qitilmoqda...
python src/2_model_training.py

echo.
echo ============================================
echo   O'rnatish tugadi!
echo   Ishga tushurish: python launcher.py
echo ============================================
pause
