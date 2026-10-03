@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
REM اسحب فيديو الكاميرا الأولى ثم الثانية وأفلتهما فوق هذا الملف
python "%~dp0montage.py" %*
echo.
pause
