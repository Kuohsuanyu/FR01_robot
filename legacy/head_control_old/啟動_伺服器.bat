@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   VR 頭部雲台 - 伺服器
echo   (攝影機橫裝請調 --rotate 0/90/180/270)
echo ============================================
python -u server.py --rotate 270
pause
