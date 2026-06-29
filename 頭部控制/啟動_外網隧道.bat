@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   Cloudflare 外網隧道（免費、免帳號）
echo   會印出一個 https://xxx.trycloudflare.com 網址
echo   手機直接開那個網址即可
echo ============================================
cloudflared.exe tunnel --url https://localhost:8443 --no-tls-verify
pause
