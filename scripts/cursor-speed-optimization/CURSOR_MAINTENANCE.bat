@echo off
chcp 65001 >nul
echo ========================================
echo   Cursor 속도 유지보수 스크립트
echo   (캐시 정리, 프로세스 정리)
echo ========================================
echo.

echo [1/3] Cursor 프로세스 종료 중...
taskkill /F /IM Cursor.exe 2>nul
taskkill /F /IM "Cursor Helper.exe" 2>nul
timeout /t 2 /nobreak >nul

echo [2/3] 캐시 폴더 정리 중 (설정은 유지됨)...
set CURSOR=%APPDATA%\Cursor
if exist "%CURSOR%\Cache" rd /s /q "%CURSOR%\Cache"
if exist "%CURSOR%\CachedData" rd /s /q "%CURSOR%\CachedData"
if exist "%CURSOR%\CachedExtensions" rd /s /q "%CURSOR%\CachedExtensions"
if exist "%CURSOR%\GPUCache" rd /s /q "%CURSOR%\GPUCache"
if exist "%CURSOR%\Code Cache" rd /s /q "%CURSOR%\Code Cache"
if exist "%CURSOR%\logs" rd /s /q "%CURSOR%\logs"
mkdir "%CURSOR%\Cache" 2>nul
mkdir "%CURSOR%\logs" 2>nul

echo [3/3] 완료.
echo.
echo Cursor를 다시 실행하세요. 채팅 히스토리 삭제로 80-90%% 속도 개선이 보고됨.
echo 채팅 히스토리까지 삭제하려면: CURSOR_FULL_RESET.bat 실행
echo.
pause
