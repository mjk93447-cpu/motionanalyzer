@echo off
chcp 65001 >nul
echo ========================================
echo   Cursor 전체 초기화 (채팅 히스토리 포함)
echo   - 80-90%% 속도 개선 보고됨
echo   - 설정은 유지됨
echo ========================================
echo.

echo Cursor를 완전히 종료한 후 실행하세요.
echo.
pause

echo [1/4] Cursor 프로세스 종료...
taskkill /F /IM Cursor.exe 2>nul
taskkill /F /IM "Cursor Helper.exe" 2>nul
timeout /t 3 /nobreak >nul

echo [2/4] workspaceStorage 삭제 (프로젝트별 채팅)...
if exist "%APPDATA%\Cursor\User\workspaceStorage" rd /s /q "%APPDATA%\Cursor\User\workspaceStorage"
mkdir "%APPDATA%\Cursor\User\workspaceStorage" 2>nul

echo [3/4] 캐시 삭제...
set CURSOR=%APPDATA%\Cursor
for %%D in (Cache CachedData CachedExtensions GPUCache "Code Cache" logs) do (
    if exist "%CURSOR%\%%D" rd /s /q "%CURSOR%\%%D"
    mkdir "%CURSOR%\%%D" 2>nul
)

echo [4/4] 완료.
echo.
echo Cursor를 다시 실행하세요. 채팅 히스토리 삭제로 80-90%% 속도 개선이 보고됨.
echo.
pause
