@echo off
chcp 65001 >nul
set "SRC=%~dp0.."
set "DEST=C:\motionanalyzer"

echo Copying from: %SRC%
echo Destination: %DEST%

mkdir "%DEST%" 2>nul
robocopy "%SRC%" "%DEST%" /E /XD .git node_modules __pycache__ .venv build dist .pytest_cache /XF *.pyc /NFL /NDL
set RC=%ERRORLEVEL%
if %RC% LSS 8 set RC=0
exit /b %RC%
