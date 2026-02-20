@echo off
REM Cursor를 NODE_OPTIONS(8GB 메모리)로 실행하여 메모리 부족 방지
REM Pro 요금제에서 추가 비용 없음

set NODE_OPTIONS=--max-old-space-size=8192
start "" "Cursor"
