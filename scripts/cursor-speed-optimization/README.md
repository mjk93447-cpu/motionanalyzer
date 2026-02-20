# Cursor 속도 최적화 (Pro 요금제, 추가 비용 없음)

웹 검색 기반 실제 속도 개선 방법을 적용한 스크립트 모음입니다.

## 적용된 설정

### 1. settings.json (자동 적용됨)
- **TypeScript**: `maxTsServerMemory` 4096MB, 프로젝트 진단 비활성화
- **파일 감시 제외**: `.venv`, `__pycache__`, `node_modules`, `data/synthetic` 등
- **검색 제외**: 동일 폴더들로 불필요한 인덱싱 감소
- **Python**: 패키지 인덱싱 깊이 제한 (numpy, torch 등)

### 2. NODE_OPTIONS (메모리 8GB)
Cursor 기본 Node.js 메모리 한도 2GB → 8GB로 상향.

**방법 A** – 바로가기에서 이 배치 실행:
```
CURSOR_START_OPTIMIZED.bat
```

**방법 B** – 사용자 환경 변수에 영구 설정:
```powershell
[System.Environment]::SetEnvironmentVariable("NODE_OPTIONS", "--max-old-space-size=8192", "User")
```
※ 다른 Node.js 앱에도 적용되므로 필요 시 제거 가능.

---

## 유지보수 스크립트

| 스크립트 | 용도 | 효과 |
|----------|------|------|
| `CURSOR_MAINTENANCE.bat` | 캐시만 정리 (설정·채팅 유지) | 주간 권장 |
| `CURSOR_FULL_RESET.bat` | 채팅 히스토리 + 캐시 삭제 | **80–90% 속도 개선** 보고됨 |
| `CURSOR_RAMDISK_SETUP.ps1` | RAM 디스크 + 캐시 이동 | I/O 병목 **약 40% 감소** |

---

## RAM 디스크 설정 (선택)

**효과**: HDD에서 I/O 병목이 클 때 특히 효과적. SSD에서도 체감 가능.

**필요**: [ImDisk Toolkit](https://imdisktoolkit.com/) (무료)

**절차**:
1. Cursor 완전 종료
2. PowerShell **관리자 권한**으로 실행
3. `.\CURSOR_RAMDISK_SETUP.ps1` 실행
4. ImDisk 미설치 시: 스크립트가 수동 명령어를 출력함

**주의**: RAM 디스크는 재부팅 시 초기화됨. ImDisk에서 "부팅 시 자동 마운트" 설정 권장.

---

## 일상 권장 사항

- **매일**: 탭 10개 이하, 4시간마다 Cursor 재시작
- **매주**: `CURSOR_MAINTENANCE.bat` 실행
- **느려질 때**: `CURSOR_FULL_RESET.bat` 실행 (채팅 삭제)

---

## 참고

- [BoostDevSpeed: Cursor AI 7 Fixes 2025](https://boostdevspeed.com/blog/cursor-ai-slow-performance-7-fixes-2025)
- [Cursor Forum: tmpfs hack for Linux](https://forum.cursor.com/t/cursor-1-3-x-slow-lots-of-i-o-tmpfs-hack-for-linux/124356)
