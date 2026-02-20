# v0.2.0 릴리즈 생성 가이드

## 완료된 작업

✅ Git 커밋 완료 (커밋 해시: 32b10d1)  
✅ Git 태그 생성 완료 (v0.2.0)  
✅ GitHub에 푸시 완료 (main 브랜치 및 태그)  
✅ GitHub Actions 빌드 실행 중 (워크플로우 ID: 22138160577)  
✅ 릴리즈 노트 준비 완료 (`RELEASE_NOTES_v0.2.0.md`)

## 다음 단계: GitHub 릴리즈 생성

### 방법 1: GitHub 웹사이트에서 수동 생성 (권장)

1. **GitHub Actions 빌드 완료 대기**
   - 빌드 상태 확인: https://github.com/mjk93447-cpu/motionanalyzer/actions/runs/22138160577
   - 빌드가 완료되면 EXE 파일이 아티팩트로 생성됩니다

2. **릴리즈 생성**
   - https://github.com/mjk93447-cpu/motionanalyzer/releases/new 접속
   - **Tag**: `v0.2.0` 선택
   - **Release title**: `v0.2.0` 입력
   - **Description**: `RELEASE_NOTES_v0.2.0.md` 파일 내용 복사/붙여넣기
   - **Attach binaries**: GitHub Actions 아티팩트에서 다운로드한 EXE 파일 업로드
     - `motionanalyzer-gui.exe` (경량 버전)
     - `motionanalyzer-gui-ml.exe` (ML 포함 버전)
   - **Publish release** 클릭

### 방법 2: PowerShell 스크립트 사용 (GitHub 토큰 필요)

1. **GitHub 토큰 생성**
   - https://github.com/settings/tokens 접속
   - "Generate new token (classic)" 클릭
   - `repo` 스코프 선택
   - 토큰 생성 후 복사

2. **환경 변수 설정**
   ```powershell
   $env:GITHUB_TOKEN = "your-token-here"
   ```

3. **릴리즈 생성 스크립트 실행**
   ```powershell
   .\scripts\create_release.ps1 -Tag v0.2.0 -ReleaseNotesPath RELEASE_NOTES_v0.2.0.md
   ```

4. **EXE 파일 업로드**
   - GitHub Actions 빌드 완료 후 아티팩트 다운로드
   - 릴리즈 페이지에서 "Edit release" 클릭
   - EXE 파일 업로드

## GitHub Actions 빌드 확인

현재 빌드 상태: **실행 중** (in_progress)

빌드 URL: https://github.com/mjk93447-cpu/motionanalyzer/actions/runs/22138160577

빌드가 완료되면:
1. Actions 페이지에서 아티팩트 다운로드
2. 릴리즈 생성 시 EXE 파일 업로드

## 릴리즈 노트

`RELEASE_NOTES_v0.2.0.md` 파일에 완전한 릴리즈 노트가 준비되어 있습니다.

## 체크리스트

- [x] Git 커밋 완료
- [x] Git 태그 생성 완료
- [x] GitHub에 푸시 완료
- [x] GitHub Actions 빌드 실행 중
- [x] 릴리즈 노트 준비 완료
- [ ] GitHub Actions 빌드 완료 대기
- [ ] GitHub 릴리즈 생성
- [ ] EXE 파일 업로드
