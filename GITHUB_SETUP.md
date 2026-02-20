# GitHub 설정 및 푸시

## 1. GitHub 저장소 생성

1. [GitHub](https://github.com/new)에서 새 저장소 생성
2. 저장소 이름: `motionanalyzer` (또는 원하는 이름)
3. **Initialize with README** 체크 해제 (이미 로컬에 있음)
4. Create repository

## 2. 원격 저장소 연결 및 푸시

```powershell
cd c:\motionanalyzer

# 원격 저장소 추가 (YOUR_USERNAME을 GitHub 사용자명으로 변경)
git remote add origin https://github.com/YOUR_USERNAME/motionanalyzer.git

# main 브랜치 푸시
git push -u origin main
```

또는 GitHub CLI 사용 시:
```powershell
gh repo create motionanalyzer --private --source=. --push
```

## 3. GitHub Actions (EXE 빌드)

푸시 후 `.github/workflows/build-windows-exe.yml`이 자동 실행됩니다.

- **트리거**: `main` 브랜치 push, `v*` 태그, 또는 수동 (Actions > build-windows-exe > Run workflow)
- **아티팩트**: `motionanalyzer-windows-exe` (dist/ 폴더)
  - `motionanalyzer-gui.exe` (경량, ML 미포함)
  - `motionanalyzer-gui-ml.exe` (PyTorch 포함, DREAM/PatchCore 사용 가능)

## 4. 합성 데이터 생성 (로컬)

저장소에는 `data/synthetic/`가 포함되지 않습니다 (용량). 로컬에서 생성:

```powershell
python scripts/generate_ml_dataset.py
python scripts/analyze_crack_detection.py
```
