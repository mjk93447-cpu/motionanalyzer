# 사내망 배포·실데이터 추가학습 워크플로 (데이터 유출 없음)

## 원칙

- **합성 pretrain**은 GitHub Actions 아티팩트 또는 `release/models`에 포함된 **균형 pretrain**을 사용합니다.
- **실데이터 추가학습(Refine)** 은 **사내망 PC의 GUI**에서만 수행합니다. 원본 영상/edge TXT가 외부로 나가지 않습니다.

## 1. 사내망 PC에 설치

GitHub Actions → `build-windows-exe` → 아티팩트 **`motionanalyzer-turnkey`** (또는 `motionanalyzer-windows-exe` + `motionanalyzer-pretrained-models`) 다운로드.

압축 해제 후:

```text
install/
  motionanalyzer-gui-ml.exe
  models/
    draem_model.pt
    patchcore_model.npz
    bundle_manifest.json
  INSTALL_TURNKEY.md
  INTERNAL_DEPLOYMENT_WORKFLOW.md
```

첫 실행 시 EXE가 내장/동봉 모델을 `%APPDATA%/motionanalyzer/models/`로 복사합니다.

선택: 모델만 별도 경로에 둘 때

```powershell
setx MOTIONANALYZER_MODELS_DIR "D:\motionanalyzer\models"
```

## 2. 사내망에서 실데이터 추가학습 (GUI)

1. **Data** 탭: `data/raw`에 edge TXT ingest → preflight → manifest (`data/real/ml_<site>/`)
2. **ML** 탭: manifest 경로 설정 → **Load manifest normals** → **Refine pretrained (real normal)**
3. **Paper/CLI preset** → **Prepare Data** → 모드 **PatchCore** → **Run**
4. (선택) **Evaluate PatchCore** 로 crack/정상 혼동행렬 확인
5. **Analyze** 탭: `patchcore` 또는 `ensemble`으로 현장 번들 추론

자세한 절차: [PATCHCORE_REALDATA_FINETUNE.md](PATCHCORE_REALDATA_FINETUNE.md)

## 3. 검증

```powershell
.\verify_turnkey_setup.ps1 -ModelsDir "%APPDATA%\motionanalyzer\models"
```

체크리스트: [GUI_ML_E2E_TEST_CHECKLIST.md](GUI_ML_E2E_TEST_CHECKLIST.md)

## Pretrain 근거

- 데이터셋: `ml_pretrain_balanced_3k_60f` (7:2:1, NG 균형)
- 메트릭: `reports/balanced_pretrain_metrics.json`
- 번들 메타: `release/TURNKEY_BUNDLE.json`
