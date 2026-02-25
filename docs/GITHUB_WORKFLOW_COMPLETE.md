# GitHub 활용 완전 가이드

**목적**: 별도 지시 없이 중간 백업, 브랜치, 커밋, 아티팩트를 자유롭게 활용.

---

## 1. 빠른 참조

| 기능 | 명령 |
|------|------|
| **중간 백업** | `.\scripts\git_backup.ps1` |
| **체크포인트 커밋** | `.\scripts\git_checkpoint.ps1 -Message "메시지"` |
| **통합 워크플로우** | `.\scripts\git_workflow.ps1 -Action backup` |
| **상태 확인** | `.\scripts\git_workflow.ps1 -Action status` |
| **평가** | `.\scripts\evaluate_github_setup.ps1` |

---

## 2. 백업 (backup)

현재 작업을 `backup/YYYY-MM-DD-HHmm` 브랜치에 커밋·푸시 후 현재 브랜치에 merge.

```powershell
.\scripts\git_backup.ps1
.\scripts\git_backup.ps1 -Message "WIP: dataset cleanup"
```

---

## 3. 체크포인트 (commit)

현재 브랜치에 즉시 커밋.

```powershell
.\scripts\git_checkpoint.ps1 -Message "feat: add MCP fetch"
```

---

## 4. 브랜치

```powershell
# 새 feature 브랜치
.\scripts\git_workflow.ps1 -Action branch -BranchName "feature/dataset-naming"

# 또는 직접
git checkout -b feature/my-feature
```

---

## 5. 아티팩트 (GitHub Actions)

- **트리거**: `main` push, `v*` 태그, workflow_dispatch
- **아티팩트**: `motionanalyzer-windows-exe` (retention 30일)
- **다운로드**: Actions → Run → Artifacts

---

## 6. 평가

```powershell
.\scripts\evaluate_github_setup.ps1
```

0-100점. 80점 이상 권장.
