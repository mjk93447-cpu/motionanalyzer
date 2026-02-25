# AI 에이전트 검증 및 캐싱 전략

**작성일**: 2026년 2월 25일  
**목적**: 에이전트 기능을 객관적 지표로 검증하고, 캐싱·성능 전략을 명시

---

## 1. 검증 스크립트

### 1.1 실행

```powershell
cd c:\motionanalyzer
.\scripts\verify_agent_handoff.ps1
```

**옵션**:
- `-Quick`: ML/GPU, QA 게이트 실행 생략 (기본 테스트만)
- `-OutputJson <path>`: 결과를 JSON으로 저장

### 1.2 검증 항목 (객관적 지표)

| ID | 항목 | 통과 조건 |
|----|------|-----------|
| docs | 핵심 문서 존재 | 6개 문서 모두 존재 |
| cursor-config | Skills/Rules 존재 | 4개 항목 존재 |
| corpus-index | corpus-index.json | 유효한 JSON |
| venv | 가상환경 | .venv 또는 .venv-gpu 존재 |
| import | motionanalyzer import | import 성공 |
| pytest | pytest 기본 | exit 0, 테스트 통과 |
| cli-doctor | CLI doctor | exit 0, "ready" 포함 |
| synthetic-smoke | 합성 데이터 | gen-synthetic + validate-synthetic 성공 |
| cuda | PyTorch CUDA | torch.cuda.is_available() (Quick 시 스킵) |
| qa-gate | QA 게이트 | evaluate_synthetic_dataset_quality 성공 (Quick 시 스킵) |
| shell-tools | Shell 도구 | 5개 스크립트 존재 |
| agent-tools | Agent tools | scripts/agent_tools 존재 |

### 1.3 점수

- **점수** = 100 × (passed / total)
- **목표**: 80/100 이상
- **출력**: 콘솔 + 선택적 JSON
- **검증 완료**: 100/100 (2026-02-25, Quick 모드)

---

## 2. 캐싱 전략

### 2.1 Cursor IDE 캐시

| 위치 | 용도 | 정리 방법 |
|------|------|-----------|
| `%APPDATA%\Cursor\Cache` | 임시 캐시 | RUN_ALL_OPTIMIZATIONS.ps1 |
| `%APPDATA%\Cursor\CachedData` | 에디터 캐시 | 동일 |
| `%APPDATA%\Cursor\User\workspaceStorage` | 워크스페이스 저장소 | 동일 |
| `%APPDATA%\Cursor\GPUCache` | GPU 캐시 | 동일 |

**캐시 과다 시**: `scripts/cursor-speed-optimization/RUN_ALL_OPTIMIZATIONS.ps1` (Cursor 종료 후 실행)

### 2.2 RAM 디스크 (선택)

- **도구**: ImDisk Toolkit (https://imdisktoolkit.com/)
- **스크립트**: `scripts/cursor-speed-optimization/CURSOR_RAMDISK_SETUP.ps1`
- **효과**: I/O 병목 40% 이상 감소 (HDD에서 특히 효과적)

### 2.3 NODE_OPTIONS

- `RUN_ALL_OPTIMIZATIONS.ps1` 실행 시 `NODE_OPTIONS=--max-old-space-size=8192` 설정
- Cursor/Node 메모리 상한 확대

### 2.4 프로젝트 인덱스

- `indexes/corpus-index.json`: canonical_read_order 기반 문서 인덱스
- 에이전트가 문서를 빠르게 찾을 수 있도록 구조화

---

## 3. Skills 및 Rules

### 3.1 Skills

| Skill | 용도 |
|-------|------|
| `ai-coding-accelerator` | 일반 코딩 (shell, compaction) |
| `agent-performance` | 검증, 캐싱, 컴팩션 |

### 3.2 Rules (alwaysApply)

- `motionanalyzer.mdc`: 프로젝트 목표, 코딩 규칙
- `fpcb-domain-knowledge.mdc`: FPCB 도메인
- `cursor-tools-optimization.mdc`: skills, shell 도구 선호

---

## 4. 검증 실행 및 결과 저장

```powershell
.\scripts\verify_agent_handoff.ps1 -OutputJson reports/agent_verification.json
```

**결과 예시** (`reports/agent_verification.json`):

```json
{
  "timestamp": "2026-02-25T...",
  "summary": { "passed": 10, "failed": 0, "skipped": 2, "score": 83.3, "total": 12 },
  "checks": [ ... ]
}
```

---

## 5. 주기적 검증 권장

- **핸드오프 직후**: 전체 검증
- **주요 변경 후**: `-Quick` 또는 전체
- **배포 전**: 전체 검증 + 점수 80 이상 확인
