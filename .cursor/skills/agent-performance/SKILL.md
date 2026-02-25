---
name: agent-performance
description: 에이전트 성능 최적화 - 캐싱, 검증, 컴팩션. 핸드오프 후 또는 장기 세션에서 적용.
---

# Agent Performance Optimization

## When to Use

- 새 에이전트로 핸드오프 후 즉시 검증할 때
- 장기 코딩 세션에서 컨텍스트가 비대해질 때
- Cursor 속도 저하(캐시 과다)가 의심될 때

## 1. 검증 (핸드오프 직후)

```powershell
cd c:\motionanalyzer
.\scripts\verify_agent_handoff.ps1 -OutputJson reports/agent_verification.json
```

**목표**: 점수 80/100 이상. 실패 항목은 순서대로 수정.

## 2. 캐싱 전략

### 2.1 Cursor 캐시 관리

- **과다 시**: `scripts/cursor-speed-optimization/RUN_ALL_OPTIMIZATIONS.ps1` (Cursor 종료 후 실행)
- **RAM 디스크**: ImDisk 설치 시 `CURSOR_RAMDISK_SETUP.ps1`로 I/O 병목 감소

### 2.2 프로젝트 캐시

- `indexes/corpus-index.json`: 문서 인덱스 (수동 갱신)
- `reports/`: 분석 결과 캐시 — 오래된 것은 `reports/archive/`로 이동
- `data/synthetic/`: 합성 데이터 — 대용량은 `artifacts/archive/` 정리

### 2.3 검증 결과 캐시

- `reports/agent_verification.json`: 마지막 검증 결과
- 변경 후 재검증: `verify_agent_handoff.ps1 -OutputJson reports/agent_verification.json`

## 3. Shell 도구 (결정적 실행)

| 도구 | 용도 |
|------|------|
| `scripts/agent_tools/run_tests.ps1` | pytest (venv 자동 선택) |
| `scripts/agent_tools/run_qa_gate.ps1` | QA 게이트 일괄 |
| `scripts/agent_tools/compact_context.ps1` | 컴팩션 체크리스트 |

## 4. 컴팩션 워크플로우 (장기 세션)

1. **마일스톤 완료 시**: 해결 내용 요약, 다음 목표 명시
2. **지속 지침**: rules/skills에 이동, 채팅에 반복하지 않음
3. **대량 로그 후**: 새 스레드로 전환 고려
4. **핵심 결정**: docs/ 또는 CHANGELOG에 기록

## 5. 읽기 순서 (캐시 워밍)

1. `docs/AGENT_HANDOFF_QUICK_START.md`
2. `docs/INDEX.md` → `PROJECT_GOALS` → `DEVELOPMENT_ROADMAP_FINAL`
3. `indexes/corpus-index.json` (canonical_read_order)
