# AI Agent Setup Iterative Refinement Loop

**목적**: 에이전트 사전 세팅을 상위 4% (96점 이상) 수준으로 유지하기 위한 반복 개선 프로세스

---

## 1. 평가 모델

**스크립트**: `scripts/evaluate_agent_setup.py`

**8개 차원 (총 110점 → 100% 환산)**:

| 차원 | 배점 | 핵심 항목 |
|------|------|-----------|
| Documentation | 15 | Handoff, Read order, Core docs, Next steps, Corpus |
| Skills & Rules | 20 | 2+ skills, Rules, agent-performance, Actionable |
| Tooling | 15 | One-command setup, Verification, Agent tools, Scripts |
| Environment | 15 | Venv, Import, Tests, CLI doctor, Synthetic smoke |
| Cache & Performance | 10 | Caching doc, Optimization scripts, Corpus, NODE/RAM |
| Roadmap Alignment | 15 | Roadmap, Phase status, QA gate, Goals, Known issues |
| Robustness | 10 | Graceful degradation, GUI fallback, Bash, MCP |
| Iterative Improvement | 10 | Evaluation model, Refinement doc, JSON, Target 96 |

**목표**: 96% 이상 (상위 4%)

---

## 2. 반복 개선 루프

```
┌─────────────────────────────────────────────────────────────┐
│  ITERATIVE REFINEMENT LOOP                                  │
├─────────────────────────────────────────────────────────────┤
│  1. Run: python scripts/evaluate_agent_setup.py -o reports/  │
│         agent_setup_evaluation.json                          │
│  2. If score < 96:                                           │
│     a. Inspect reports/agent_setup_evaluation.json            │
│     b. Identify dimensions with lowest pct                   │
│     c. Fix gaps (docs, scripts, skills, rules)              │
│     d. Re-run verify_agent_handoff.ps1                      │
│     e. Re-run evaluate_agent_setup.py                       │
│     f. Goto 2 until score >= 96                             │
│  3. If score >= 96: DONE                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 실행

### 3.1 Quick 모드 (런타임 생략)

```powershell
python scripts/evaluate_agent_setup.py --quick -o reports/agent_setup_evaluation.json
```

### 3.2 전체 모드 (Import, pytest, synthetic 포함)

```powershell
python scripts/evaluate_agent_setup.py -o reports/agent_setup_evaluation.json
```

### 3.3 자동 반복 (PowerShell)

```powershell
.\scripts\agent_tools\run_refinement_loop.ps1
```

---

## 4. 갭 분석

점수 미달 시 `reports/agent_setup_evaluation.json`의 `details` 배열에서 `score < max`인 항목 확인.

| ID 패턴 | 대응 조치 |
|---------|-----------|
| D1.* | 문서 추가/수정, corpus-index 갱신 |
| D2.* | Skills/Rules 추가, agent-performance 보강 |
| D3.* | 스크립트 추가, verify_agent_handoff 보강 |
| D4.* | venv 설정, pytest, CLI 검증 |
| D5.* | AGENT_VERIFICATION_AND_CACHING.md 보강 |
| D6.* | 로드맵 링크, QA 게이트, 다음 단계 명시 |
| D7.* | run_gui fallback, Bash script, MCP |
| D8.* | evaluate_agent_setup.py, 본 문서 |

---

## 5. 완료 기준

- **Score >= 96%** (110점 만점 시 105.6 이상)
- **verify_agent_handoff.ps1**: 80/100 이상
- **모든 차원 70% 이상**
