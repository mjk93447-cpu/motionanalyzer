# Agent Tools

에이전트 성능 최적화를 위한 결정적·재현 가능한 쉘 도구.

| 스크립트 | 용도 |
|----------|------|
| `run_tests.ps1` | pytest 실행 (venv 자동 선택) |
| `run_qa_gate.ps1` | QA 게이트 일괄 실행 |
| `compact_context.ps1` | 컨텍스트 컴팩션 체크리스트 출력 |
| `run_refinement_loop.ps1` | 96% 달성까지 반복 평가 |
| `run_scenario_refinement_loop.ps1` | 시나리오 준비도 96% 달성까지 반복 |
| `run_scenario_S01.ps1` ~ `run_scenario_S15.ps1` | 시나리오별 원클릭 실행 |

사용: `.\scripts\agent_tools\<script>.ps1`
