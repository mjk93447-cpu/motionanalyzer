# AI Modeling Playbook (for Vibe Coding)

## When implementing or reviewing model code

Use this checklist in order:

1. 문제를 `구조(geometry) / 재료(material) / 구동(process)`로 분해했는가
2. 지표가 `형상, 곡률, 시간응답, 변형률 surrogate`를 포함하는가
3. 시나리오별 실패 시그니처가 코드 파라미터와 1:1 매핑되는가
4. 생성 산출물에 메타데이터(파라미터/가정/버전)가 남는가
5. 검증 실패 메시지가 원인(곡률 집중 부족, 최종각도 부족 등)을 말하는가

## Prompt Template for AI during development

```text
Task: Improve FPCB side-view bending model for scenario <SCENARIO>.

Constraints:
- Use physics-informed surrogate, not arbitrary random curves.
- Keep outputs backward compatible: frame_*.txt + fps.txt.
- Also emit diagnostics: frame_metrics.csv + metadata.json.
- Add/adjust validation rules tied to physical signature.

Focus:
- Geometry: straight -> arc -> U-like progression.
- Mechanics: kappa(s,t), theta(t), damping/response.
- Materials: stiffness increase (thick/over-cure), local softening (crack).

Deliver:
- Code changes
- Tests proving scenario signature
- Short note on assumptions and limitations
```

## Review Questions for Each PR

- 물리 가정이 코드/문서에 명시되어 있는가
- 파라미터 하나를 바꾸면 어떤 관측지표가 바뀌는지 설명 가능한가
- 실제 공정 로그와 비교할 연결 지점(지표/단위)이 있는가
- 시나리오 간 분리가 충분한가(오분류 가능성 낮은가)
