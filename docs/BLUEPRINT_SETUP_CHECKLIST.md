# Blueprint Setup Checklist

## Goal

청사진 구현 전 필수 기반(분석 엔진, GUI, 합성/실데이터 테스트 체계, 보안운영)을 완료했는지 확인한다.

## Completed

- [x] Python 프로젝트 구조 및 품질도구(ruff/mypy/pytest/pre-commit)
- [x] FPCB 물리/화학 기반 합성데이터 생성기(시나리오 5종)
- [x] 합성데이터 선검증(시나리오별 규칙)
- [x] 실데이터 preflight 점검(포맷/연속성/index 일관성)
- [x] 벡터 분석 엔진(위치/속도/가속도/curvature surrogate)
- [x] 표준 결과 export (`vectors.csv`, `vectors.txt`, `summary.json`)
- [x] 결과 비교 기능(`compare-runs`)
- [x] Streamlit GUI 분석/비교 탭
- [x] Windows exe 빌드 파이프라인(GitHub Actions + local build script)
- [x] 사내 실환경 테스트 시나리오 문서
- [x] 비식별 로그 템플릿 및 보안 운영 정책

## Next Development Targets (Functional Completion)

- [ ] crack/predamage/thick/uv 판정 규칙을 summary 기반 자동 분류기로 확장
- [ ] GUI에서 결과 업로드 후 동시 비교 시각화(2개 이상 런)
- [ ] 실데이터 캘리브레이션 기반 임계값 재설정
- [ ] 분석 실패 시 자동 진단 리포트 생성
