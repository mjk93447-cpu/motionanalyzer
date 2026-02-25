# 컨텍스트 컴팩션 체크리스트 (에이전트 장기 세션용)
# 사용: .\scripts\agent_tools\compact_context.ps1
# 출력: 마일스톤 완료 후 적용할 체크리스트

$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$out = @"
=== 컨텍스트 컴팩션 체크리스트 ===
(장기 세션, 대량 로그 후 적용)

1. 마일스톤 완료 요약 작성
2. 해결된 분기(branch) 정리
3. 지속 지침은 .cursor/rules 또는 skills에 이동 (채팅에 반복 X)
4. 필요 시 새 스레드로 전환
5. 핵심 결정사항을 docs/ 또는 CHANGELOG에 기록

참고: .cursor/skills/ai-coding-accelerator/SKILL.md
"@
Write-Host $out -ForegroundColor Cyan
