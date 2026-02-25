# MCP (Model Context Protocol) 설정 가이드

**목적**: 로드맵 개발 및 연구 논문 작성에 유리하도록 Cursor Tools & MCP를 적극 활용하기 위한 설정 및 사용 가이드.

---

## 1. Tools & MCP 활성화 확인

1. Cursor에서 **Ctrl + ,** (설정) 열기
2. 왼쪽 메뉴에서 **Tools & MCP** 선택
3. 등록된 MCP 서버 목록 확인
   - 프로젝트 루트 `.cursor/mcp.json`에 정의된 서버가 자동 로드됨
   - 서버가 보이지 않으면 Cursor **완전 종료 후 재시작**

> **참고**: MCP는 설정 파일이 있으면 기본 활성화됨. 별도 `settings.json` 키는 없음.

---

## 2. 현재 설정된 MCP 서버

| 서버 | 용도 | 명령 |
|------|------|------|
| **filesystem-workspace** | 워크스페이스 파일 읽기/쓰기 | `npx -y @modelcontextprotocol/server-filesystem .` |

### Tools & MCP UI (라디오 버튼)

- **Settings → Tools & MCP** 에서 각 서버를 **켜기/끄기** 할 수 있음
- 새로 추가된 서버는 기본적으로 **꺼져 있을 수 있음** → 라디오 버튼으로 직접 켜야 함
- `mcp.json`에는 enabled/disabled 설정이 없음 (UI에서만 제어)

### fetch 서버 제거 사유

`mcp-fetch-node`는 **HTTP/SSE 서버**(포트 8080)로 동작하므로, Cursor가 기대하는 **stdio 기반 MCP**와 호환되지 않아 에러 발생.  
→ fetch가 필요하면 Cursor 내장 `mcp_web_fetch` 도구 사용, 또는 아래 대안 참고.

### fetch 대안 (선택)

| 방법 | 설명 |
|------|------|
| **Cursor 내장** | 채팅 시 AI가 `mcp_web_fetch` 도구로 URL fetch (별도 설정 불필요) |
| **uvx mcp-server-fetch** | `uv` 설치 후 `uvx mcp-server-fetch` (stdio 기반) — `mcp.json`에 command 추가 |

### 활용 시나리오

- **로드맵 개발**: `docs/`, `reports/` 등 프로젝트 문서 탐색·수정
- **논문 작성**: Cursor 내장 fetch 또는 MCP fetch로 arXiv, 학회 사이트 URL 가져오기

---

## 3. 설정 파일 위치

- **프로젝트별**: `c:\motionanalyzer\.cursor\mcp.json` (팀 공유 가능)
- **전역**: `~/.cursor/mcp.json` (사용자 전역)

---

## 4. Paper Banana MCP (선택)

논문용 다이어그램·플롯 생성 시 [Paper Banana MCP](https://github.com/llmsresearch/paperbanana)를 사용하려면 `.cursor/mcp.json`에 다음을 추가:

```json
"paperbanana": {
  "command": "uvx",
  "args": ["--from", "paperbanana[mcp]", "paperbanana-mcp"],
  "env": { "GOOGLE_API_KEY": "your-google-api-key" }
}
```

- **전제**: `pip install paperbanana` 또는 `uvx` 사용 가능, `GOOGLE_API_KEY` 설정
- 상세: `docs/PAPER_WRITING_AND_PAPERBANANA_PLAN.md` 참고

---

## 5. Cursor 재시작

MCP 서버 변경 후 **반드시 Cursor를 완전 종료 후 재시작**해야 적용됨.

---

## 6. 요구 사항

- **Node.js 18+** (npx 사용)
- **Python 3.x** (Paper Banana 사용 시)
