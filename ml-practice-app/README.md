# ML Practice App

ML 인터뷰 준비를 위한 macOS 앱.
하루에 N개의 문제를 push 받아 자기 테스트하고, Claude Code로 피드백을 받을 수 있습니다.

## Features

- **Daily Quiz**: 매일 설정한 수만큼 문제를 자동 배정 (in-progress 우선)
- **Progress Tracking**: 문제별 상태 관리 (Unsolved / In Progress / Solved)
- **Syntax Highlighted Code**: Python 코드를 컬러 하이라이팅으로 표시
- **Claude Code Integration**: 버튼 한 번으로 Terminal에서 Claude Code 실행
- **Dashboard**: 카테고리별 진행률, 동기부여 메시지
- **Daily Notifications**: 설정한 시간에 macOS 알림

## Build & Run

```bash
cd ml-practice-app
swift build
.build/debug/MLPractice
```

## First Launch

1. 앱 실행 시 "Choose Directory..." 버튼 클릭
2. `implementation-practice/` 폴더 선택
3. 자동으로 문제 목록 로드 + 오늘의 문제 배정

## Usage

- **사이드바**: Today's Practice (오늘의 문제) + 카테고리별 전체 문제
- **코드 뷰어**: 문제 파일의 코드를 syntax highlighting으로 표시
- **상태 변경**: 문제별로 Unsolved / In Progress / Solved 토글
- **Claude Feedback**: "Claude Feedback" 버튼 → Terminal에서 Claude Code 실행
- **Settings** (Cmd+,): 하루 문제 수, 알림 시간 설정

## Requirements

- macOS 14 (Sonoma) 이상
- Swift 5.9+
- [Claude Code](https://claude.ai/claude-code) (피드백 기능 사용 시)
