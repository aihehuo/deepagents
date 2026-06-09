# TOOLS.md - Local Notes

Skills define how tools work. This file is for environment-specific details.

## Deployment

- Agent id: `andy01`
- Persona: 吴探长（餐饮商业探店顾问）
- Primary model: DeepSeek V4 Flash (`deepseek/deepseek-v4-flash`)
- Primary channel: Feishu

## Knowledge Base

- Path: `kb/`（index + raw + chunks）
- Search skill: `skills/local/wu-tanchang-kb`
- Rebuild: `python3 scripts/build_wu_kb.py`（from `claws/agent01/`）

## What Goes Here

- SSH hosts and aliases
- API endpoints and credentials references (not secrets themselves)
- Device names, camera names, TTS voice preferences
- Anything unique to this server or workspace

---

Add whatever helps you do your job. This is your cheat sheet.
