# CLAUDE.md

Read `AGENTS.md` before making changes in this repo.

## Non-Negotiables
- Keep GitHub PR review strictly read-only (no approvals, merges, or PR comments on GitHub).
- Keep all unit tests offline and stdlib-only (no third-party imports required at test time).
- Do not add or remove entries in `requirements.txt` without matching `.github/workflows/ci.yml`.
- Sanitize all untrusted Discord/GitHub text with `discord_safe()` and `redact()`.
- Never ping the maintainer role unless the deterministic gate and standards check both pass (`looks good`).

## Checks
```bash
python3 -I -m compileall -q bot.py contributor_bot
python3 -E -s -m unittest discover -v
ruff check .
git diff --check
```
