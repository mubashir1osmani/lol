# lol (LiteLLM Contributor Bot) Agent Instructions

## Layout
- `bot.py`: process entrypoint.
- `contributor_bot/`: bot modules (`api.py`, `config.py`, `gateway.py`, `interactions.py`, `matcher.py`, `poller.py`, `pr_board.py`, `questions.py`, `review_agent.py`, `review_github.py`, `review_job.py`, `review_llm.py`, `reviews.py`, `roles_sync.py`, `routing.py`, `schedule.py`, `store.py`, `tracing.py`, `weekly.py`).
- `review_standards/`: LiteLLM review standards loaded by the PR review agent (`LITELLM_STANDARDS.md`) and proposed upstream agent docs.
- `tests/`: stdlib `unittest` suite that runs offline with zero third-party packages installed.

## Hard Security and Safety Invariants
- GitHub reads stay read-only (`contributor_bot/api.py` and `contributor_bot/review_github.py` must never approve, merge, or post GitHub PR comments). The only GitHub write path is the opt-in Projects v2 sync in `contributor_bot/pr_board.py`.
- Never allow untrusted GitHub diffs, PR descriptions, review comments, or Discord messages to trigger `@everyone`, `@here`, or arbitrary `<@&role>` pings. Pass untrusted text through `discord_safe()` and `redact()`.
- Never ping the maintainer role unless `submit_verdict` has already verified a `looks good` verdict with a clean deterministic gate.
- Never log, print, or post tokens, API keys, or secrets. Keep `redact()` on all agent output.
- Do not change `requirements.txt` unless the CI allowlist in `.github/workflows/ci.yml` is intentionally updated to match.

## Code Quality Rules
- Keep changes small, focused, and in the right module. Pure logic (formatting, gates, parsing, matching) stays free of network and SDK imports so offline tests stay fast and isolated.
- Import optional third-party SDKs (`discord`, `litellm`, `agents`, `opentelemetry`) lazily inside runtime paths so `python3 -E -s -m unittest discover -v` passes with stdlib only.
- Avoid broad silent `except Exception: pass` blocks. Catch specific exceptions where practical, and log warnings when falling back intentionally.
- Use type annotations on new and modified functions, clear names, and no dead or commented-out code.
- Do not add narrating comments; only comment non-obvious invariants or constraints.
- For PR review output (`contributor_bot/reviews.py` and `contributor_bot/review_agent.py`): verdict first (`looks good`, `needs work`, or `doesn't meet our standards`), one line when clean, at most 5 short `file:line` bullets otherwise, no praise, no PR summary, no headings, no emojis, no em-dashes.

## Verification Before Pushing
Run these checks and keep them green:
```bash
python3 -I -m compileall -q bot.py contributor_bot
python3 -E -s -m unittest discover -v
ruff check .
git diff --check
```
