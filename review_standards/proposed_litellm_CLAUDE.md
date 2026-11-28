# CLAUDE.md

Proposed `BerriAI/litellm/CLAUDE.md`. Read `AGENTS.md` (and `tests/unit/AGENTS.md` when touching tests) before making changes.

## Hard Rules
- Register OpenAI-compatible providers in `litellm/llms/openai_like/providers.json` instead of writing a custom Python provider. Put standalone providers in `litellm/llms/<provider>/<endpoint>/transformation.py` using `litellm/llms/base_llm/<endpoint>/`. Never add bespoke provider functions to `litellm/__init__.py` or pile provider branches into `litellm/main.py` or `litellm/utils.py`.
- Put model capabilities, context limits, and pricing in `model_prices_and_context_window.json`, not hardcoded Python lists.
- Use LiteLLM's `HTTPHandler`/`AsyncHTTPHandler` (`litellm/llms/custom_httpx/http_handler.py`) and LiteLLM exception mapping. Never import private third-party internals (such as `httpx._types`).
- Use explicit typed FastAPI routes and frozen Pydantic models. No wildcard `/{endpoint:path}` regex routers and no `response_model=dict[str, object]`.
- No inline (function-level) imports unless required by circular-import/lint rules. No code comments unless strictly required for a non-obvious invariant. No AI-generated `.md` files in PRs.
- No broad `except Exception`, no silent `try/except/pass`, no `Any`, no `typing.cast`, no `# type: ignore`, no dead code, no `print()`.
- Keep diffs small and scoped; guard shared paths/defaults against regressions with unit and e2e tests. Do not touch unrelated files, Terraform, README, or generated schemas unless required.
- Write unit tests in `tests/unit/<matching_path>` mocking the HTTP boundary (`httpx.MockTransport` / `respx`) and asserting observable behavior (no mock-echo tests).

## Commands
Run before finishing:
```bash
make lint
make lint-type-discipline
make lint-test-quality
make lint-basedpyright
make check-circular-imports
pytest tests/unit/<affected_path>
```
