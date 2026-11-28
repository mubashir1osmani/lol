# LiteLLM Agent Instructions

Proposed replacement/extension for `BerriAI/litellm/AGENTS.md`. Encodes architecture, code-quality standards, and local lint/test gates.

## Architecture and Placement
- For OpenAI-compatible providers, register them in `litellm/llms/openai_like/providers.json` (loaded by `litellm/llms/openai_like/json_loader.py` and `dynamic_config.py`) plus `model_prices_and_context_window.json` rather than writing a custom Python provider package.
- Put standalone (non-OpenAI-compatible) provider logic under `litellm/llms/<provider>/<endpoint>/transformation.py` implementing the base config under `litellm/llms/base_llm/<endpoint>/` (for example `litellm/llms/base_llm/chat/transformation.py`, `litellm/llms/base_llm/batches/`, `litellm/llms/base_llm/audio_transcription/`).
- Do not add provider-specific top-level SDK exports to `litellm/__init__.py` when standard unified endpoints or pass-through apply.
- Do not dump provider branches or one-off helpers into `litellm/main.py`, `litellm/utils.py`, or `litellm/proxy/proxy_server.py`.
- Call public dispatch entry points (`litellm.completion`, `litellm.responses.dispatch.responses`, `litellm.messages.anthropic_messages`), not internal un-dispatched functions (`.github/linters/ruff-strict.toml` `TID251`).
- Keep Rust bindings behind Python repositories in `litellm/repositories/` (`pyproject.toml` `TID251`).

## Reuse Existing Plumbing
- Reuse shared utilities before writing new helpers: `litellm/llms/<provider>/common_utils.py`, `litellm/llms/custom_httpx/http_handler.py` (`HTTPHandler`, `AsyncHTTPHandler`), `litellm/litellm_core_utils/core_helpers.py`, `litellm/litellm_core_utils/prompt_templates/factory.py`, and `litellm/litellm_core_utils/url_utils.py`.
- Send outbound HTTP through LiteLLM's HTTP handlers and map provider errors through LiteLLM exceptions (`litellm/llms/base_llm/base_utils.py`). Do not bypass handlers with raw `.client.send(...)` or return raw `httpx.Response` from SDK functions.
- Never import private third-party modules (for example `httpx._types`).
- Put model capabilities, pricing, and supported endpoints in `model_prices_and_context_window.json` (and `litellm/model_prices_and_context_window_backup.json` when updated together), never hardcoded model lists or prefix switches in Python code.

## Proxy Endpoints
- Define explicit FastAPI routes with typed Pydantic request and response models under `litellm/proxy/<feature>/`.
- Do not build catch-all `/{endpoint:path}` handlers with manual regex routing, manual 404/405 dispatch, or `response_model=dict[str, object]`.
- Do not change shared pass-through signatures (`litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`) for one provider unless it genuinely generalizes.

## Error Handling and Type Discipline
- Never write bare `except:`, blind `except Exception:` (`BLE`), or silent `try/except/pass` or `try/except/continue` (`S110`, `S112`).
- Follow strict `basedpyright` (`pyproject.toml` `[tool.pyright]`) and `litellm/scripts/check_type_discipline.py`:
  - No `Any` (`reportAny`, `reportExplicitAny`, `TID251`), no `typing.cast` (`LIT006`), no `TypeGuard`/`TypeIs` (`LIT007`), avoid `**kwargs` (`LIT008`).
  - No `# type: ignore` (`LIT009`). Only use `# pyright: ignore[rule]  # reason` when unavoidable (`LIT003`, `LIT004`). Do not use `TypeAdapter(..., arbitrary_types_allowed=True)` hacks to hide untyped calls.
  - Use immutable types in annotations (`tuple`, `Sequence`, `Mapping`, `frozenset` per `LIT001`), `: Final` bindings (`LIT010`), `ReadOnly[...]` TypedDict fields (`LIT012`), frozen Pydantic models (`ConfigDict(frozen=True)` per `LIT015`), and do not rebind or mutate parameters (`LIT011`).
  - Keep functions simple and readable: max 1 `for` + 1 `if` per comprehension (`LIT014`), complexity <= 15 (`C901`), <= 5 args (`PLR0913`), line length <= 120. No clever nested dict-unpacking spreads or `print()` (`T20`).

## Comments, Diffs, and Writing Style
- Say no to comments in code unless strictly required for a non-obvious invariant, a tool directive with a reason, or an unavoidable TODO/FIXME with an issue link. Never leave dead or commented-out code.
- Put imports at module top level; do not add inline (function-level) imports unless required by lint or circular-import rules.
- Do not add or change `.md` files unless explicitly required; keep the codebase free of AI-generated markdown files.
- Keep PRs small and focused on one logical change. Watch for regressions when touching shared code paths, defaults, or signatures, and include regression tests. Do not reformat untouched files or bundle unrelated Terraform, README, or generated schema churn.
- Fill out PR descriptions plainly and accurately; never leave template placeholders behind.
- In docs, PR text, and user-facing messages: use plain, direct engineering language. No emojis, no em-dashes, no "It's not X, it's Y", no fluff or hype.

## Testing
- Read `tests/unit/AGENTS.md` before writing tests.
- Mirror source paths under `tests/unit/<path>`. Extend existing test files for bug fixes. Register new test directories in `.github/workflows/test-unit.yml` (`tests/unit/test_assert_ci_coverage.py`).
- Mock at the HTTP boundary (`httpx.MockTransport` / `respx`). Never hit live networks or patch LiteLLM internals when an HTTP fake works (`TQ002`).
- Pass `litellm/scripts/test_quality_gate.py`: assert observable behavior (`TQ001`, `TQ002`), no `sys.path.insert` (`TQ003`), use `monkeypatch.setenv` (`TQ004`), no `litellm.<attr> =` leaks (`TQ005`), no credential-gated `pytest.skip` (`TQ006`), isolated child interpreters (`TQ009`).

## Required Verification Before Pushing
Run the `Makefile` checks on your branch:
```bash
make lint
make lint-type-discipline
make lint-test-quality
make lint-basedpyright
make check-circular-imports
make check-import-safety
pytest tests/unit/<affected_path>
```
Or run `make check` to execute the combined gate suite.
