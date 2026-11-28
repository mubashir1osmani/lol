# LiteLLM Code Quality Standards

Concrete bar for pull requests to `BerriAI/litellm`. Every rule below is grounded in the repository's existing layout, lint config, and CI gates.

## 1. Put Code in the Right Layer and Register Providers Properly
- **OpenAI-compatible providers**: Register them in `litellm/llms/openai_like/providers.json` (loaded by `litellm/llms/openai_like/json_loader.py` and `litellm/llms/openai_like/dynamic_config.py`, plus `model_prices_and_context_window.json` as required). Do not build a custom Python provider package for a provider that is OpenAI-compatible; a full custom implementation for an OpenAI-like provider is a finding.
- **Standalone (non-OpenAI-compatible) providers**: Put them under `litellm/llms/<provider>/<endpoint>/transformation.py` implementing the base config under `litellm/llms/base_llm/<endpoint>/` (for example `litellm/llms/base_llm/chat/transformation.py`, `litellm/llms/base_llm/batches/`, `litellm/llms/base_llm/audio_transcription/`) to convert the provider API to and from OpenAI format.
- Never add provider-specific top-level SDK functions to `litellm/__init__.py` (for example `litellm.<provider>_batch`) when standard unified endpoints or pass-through apply.
- Do not dump provider branches, helpers, or one-off fixes into `litellm/main.py` or `litellm/utils.py`.
- Call public dispatch entry points (`litellm.completion`, `litellm.responses.dispatch.responses`, `litellm.messages.anthropic_messages`), not internal un-dispatched functions (enforced via `TID251` in `.github/linters/ruff-strict.toml`).
- Keep Rust bindings behind Python repositories in `litellm/repositories/` (`pyproject.toml` `TID251`).

## 2. Reuse Existing Helpers and HTTP Plumbing
- Check shared helpers before writing new code: `litellm/llms/<provider>/common_utils.py`, `litellm/llms/custom_httpx/http_handler.py` (`HTTPHandler`, `AsyncHTTPHandler`), `litellm/litellm_core_utils/core_helpers.py`, `litellm/litellm_core_utils/prompt_templates/factory.py`, and `litellm/litellm_core_utils/url_utils.py`.
- Route outbound requests through LiteLLM's HTTP handlers and map errors to LiteLLM exceptions (`litellm/llms/base_llm/base_utils.py`) instead of calling raw `.client.send(...)` and returning raw `httpx.Response` or raising raw `httpx.HTTPStatusError`.
- Never import private third-party internals (for example `from httpx._types import ...`).
- Do not hardcode provider base URLs or duplicate language/model normalization when the provider module already owns them (for example `litellm/llms/<provider>/common_utils.py`).

## 3. Model Capabilities and Pricing Belong in JSON
- Put model capabilities, context windows, supported endpoints, and pricing in `model_prices_and_context_window.json` (and `litellm/model_prices_and_context_window_backup.json` when updated together), and look them up through `litellm/litellm_core_utils/model_cost_table.py` or `litellm.get_model_info`.
- Never hardcode model allowlists, model-name prefix checks, or regex capability switches in Python code where the cost map applies.

## 4. Proxy Routes Must Be Explicit and Typed
- Put new proxy routes in dedicated modules under `litellm/proxy/` rather than bloating `litellm/proxy/proxy_server.py`.
- Define explicit FastAPI routes with typed Pydantic request/response models. Do not build catch-all `/{endpoint:path}` handlers with hand-rolled regex path parsing, manual 404/405 dispatch, or `response_model=dict[str, object]`.
- Do not modify shared core pass-through signatures (`litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`) for a single niche provider unless the change genuinely generalizes across providers.

## 5. Error Handling and Regression Safety
- No bare `except:` and no blind `except Exception:` that hides failures (`BLE` in `.github/linters/ruff-strict.toml`).
- No silent `try/except/pass` or `try/except/continue` (`S110`, `S112`). Catch specific exception types, log with context when recovery is intentional, and let unexpected failures surface.
- Validate inputs explicitly and fail loudly rather than silently coercing wrong types (for example wrapping a bare `str` as a 1-tuple for a sequence field).
- **Watch for regressions**: Even a small OSS PR can break existing users. Flag changes to shared code paths (`litellm/main.py`, `litellm/utils.py`, `litellm/__init__.py`, `litellm/router.py`, `litellm/litellm_core_utils/`, `litellm/llms/base_llm/`, `litellm/llms/custom_httpx/`, `litellm/proxy/proxy_server.py`, `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`), default behavior, function signatures, or parameters that could break existing callers, and ask for regression tests.

## 6. Strict Typing, Clean Imports, Immutability, and Readable Code
Enforced by `basedpyright` (`pyproject.toml` `[tool.pyright]`), `.github/linters/ruff-strict.toml`, and `litellm/scripts/check_type_discipline.py` / `litellm/scripts/type_discipline_gate.py`:
- **No inline (function-level) imports**: Put imports at module top level. Inline imports inside functions/methods are a finding unless required by lint or circular-import rules (and explicitly marked as such).
- No `Any` (`reportAny`, `reportExplicitAny`, `TID251`), no `typing.cast` (`LIT006`), no `TypeGuard`/`TypeIs` (`LIT007`), and avoid `**kwargs` sprawl (`LIT008`).
- No `# type: ignore` (`LIT009`). Only use `# pyright: ignore[rule]  # reason` when genuinely unavoidable (`LIT003`, `LIT004`). Never wrap untyped calls in `TypeAdapter(..., arbitrary_types_allowed=True)` hacks to silence the type checker.
- Prefer immutable types in annotations (`tuple`, `Sequence`, `Mapping`, `frozenset` per `LIT001`), `: Final` bindings (`LIT010`), `ReadOnly[...]` TypedDict fields (`LIT012`), frozen Pydantic models (`ConfigDict(frozen=True)` per `LIT015`), and no parameter rebinding/mutation (`LIT011`).
- Keep code plain and readable: max 1 `for` and 1 `if` per comprehension (`LIT014`), complexity <= 15 (`C901`), <= 5 function arguments (`PLR0913`), line length <= 120. Avoid clever nested `**({...} if ... else {})` spreads, `sum(map(bool, ...))` tricks, or `PureWindowsPath` hacks when straightforward control flow is clearer.

## 7. No Code Comments, No Dead Code, No Debug Prints
- **Say no to comments in code** unless absolutely necessary to document a non-obvious invariant (or a required tool directive with a reason, or an unavoidable TODO/FIXME with an issue link). Flag added comments in code diffs.
- Remove unused imports, unused variables, commented-out code, and `print()` statements (`T20`).

## 8. Small, Focused Diffs, No `.md` Files, and Clean PR Hygiene
- **No `.md` files in PRs**: Keep the codebase free of AI-generated markdown files. Any added or modified `.md` file in a PR diff is a finding.
- One logical change per PR. Keep diffs tight and scoped to the files actually needed.
- No unrelated reformatting, no drive-by edits to Terraform (`deploy/`), root `README.md`, or generated OpenAPI/UI schemas unless directly required by the change.
- PR description must explain what changed and how it was verified. Never leave unfilled template placeholders (such as `[insert handwritten description here]`) or ship known-broken endpoints.

## 9. Tests: Unit + E2E Coverage and Third-Party Provider Proof
Enforced by `tests/unit/AGENTS.md` and `litellm/scripts/check_test_quality.py` / `litellm/scripts/test_quality_gate.py`:
- **PRs must be well tested (unit + e2e where applicable)**: Missing tests for code changes is a finding. Include regression tests when changing shared code paths or defaults.
- **New third-party providers require contributor proof**: Maintainers usually do not have credentials to run live calls against a new third-party provider. State plainly that we do not have credentials to test it and ask the contributor for proof (for example e2e run output).
- Place unit tests under `tests/unit/<path>` mirroring `litellm/<path>`. Extend existing test files for bug fixes. Register any new test directory in `.github/workflows/test-unit.yml` (verified by `tests/unit/test_assert_ci_coverage.py`).
- Mock at the HTTP boundary (`httpx.MockTransport` or `respx`). No live network calls, no sleeping, no real clocks.
- Assert observable behavior (return values, transformed payloads, raised exceptions). Do not write zero-assert tests (`TQ001`) or mock-echo tests that only check `assert_called_once_with` on patched internals (`TQ002`).
- Never use `sys.path.insert` (`TQ003`), raw `os.environ[...] =` (`TQ004`, use `monkeypatch.setenv`), `litellm.<attr> =` global leaks (`TQ005`), credential-gated `pytest.skip` (`TQ006`), or un-isolated child interpreters (`TQ009`).

## 10. Security and Local Lint Gates
- Never log, return, or hardcode API keys, bearer tokens, or credentials. Never use `eval`, `exec`, or unsafe deserialization on untrusted data.
- Pass the repo checks in `Makefile` before asking for review: `make check` / `make lint` (`ruff`, `basedpyright`, `type_discipline_gate.py`, `test_quality_gate.py`, `check-circular-imports`, `check-import-safety`).
