"""Tests for LiteLLM review standards loading, #54805-like rejection, and concise output format."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from contributor_bot.config import Config
from contributor_bot.review_agent import (
    SYSTEM_PROMPT,
    ReviewContext,
    build_system_prompt,
    execute_tool,
    tool_ping_maintainer,
    tool_submit_verdict,
)
from contributor_bot.review_github import Snapshot
from contributor_bot.review_job import perform_review
from contributor_bot.review_llm import parse_review_json
from contributor_bot.reviews import (
    DEFAULT_REVIEW_BOTS,
    MAX_REVIEW_BULLETS,
    VERDICT_BELOW_STANDARDS,
    VERDICT_LOOKS_GOOD,
    VERDICT_NEEDS_WORK,
    analyze_pr_files,
    assess_gate,
    clean_review_text,
    discord_length,
    files_from_unified_diff,
    load_review_standards,
    normalize_verdict,
    render_review,
    resolve_verdict,
    scan_md_files,
    scan_missing_tests,
    scan_patch_added_comments,
    scan_patch_inline_imports,
)
from contributor_bot.store import Store

ROOT = Path(__file__).resolve().parents[1]
HEAD = "ab" * 20
BASE = "cd" * 20
BOTS = DEFAULT_REVIEW_BOTS

_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
_FORBIDDEN_HEADINGS = ("written review:", "risks:", "missing tests:", "checklist:", "fix these before a maintainer looks:")
_FORBIDDEN_PRAISE = ("great pr", "nice work", "thanks for the contribution", "overall looks solid")


def _clean_reviews() -> list[dict]:
    return [
        {"user": {"login": login}, "commit_id": HEAD, "state": "COMMENTED", "body": "ok"}
        for login in BOTS
    ]


def _fake_sdk(on_run):
    calls: dict = {"n": 0, "instructions": None}

    class LitellmModel:
        def __init__(self, model, **kwargs):
            calls["model"] = model

    class ModelSettings:
        def __init__(self, **kwargs):
            calls["settings"] = kwargs

    class Agent:
        def __init__(self, **kwargs):
            calls["instructions"] = kwargs.get("instructions")

    class Runner:
        @staticmethod
        async def run(agent, incoming, context=None, max_turns=None, hooks=None):
            calls["n"] += 1
            on_run(context)
            return type("Result", (), {"final_output": "done"})()

    def function_tool(func=None, **kwargs):
        def deco(fn):
            return fn
        return deco(func) if callable(func) else deco

    class RunHooks:
        pass

    class RunContextWrapper:
        pass

    sdk = type("SDK", (), {})()
    sdk.Agent = Agent
    sdk.Runner = Runner
    sdk.ModelSettings = ModelSettings
    sdk.LitellmModel = LitellmModel
    sdk.function_tool = function_tool
    sdk.RunContextWrapper = RunContextWrapper
    sdk.RunHooks = RunHooks
    sdk.MaxTurnsExceeded = type("MaxTurnsExceeded", (Exception,), {})
    sdk.set_tracing_disabled = lambda disabled: None
    return sdk, calls


# Modeled directly on BerriAI/litellm#54805: green CI/bots, but ugly sprawling code.
PR_54805_LIKE_FILES = [
    {
        "filename": "litellm/__init__.py",
        "status": "modified",
        "additions": 2,
        "deletions": 0,
        "patch": (
            "@@ -1691,0 +1692,2 @@\n"
            "+from .llms.gnani.batches.jobs import agnani_batch as agnani_batch, gnani_batch as gnani_batch\n"
        ),
    },
    {
        "filename": "litellm/llms/gnani/batches/jobs.py",
        "status": "added",
        "additions": 165,
        "deletions": 0,
        "patch": (
            "@@ -0,0 +1,40 @@\n"
            "+from pathlib import PureWindowsPath\n"
            "+from httpx._types import RequestFiles\n"
            "+DEFAULT_BASE_URL = 'https://api.vachana.ai'\n"
            "+def create_batch(audio_file, language):\n"
            "+    name = PureWindowsPath(audio_file).name\n"
            "+    params = {**({'language': language} if language else {}), **({'extra': 1} if True else {})}\n"
            "+    client = get_httpx_client().client  # pyright: ignore[reportUnknownVariableType]\n"
            "+    resp = client.send(req)\n"
            "+    resp.raise_for_status()\n"
            "+    return resp\n"
        ),
    },
    {
        "filename": "litellm/proxy/pass_through_endpoints/gnani_batch.py",
        "status": "added",
        "additions": 110,
        "deletions": 0,
        "patch": (
            "@@ -0,0 +1,35 @@\n"
            "+_JOB_PATH = re.compile(r'^[0-9a-f-]+(?:/(?:status|result))?$')\n"
            "+_RESPONSE = TypeAdapter(Response, config=ConfigDict(arbitrary_types_allowed=True))\n"
            "+@router.get('/{endpoint:path}', response_model=dict[str, object])\n"
            "+@router.post('/{endpoint:path}', response_model=dict[str, object])\n"
            "+async def gnani_batch_passthrough(endpoint: str, request: Request):\n"
            "+    if not _JOB_PATH.match(endpoint):\n"
            "+        raise HTTPException(status_code=404)\n"
        ),
    },
    {
        "filename": "litellm/proxy/pass_through_endpoints/pass_through_endpoints.py",
        "status": "modified",
        "additions": 18,
        "deletions": 4,
        "patch": (
            "@@ -1730,6 +1734,12 @@ def pass_through_request(\n"
            "+    upstream_headers_filter=None,\n"
            "+    custom_multipart_override=False,\n"
        ),
    },
    {
        "filename": "deploy/terraform/aws/locals.tf",
        "status": "modified",
        "additions": 4,
        "deletions": 0,
        "patch": "@@ -10,0 +11,4 @@\n+  gnani_batch_enabled = true\n",
    },
    {
        "filename": "README.md",
        "status": "modified",
        "additions": 12,
        "deletions": 1,
        "patch": "@@ -40,0 +41,12 @@\n+Gnani batch endpoint\n",
    },
    {
        "filename": "ui/litellm-dashboard/src/types/schema.d.ts",
        "status": "modified",
        "additions": 350,
        "deletions": 10,
        "patch": "@@ -1,3 +1,350 @@\n+// generated openapi churn\n",
    },
]

PR_54805_LIKE_BODY = (
    "[insert handwritten description here]\n\n"
    "Adds Gnani Prisma v2.5 and v3.0 batch transcription. "
    "Note: presigned upload still returns 404 from Gnani."
)


class StandardsDocTests(unittest.TestCase):
    def test_standards_and_repo_agent_docs_exist_and_cite_real_litellm_paths(self):
        for rel in (
            "review_standards/LITELLM_STANDARDS.md",
            "review_standards/proposed_litellm_AGENTS.md",
            "review_standards/proposed_litellm_CLAUDE.md",
            "AGENTS.md",
            "CLAUDE.md",
            "ruff.toml",
        ):
            path = ROOT / rel
            self.assertTrue(path.is_file(), f"missing {rel}")
            self.assertGreater(len(path.read_text(encoding="utf-8").strip()), 100)

        standards = load_review_standards()
        for expected in (
            "litellm/llms/openai_like/providers.json",
            "litellm/llms/<provider>/<endpoint>/transformation.py",
            "litellm/__init__.py",
            "litellm/main.py",
            "litellm/llms/custom_httpx/http_handler.py",
            "model_prices_and_context_window.json",
            "litellm/proxy/pass_through_endpoints/pass_through_endpoints.py",
            ".github/linters/ruff-strict.toml",
            "tests/unit/AGENTS.md",
            "httpx._types",
            "No `.md` files in PRs",
            "No inline (function-level) imports",
            "Say no to comments in code",
            "credentials to test it",
            "regression tests",
        ):
            self.assertIn(expected, standards)

    def test_system_prompt_embeds_standards_and_direct_output_rules(self):
        prompt = build_system_prompt()
        self.assertEqual(prompt, SYSTEM_PROMPT)
        self.assertIn("=== LITELLM CODE QUALITY STANDARDS ===", prompt)
        self.assertIn("read_file_diff", prompt)
        self.assertIn(VERDICT_LOOKS_GOOD, prompt)
        self.assertIn(VERDICT_NEEDS_WORK, prompt)
        self.assertIn(VERDICT_BELOW_STANDARDS, prompt)
        self.assertIn("untrusted", prompt.lower())
        self.assertIn("no emojis", prompt.lower())
        self.assertIn("no headings", prompt.lower())
        self.assertIn("model_prices_and_context_window.json", prompt)
        self.assertIn("litellm/llms/openai_like/providers.json", prompt)
        self.assertIn("do not have credentials to test it", prompt.lower())
        self.assertIn("regression", prompt.lower())
        self.assertIn("inline (function-level) imports", prompt.lower())
        self.assertIn("standards_signals", prompt)


class OutputFormatTests(unittest.TestCase):
    def test_clean_review_is_one_line_with_no_fluff_or_emojis(self):
        rendered = render_review(VERDICT_LOOKS_GOOD)
        self.assertEqual(rendered, "looks good")
        self.assertEqual(len(rendered.splitlines()), 1)
        self.assertIsNone(_EMOJI_RE.search(rendered))
        self.assertNotIn("—", rendered)

    def test_fluff_praise_headings_emojis_and_excess_bullets_are_stripped(self):
        noisy_bullets = [
            "Great PR! ✅ litellm/main.py:12 — provider branch belongs in litellm/llms/foo/",
            "Thanks for the contribution! written review: litellm/utils.py:44 - broad except Exception swallows errors 🚀",
            "## Summary: litellm/foo.py:8 - imports private httpx._types.RequestFiles",
            "risks: litellm/bar.py:19 - hardcoded model list instead of model_prices_and_context_window.json",
            "litellm/baz.py:30 - missing unit test under tests/unit/",
            "litellm/extra.py:99 - sixth bullet must be dropped by cap",
            "litellm/extra2.py:101 - seventh bullet must be dropped by cap",
        ]
        rendered = render_review(VERDICT_BELOW_STANDARDS, noisy_bullets)
        lines = rendered.splitlines()
        self.assertEqual(lines[0], VERDICT_BELOW_STANDARDS)
        bullet_lines = lines[1:]
        self.assertEqual(len(bullet_lines), MAX_REVIEW_BULLETS)
        self.assertTrue(all(line.startswith("- ") for line in bullet_lines))
        self.assertIsNone(_EMOJI_RE.search(rendered))
        self.assertNotIn("—", rendered)
        lowered = rendered.lower()
        for heading in _FORBIDDEN_HEADINGS:
            self.assertNotIn(heading, lowered)
        for praise in _FORBIDDEN_PRAISE:
            self.assertNotIn(praise, lowered)
        self.assertNotIn("sixth bullet", rendered)
        self.assertLessEqual(discord_length(rendered), 2000)

    def test_clean_review_text_blocks_mentions_and_normalize_verdict(self):
        cleaned = clean_review_text("Great PR! @everyone <@&999> fix litellm/a.py:1 — bad catch ✅")
        self.assertNotIn("@everyone", cleaned)
        self.assertNotIn("<@&999>", cleaned)
        self.assertNotIn("Great PR", cleaned)
        self.assertNotIn("✅", cleaned)
        self.assertNotIn("—", cleaned)
        self.assertEqual(normalize_verdict("looks good"), VERDICT_LOOKS_GOOD)
        self.assertEqual(normalize_verdict("Needs Work."), VERDICT_NEEDS_WORK)
        self.assertEqual(normalize_verdict("doesn\u2019t meet our standards!"), VERDICT_BELOW_STANDARDS)
        self.assertEqual(normalize_verdict("doesnt meet our standards"), VERDICT_BELOW_STANDARDS)

    def test_deterministic_gate_prevents_looks_good_when_ci_fails(self):
        gate = assess_gate(
            bots=BOTS,
            head_sha=HEAD,
            draft=False,
            merged=False,
            state="open",
            mergeable=True,
            mergeable_state="clean",
            reviews=_clean_reviews(),
            comments=[],
            threads=[],
            check_runs=[{"name": "unit-tests", "status": "completed", "conclusion": "failure"}],
            combined_state=None,
        )
        verdict, ready, bullets = resolve_verdict(gate, requested_verdict=VERDICT_LOOKS_GOOD, findings=[])
        self.assertEqual(verdict, VERDICT_NEEDS_WORK)
        self.assertFalse(ready)
        self.assertTrue(any("ci failing: unit-tests" in b for b in bullets))

    def test_review_llm_parses_standards_verdict_and_findings(self):
        payload = json.dumps({
            "verdict": "doesn't meet our standards",
            "findings": [
                {"path": "litellm/__init__.py", "line": 1694, "reason": "exports provider-specific gnani_batch at top level"},
                {"path": "litellm/llms/gnani/batches/jobs.py", "line": 11, "reason": "imports private httpx._types"},
            ],
        })
        parsed = parse_review_json(payload)
        self.assertEqual(parsed["verdict"], VERDICT_BELOW_STANDARDS)
        self.assertEqual(len(parsed["findings"]), 2)
        self.assertIn("litellm/__init__.py:1694", parsed["findings"][0])


def _inspect_fixture_against_standards(ctx: ReviewContext) -> None:
    """Simulated standards-checking agent that inspects the #54805-like diff via tools."""
    bundle = json.loads(execute_tool(ctx, "review_bundle", {}))
    findings: list[dict] = []
    for entry in bundle.get("files") or []:
        path = entry.get("path") or ""
        if not path.endswith(".py"):
            continue
        diff_payload = json.loads(execute_tool(ctx, "read_file_diff", {"path": path}))
        patch = diff_payload.get("patch") or ""
        if path == "litellm/__init__.py" and "gnani_batch" in patch:
            findings.append({
                "path": path,
                "line": 1694,
                "reason": "exports bespoke top-level gnani_batch/agnani_batch instead of using the standard batch transformation layer",
            })
        if "httpx._types" in patch or ".client.send(" in patch:
            findings.append({
                "path": path,
                "line": 11,
                "reason": "imports private httpx._types and bypasses LiteLLM HTTPHandler and exception mapping",
            })
        if "/{endpoint:path}" in patch and "dict[str, object]" in patch:
            findings.append({
                "path": path,
                "line": 22,
                "reason": "catch-all regex router with response_model=dict[str, object] and TypeAdapter hack instead of explicit typed FastAPI routes",
            })
        if path.endswith("pass_through_endpoints.py") and "upstream_headers_filter" in patch:
            findings.append({
                "path": path,
                "line": 1734,
                "reason": "mutates shared pass_through_request signature and spans Terraform/README/generated schema churn for one provider",
            })
    if "[insert handwritten description here]" in (bundle.get("body_preview") or ""):
        findings.append({
            "path": "litellm/llms/gnani/batches/jobs.py",
            "line": 24,
            "reason": "hardcodes provider URL with PureWindowsPath/dict-unpacking hacks and leaves unfilled PR template placeholder",
        })
    tool_submit_verdict(ctx, {
        "verdict": VERDICT_BELOW_STANDARDS if len(findings) >= 2 else VERDICT_LOOKS_GOOD,
        "findings": findings,
    })


class PR54805UglyCodeFixtureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(str(Path(self.directory.name) / "roles.db"))
        self.addCleanup(self.store.close)
        self.config = replace(
            Config("token", "123", "456", github_token="gh"),
            pr_review_enabled=True,
            review_channel_ids=("321",),
            review_model="anthropic/claude-sonnet-4-5",
            review_bot_logins=BOTS,
            review_maintainer_role_id="555",
            review_cooldown_seconds=600,
            discord_guild_id="333",
            discord_application_id="444",
        )

    def _ugly_snapshot(self) -> Snapshot:
        return Snapshot(
            number=54805,
            title="feat(gnani): add Prisma v2.5 and v3.0 batch transcription",
            head_sha=HEAD,
            base_sha=BASE,
            draft=False,
            merged=False,
            state="open",
            mergeable=True,
            mergeable_state="clean",
            author_login="someone",
            author_id=42,
            reviews=_clean_reviews(),
            comments=[],
            threads=[],
            check_runs=[{"name": "unit-tests", "status": "completed", "conclusion": "success"}],
            combined_state=None,
            body=PR_54805_LIKE_BODY,
        )

    def test_ugly_54805_fixture_gets_blunt_below_standards_verdict_even_with_green_ci(self):
        self.store.link_account("111", "someone", 42, "2026-11-01T00:00:00+00:00")
        captured: dict = {}

        def on_run(ctx: ReviewContext) -> None:
            _inspect_fixture_against_standards(ctx)
            captured["ctx"] = ctx
            captured["ping_attempt"] = json.loads(tool_ping_maintainer(ctx, {}))

        sdk, calls = _fake_sdk(on_run)
        with mock.patch("contributor_bot.review_agent.gather_snapshot", return_value=self._ugly_snapshot()), \
                mock.patch("contributor_bot.review_agent.github_get_json", return_value=PR_54805_LIKE_FILES), \
                mock.patch("contributor_bot.review_agent._load_sdk", return_value=sdk), \
                mock.patch("contributor_bot.review_job.fetch_recent_messages", return_value=[]), \
                mock.patch("contributor_bot.review_job.start_public_thread", return_value={"id": "900"}), \
                mock.patch("contributor_bot.review_agent.post_message", return_value={"id": "1", "channel_id": "900"}) as post, \
                mock.patch("contributor_bot.api.post_issue_comment", side_effect=AssertionError("github write")):
            result = perform_review(self.config, self.store, 54805, channel_id="321", message_id="654", source="new")

        self.assertEqual(result["status"], "posted")
        self.assertIn("=== LITELLM CODE QUALITY STANDARDS ===", calls["instructions"])

        ctx = captured["ctx"]
        self.assertEqual(ctx.verdict, VERDICT_BELOW_STANDARDS)
        self.assertFalse(ctx.verdict_ready)
        self.assertIn("refusing", captured["ping_attempt"]["error"])

        contents = [call.args[2]["content"] for call in post.call_args_list]
        review_msg = contents[0]
        lines = review_msg.splitlines()
        self.assertEqual(lines[0], VERDICT_BELOW_STANDARDS)

        bullets = lines[1:]
        self.assertGreaterEqual(len(bullets), 3)
        self.assertLessEqual(len(bullets), MAX_REVIEW_BULLETS)
        self.assertTrue(all(line.startswith("- ") for line in bullets))
        self.assertIn("litellm/__init__.py:1694", review_msg)
        self.assertIn("litellm/llms/gnani/batches/jobs.py:11", review_msg)
        self.assertIn("litellm/proxy/pass_through_endpoints/gnani_batch.py:22", review_msg)

        self.assertIsNone(_EMOJI_RE.search(review_msg))
        self.assertNotIn("—", review_msg)
        lowered = review_msg.lower()
        for heading in _FORBIDDEN_HEADINGS:
            self.assertNotIn(heading, lowered)
        for praise in _FORBIDDEN_PRAISE:
            self.assertNotIn(praise, lowered)
        self.assertNotIn("<@&555>", "\n".join(contents))
        self.assertTrue(all(call.kwargs.get("mention_roles") in (None, []) for call in post.call_args_list))


class StrictRuleChecksTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(str(Path(self.directory.name) / "roles.db"))
        self.addCleanup(self.store.close)
        self.config = replace(
            Config("token", "123", "456", github_token="gh"),
            pr_review_enabled=True,
            review_channel_ids=("321",),
            review_model="anthropic/claude-sonnet-4-5",
            review_bot_logins=BOTS,
            review_maintainer_role_id="555",
            review_cooldown_seconds=600,
            discord_guild_id="333",
            discord_application_id="444",
        )

    def _green_snapshot(self, *, diff: str = "") -> Snapshot:
        return Snapshot(
            number=55001,
            title="fix helper",
            head_sha=HEAD,
            base_sha=BASE,
            draft=False,
            merged=False,
            state="open",
            mergeable=True,
            mergeable_state="clean",
            author_login="contrib",
            author_id=7,
            reviews=_clean_reviews(),
            comments=[],
            threads=[],
            check_runs=[{"name": "unit-tests", "status": "completed", "conclusion": "success"}],
            combined_state=None,
            diff=diff,
        )

    def test_rule1_added_comments_detected_and_directives_ignored(self):
        patch = (
            "@@ -10,2 +10,13 @@ def transform(payload):\n"
            "     x = 1\n"
            "+    # increment counter by one\n"
            "+    x += 1  # bump x\n"
            "+    y = '# not a comment inside string'\n"
            "+    z = x  # pyright: ignore[reportUnknownVariableType]  # sdk returns Any\n"
            "+    w = x  # noqa: E501\n"
            "+    # TODO(#1234): remove after migration\n"
            '+    doc = """\n'
            "+    # comment inside triple-quoted docstring\n"
            '+    """\n'
        )
        found = scan_patch_added_comments("litellm/llms/foo/chat/transformation.py", patch)
        self.assertEqual([item["line"] for item in found], [11, 12])
        self.assertIn("increment counter", found[0]["text"])
        self.assertIn("bump x", found[1]["text"])

    def test_rule2_md_files_deterministically_flagged_and_block_looks_good(self):
        files = [
            {"filename": "NOTES.md", "status": "added", "patch": "@@ -0,0 +1 @@\n+# AI notes\n"},
            {"filename": "docs/GUIDE.md", "status": "modified", "patch": "@@ -1 +1 @@\n+updated\n"},
            {"filename": "OLD_AI_SLOP.md", "status": "removed", "patch": ""},
            {"filename": "tests/unit/test_ok.py", "status": "modified", "patch": "@@ -1 +1 @@\n+assert True\n"},
        ]
        md_findings = scan_md_files(files)
        self.assertEqual([item["path"] for item in md_findings], ["NOTES.md", "docs/GUIDE.md"])

        ctx = ReviewContext(config=self.config, store=self.store, number=55001, thread_id="900")
        with mock.patch("contributor_bot.review_agent.gather_snapshot", return_value=self._green_snapshot()), \
                mock.patch("contributor_bot.review_agent.github_get_json", return_value=files), \
                mock.patch("contributor_bot.review_agent.post_message", return_value={"id": "1", "channel_id": "900"}) as post:
            bundle = json.loads(execute_tool(ctx, "review_bundle", {}))
            self.assertEqual(bundle["standards_signals"]["md_files"], ["NOTES.md", "docs/GUIDE.md"])
            # Even if the model tries to say "looks good", deterministic .md check blocks it.
            verdict_res = json.loads(tool_submit_verdict(ctx, {"verdict": VERDICT_LOOKS_GOOD, "findings": []}))
            ping_res = json.loads(tool_ping_maintainer(ctx, {}))

        self.assertEqual(verdict_res["verdict"], VERDICT_NEEDS_WORK)
        self.assertFalse(verdict_res["ready"])
        self.assertIn("refusing", ping_res["error"])
        posted = post.call_args_list[0].args[2]["content"]
        self.assertTrue(posted.startswith("needs work\n- "))
        self.assertIn("NOTES.md", posted)
        self.assertNotIn("OLD_AI_SLOP.md", posted)

    def test_rule2_md_files_caught_via_unified_diff_fallback(self):
        diff = (
            "diff --git a/SUMMARY.md b/SUMMARY.md\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/SUMMARY.md\n"
            "@@ -0,0 +1 @@\n"
            "+# generated summary\n"
        )
        entries = files_from_unified_diff(diff)
        self.assertEqual(entries[0]["filename"], "SUMMARY.md")
        self.assertEqual(entries[0]["status"], "added")
        ctx = ReviewContext(config=self.config, store=self.store, number=55001, thread_id="900")
        with mock.patch("contributor_bot.review_agent.gather_snapshot", return_value=self._green_snapshot(diff=diff)), \
                mock.patch("contributor_bot.review_agent.post_message", return_value={"id": "1", "channel_id": "900"}):
            res = json.loads(tool_submit_verdict(ctx, {"verdict": VERDICT_LOOKS_GOOD, "findings": []}))
        self.assertEqual(res["verdict"], VERDICT_NEEDS_WORK)
        self.assertFalse(res["ready"])
        self.assertTrue(any("SUMMARY.md" in reason for reason in res["reasons"]))

    def test_rule3_and_rule4_and_rule5_signals_in_review_bundle(self):
        files = [
            {
                "filename": "litellm/llms/acme/chat/transformation.py",
                "status": "added",
                "patch": "@@ -0,0 +1,4 @@\n+# custom client\n+class AcmeConfig:\n+    pass\n",
            },
            {
                "filename": "litellm/llms/openai_like/providers.json",
                "status": "modified",
                "patch": "@@ -1 +1 @@\n+{}\n",
            },
            {
                "filename": "litellm/main.py",
                "status": "modified",
                "patch": "@@ -100,2 +100,3 @@\n+    default_timeout = 5\n",
            },
        ]
        analysis = analyze_pr_files(files)
        self.assertEqual(analysis["new_provider_dirs"], ["acme"])
        self.assertTrue(analysis["touches_openai_like_providers_json"])
        self.assertEqual(analysis["shared_paths_touched"], ["litellm/main.py"])
        self.assertTrue(analysis["missing_tests"])
        self.assertEqual(len(analysis["added_comments"]), 1)
        self.assertEqual(analysis["added_comments"][0]["path"], "litellm/llms/acme/chat/transformation.py")

        ctx = ReviewContext(config=self.config, store=self.store, number=55001, thread_id="900")
        with mock.patch("contributor_bot.review_agent.gather_snapshot", return_value=self._green_snapshot()), \
                mock.patch("contributor_bot.review_agent.github_get_json", return_value=files):
            bundle = json.loads(execute_tool(ctx, "review_bundle", {}))
        signals = bundle["standards_signals"]
        self.assertEqual(signals["new_provider_dirs"], ["acme"])
        self.assertEqual(signals["shared_paths_touched"], ["litellm/main.py"])
        self.assertTrue(signals["missing_tests"])

    def test_rule5_missing_tests_deterministically_blocks_looks_good(self):
        code_only = [
            {"filename": "litellm/utils.py", "status": "modified", "patch": "@@ -1 +1 @@\n+x = 1\n"},
        ]
        self.assertEqual(len(scan_missing_tests(code_only)), 1)
        with_tests = code_only + [
            {"filename": "tests/unit/test_utils.py", "status": "modified", "patch": "@@ -1 +1 @@\n+assert True\n"},
        ]
        self.assertEqual(scan_missing_tests(with_tests), [])

        ctx = ReviewContext(config=self.config, store=self.store, number=55001, thread_id="900")
        with mock.patch("contributor_bot.review_agent.gather_snapshot", return_value=self._green_snapshot()), \
                mock.patch("contributor_bot.review_agent.github_get_json", return_value=code_only), \
                mock.patch("contributor_bot.review_agent.post_message", return_value={"id": "1", "channel_id": "900"}) as post:
            execute_tool(ctx, "review_bundle", {})
            res = json.loads(tool_submit_verdict(ctx, {"verdict": VERDICT_LOOKS_GOOD, "findings": []}))
        self.assertEqual(res["verdict"], VERDICT_NEEDS_WORK)
        self.assertFalse(res["ready"])
        self.assertIn("no tests added or updated", post.call_args_list[0].args[2]["content"])

    def test_rule6_inline_imports_deterministically_flagged_with_exemptions(self):
        patch = (
            "@@ -1,4 +1,22 @@\n"
            "+import os\n"
            "+if TYPE_CHECKING:\n"
            "+    from litellm.types.utils import ModelResponse\n"
            "+try:\n"
            "+    import uvloop\n"
            "+except ImportError:\n"
            "+    import asyncio\n"
            "+def handle_request():\n"
            "+    import json\n"
            "+    from litellm.utils import get_model_info\n"
            "+    from litellm.main import completion  # circular import\n"
            "+    import litellm  # noqa: PLC0415\n"
            '+    snippet = """\n'
            "+    import fake_in_docstring\n"
            '+    """\n'
        )
        found = scan_patch_inline_imports("litellm/llms/foo/chat/transformation.py", patch)
        self.assertEqual([item["line"] for item in found], [9, 10])
        self.assertTrue(all("inline import" in item["reason"] for item in found))

        # Hunk starting inside an existing TYPE_CHECKING block should stay exempt.
        tc_hunk = (
            "@@ -12,2 +12,3 @@ if TYPE_CHECKING:\n"
            "     from typing import Any\n"
            "+    from httpx import Client\n"
        )
        self.assertEqual(scan_patch_inline_imports("litellm/llms/foo/types.py", tc_hunk), [])

        # End-to-end: green CI + tests present + LLM says looks good, still blocked by inline import!
        files = [
            {"filename": "litellm/llms/foo/chat/transformation.py", "status": "modified", "patch": patch},
            {"filename": "tests/unit/llms/foo/test_transformation.py", "status": "added", "patch": "@@ -0,0 +1 @@\n+assert True\n"},
        ]
        ctx = ReviewContext(config=self.config, store=self.store, number=55001, thread_id="900")
        with mock.patch("contributor_bot.review_agent.gather_snapshot", return_value=self._green_snapshot()), \
                mock.patch("contributor_bot.review_agent.github_get_json", return_value=files), \
                mock.patch("contributor_bot.review_agent.post_message", return_value={"id": "1", "channel_id": "900"}) as post:
            bundle = json.loads(execute_tool(ctx, "review_bundle", {}))
            self.assertEqual(len(bundle["standards_signals"]["inline_imports"]), 2)
            res = json.loads(tool_submit_verdict(ctx, {"verdict": VERDICT_LOOKS_GOOD, "findings": []}))
            ping = json.loads(tool_ping_maintainer(ctx, {}))

        self.assertEqual(res["verdict"], VERDICT_NEEDS_WORK)
        self.assertFalse(res["ready"])
        self.assertIn("refusing", ping["error"])
        posted = post.call_args_list[0].args[2]["content"]
        self.assertIn("litellm/llms/foo/chat/transformation.py:9", posted)
        self.assertIn("inline import", posted)

    def test_deterministic_findings_deduplicate_against_agent_findings_and_respect_5_bullet_cap(self):
        gate = assess_gate(
            bots=BOTS,
            head_sha=HEAD,
            draft=False,
            merged=False,
            state="open",
            mergeable=True,
            mergeable_state="clean",
            reviews=_clean_reviews(),
            comments=[],
            threads=[],
            check_runs=[{"name": "unit-tests", "status": "completed", "conclusion": "success"}],
            combined_state=None,
        )
        agent_findings = [
            {"path": "NOTES.md", "line": None, "reason": "remove AI-generated markdown file"},
            {"path": "litellm/foo.py", "line": 14, "reason": "move inline import to top of file"},
        ]
        det_findings = [
            {"path": "NOTES.md", "line": None, "reason": "no .md files in PRs; remove added/changed markdown file", "rule": "no_md_files"},
            {"path": "litellm/foo.py", "line": 14, "reason": "inline import inside function; move to top-level unless required for circular imports", "rule": "no_inline_imports"},
            {"path": "litellm/foo.py", "line": None, "reason": "no tests added or updated; add unit and e2e tests covering this change", "rule": "missing_tests"},
        ]
        verdict, ready, bullets = resolve_verdict(
            gate,
            requested_verdict=VERDICT_NEEDS_WORK,
            findings=agent_findings,
            deterministic_findings=det_findings,
        )
        self.assertEqual(verdict, VERDICT_NEEDS_WORK)
        self.assertFalse(ready)
        # NOTES.md and litellm/foo.py:14 are deduplicated; missing_tests is appended -> 3 bullets total.
        self.assertEqual(len(bullets), 3)
        self.assertLessEqual(len(bullets), MAX_REVIEW_BULLETS)
        self.assertTrue(any("no tests added or updated" in b for b in bullets))


if __name__ == "__main__":
    unittest.main()
