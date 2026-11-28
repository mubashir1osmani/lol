"""Tests for the three-agent Aria architecture, single message router,
support_agent with mocked LiteLLM KB, PR review lifecycle tracking,
and OpenInference AGENT span verification with an in-memory span exporter.
"""

from __future__ import annotations

import contextvars
import json
import tempfile
import types
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from contributor_bot import engagement_agent, message_router, pr_review_agent, support_agent
from contributor_bot.config import Config, get_config
from contributor_bot.matcher import DiscordMemberInfo
from contributor_bot.review_agent import execute_tool
from contributor_bot.schedule import UTC
from contributor_bot.store import Store
from contributor_bot.support_agent import (
    LiteLLMKnowledgeBase,
    answer_support_question,
    format_support_reply,
    reset_support_cooldowns,
)
from contributor_bot.tracing import (
    AGENT_ENGAGEMENT,
    AGENT_PR_REVIEW,
    AGENT_SUPPORT,
    OPENINFERENCE_AGENT_NAME_ATTR,
    OPENINFERENCE_SPAN_KIND_AGENT,
    OPENINFERENCE_SPAN_KIND_ATTR,
    OPENINFERENCE_SPAN_KIND_CHAIN,
    OPENINFERENCE_SPAN_KIND_RETRIEVER,
    OPENINFERENCE_SPAN_KIND_TOOL,
    OPENINFERENCE_TOOL_NAME_ATTR,
    TOP_LEVEL_AGENTS,
    _reset_tracing_state,
    install_tracer_provider,
)


# ---------------------------------------------------------------------------
# In-memory span exporter harness
# Uses real opentelemetry-sdk InMemorySpanExporter when installed, and a
# faithful stdlib InMemorySpanExporter when running offline in CI.
# ---------------------------------------------------------------------------

_FALLBACK_ACTIVE_SPAN: contextvars.ContextVar = contextvars.ContextVar("_inmem_active_span", default=None)


class _StdlibRecordedSpan:
    def __init__(self, name: str, parent=None) -> None:
        self.name = name
        self.parent = parent
        self.attributes: dict[str, object] = {}
        self.status = None
        self.exceptions: list[Exception] = []

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def record_exception(self, exc: Exception) -> None:
        self.exceptions.append(exc)

    def set_status(self, status) -> None:
        self.status = status


class _StdlibInMemorySpanExporter:
    def __init__(self) -> None:
        self._spans: list[_StdlibRecordedSpan] = []

    def get_finished_spans(self) -> list[_StdlibRecordedSpan]:
        return list(self._spans)

    def clear(self) -> None:
        self._spans.clear()


class _StdlibTracer:
    def __init__(self, exporter: _StdlibInMemorySpanExporter) -> None:
        self._exporter = exporter

    @contextmanager
    def start_as_current_span(self, name: str):
        parent = _FALLBACK_ACTIVE_SPAN.get()
        span = _StdlibRecordedSpan(name, parent=parent)
        token = _FALLBACK_ACTIVE_SPAN.set(span)
        try:
            yield span
        finally:
            _FALLBACK_ACTIVE_SPAN.reset(token)
            self._exporter._spans.append(span)


class _StdlibTracerProvider:
    def __init__(self, exporter: _StdlibInMemorySpanExporter) -> None:
        self._exporter = exporter
        self.flush_count = 0

    def get_tracer(self, _name: str):
        return _StdlibTracer(self._exporter)

    def force_flush(self, timeout_millis=None) -> bool:
        self.flush_count += 1
        return True

    def shutdown(self) -> None:
        pass


def _install_in_memory_exporter():
    """Install an in-memory span exporter (real OTel SDK if available, else stdlib)."""
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider(resource=Resource.create({"service.name": "aria-litellm-discord"}))
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        install_tracer_provider(provider)
        return exporter, provider, True
    except ImportError:
        exporter = _StdlibInMemorySpanExporter()
        provider = _StdlibTracerProvider(exporter)
        install_tracer_provider(provider)
        return exporter, provider, False


def _span_attrs(span) -> dict:
    return dict(getattr(span, "attributes", None) or {})


def _span_parent_name(span, all_spans) -> str | None:
    parent = getattr(span, "parent", None)
    if parent is None:
        return None
    if isinstance(parent, _StdlibRecordedSpan):
        return parent.name
    # Real OTel SpanContext parent: match by span_id.
    parent_id = getattr(parent, "span_id", None)
    if parent_id is None:
        return None
    for candidate in all_spans:
        ctx = getattr(candidate, "context", None)
        if ctx is not None and getattr(ctx, "span_id", None) == parent_id:
            return candidate.name
    return None


MOCK_LLMS_TXT = """# LiteLLM Docs

## Proxy Server
- [Proxy Quick Start](https://docs.litellm.ai/docs/proxy/quick_start): Start the LiteLLM Proxy with config.yaml and set model_list and router_settings.
- [Router Fallbacks and Retries](https://docs.litellm.ai/docs/proxy/reliability): Configure fallbacks, num_retries, and timeout on the LiteLLM Proxy.
- [Virtual Keys and Budgets](https://docs.litellm.ai/docs/proxy/virtual_keys): Create virtual keys with max_budget and rpm_limit using /key/generate.
"""

MOCK_BLOG_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>LiteLLM Blog</title>
  <item>
    <title>Prompt Caching with Anthropic and OpenAI</title>
    <link>https://docs.litellm.ai/blog/prompt-caching</link>
    <description>How to enable prompt caching in LiteLLM Proxy using cache_control_injection_points.</description>
  </item>
</channel></rss>
"""

MOCK_README = """# LiteLLM

Call 100+ LLM APIs using the OpenAI format.

## Installation

Install LiteLLM with `pip install litellm` or `pip install 'litellm[proxy]'` for the proxy server.

## Quick Start

Set `OPENAI_API_KEY` and call `litellm.completion(model="openai/gpt-4o", messages=[...])`.
"""

MOCK_RELIABILITY_MD = """# Router Reliability

To configure fallbacks and retries in `config.yaml`:

```yaml
router_settings:
  num_retries: 3
  timeout: 30
  fallbacks:
    - {"gpt-4o": ["claude-3-5-sonnet"]}
```
"""


def _make_mock_kb(store: Store | None = None, cache_path: Path | None = None) -> LiteLLMKnowledgeBase:
    kb = LiteLLMKnowledgeBase(
        store=store,
        cache_path=cache_path,
        repository="BerriAI/litellm",
        refresh_hours=12,
        llms_txt_fetcher=lambda: MOCK_LLMS_TXT,
        blog_rss_fetcher=lambda: MOCK_BLOG_RSS,
        readme_fetcher=lambda: MOCK_README,
        doc_page_fetcher=lambda url: MOCK_RELIABILITY_MD if "reliability" in url else "# Quick Start\nUse config.yaml.",
    )
    kb.refresh(now=datetime(2026, 11, 28, 12, 0, tzinfo=UTC))
    return kb


def _base_config(db_path: str, **overrides) -> Config:
    defaults = {
        "discord_token": "tok-test",
        "contributions_channel_id": "111111111111111111",
        "announcements_channel_id": "222222222222222222",
        "github_repository": "BerriAI/litellm",
        "roles_db_path": db_path,
        "pr_review_enabled": True,
        "review_channel_ids": ("333333333333333333",),
        "review_model": "anthropic/claude-sonnet-4-5",
        "review_classifier_model": "anthropic/claude-haiku-4-5",
        "review_classify": False,
        "support_agent_enabled": True,
        "support_agent_channel_ids": ("444444444444444444", "333333333333333333"),
        "support_agent_model": "anthropic/claude-haiku-4-5",
        "support_kb_refresh_hours": 12,
        "fast_model": "anthropic/claude-haiku-4-5",
    }
    defaults.update(overrides)
    return Config(**defaults)


class ThreeAgentsArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_tracing_state()
        reset_support_cooldowns()
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "state.db")
        self.store = Store(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        _reset_tracing_state()
        reset_support_cooldowns()

    def test_each_agent_module_exposes_name_prompt_tools_and_entry_points(self) -> None:
        self.assertEqual(pr_review_agent.AGENT_NAME, "pr_review_agent")
        self.assertTrue(pr_review_agent.PROMPT)
        self.assertTrue(pr_review_agent.TOOLS)
        self.assertTrue(callable(pr_review_agent.handle_routed_message))
        self.assertTrue(callable(pr_review_agent.handle_thread_followup))
        self.assertTrue(callable(pr_review_agent.run_review))
        self.assertTrue(callable(pr_review_agent.poll_review_thread_lifecycle))

        self.assertEqual(support_agent.AGENT_NAME, "support_agent")
        self.assertTrue(support_agent.PROMPT)
        self.assertTrue(support_agent.TOOLS)
        self.assertTrue(callable(support_agent.handle_routed_message))
        self.assertTrue(callable(support_agent.answer_support_question))
        self.assertTrue(callable(support_agent.refresh_knowledge_base))

        self.assertEqual(engagement_agent.AGENT_NAME, "engagement_agent")
        self.assertTrue(engagement_agent.PROMPT)
        self.assertTrue(engagement_agent.TOOLS)
        self.assertTrue(callable(engagement_agent.run_role_matching))
        self.assertTrue(callable(engagement_agent.run_weekly_summary))
        self.assertTrue(callable(engagement_agent.run_daily_roles))

    def test_in_memory_exporter_records_exact_three_agent_spans(self) -> None:
        exporter, _provider, _is_real = _install_in_memory_exporter()
        config = _base_config(self.db_path)
        kb = _make_mock_kb(self.store)
        exporter.clear()

        # 1. pr_review_agent: run_review + execute_tool child TOOL span
        class _DummyRunner:
            @staticmethod
            async def run(_agent, _prompt, context=None, max_turns=None, hooks=None):
                execute_tool(context, "lookup_contributor", {"github_login": "octocat"})
                return types.SimpleNamespace(final_output="lgtm")

        sdk = types.SimpleNamespace(
            Agent=lambda **kw: types.SimpleNamespace(**kw),
            LitellmModel=lambda **kw: types.SimpleNamespace(**kw),
            ModelSettings=lambda **kw: types.SimpleNamespace(**kw),
            Runner=_DummyRunner,
            RunHooks=object,
            RunContextWrapper=object,
            function_tool=lambda fn=None, **kw: fn if callable(fn) else (lambda f: f),
            set_tracing_disabled=lambda _flag: None,
        )
        with mock.patch("contributor_bot.review_agent._load_sdk", return_value=sdk), \
             mock.patch("contributor_bot.review_agent._post_text"), \
             mock.patch("contributor_bot.review_agent.tool_lookup_contributor", return_value='{"github_login":"octocat"}'), \
             mock.patch.dict("contributor_bot.review_agent._HANDLERS", {"lookup_contributor": lambda _ctx, _args: '{"github_login":"octocat"}'}):
            pr_review_agent.run_review(config, self.store, 1234, "999888777", mode="review")

        # 2. support_agent: routed support question with mocked KB
        posted_support: list[tuple[str, str]] = []
        message_router.route_and_dispatch_message(
            config,
            self.store,
            "444444444444444444",
            {
                "id": "500000000000000001",
                "author": {"id": "700000000000000001", "bot": False},
                "content": "how do i configure fallbacks and num_retries on the litellm proxy?",
            },
            kb=kb,
            support_complete=lambda _sys, _usr: json.dumps({
                "action": "reply",
                "confidence": 0.92,
                "reply": "set `num_retries` and `fallbacks` under `router_settings` in your `config.yaml`: https://docs.litellm.ai/docs/proxy/reliability",
                "citations": ["https://docs.litellm.ai/docs/proxy/reliability"],
            }),
            support_poster=lambda cid, text: posted_support.append((cid, text)),
        )

        # 3. engagement_agent: role_matcher + weekly_summary child spans
        engagement_agent.run_role_matching(
            [{"login": "alice-dev", "user_id": 10, "pulls": [{"number": 50, "title": "fix: timeout"}]}],
            config,
            self.store,
            members=[
                DiscordMemberInfo(discord_id="800000000000000001", username="alice-dev", global_name="Alice Dev"),
            ],
            skip_github_enrichment=True,
            now=datetime(2026, 11, 28, 12, 0, tzinfo=UTC),
        )
        engagement_agent.summarize_weekly_prs(
            config,
            [{"pulls": [{"number": 50, "title": "fix: proxy timeout"}]}],
            summarizer=lambda _model, _msgs: json.dumps({"50": "fixed proxy timeout handling"}),
        )

        spans = exporter.get_finished_spans()
        agent_spans = [
            s for s in spans
            if _span_attrs(s).get(OPENINFERENCE_SPAN_KIND_ATTR) == OPENINFERENCE_SPAN_KIND_AGENT
        ]
        agent_names = {_span_attrs(s).get(OPENINFERENCE_AGENT_NAME_ATTR) for s in agent_spans}
        self.assertEqual(agent_names, TOP_LEVEL_AGENTS)
        self.assertEqual(agent_names, {AGENT_PR_REVIEW, AGENT_SUPPORT, AGENT_ENGAGEMENT})

        # Verify child spans nest under the right agents and NEVER have agent.name.
        by_name = {s.name: s for s in spans}
        self.assertIn("pr_review_session", by_name)
        self.assertEqual(_span_parent_name(by_name["pr_review_session"], spans), AGENT_PR_REVIEW)
        self.assertIn("lookup_contributor", by_name)
        self.assertEqual(_span_attrs(by_name["lookup_contributor"]).get(OPENINFERENCE_SPAN_KIND_ATTR), OPENINFERENCE_SPAN_KIND_TOOL)
        self.assertEqual(_span_attrs(by_name["lookup_contributor"]).get(OPENINFERENCE_TOOL_NAME_ATTR), "lookup_contributor")
        self.assertEqual(_span_parent_name(by_name["lookup_contributor"], spans), "pr_review_session")

        self.assertIn("message_router", by_name)
        self.assertEqual(_span_attrs(by_name["message_router"]).get(OPENINFERENCE_SPAN_KIND_ATTR), OPENINFERENCE_SPAN_KIND_CHAIN)
        self.assertEqual(_span_parent_name(by_name["message_router"], spans), AGENT_SUPPORT)
        self.assertIn("kb_retrieve", by_name)
        self.assertEqual(_span_attrs(by_name["kb_retrieve"]).get(OPENINFERENCE_SPAN_KIND_ATTR), OPENINFERENCE_SPAN_KIND_RETRIEVER)
        self.assertEqual(_span_parent_name(by_name["kb_retrieve"], spans), AGENT_SUPPORT)
        self.assertIn("post_support_reply", by_name)
        self.assertEqual(_span_parent_name(by_name["post_support_reply"], spans), AGENT_SUPPORT)

        self.assertIn("role_matcher", by_name)
        self.assertEqual(_span_attrs(by_name["role_matcher"]).get(OPENINFERENCE_SPAN_KIND_ATTR), OPENINFERENCE_SPAN_KIND_CHAIN)
        self.assertEqual(_span_parent_name(by_name["role_matcher"], spans), AGENT_ENGAGEMENT)
        self.assertNotIn(OPENINFERENCE_AGENT_NAME_ATTR, _span_attrs(by_name["role_matcher"]))

        self.assertIn("weekly_summary", by_name)
        self.assertEqual(_span_attrs(by_name["weekly_summary"]).get(OPENINFERENCE_SPAN_KIND_ATTR), OPENINFERENCE_SPAN_KIND_CHAIN)
        self.assertEqual(_span_parent_name(by_name["weekly_summary"], spans), AGENT_ENGAGEMENT)
        self.assertNotIn(OPENINFERENCE_AGENT_NAME_ATTR, _span_attrs(by_name["weekly_summary"]))
        self.assertEqual(len(posted_support), 1)


class MessageRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_tracing_state()
        reset_support_cooldowns()
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "state.db")
        self.store = Store(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        _reset_tracing_state()
        reset_support_cooldowns()

    def test_routes_review_request_to_pr_review_agent_with_child_router_span(self) -> None:
        exporter, _provider, _ = _install_in_memory_exporter()
        config = _base_config(self.db_path)
        msg = {
            "id": "510000000000000001",
            "author": {"id": "710000000000000001", "bot": False},
            "content": "can you review https://github.com/BerriAI/litellm/pull/9999?",
        }
        with mock.patch("contributor_bot.review_job._spawn_review") as spawn, \
             mock.patch("contributor_bot.review_job.acknowledge_message"):
            decision = message_router.route_and_dispatch_message(
                config, self.store, "333333333333333333", msg
            )
        self.assertEqual(decision.target, AGENT_PR_REVIEW)
        self.assertEqual(decision.pr_numbers, (9999,))
        spawn.assert_called_once()

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if _span_attrs(s).get(OPENINFERENCE_SPAN_KIND_ATTR) == OPENINFERENCE_SPAN_KIND_AGENT]
        self.assertEqual([_span_attrs(s).get(OPENINFERENCE_AGENT_NAME_ATTR) for s in agent_spans], [AGENT_PR_REVIEW])
        router_spans = [s for s in spans if s.name == "message_router"]
        self.assertEqual(len(router_spans), 1)
        self.assertEqual(_span_parent_name(router_spans[0], spans), AGENT_PR_REVIEW)

    def test_routes_support_question_to_support_agent_and_skips_chitchat_with_zero_spans(self) -> None:
        exporter, _provider, _ = _install_in_memory_exporter()
        config = _base_config(self.db_path)
        kb = _make_mock_kb(self.store)
        exporter.clear()

        # Chit-chat -> none, ZERO spans emitted!
        for chatter in ("lol", "thanks!", "nice work everyone", "gm"):
            decision = message_router.route_and_dispatch_message(
                config,
                self.store,
                "444444444444444444",
                {"id": "520000000000000001", "author": {"id": "1", "bot": False}, "content": chatter},
                kb=kb,
            )
            self.assertEqual(decision.target, message_router.ROUTE_NONE)
        self.assertEqual(list(exporter.get_finished_spans()), [])

        # Support question in allowlisted channel -> support_agent + message_router child span!
        posted: list[str] = []
        decision = message_router.route_and_dispatch_message(
            config,
            self.store,
            "444444444444444444",
            {
                "id": "520000000000000002",
                "author": {"id": "2", "bot": False},
                "content": "how do i create virtual keys with max_budget in litellm proxy?",
            },
            kb=kb,
            support_complete=lambda _s, _u: json.dumps({
                "action": "reply",
                "confidence": 0.9,
                "reply": "hit `/key/generate` with `max_budget` and `rpm_limit`. see https://docs.litellm.ai/docs/proxy/virtual_keys",
                "citations": ["https://docs.litellm.ai/docs/proxy/virtual_keys"],
            }),
            support_poster=lambda _cid, body: posted.append(body),
        )
        self.assertEqual(decision.target, AGENT_SUPPORT)
        self.assertEqual(len(posted), 1)
        self.assertIn("https://docs.litellm.ai/docs/proxy/virtual_keys", posted[0])

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if _span_attrs(s).get(OPENINFERENCE_SPAN_KIND_ATTR) == OPENINFERENCE_SPAN_KIND_AGENT]
        self.assertEqual([_span_attrs(s).get(OPENINFERENCE_AGENT_NAME_ATTR) for s in agent_spans], [AGENT_SUPPORT])
        router_spans = [s for s in spans if s.name == "message_router"]
        self.assertEqual(len(router_spans), 1)
        self.assertEqual(_span_parent_name(router_spans[0], spans), AGENT_SUPPORT)

    def test_support_agent_gating_env_flag_and_channel_allowlist(self) -> None:
        exporter, _provider, _ = _install_in_memory_exporter()
        question_msg = {
            "id": "530000000000000001",
            "author": {"id": "3", "bot": False},
            "content": "how do i set fallbacks on the litellm proxy?",
        }

        # Disabled by default -> none, zero spans.
        disabled_cfg = _base_config(self.db_path, support_agent_enabled=False)
        d1 = message_router.route_and_dispatch_message(disabled_cfg, self.store, "444444444444444444", question_msg)
        self.assertEqual(d1.target, message_router.ROUTE_NONE)

        # Enabled, but channel not in SUPPORT_AGENT_CHANNEL_IDS -> none, zero spans.
        other_channel_cfg = _base_config(self.db_path, support_agent_enabled=True, support_agent_channel_ids=("444444444444444444",))
        d2 = message_router.route_and_dispatch_message(other_channel_cfg, self.store, "999999999999999999", question_msg)
        self.assertEqual(d2.target, message_router.ROUTE_NONE)

        # Enabled, but PR URL in a support-only channel (review disabled) -> never misrouted to support_agent.
        pr_only_cfg = _base_config(self.db_path, pr_review_enabled=False, support_agent_enabled=True, support_agent_channel_ids=("444444444444444444",))
        pr_msg = {
            "id": "530000000000000002",
            "author": {"id": "3", "bot": False},
            "content": "please review https://github.com/BerriAI/litellm/pull/7777",
        }
        d3 = message_router.route_and_dispatch_message(pr_only_cfg, self.store, "444444444444444444", pr_msg)
        self.assertEqual(d3.target, message_router.ROUTE_NONE)
        self.assertEqual(list(exporter.get_finished_spans()), [])


class SupportAgentKBTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_tracing_state()
        reset_support_cooldowns()
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "state.db")
        self.store = Store(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        _reset_tracing_state()
        reset_support_cooldowns()

    def test_kb_caches_in_sqlite_and_refreshes_only_when_stale(self) -> None:
        calls = {"llms": 0, "blog": 0, "readme": 0}

        def fetch_llms():
            calls["llms"] += 1
            return MOCK_LLMS_TXT

        def fetch_blog():
            calls["blog"] += 1
            return MOCK_BLOG_RSS

        def fetch_readme():
            calls["readme"] += 1
            return MOCK_README

        kb = LiteLLMKnowledgeBase(
            store=self.store,
            refresh_hours=12,
            llms_txt_fetcher=fetch_llms,
            blog_rss_fetcher=fetch_blog,
            readme_fetcher=fetch_readme,
        )
        t0 = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)
        count = kb.refresh(now=t0)
        self.assertGreaterEqual(count, 5)
        self.assertEqual(calls, {"llms": 1, "blog": 1, "readme": 1})
        self.assertFalse(kb.is_stale(now=t0 + timedelta(hours=6)))
        self.assertTrue(kb.is_stale(now=t0 + timedelta(hours=13)))

        # Second KB instance reading from same SQLite store uses cache without refetching.
        kb2 = LiteLLMKnowledgeBase(
            store=self.store,
            refresh_hours=12,
            llms_txt_fetcher=fetch_llms,
            blog_rss_fetcher=fetch_blog,
            readme_fetcher=fetch_readme,
        )
        loaded = kb2.ensure_loaded(now=t0 + timedelta(hours=2), refresh_if_needed=True)
        self.assertEqual(len(loaded), count)
        self.assertEqual(calls, {"llms": 1, "blog": 1, "readme": 1})

        # Search retrieves docs, blog, and README entries accurately.
        hits = kb2.search("prompt caching cache_control_injection_points", refresh_if_needed=False)
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc.url, "https://docs.litellm.ai/blog/prompt-caching")

    def test_kb_disk_cache_fallback_without_store(self) -> None:
        cache_file = Path(self.tmp.name) / "kb_cache.json"
        kb = _make_mock_kb(store=None, cache_path=cache_file)
        self.assertTrue(cache_file.exists())
        hits = kb.search("router fallbacks num_retries", refresh_if_needed=False)
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc.url, "https://docs.litellm.ai/docs/proxy/reliability")

    def test_support_agent_voice_formatting_citations_and_low_confidence_silence(self) -> None:
        config = _base_config(self.db_path)
        kb = _make_mock_kb(self.store)

        # Formatter strips headings, emojis, AI fluff, and attaches doc citation.
        cleaned = format_support_reply(
            "Great question! ## Router Fallbacks 🚀\nSet `num_retries` under `router_settings` in `config.yaml`.",
            cited_urls=["https://docs.litellm.ai/docs/proxy/reliability"],
        )
        self.assertNotIn("##", cleaned)
        self.assertNotIn("🚀", cleaned)
        self.assertFalse(cleaned.lower().startswith("great question"))
        self.assertIn("https://docs.litellm.ai/docs/proxy/reliability", cleaned)

        # Low-confidence model response stays silent (does NOT post or invent config).
        posted: list[str] = []
        low_conf = answer_support_question(
            config,
            "444444444444444444",
            "how do i configure fallbacks on litellm proxy?",
            store=self.store,
            kb=kb,
            bypass_cooldown=True,
            complete=lambda _s, _u: json.dumps({
                "action": "reply",
                "confidence": 0.35,
                "reply": "maybe try magic_fallback_flag: true",
                "citations": [],
            }),
            poster=lambda _c, text: posted.append(text),
        )
        self.assertEqual(low_conf.action, "silent")
        self.assertFalse(low_conf.posted)
        self.assertEqual(posted, [])

        # flag_maintainer action posts short note without inventing config.
        flagged = answer_support_question(
            config,
            "444444444444444444",
            "my proxy segfaults on arm64 with custom ssl certs, how do i fix it?",
            store=self.store,
            kb=kb,
            bypass_cooldown=True,
            complete=lambda _s, _u: json.dumps({
                "action": "flag_maintainer",
                "confidence": 0.4,
                "reply": "not totally sure on arm64 ssl crashes — i'll flag a maintainer to take a look.",
                "citations": [],
            }),
            poster=lambda _c, text: posted.append(text),
        )
        self.assertEqual(flagged.action, "flag_maintainer")
        self.assertTrue(flagged.posted)
        self.assertEqual(len(posted), 1)
        self.assertIn("maintainer", posted[0].lower())

    def test_support_agent_env_config_defaults_and_validation(self) -> None:
        base_env = {
            "DISCORD_BOT_TOKEN": "tok",
            "OSS_CONTRIBUTIONS_CHANNEL_ID": "111",
            "ANNOUNCEMENTS_CHANNEL_ID": "222",
        }
        default_cfg = get_config(base_env)
        self.assertFalse(default_cfg.support_agent_enabled)
        self.assertEqual(default_cfg.support_agent_channel_ids, ())
        self.assertEqual(default_cfg.support_kb_refresh_hours, 12)
        self.assertEqual(default_cfg.review_lifecycle_poll_seconds, 600)

        enabled_cfg = get_config(
            base_env
            | {
                "SUPPORT_AGENT_ENABLED": "true",
                "SUPPORT_AGENT_CHANNEL_IDS": "444444444444444444, 555555555555555555",
                "SUPPORT_AGENT_MODEL": "anthropic/claude-haiku-4-5",
                "SUPPORT_KB_REFRESH_HOURS": "6",
                "REVIEW_LIFECYCLE_POLL_SECONDS": "300",
            }
        )
        self.assertTrue(enabled_cfg.support_agent_enabled)
        self.assertEqual(enabled_cfg.support_agent_channel_ids, ("444444444444444444", "555555555555555555"))
        self.assertEqual(enabled_cfg.support_agent_model, "anthropic/claude-haiku-4-5")
        self.assertEqual(enabled_cfg.support_kb_refresh_hours, 6)
        self.assertEqual(enabled_cfg.review_lifecycle_poll_seconds, 300)

        with self.assertRaisesRegex(ValueError, "SUPPORT_AGENT_CHANNEL_IDS"):
            get_config(base_env | {"SUPPORT_AGENT_CHANNEL_IDS": "not-a-channel-id"})


class PRReviewLifecycleTrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "state.db")
        self.store = Store(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_tracks_pr_through_merge_and_release_updates_once_each(self) -> None:
        config = _base_config(self.db_path)
        self.store.save_review_thread(
            repository="BerriAI/litellm",
            number=55001,
            channel_id="333333333333333333",
            message_id="600000000000000001",
            thread_id="700000000000000001",
            created_at="2026-11-26T00:00:00+00:00",
        )
        self.store.save_review_thread(
            repository="BerriAI/litellm",
            number=55002,
            channel_id="333333333333333333",
            message_id="600000000000000002",
            thread_id="700000000000000002",
            created_at="2026-11-26T00:00:00+00:00",
        )

        pr_state = {
            55001: {"number": 55001, "state": "open", "merged": False},
            55002: {"number": 55002, "state": "closed", "merged": False, "closed_at": "2026-11-27T01:00:00+00:00"},
        }
        gh_releases: list[dict] = []
        posted: list[tuple[str, str]] = []

        # Tick 1: #55001 still open, #55002 closed without merging -> stop tracking #55002, no messages posted.
        res1 = pr_review_agent.poll_review_thread_lifecycle(
            config,
            self.store,
            now=datetime(2026, 11, 27, 12, 0, tzinfo=UTC),
            pull_fetcher=lambda _repo, num: pr_state[num],
            github_releases_fetcher=lambda _repo: gh_releases,
            release_entries_fetcher=dict,
            message_poster=lambda tid, body: posted.append((tid, body)),
        )
        self.assertEqual(res1["merged_updates"], 0)
        self.assertEqual(res1["closed_unmerged"], 1)
        self.assertEqual(posted, [])

        # Tick 2: #55001 is merged, not yet released -> posts ONE merge update.
        pr_state[55001] = {
            "number": 55001,
            "state": "closed",
            "merged": True,
            "merged_at": "2026-11-28T10:00:00+00:00",
            "merge_commit_sha": "a" * 40,
        }
        res2 = pr_review_agent.poll_review_thread_lifecycle(
            config,
            self.store,
            now=datetime(2026, 11, 28, 11, 0, tzinfo=UTC),
            pull_fetcher=lambda _repo, num: pr_state[num],
            github_releases_fetcher=lambda _repo: gh_releases,
            release_entries_fetcher=dict,
            message_poster=lambda tid, body: posted.append((tid, body)),
        )
        self.assertEqual(res2["merged_updates"], 1)
        self.assertEqual(res2["release_updates"], 0)
        self.assertEqual(len(posted), 1)
        self.assertIn("merged", posted[0][1].lower())

        # Tick 3: Still no release -> no duplicate merge update posted.
        res3 = pr_review_agent.poll_review_thread_lifecycle(
            config,
            self.store,
            now=datetime(2026, 11, 28, 12, 0, tzinfo=UTC),
            pull_fetcher=lambda _repo, num: pr_state[num],
            github_releases_fetcher=lambda _repo: gh_releases,
            release_entries_fetcher=dict,
            message_poster=lambda tid, body: posted.append((tid, body)),
        )
        self.assertEqual(res3["merged_updates"], 0)
        self.assertEqual(res3["release_updates"], 0)
        self.assertEqual(len(posted), 1)

        # Tick 4: LiteLLM release v1.80.0 includes #55001 -> posts ONE release update and completes tracking.
        gh_releases.append({
            "tag_name": "v1.80.0",
            "published_at": "2026-11-28T15:00:00+00:00",
            "html_url": "https://github.com/BerriAI/litellm/releases/tag/v1.80.0",
            "body": "Includes fix for router timeout (#55001)",
        })
        res4 = pr_review_agent.poll_review_thread_lifecycle(
            config,
            self.store,
            now=datetime(2026, 11, 28, 16, 0, tzinfo=UTC),
            pull_fetcher=lambda _repo, num: pr_state[num],
            github_releases_fetcher=lambda _repo: gh_releases,
            release_entries_fetcher=dict,
            message_poster=lambda tid, body: posted.append((tid, body)),
        )
        self.assertEqual(res4["release_updates"], 1)
        self.assertEqual(len(posted), 2)
        self.assertIn("v1.80.0", posted[1][1])
        self.assertIn("https://github.com/BerriAI/litellm/releases/tag/v1.80.0", posted[1][1])

        # Tick 5: Thread is fully complete -> zero remaining active lifecycle threads.
        self.assertEqual(self.store.active_lifecycle_review_threads("BerriAI/litellm"), [])


if __name__ == "__main__":
    unittest.main()
