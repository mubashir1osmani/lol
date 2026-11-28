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


class _StdlibRecordedEvent:
    def __init__(self, name: str, attributes: dict | None = None) -> None:
        self.name = name
        self.attributes: dict[str, object] = dict(attributes or {})


class _StdlibRecordedSpan:
    def __init__(self, name: str, parent=None) -> None:
        self.name = name
        self.parent = parent
        self.attributes: dict[str, object] = {}
        self.events: list[_StdlibRecordedEvent] = []
        self.status = None
        self.exceptions: list[Exception] = []

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        self.events.append(_StdlibRecordedEvent(name, attributes))

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


def _span_parent_obj(span, all_spans):
    parent = getattr(span, "parent", None)
    if parent is None:
        return None
    if isinstance(parent, _StdlibRecordedSpan):
        return parent
    parent_id = getattr(parent, "span_id", None)
    if parent_id is None:
        return None
    for candidate in all_spans:
        ctx = getattr(candidate, "context", None)
        if ctx is not None and getattr(ctx, "span_id", None) == parent_id:
            return candidate
    return None


def _span_parent_name(span, all_spans) -> str | None:
    parent = _span_parent_obj(span, all_spans)
    return parent.name if parent is not None else None


def _span_events(span) -> list[tuple[str, dict]]:
    return [
        (ev.name, dict(getattr(ev, "attributes", None) or {}))
        for ev in (getattr(span, "events", None) or ())
    ]


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


class LensObservabilityParityTests(unittest.TestCase):
    """Verify full log -> span observability parity across all three agents
    using an in-memory span exporter (real OTel SDK when available, stdlib otherwise).
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "state.db")
        self.store = Store(self.db_path)
        _reset_tracing_state()
        reset_support_cooldowns()
        self.exporter, self.provider, self.using_real_otel = _install_in_memory_exporter()

    def tearDown(self) -> None:
        _reset_tracing_state()
        reset_support_cooldowns()
        self.store.close()
        self.tmp.cleanup()

    def test_engagement_agent_role_match_and_grant_spans_attributes_events_and_nested_llm(self) -> None:
        from contributor_bot.matcher import _compute_attempt_signature, score_author_against_members, GitHubAuthorProfile
        from contributor_bot.roles_sync import poll_merged_prs_for_roles
        from contributor_bot.tracing import OPENINFERENCE_INPUT_VALUE_ATTR, OPENINFERENCE_OUTPUT_VALUE_ATTR, child_span

        config = _base_config(
            self.db_path,
            contributor_roles_enabled=True,
            roles_announce_enabled=False,
            discord_guild_id="999000111222333444",
            role_ids={"contributor": "555000111222333444"},
            match_model="claude-sonnet-4-5",
        )
        now = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)
        members = [
            DiscordMemberInfo(discord_id="100000000000000001", username="alex_rivera_dev", display_name="Alex Rivera"),
            DiscordMemberInfo(discord_id="100000000000000002", username="jordan_lin", display_name="Jordan Lin"),
            DiscordMemberInfo(discord_id="100000000000000003", username="sam_k", display_name="Sam K"),
        ]

        # Seed 1 already-existing PR in SQLite (outside the 7d match window) so prs_existing=1 is exercised.
        self.store.upsert_pull(
            repository="BerriAI/litellm",
            number=900,
            github_login="prior-dev",
            github_user_id=900,
            merged_at="2026-10-01T01:00:00+00:00",
            title="Already ingested PR",
            labels=[],
            files=["litellm/utils.py"],
        )

        # Seed a fresh negative cache entry for "cached-neg-user" so skip_reason="cached_negative" runs.
        cached_prof = GitHubAuthorProfile(login="cached-neg-user", name="Sam K", bio="rust dev")
        _s, cached_cands = score_author_against_members(cached_prof, members)
        cached_sig = _compute_attempt_signature(cached_prof, cached_cands, [], [])
        self.store.save_match_attempt(
            "cached-neg-user",
            attempted_at="2026-11-27T12:00:00+00:00",
            signature=cached_sig,
            outcome="no_match",
        )

        # 5 PRs discovered in the poll window:
        # #900 existing, #901 excluded bot, #902 alex-builds (LLM matched),
        # #903 jordan-other (LLM none_of_these), #904 cached-neg-user (cached_negative),
        # #905 zzz-unmatched (no_signal_match), #906 broken-enrich (enrichment_failed).
        prs = {
            900: {
                "number": 900,
                "title": "Already ingested PR",
                "merged": True,
                "merged_at": "2026-10-01T01:00:00+00:00",
                "author_association": "CONTRIBUTOR",
                "user": {"login": "prior-dev", "id": 900, "type": "User"},
            },
            901: {
                "number": 901,
                "title": "Bump deps",
                "merged": True,
                "merged_at": "2026-11-28T01:00:00+00:00",
                "author_association": "NONE",
                "user": {"login": "dependabot[bot]", "id": 901, "type": "Bot"},
            },
            902: {
                "number": 902,
                "title": "Add provider fallback",
                "merged": True,
                "merged_at": "2026-11-28T02:00:00+00:00",
                "author_association": "CONTRIBUTOR",
                "user": {"login": "alex-builds", "id": 902, "type": "User"},
            },
            903: {
                "number": 903,
                "title": "Fix streaming parser",
                "merged": True,
                "merged_at": "2026-11-28T03:00:00+00:00",
                "author_association": "CONTRIBUTOR",
                "user": {"login": "jordan-other", "id": 903, "type": "User"},
            },
            904: {
                "number": 904,
                "title": "Docs typo",
                "merged": True,
                "merged_at": "2026-11-28T04:00:00+00:00",
                "author_association": "CONTRIBUTOR",
                "user": {"login": "cached-neg-user", "id": 904, "type": "User"},
            },
            905: {
                "number": 905,
                "title": "Unrelated tweak",
                "merged": True,
                "merged_at": "2026-11-28T05:00:00+00:00",
                "author_association": "CONTRIBUTOR",
                "user": {"login": "zzz-unmatched", "id": 905, "type": "User"},
            },
            906: {
                "number": 906,
                "title": "Enrichment fail case",
                "merged": True,
                "merged_at": "2026-11-28T06:00:00+00:00",
                "author_association": "CONTRIBUTOR",
                "user": {"login": "broken-enrich", "id": 906, "type": "User"},
            },
        }

        profiles = {
            "alex-builds": {"login": "alex-builds", "name": "Alex Rivera", "bio": "building litellm adapters", "company": "", "twitterUsername": None, "websiteUrl": None, "location": None, "email": None, "socialAccounts": {"nodes": []}, "readme": None},
            "jordan-other": {"login": "jordan-other", "name": "Jordan Lin", "bio": "backend engineer", "company": "", "twitterUsername": None, "websiteUrl": None, "location": None, "email": None, "socialAccounts": {"nodes": []}, "readme": None},
            "cached-neg-user": {"login": "cached-neg-user", "name": "Sam K", "bio": "rust dev", "company": "", "twitterUsername": None, "websiteUrl": None, "location": None, "email": None, "socialAccounts": {"nodes": []}, "readme": None},
            "zzz-unmatched": {"login": "zzz-unmatched", "name": "Qux Bar", "bio": "something else", "company": "", "twitterUsername": None, "websiteUrl": None, "location": None, "email": None, "socialAccounts": {"nodes": []}, "readme": None},
        }

        def fake_graphql(_token, query, variables):
            if "BatchProfiles" not in query:
                return {"data": {}}
            nodes = {}
            errors = []
            for k, login in (variables or {}).items():
                idx = k[1:]
                if login == "broken-enrich":
                    nodes[f"u{idx}"] = None
                    errors.append({"path": [f"u{idx}"], "type": "INTERNAL", "message": "transient error"})
                else:
                    nodes[f"u{idx}"] = profiles.get(login)
            result = {"data": nodes}
            if errors:
                result["errors"] = errors
            return result

        def fake_rest_github(url, _token):
            if "broken-enrich" in url:
                raise RuntimeError("transient rest error")
            for login, node in profiles.items():
                if url.endswith(f"/users/{login}"):
                    return {"login": login, "name": node["name"], "bio": node["bio"]}
            return []

        def fake_llm_complete(model, messages):
            user_prompt = str(messages[-1]["content"]) if messages else ""
            # Simulate LiteLLMInstrumentor creating an LLM completion span inside _select_with_llm.
            with child_span(
                "litellm.completion",
                kind="LLM",
                input_value={"model": model},
                attributes={"llm.model_name": model},
            ) as llm_span:
                if "alex-builds" in user_prompt:
                    reply = json.dumps({
                        "discord_id": "100000000000000001",
                        "confidence": 0.94,
                        "reason": "Full name Alex Rivera matches alex_rivera_dev and bio confirms LiteLLM work.",
                        "evidence": ["full_name:Alex Rivera", "handle_token:alex"],
                    })
                else:
                    reply = json.dumps({
                        "discord_id": None,
                        "confidence": 0.15,
                        "reason": "Common name Jordan Lin with no corroborating handle or bio signal.",
                        "evidence": [],
                    })
                llm_span.set_output(reply)
                return reply

        role_api_calls: list[tuple[str, str]] = []

        with mock.patch("contributor_bot.rewards.add_member_role", side_effect=lambda _g, uid, rid, _t, **_kw: role_api_calls.append((uid, rid))):
            summary = poll_merged_prs_for_roles(
                config,
                self.store,
                now=now,
                search_merged_fn=lambda _r, _s, _e, _t, **_kw: list(prs.values()),
                pull_fetcher=lambda _r, num, _t: prs[num],
                files_fetcher=lambda _r, _num, _t: ["litellm/proxy/proxy_server.py"],
                org_members_fetcher=lambda _org, _t: frozenset(),
                members=members,
                github_getter=fake_rest_github,
                graphql_caller=fake_graphql,
                haiku_complete=fake_llm_complete,
                sleep_fn=lambda _s: None,
            )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.roles_granted, 1)
        spans = self.exporter.get_finished_spans()
        agent_spans = [s for s in spans if _span_attrs(s).get(OPENINFERENCE_SPAN_KIND_ATTR) == OPENINFERENCE_SPAN_KIND_AGENT]
        self.assertEqual(len(agent_spans), 1)
        root = agent_spans[0]
        self.assertEqual(root.name, AGENT_ENGAGEMENT)
        root_attrs = _span_attrs(root)

        # Run summary attributes on engagement_agent AGENT span.
        self.assertEqual(root_attrs["engagement.contributors"], 5)
        self.assertEqual(root_attrs["engagement.members"], 3)
        self.assertGreaterEqual(root_attrs["engagement.candidates"], 1)
        self.assertEqual(root_attrs["engagement.matched"], 1)
        self.assertEqual(root_attrs["engagement.skipped"], 4)
        skip_reasons = json.loads(root_attrs["engagement.skip_reasons"])
        self.assertEqual(skip_reasons.get("llm_none_of_these"), 1)
        self.assertEqual(skip_reasons.get("cached_negative"), 1)
        self.assertEqual(skip_reasons.get("no_signal_match"), 1)
        self.assertEqual(skip_reasons.get("enrichment_failed"), 1)
        methods = json.loads(root_attrs["engagement.methods"])
        self.assertEqual(methods.get("llm"), 1)
        self.assertEqual(root_attrs["engagement.roles_granted"], 1)
        self.assertEqual(root_attrs["engagement.prs_discovered"], 7)
        self.assertEqual(root_attrs["engagement.prs_stored"], 5)
        self.assertEqual(root_attrs["engagement.prs_existing"], 1)
        self.assertEqual(root_attrs["engagement.prs_excluded"], 1)
        self.assertIn("engagement.duration_seconds", root_attrs)
        self.assertIs(root_attrs["engagement.announce_enabled"], False)
        self.assertIn(OPENINFERENCE_INPUT_VALUE_ATTR, root_attrs)
        self.assertIn(OPENINFERENCE_OUTPUT_VALUE_ATTR, root_attrs)

        # Per-contributor match_contributor spans under role_matcher -> engagement_agent.
        rm_spans = [s for s in spans if s.name == "role_matcher"]
        self.assertEqual(len(rm_spans), 1)
        self.assertIs(_span_parent_obj(rm_spans[0], spans), root)

        contrib_spans = {
            _span_attrs(s)["contributor.github_login"]: s
            for s in spans
            if s.name == "match_contributor"
        }
        self.assertEqual(
            set(contrib_spans.keys()),
            {"alex-builds", "jordan-other", "cached-neg-user", "zzz-unmatched", "broken-enrich"},
        )
        for cspan in contrib_spans.values():
            self.assertIs(_span_parent_obj(cspan, spans), rm_spans[0])
            attrs = _span_attrs(cspan)
            self.assertIn("contributor.candidate_count", attrs)
            self.assertIn("contributor.top_score", attrs)
            self.assertIn("contributor.method", attrs)
            self.assertIn("contributor.outcome", attrs)
            self.assertIn(OPENINFERENCE_INPUT_VALUE_ATTR, attrs)
            self.assertIn(OPENINFERENCE_OUTPUT_VALUE_ATTR, attrs)

        alex_attrs = _span_attrs(contrib_spans["alex-builds"])
        self.assertEqual(alex_attrs["contributor.outcome"], "matched")
        self.assertEqual(alex_attrs["contributor.method"], "llm")
        self.assertEqual(alex_attrs["contributor.discord_id"], "100000000000000001")
        self.assertIn("Full name Alex Rivera matches alex_rivera_dev", alex_attrs[OPENINFERENCE_OUTPUT_VALUE_ATTR])

        jordan_attrs = _span_attrs(contrib_spans["jordan-other"])
        self.assertEqual(jordan_attrs["contributor.outcome"], "skipped")
        self.assertEqual(jordan_attrs["contributor.skip_reason"], "llm_none_of_these")
        self.assertIn("Common name Jordan Lin", jordan_attrs[OPENINFERENCE_OUTPUT_VALUE_ATTR])

        self.assertEqual(_span_attrs(contrib_spans["cached-neg-user"])["contributor.skip_reason"], "cached_negative")
        self.assertEqual(_span_attrs(contrib_spans["zzz-unmatched"])["contributor.skip_reason"], "no_signal_match")
        self.assertEqual(_span_attrs(contrib_spans["broken-enrich"])["contributor.skip_reason"], "enrichment_failed")

        # LLM completions MUST nest directly under their own contributor's match_contributor span (no orphans!).
        llm_spans = [s for s in spans if _span_attrs(s).get(OPENINFERENCE_SPAN_KIND_ATTR) == "LLM"]
        self.assertEqual(len(llm_spans), 2)
        llm_parents = {_span_attrs(_span_parent_obj(ls, spans))["contributor.github_login"] for ls in llm_spans}
        self.assertEqual(llm_parents, {"alex-builds", "jordan-other"})

        # Role grant span + role_granted event with full metadata & evidence list.
        grant_spans = [s for s in spans if s.name == "grant_role"]
        self.assertEqual(len(grant_spans), 1)
        gspan = grant_spans[0]
        self.assertIs(_span_parent_obj(gspan, spans), root)
        gattrs = _span_attrs(gspan)
        self.assertEqual(gattrs[OPENINFERENCE_SPAN_KIND_ATTR], OPENINFERENCE_SPAN_KIND_TOOL)
        self.assertEqual(gattrs["grant.discord_id"], "100000000000000001")
        self.assertEqual(gattrs["grant.github_login"], "alex-builds")
        self.assertEqual(gattrs["grant.role_id"], "555000111222333444")
        self.assertTrue(gattrs["grant.role_name"])
        self.assertEqual(gattrs["grant.method"], "llm")
        self.assertGreaterEqual(float(gattrs["grant.confidence"]), 0.9)
        grant_evidence = json.loads(gattrs["grant.evidence"])
        self.assertIsInstance(grant_evidence, list)
        self.assertGreaterEqual(len(grant_evidence), 1)
        grant_events = dict(_span_events(gspan))
        self.assertIn("role_granted", grant_events)
        self.assertEqual(grant_events["role_granted"]["grant.discord_id"], "100000000000000001")

        # Log -> span event parity for match_skipped, match_summary, roles_poll_complete, roles_granted_silent.
        rm_event_names = [name for name, _ in _span_events(rm_spans[0])]
        self.assertIn("match_skipped", rm_event_names)
        self.assertIn("match_summary", rm_event_names)
        root_event_names = [name for name, _ in _span_events(root)]
        self.assertIn("roles_poll_complete", root_event_names)
        self.assertIn("roles_granted_silent", root_event_names)

    def test_pr_review_agent_observability_decision_verdict_findings_and_skips(self) -> None:
        from contributor_bot import review_job
        from contributor_bot.review_agent import AgentResult, ReviewContext, tool_submit_verdict
        from contributor_bot.tracing import OPENINFERENCE_INPUT_VALUE_ATTR, OPENINFERENCE_OUTPUT_VALUE_ATTR

        config = _base_config(
            self.db_path,
            pr_review_enabled=True,
            review_channel_ids=("333333333333333333",),
            review_cooldown_seconds=3600,
            review_daily_run_budget=1,
        )
        now_iso = datetime.now(UTC).isoformat()

        # 1. Request detection + cooldown skip in handle_routed_message.
        self.store.save_review_thread(
            repository="BerriAI/litellm",
            number=7701,
            channel_id="333333333333333333",
            message_id="600000000000000010",
            thread_id="700000000000000010",
            created_at=now_iso,
        )
        self.store.mark_review_done("BerriAI/litellm", 7701, now_iso, "deadbeef")
        msg = {
            "id": "600000000000000011",
            "content": "please review https://github.com/BerriAI/litellm/pull/7701",
            "author": {"id": "111", "bot": False},
        }
        pr_review_agent.handle_routed_message(
            config,
            self.store,
            "333333333333333333",
            msg,
            route_info={"target": AGENT_PR_REVIEW, "reason": "pr_review_request", "pr_numbers": [7701]},
        )

        # 2. Full review posting a verdict with findings + posted message id.
        def fake_run_sdk(ctx: ReviewContext, _task: str, _cfg: Config) -> AgentResult:
            ctx.snapshot = types.SimpleNamespace(
                number=7702,
                title="Add router retry",
                body="",
                author="dev",
                author_id=12345,
                state="open",
                draft=False,
                merged=False,
                head_sha="abc1234",
                mergeable=True,
                mergeable_state="clean",
                reviews=[],
                comments=[],
                threads=[],
                check_runs=[],
                combined_state="success",
                diff="",
            )
            ctx.files = []
            with mock.patch("contributor_bot.review_agent.post_message", return_value={"id": "999888777"}):
                tool_submit_verdict(
                    ctx,
                    {
                        "verdict": "needs_changes",
                        "findings": [
                            {
                                "issue": "missing unit test in tests/test_router.py",
                                "suggestion": "add a pytest case covering fallback retry exhaustion in tests/test_router.py",
                                "file": "tests/test_router.py",
                            },
                            {
                                "issue": "unused import in proxy_server.py",
                                "suggestion": "remove the unused os import at top of litellm/proxy/proxy_server.py",
                                "file": "litellm/proxy/proxy_server.py",
                            },
                        ],
                    },
                )
            return AgentResult("posted verdict", steps=2)

        with mock.patch("contributor_bot.review_job._ensure_thread", return_value="700000000000000012"):
            with mock.patch("contributor_bot.review_job.fetch_recent_messages", return_value=[]):
                with mock.patch("contributor_bot.review_agent._run_sdk_session", side_effect=fake_run_sdk):
                    res_ok = review_job.perform_review(
                        config,
                        self.store,
                        7702,
                        channel_id="333333333333333333",
                        message_id="600000000000000012",
                        source="new",
                    )
        self.assertEqual(res_ok["status"], "posted")
        self.assertEqual(res_ok["posted_message_id"], "999888777")

        # 3. Daily budget skip on a second PR (#7703).
        res_budget = review_job.perform_review(
            config,
            self.store,
            7703,
            channel_id="333333333333333333",
            message_id="600000000000000013",
            source="new",
        )
        self.assertEqual(res_budget["status"], "budget")

        # 4. In-flight dedupe skip when lock is already held.
        self.assertTrue(review_job._try_lock("BerriAI/litellm", 7704))
        try:
            res_busy = review_job.perform_review(config, self.store, 7704, source="new")
        finally:
            review_job._unlock("BerriAI/litellm", 7704)
        self.assertEqual(res_busy["status"], "busy")

        spans = self.exporter.get_finished_spans()
        agent_spans = [s for s in spans if _span_attrs(s).get(OPENINFERENCE_SPAN_KIND_ATTR) == OPENINFERENCE_SPAN_KIND_AGENT]
        self.assertEqual(len(agent_spans), 4)
        self.assertTrue(all(s.name == AGENT_PR_REVIEW for s in agent_spans))

        # Verify cooldown skip span (run 1).
        cd_attrs = _span_attrs(agent_spans[0])
        self.assertIs(cd_attrs["review.request_detected"], True)
        self.assertEqual(json.loads(cd_attrs["review.pr_numbers"]), [7701])
        self.assertEqual(cd_attrs["review.skip_reason"], "cooldown")
        self.assertIn("review_skipped", dict(_span_events(agent_spans[0])))
        self.assertIn(OPENINFERENCE_INPUT_VALUE_ATTR, cd_attrs)
        self.assertIn(OPENINFERENCE_OUTPUT_VALUE_ATTR, cd_attrs)

        # Verify posted verdict span (run 2).
        ok_attrs = _span_attrs(agent_spans[1])
        self.assertEqual(ok_attrs["pr.number"], 7702)
        self.assertEqual(ok_attrs["review.verdict"], "needs work")
        self.assertEqual(ok_attrs["review.findings_count"], 2)
        self.assertEqual(len(json.loads(ok_attrs["review.findings"])), 2)
        self.assertEqual(ok_attrs["review.posted_message_id"], "999888777")
        self.assertIn(OPENINFERENCE_OUTPUT_VALUE_ATTR, ok_attrs)
        ok_events = dict(_span_events(agent_spans[1]))
        self.assertIn("review_started", ok_events)
        self.assertIn("review_verdict_posted", ok_events)
        session_spans = [s for s in spans if s.name == "pr_review_session"]
        self.assertEqual(len(session_spans), 1)
        self.assertIn("review_verdict_submitted", dict(_span_events(session_spans[0])))

        # Verify daily budget skip (run 3) and in-flight dedupe skip (run 4).
        self.assertEqual(_span_attrs(agent_spans[2])["review.skip_reason"], "daily_budget")
        self.assertIn("review_skipped", dict(_span_events(agent_spans[2])))
        self.assertEqual(_span_attrs(agent_spans[3])["review.skip_reason"], "in_flight_dedupe")
        self.assertIn("review_skipped", dict(_span_events(agent_spans[3])))

    def test_support_agent_observability_detection_kb_retrieval_answer_and_skips(self) -> None:
        from contributor_bot.tracing import OPENINFERENCE_INPUT_VALUE_ATTR, OPENINFERENCE_OUTPUT_VALUE_ATTR

        config = _base_config(
            self.db_path,
            support_agent_enabled=True,
            support_agent_channel_ids=("444444444444444444",),
        )
        kb = _make_mock_kb(store=self.store)
        kb.refresh()

        # 1. Answered question with KB retrieval and returned Discord message ID.
        ans = answer_support_question(
            config,
            "444444444444444444",
            "How do I configure router fallbacks and num_retries in LiteLLM proxy?",
            store=self.store,
            kb=kb,
            complete=lambda _s, _u: json.dumps({
                "action": "reply",
                "confidence": 0.92,
                "reply": "set `num_retries` and `fallbacks` under `router_settings` in config.yaml.",
                "citations": ["https://docs.litellm.ai/docs/proxy/reliability"],
            }),
            poster=lambda _c, _text: {"id": "8880001"},
            route_info={"target": AGENT_SUPPORT, "reason": "support_question"},
        )
        self.assertEqual(ans.action, "reply")
        self.assertEqual(ans.posted_message_id, "8880001")

        # 2. Second message in same channel hits channel_cooldown (still recorded on a span!).
        cooled = answer_support_question(
            config,
            "444444444444444444",
            "How do I create virtual keys with max_budget in LiteLLM?",
            store=self.store,
            kb=kb,
            route_info={"target": AGENT_SUPPORT, "reason": "support_question"},
        )
        self.assertEqual(cooled.action, "skipped")
        self.assertEqual(cooled.reason, "channel_cooldown")

        # 3. Low-confidence skip (bypass_cooldown=True).
        low = answer_support_question(
            config,
            "444444444444444444",
            "How do I configure router fallbacks in LiteLLM?",
            store=self.store,
            kb=kb,
            bypass_cooldown=True,
            complete=lambda _s, _u: json.dumps({
                "action": "reply",
                "confidence": 0.25,
                "reply": "maybe try something",
                "citations": [],
            }),
            poster=lambda _c, _text: {"id": "never"},
        )
        self.assertEqual(low.action, "silent")
        self.assertEqual(low.reason, "low_confidence")

        spans = self.exporter.get_finished_spans()
        agent_spans = [s for s in spans if _span_attrs(s).get(OPENINFERENCE_SPAN_KIND_ATTR) == OPENINFERENCE_SPAN_KIND_AGENT]
        self.assertEqual(len(agent_spans), 3)
        self.assertTrue(all(s.name == AGENT_SUPPORT for s in agent_spans))

        # Run 1 assertions: answered.
        a1 = _span_attrs(agent_spans[0])
        self.assertIs(a1["support.question_detected"], True)
        self.assertGreaterEqual(int(a1["support.kb_hits_count"]), 1)
        self.assertIn("https://docs.litellm.ai/docs/proxy/reliability", json.loads(a1["support.kb_urls"]))
        self.assertEqual(a1["support.outcome"], "reply")
        self.assertIs(a1["support.posted"], True)
        self.assertEqual(a1["support.posted_message_id"], "8880001")
        self.assertIn(OPENINFERENCE_INPUT_VALUE_ATTR, a1)
        self.assertIn(OPENINFERENCE_OUTPUT_VALUE_ATTR, a1)
        e1 = dict(_span_events(agent_spans[0]))
        self.assertIn("support_kb_retrieved", e1)
        self.assertIn("support_answered", e1)

        # Run 2 assertions: channel_cooldown skip.
        a2 = _span_attrs(agent_spans[1])
        self.assertIs(a2["support.question_detected"], True)
        self.assertEqual(a2["support.outcome"], "skipped")
        self.assertEqual(a2["support.skip_reason"], "channel_cooldown")
        self.assertIn("support_skipped", dict(_span_events(agent_spans[1])))

        # Run 3 assertions: low_confidence skip.
        a3 = _span_attrs(agent_spans[2])
        self.assertEqual(a3["support.outcome"], "silent")
        self.assertEqual(a3["support.skip_reason"], "low_confidence")
        self.assertIn("support_skipped", dict(_span_events(agent_spans[2])))


class PRReviewThreadRehydrationAndRecheckTests(unittest.TestCase):
    """Tests for post-deploy SQLite wipe resilience: review_threads rehydration
    in recover_pending_reviews() and handle_thread_message(), @Aria re-review
    mentions, expanded wants_recheck() phrases, and ROLES_DB_PATH config.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "state.db")
        self.store = Store(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_recover_pending_reviews_rehydrates_review_threads_when_skipping_existing_bot_review(self) -> None:
        from contributor_bot import review_job

        config = _base_config(
            self.db_path,
            pr_review_enabled=True,
            review_channel_ids=("1391835163780186154",),
        )
        now = datetime(2026, 11, 28, 15, 0, 0, tzinfo=UTC)
        starter_id = "1576292666428104857"
        channel_messages = [
            {
                "id": starter_id,
                "timestamp": "2026-11-28T13:10:00+00:00",
                "author": {"id": "222000000000000001", "bot": False},
                "content": (
                    "please review https://github.com/BerriAI/litellm/pull/54805 "
                    "and https://github.com/BerriAI/litellm/pull/54852"
                ),
                "thread": {"id": starter_id},
            },
        ]
        thread_messages = [
            {
                "id": "1576293000000000001",
                "timestamp": "2026-11-28T13:15:00+00:00",
                "author": {"id": "999999999999999999", "bot": True},
                "content": "#54805 needs work\n- missing unit test",
            },
            {
                "id": "1576293000000000002",
                "timestamp": "2026-11-28T13:20:00+00:00",
                "author": {"id": "999999999999999999", "bot": True},
                "content": "#54852 looks good",
            },
        ]

        def fake_fetch_recent(target_id: str, _token: str, limit: int = 50) -> list[dict]:
            if target_id == "1391835163780186154":
                return list(channel_messages)
            if target_id == starter_id:
                return list(thread_messages)
            return []

        with mock.patch.object(review_job, "fetch_recent_messages", side_effect=fake_fetch_recent):
            queued = review_job.recover_pending_reviews(config, self.store, now=now)

        # Must NOT duplicate reviews that already exist in Discord.
        self.assertEqual(queued, [])
        # Both PRs must now be rehydrated in review_threads!
        rec_54805 = self.store.review_thread("BerriAI/litellm", 54805)
        rec_54852 = self.store.review_thread("BerriAI/litellm", 54852)
        self.assertIsNotNone(rec_54805)
        self.assertIsNotNone(rec_54852)
        self.assertEqual(rec_54805["thread_id"], starter_id)
        self.assertEqual(rec_54805["channel_id"], "1391835163780186154")
        self.assertEqual(rec_54805["message_id"], starter_id)
        self.assertEqual(rec_54805["last_reviewed_at"], "2026-11-28T13:15:00+00:00")
        self.assertEqual(rec_54852["thread_id"], starter_id)
        self.assertEqual(rec_54852["last_reviewed_at"], "2026-11-28T13:20:00+00:00")

    def test_handle_thread_message_rehydrates_from_starter_and_handles_aria_mention(self) -> None:
        from contributor_bot import review_job

        config = _base_config(
            self.db_path,
            pr_review_enabled=True,
            review_channel_ids=("1391835163780186154",),
            discord_application_id="999999999999999999",
        )
        thread_id = "1576292666428104857"
        starter_msg = {
            "id": thread_id,
            "timestamp": "2026-11-27T10:00:00+00:00",
            "author": {"id": "222000000000000001", "bot": False},
            "content": (
                "please review https://github.com/BerriAI/litellm/pull/54805 "
                "and https://github.com/BerriAI/litellm/pull/54852"
            ),
        }
        bot_messages = [
            {
                "id": "1576293000000000010",
                "timestamp": "2026-11-27T10:05:00+00:00",
                "author": {"id": "999999999999999999", "bot": True},
                "content": "#54805 needs work\n- add a test",
            },
        ]
        reviewed: list[tuple[int, str]] = []

        with (
            mock.patch.object(
                review_job,
                "fetch_channel",
                return_value={"id": thread_id, "parent_id": "1391835163780186154", "name": "pr #54805 review"},
            ),
            mock.patch.object(review_job, "fetch_message", return_value=starter_msg),
            mock.patch.object(review_job, "fetch_recent_messages", return_value=bot_messages),
            mock.patch.object(review_job, "acknowledge_message"),
            mock.patch.object(
                review_job,
                "perform_review",
                side_effect=lambda _cfg, _st, num, **kw: reviewed.append((num, kw.get("source", ""))),
            ),
        ):
            # DB is completely empty (simulating deploy wipe), and user @mentions Aria inside the thread!
            pr_review_agent.handle_thread_followup(
                config,
                self.store,
                {
                    "id": "1576300000000000001",
                    "channel_id": thread_id,
                    "author": {"id": "222000000000000001", "bot": False},
                    "content": "<@999999999999999999> can you look at #54805?",
                    "mentions": [{"id": "999999999999999999", "bot": True, "username": "Aria"}],
                },
            )

        # Store rehydrated for BOTH PRs, and @Aria mention triggered re-review for #54805!
        self.assertEqual([r["number"] for r in self.store.review_threads_by_discord(thread_id)], [54805, 54852])
        self.assertEqual(reviewed, [(54805, "recheck")])

    def test_handle_thread_message_rehydrates_from_bot_messages_when_starter_unavailable_and_logs_unknown_skip(self) -> None:
        from contributor_bot import review_job

        config = _base_config(
            self.db_path,
            pr_review_enabled=True,
            review_channel_ids=("1391835163780186154",),
        )
        thread_id = "1576292666428104900"
        bot_only_messages = [
            {
                "id": "1576293000000000020",
                "timestamp": "2026-11-27T11:00:00+00:00",
                "author": {"id": "999999999999999999", "bot": True},
                "content": "needs work on #54852 - please add error handling",
            },
        ]
        reviewed: list[tuple[int, str]] = []

        with (
            mock.patch.object(
                review_job,
                "fetch_channel",
                return_value={"id": thread_id, "parent_id": "1391835163780186154"},
            ),
            mock.patch.object(review_job, "fetch_message", side_effect=RuntimeError("HTTP 404")),
            mock.patch.object(review_job, "fetch_recent_messages", return_value=bot_only_messages),
            mock.patch.object(review_job, "acknowledge_message"),
            mock.patch.object(
                review_job,
                "perform_review",
                side_effect=lambda _cfg, _st, num, **kw: reviewed.append((num, kw.get("source", ""))),
            ),
        ):
            review_job.handle_thread_message(
                config,
                self.store,
                {
                    "id": "1576300000000000002",
                    "channel_id": thread_id,
                    "author": {"id": "222000000000000001", "bot": False},
                    "content": "@Aria review again",
                },
            )

        self.assertEqual([r["number"] for r in self.store.review_threads_by_discord(thread_id)], [54852])
        self.assertEqual(reviewed, [(54852, "recheck")])

        # Completely unrelated thread that cannot be rehydrated: logs INFO skip and does not review.
        unknown_thread_id = "1576292666428104999"
        with (
            mock.patch.object(review_job, "fetch_channel", return_value={"id": unknown_thread_id, "parent_id": "999999"}),
            mock.patch.object(review_job, "fetch_message", return_value={"id": unknown_thread_id, "content": "random chat"}),
            mock.patch.object(review_job, "fetch_recent_messages", return_value=[]),
            mock.patch.object(review_job, "perform_review") as mock_review,
            self.assertLogs(level="INFO") as cm,
        ):
            review_job.handle_thread_message(
                config,
                self.store,
                {
                    "id": "1576300000000000003",
                    "channel_id": unknown_thread_id,
                    "author": {"id": "222000000000000001", "bot": False},
                    "content": "just chatting",
                },
            )
        mock_review.assert_not_called()
        self.assertTrue(any(unknown_thread_id in line and "Skipping" in line for line in cm.output))

    def test_wants_recheck_matches_review_again_and_rereview_variants(self) -> None:
        from contributor_bot.reviews import wants_recheck

        for phrase in (
            "review again",
            "review it again",
            "please review again",
            "re-review",
            "rereview",
            "Can you please review it again?",
            "pushed updates, re-review when ready",
            "check again",
            "recheck",
            "fixed it",
        ):
            self.assertTrue(wants_recheck(phrase), f"Expected wants_recheck({phrase!r}) to be True")

        for non_recheck in (
            "thanks for the review!",
            "what does this failure mean?",
            "sounds good",
        ):
            self.assertFalse(wants_recheck(non_recheck), f"Expected wants_recheck({non_recheck!r}) to be False")

    def test_roles_db_path_env_config_and_default(self) -> None:
        base_env = {
            "DISCORD_BOT_TOKEN": "tok",
            "OSS_CONTRIBUTIONS_CHANNEL_ID": "111",
            "ANNOUNCEMENTS_CHANNEL_ID": "222",
        }
        self.assertEqual(get_config(base_env).roles_db_path, ".data/contributor_roles.db")
        custom = get_config(base_env | {"ROLES_DB_PATH": "/opt/render/project/src/.data/custom.db"})
        self.assertEqual(custom.roles_db_path, "/opt/render/project/src/.data/custom.db")


if __name__ == "__main__":
    unittest.main()
