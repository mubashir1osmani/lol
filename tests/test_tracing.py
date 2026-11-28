import contextvars
import os
import sys
import types
import unittest
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest import mock

from contributor_bot.matcher import (
    CandidatePair,
    DiscordMemberInfo,
    GitHubAuthorProfile,
    _select_with_llm,
    match_community_contributors_with_diagnostics,
)
from contributor_bot.questions import answer_question, _recent as _qa_recent
from contributor_bot.review_agent import run_review_session
from contributor_bot.review_job import message_decision
from contributor_bot.review_llm import complete_litellm
from contributor_bot.routing import decide, reset_decision_cache
from contributor_bot.schedule import UTC
from contributor_bot.tracing import (
    AGENT_ENGAGEMENT,
    AGENT_PR_REVIEW,
    AGENT_SUPPORT,
    OPENINFERENCE_AGENT_NAME_ATTR,
    OPENINFERENCE_INPUT_VALUE_ATTR,
    OPENINFERENCE_OUTPUT_VALUE_ATTR,
    OPENINFERENCE_SESSION_ID_ATTR,
    OPENINFERENCE_SPAN_KIND_AGENT,
    OPENINFERENCE_SPAN_KIND_ATTR,
    OPENINFERENCE_SPAN_KIND_CHAIN,
    TOP_LEVEL_AGENTS,
    _reset_tracing_state,
    agent_span,
    build_sdk_trace_metadata,
    build_trace_metadata,
    configure_litellm_tracing,
    normalize_otlp_traces_endpoint,
    parse_otel_headers,
    shutdown_tracing,
    tracing_endpoint,
)
from contributor_bot.weekly import summarize_pr_titles


_ACTIVE_SPAN: contextvars.ContextVar = contextvars.ContextVar("_test_active_span", default=None)
_ACTIVE_OI_SESSION: contextvars.ContextVar = contextvars.ContextVar("_test_oi_session", default="")


class _FakeSpan:
    def __init__(self, name: str, parent=None) -> None:
        self.name = name
        self.parent = parent
        self.attributes: dict[str, object] = {}
        self.events: list[tuple[str, dict[str, object]]] = []
        self.exceptions: list[Exception] = []
        self.status = None
        self.oi_session_at_start = _ACTIVE_OI_SESSION.get()

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        self.events.append((name, dict(attributes or {})))

    def record_exception(self, exc: Exception) -> None:
        self.exceptions.append(exc)

    def set_status(self, status) -> None:
        self.status = status


class _FakeTracer:
    def __init__(self, recorded: list[_FakeSpan]) -> None:
        self.recorded = recorded

    @contextmanager
    def start_as_current_span(self, name: str):
        parent = _ACTIVE_SPAN.get()
        span = _FakeSpan(name, parent=parent)
        self.recorded.append(span)
        token = _ACTIVE_SPAN.set(span)
        try:
            yield span
        finally:
            _ACTIVE_SPAN.reset(token)


def _fake_otel_and_openinference_modules():
    """Build an in-memory fake OTel + OpenInference module tree for offline tests."""
    state = {
        "resources": [],
        "providers": [],
        "exporters": [],
        "processors": [],
        "set_provider_calls": [],
        "instrument_calls": [],
        "spans": [],
        "session_scopes": [],
    }

    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"

    class Status:
        def __init__(self, code, description=None):
            self.status_code = code
            self.description = description

    class Resource:
        def __init__(self, attrs):
            self.attributes = dict(attrs)

        @classmethod
        def create(cls, attrs):
            res = cls(attrs)
            state["resources"].append(res)
            return res

    class TracerProvider:
        def __init__(self, resource=None):
            self.resource = resource
            self.processors = []
            self.flush_called = 0
            self.shutdown_called = 0
            state["providers"].append(self)

        def add_span_processor(self, processor):
            self.processors.append(processor)

        def get_tracer(self, name):
            return _FakeTracer(state["spans"])

        def force_flush(self, timeout_millis=None):
            self.flush_called += 1
            return True

        def shutdown(self):
            self.shutdown_called += 1

    class BatchSpanProcessor:
        def __init__(self, exporter, **kwargs):
            self.exporter = exporter
            self.kwargs = dict(kwargs)
            state["processors"].append(self)

    class Compression:
        Gzip = "gzip"

    class OTLPSpanExporter:
        def __init__(self, endpoint=None, headers=None, **kwargs):
            self.endpoint = endpoint
            self.headers = headers
            self.kwargs = dict(kwargs)
            state["exporters"].append(self)

    class TraceConfig:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    class LiteLLMInstrumentor:
        def instrument(self, **kwargs):
            state["instrument_calls"].append(kwargs)

    @contextmanager
    def using_session(session_id: str):
        state["session_scopes"].append(session_id)
        token = _ACTIVE_OI_SESSION.set(session_id)
        try:
            yield
        finally:
            _ACTIVE_OI_SESSION.reset(token)

    otel_pkg = types.ModuleType("opentelemetry")
    otel_trace = types.ModuleType("opentelemetry.trace")
    otel_trace.Status = Status
    otel_trace.StatusCode = StatusCode
    otel_trace.set_tracer_provider = lambda provider: state["set_provider_calls"].append(provider)
    otel_pkg.trace = otel_trace

    otel_sdk = types.ModuleType("opentelemetry.sdk")
    otel_sdk_resources = types.ModuleType("opentelemetry.sdk.resources")
    otel_sdk_resources.Resource = Resource
    otel_sdk_trace = types.ModuleType("opentelemetry.sdk.trace")
    otel_sdk_trace.TracerProvider = TracerProvider
    otel_sdk_trace_export = types.ModuleType("opentelemetry.sdk.trace.export")
    otel_sdk_trace_export.BatchSpanProcessor = BatchSpanProcessor

    otel_exporter = types.ModuleType("opentelemetry.exporter")
    otel_exporter_otlp = types.ModuleType("opentelemetry.exporter.otlp")
    otel_exporter_otlp_proto = types.ModuleType("opentelemetry.exporter.otlp.proto")
    otel_exporter_otlp_proto_http = types.ModuleType("opentelemetry.exporter.otlp.proto.http")
    otel_exporter_otlp_proto_http.Compression = Compression
    otel_exporter_otlp_proto_http_trace = types.ModuleType(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter"
    )
    otel_exporter_otlp_proto_http_trace.OTLPSpanExporter = OTLPSpanExporter

    oi_pkg = types.ModuleType("openinference")
    oi_inst = types.ModuleType("openinference.instrumentation")
    oi_inst.using_session = using_session
    oi_inst.TraceConfig = TraceConfig
    oi_litellm = types.ModuleType("openinference.instrumentation.litellm")
    oi_litellm.LiteLLMInstrumentor = LiteLLMInstrumentor

    modules = {
        "opentelemetry": otel_pkg,
        "opentelemetry.trace": otel_trace,
        "opentelemetry.sdk": otel_sdk,
        "opentelemetry.sdk.resources": otel_sdk_resources,
        "opentelemetry.sdk.trace": otel_sdk_trace,
        "opentelemetry.sdk.trace.export": otel_sdk_trace_export,
        "opentelemetry.exporter": otel_exporter,
        "opentelemetry.exporter.otlp": otel_exporter_otlp,
        "opentelemetry.exporter.otlp.proto": otel_exporter_otlp_proto,
        "opentelemetry.exporter.otlp.proto.http": otel_exporter_otlp_proto_http,
        "opentelemetry.exporter.otlp.proto.http.trace_exporter": otel_exporter_otlp_proto_http_trace,
        "openinference": oi_pkg,
        "openinference.instrumentation": oi_inst,
        "openinference.instrumentation.litellm": oi_litellm,
    }
    return modules, state


def _fake_litellm_module(callbacks=None, *, recorded_spans: list[_FakeSpan] | None = None):
    mod = types.ModuleType("litellm")
    mod.callbacks = [] if callbacks is None else list(callbacks)
    mod.calls = []

    def completion(**kwargs):
        active_parent = _ACTIVE_SPAN.get()
        if recorded_spans is not None:
            child = _FakeSpan("litellm.completion", parent=active_parent)
            child.set_attribute("openinference.span.kind", "LLM")
            child.set_attribute("session.id", _ACTIVE_OI_SESSION.get())
            recorded_spans.append(child)
        mod.calls.append((list(mod.callbacks), kwargs, active_parent))
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    mod.completion = completion
    return mod


class TracingConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_tracing_state()

    def tearDown(self) -> None:
        _reset_tracing_state()

    def test_stays_off_when_env_vars_unset(self):
        fake = _fake_litellm_module()
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            enabled = configure_litellm_tracing(env={})
        self.assertFalse(enabled)
        self.assertEqual(fake.callbacks, [])

    def test_stays_off_for_blank_endpoint(self):
        fake = _fake_litellm_module()
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            enabled = configure_litellm_tracing(env={"OTEL_ENDPOINT": "   "})
        self.assertFalse(enabled)
        self.assertEqual(fake.callbacks, [])

    def test_stays_off_without_importing_third_party_packages(self):
        saved = {}
        for name in ("litellm", "opentelemetry", "openinference"):
            if name in sys.modules:
                saved[name] = sys.modules.pop(name)
        try:
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertFalse(configure_litellm_tracing())
                with agent_span("role_matcher", input_value={"test": 1}) as span:
                    self.assertFalse(span.active)
                    span.set_output({"ok": True})
                self.assertNotIn("litellm", sys.modules)
                self.assertNotIn("opentelemetry", sys.modules)
                self.assertNotIn("openinference", sys.modules)
        finally:
            sys.modules.update(saved)

    def test_header_parsing_preserves_literal_values_and_endpoint_normalization(self):
        self.assertEqual(
            parse_otel_headers({
                "OTEL_HEADERS": "Authorization=Bearer abc=def==, X-Custom = hello world , malformed , =missingkey",
            }),
            {
                "Authorization": "Bearer abc=def==",
                "X-Custom": "hello world",
            },
        )
        self.assertEqual(
            parse_otel_headers({"OTEL_EXPORTER_OTLP_HEADERS": "X-Api-Key=literal%20value"}),
            {"X-Api-Key": "literal%20value"},
        )
        self.assertEqual(
            normalize_otlp_traces_endpoint("https://lens-f7cz.onrender.com"),
            "https://lens-f7cz.onrender.com/v1/traces",
        )
        self.assertEqual(
            normalize_otlp_traces_endpoint("https://lens-f7cz.onrender.com/"),
            "https://lens-f7cz.onrender.com/v1/traces",
        )
        self.assertEqual(
            normalize_otlp_traces_endpoint("https://lens-f7cz.onrender.com/v1/traces"),
            "https://lens-f7cz.onrender.com/v1/traces",
        )
        self.assertEqual(
            tracing_endpoint({
                "OTEL_ENDPOINT": "https://fallback.example.com",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "https://preferred.example.com",
            }),
            "https://preferred.example.com",
        )

    def test_openinference_instrumentation_setup_does_not_add_otel_callback_and_is_idempotent(self):
        fake_litellm = _fake_litellm_module(callbacks=["existing"])
        fake_mods, state = _fake_otel_and_openinference_modules()
        fake_mods["litellm"] = fake_litellm
        env = {
            "OTEL_ENDPOINT": "https://lens-f7cz.onrender.com",
            "OTEL_HEADERS": "Authorization=Bearer secret==key, X-Tenant=berri",
            "OTEL_SERVICE_NAME": "litellm-contributor-bot",
        }
        with mock.patch.dict(sys.modules, fake_mods):
            self.assertTrue(configure_litellm_tracing(env=env))
            # Repeated call is a no-op (does not re-instrument or create a 2nd provider).
            self.assertTrue(configure_litellm_tracing(env=env))

        # CRITICAL: "otel" must NOT be added to litellm.callbacks.
        self.assertEqual(fake_litellm.callbacks, ["existing"])
        self.assertNotIn("otel", fake_litellm.callbacks)

        self.assertEqual(len(state["resources"]), 1)
        self.assertEqual(state["resources"][0].attributes, {"service.name": "litellm-contributor-bot"})
        self.assertEqual(len(state["exporters"]), 1)
        self.assertEqual(state["exporters"][0].endpoint, "https://lens-f7cz.onrender.com/v1/traces")
        self.assertEqual(
            state["exporters"][0].headers,
            {"Authorization": "Bearer secret==key", "X-Tenant": "berri"},
        )
        self.assertEqual(state["exporters"][0].kwargs.get("compression"), "gzip")
        self.assertEqual(state["exporters"][0].kwargs.get("timeout"), 30)
        self.assertEqual(len(state["providers"]), 1)
        self.assertEqual(len(state["processors"]), 1)
        self.assertEqual(state["processors"][0].kwargs.get("max_export_batch_size"), 16)
        self.assertEqual(state["processors"][0].kwargs.get("export_timeout_millis"), 30000)
        self.assertIs(state["set_provider_calls"][0], state["providers"][0])
        self.assertEqual(len(state["instrument_calls"]), 1)
        self.assertIs(state["instrument_calls"][0]["tracer_provider"], state["providers"][0])
        trace_cfg = state["instrument_calls"][0].get("config")
        self.assertIsNotNone(trace_cfg)
        self.assertTrue(trace_cfg.kwargs.get("hide_llm_invocation_parameters"))

        shutdown_tracing()
        shutdown_tracing()
        self.assertEqual(state["providers"][0].flush_called, 1)
        self.assertEqual(state["providers"][0].shutdown_called, 1)

    def test_default_service_name_is_aria_litellm_discord(self):
        fake_litellm = _fake_litellm_module()
        fake_mods, state = _fake_otel_and_openinference_modules()
        fake_mods["litellm"] = fake_litellm
        with mock.patch.dict(sys.modules, fake_mods):
            self.assertTrue(configure_litellm_tracing(env={"OTEL_ENDPOINT": "https://lens.example.com"}))
        self.assertEqual(state["resources"][0].attributes, {"service.name": "aria-litellm-discord"})

    def test_turns_on_with_otel_exporter_otlp_endpoint(self):
        fake_litellm = _fake_litellm_module()
        fake_mods, state = _fake_otel_and_openinference_modules()
        fake_mods["litellm"] = fake_litellm
        env = {"OTEL_EXPORTER_OTLP_ENDPOINT": "https://lens.example.com/v1/traces"}
        with mock.patch.dict(sys.modules, fake_mods):
            self.assertTrue(configure_litellm_tracing(env=env))
        self.assertEqual(fake_litellm.callbacks, [])
        self.assertEqual(state["exporters"][0].endpoint, "https://lens.example.com/v1/traces")
        self.assertEqual(len(state["instrument_calls"]), 1)

    def test_main_startup_enables_tracing_only_when_env_set(self):
        from contributor_bot import main as bot_main

        fake_mods_off, state_off = _fake_otel_and_openinference_modules()
        fake_mods_off["litellm"] = _fake_litellm_module()
        with mock.patch.dict(sys.modules, fake_mods_off):
            with mock.patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "tok"}, clear=True):
                with mock.patch.object(bot_main, "load_dotenv"):
                    with mock.patch.object(bot_main, "send_test_message", return_value="https://discord.com/x"):
                        bot_main.main(["--test-channel-id", "123456789012345678"])
        self.assertEqual(state_off["instrument_calls"], [])

        _reset_tracing_state()
        fake_mods_on, state_on = _fake_otel_and_openinference_modules()
        fake_litellm_on = _fake_litellm_module()
        fake_mods_on["litellm"] = fake_litellm_on
        with mock.patch.dict(sys.modules, fake_mods_on):
            with mock.patch.dict(
                os.environ,
                {"DISCORD_BOT_TOKEN": "tok", "OTEL_ENDPOINT": "https://lens.example.com/v1/traces"},
                clear=True,
            ):
                with mock.patch.object(bot_main, "load_dotenv"):
                    with mock.patch.object(bot_main, "send_test_message", return_value="https://discord.com/x"):
                        bot_main.main(["--test-channel-id", "123456789012345678"])
        self.assertEqual(len(state_on["instrument_calls"]), 1)
        self.assertEqual(fake_litellm_on.callbacks, [])

    def test_complete_litellm_instruments_openinference_without_otel_callback(self):
        fake_mods, state = _fake_otel_and_openinference_modules()
        fake_litellm = _fake_litellm_module(recorded_spans=state["spans"])
        fake_mods["litellm"] = fake_litellm
        meta = build_trace_metadata("role_matcher", session_id="role_matcher:octocat", extra={"github_login": "octocat"})
        with mock.patch.dict(sys.modules, fake_mods):
            with mock.patch.dict(os.environ, {"OTEL_ENDPOINT": "https://lens.example.com/v1/traces"}, clear=True):
                result = complete_litellm("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}], metadata=meta)
        self.assertEqual(result, '{"ok": true}')
        self.assertEqual(len(fake_litellm.calls), 1)
        callbacks_at_call, kwargs, _parent = fake_litellm.calls[0]
        self.assertEqual(callbacks_at_call, [])
        self.assertEqual(len(state["instrument_calls"]), 1)
        self.assertEqual(kwargs["metadata"]["generation_name"], "role_matcher")
        self.assertEqual(kwargs["metadata"]["agent_name"], "role_matcher")
        self.assertEqual(kwargs["metadata"]["session_id"], "role_matcher:octocat")
        self.assertEqual(kwargs["metadata"]["requester_metadata"]["github_login"], "octocat")

    def test_agent_span_records_exceptions_and_error_status(self):
        fake_mods, state = _fake_otel_and_openinference_modules()
        with mock.patch.dict(sys.modules, fake_mods):
            with mock.patch.dict(os.environ, {"OTEL_ENDPOINT": "https://lens.example.com/v1/traces"}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    with agent_span("pr_review", session_id="pr_review:repo#1", input_value="review #1"):
                        raise RuntimeError("boom")
        self.assertEqual(len(state["spans"]), 1)
        span = state["spans"][0]
        self.assertEqual(span.attributes[OPENINFERENCE_SPAN_KIND_ATTR], OPENINFERENCE_SPAN_KIND_AGENT)
        self.assertEqual(len(span.exceptions), 1)
        self.assertEqual(span.status.status_code, "ERROR")


class AgentSpanWrappingTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_tracing_state()
        reset_decision_cache()
        _qa_recent.clear()

    def tearDown(self) -> None:
        _reset_tracing_state()
        reset_decision_cache()
        _qa_recent.clear()

    def test_role_matcher_wraps_run_in_agent_span_and_propagates_context_across_threads(self):
        fake_mods, state = _fake_otel_and_openinference_modules()
        fake_litellm = _fake_litellm_module(recorded_spans=state["spans"])
        fake_mods["litellm"] = fake_litellm

        config = types.SimpleNamespace(
            github_repository="BerriAI/litellm",
            github_token="ghp_test",
            match_model="anthropic/claude-sonnet-4-5",
            review_model="anthropic/claude-sonnet-4-5",
            github_discord_map={},
            discord_guild_id="300000000000000001",
            discord_token="tok",
        )
        # Two ambiguous authors (each with 2 plausible Discord candidates) so
        # _disambiguate_with_reasons exercises ThreadPoolExecutor(workers=2).
        groups = [
            {"login": "alex-builds", "user_id": 1, "pulls": [{"number": 101, "title": "fix a"}]},
            {"login": "jordan-codes", "user_id": 2, "pulls": [{"number": 102, "title": "fix b"}]},
        ]
        members = [
            DiscordMemberInfo(discord_id="100000000000000001", username="alex_one", global_name="Alex Rivera"),
            DiscordMemberInfo(discord_id="100000000000000002", username="alex_two", global_name="Alex Rivera"),
            DiscordMemberInfo(discord_id="100000000000000003", username="jordan_one", global_name="Jordan Lee"),
            DiscordMemberInfo(discord_id="100000000000000004", username="jordan_two", global_name="Jordan Lee"),
        ]

        def fake_graphql(_token, query, _variables):
            if "BatchProfiles" in query:
                return {
                    "data": {
                        "u0": {
                            "login": "alex-builds", "name": "Alex Rivera", "bio": "backend", "company": "",
                            "email": "", "twitterUsername": None, "websiteUrl": None,
                            "socialAccounts": {"nodes": []}, "profileRepo": None,
                        },
                        "u1": {
                            "login": "jordan-codes", "name": "Jordan Lee", "bio": "infra", "company": "",
                            "email": "", "twitterUsername": None, "websiteUrl": None,
                            "socialAccounts": {"nodes": []}, "profileRepo": None,
                        },
                    }
                }
            return {"data": {"repository": {}}}

        with mock.patch.dict(sys.modules, fake_mods):
            with mock.patch.dict(
                os.environ,
                {"OTEL_ENDPOINT": "https://lens.example.com/v1/traces", "ANTHROPIC_API_KEY": "test-key"},
                clear=True,
            ):
                match_community_contributors_with_diagnostics(
                    groups,
                    config,
                    store=None,
                    now=datetime(2026, 11, 28, 12, 0, tzinfo=UTC),
                    members=members,
                    graphql_caller=fake_graphql,
                )

        agent_spans = [s for s in state["spans"] if s.attributes.get(OPENINFERENCE_SPAN_KIND_ATTR) == OPENINFERENCE_SPAN_KIND_AGENT]
        chain_spans = [s for s in state["spans"] if s.attributes.get(OPENINFERENCE_SPAN_KIND_ATTR) == OPENINFERENCE_SPAN_KIND_CHAIN]
        llm_spans = [s for s in state["spans"] if s.attributes.get(OPENINFERENCE_SPAN_KIND_ATTR) == "LLM"]
        self.assertEqual(len(agent_spans), 1)
        root = agent_spans[0]
        self.assertEqual(root.name, AGENT_ENGAGEMENT)
        self.assertEqual(root.attributes[OPENINFERENCE_AGENT_NAME_ATTR], AGENT_ENGAGEMENT)
        self.assertEqual(root.attributes[OPENINFERENCE_SESSION_ID_ATTR], "engagement_agent:role_matcher:BerriAI/litellm:2026-11-28")
        self.assertIn("alex-builds", str(root.attributes.get(OPENINFERENCE_INPUT_VALUE_ATTR)))
        self.assertIn("matched", str(root.attributes.get(OPENINFERENCE_OUTPUT_VALUE_ATTR)))
        # role_matcher is a CHAIN child span of engagement_agent (NOT a separate top-level AGENT),
        # and each contributor gets a match_contributor CHAIN child span under role_matcher.
        rm_spans = [s for s in chain_spans if s.name == "role_matcher"]
        contrib_spans = [s for s in chain_spans if s.name == "match_contributor"]
        self.assertEqual(len(rm_spans), 1)
        rm_span = rm_spans[0]
        self.assertIs(rm_span.parent, root)
        self.assertNotIn(OPENINFERENCE_AGENT_NAME_ATTR, rm_span.attributes)
        self.assertEqual(len(contrib_spans), 2)
        for cspan in contrib_spans:
            self.assertIs(cspan.parent, rm_span)
            self.assertNotIn(OPENINFERENCE_AGENT_NAME_ATTR, cspan.attributes)
        # Every parallel LLM span created in ThreadPoolExecutor nested under its contributor's
        # match_contributor -> role_matcher -> engagement_agent!
        self.assertGreaterEqual(len(llm_spans), 2)
        for child in llm_spans:
            self.assertIn(child.parent, contrib_spans)
            self.assertIs(child.parent.parent, rm_span)
            self.assertEqual(child.attributes["session.id"], "engagement_agent:role_matcher:BerriAI/litellm:2026-11-28")
        # Root AGENT span exit flushed the TracerProvider.
        self.assertGreaterEqual(state["providers"][0].flush_called, 1)

    def test_pr_review_session_wraps_in_agent_span_and_nests_sdk_run(self):
        fake_mods, state = _fake_otel_and_openinference_modules()
        captured = {}

        class LitellmModel:
            def __init__(self, model, **kwargs):
                captured["model"] = model

        class ModelSettings:
            def __init__(self, **kwargs):
                captured["settings"] = kwargs

        class Agent:
            def __init__(self, **kwargs):
                captured["agent_name"] = kwargs.get("name")

        class Runner:
            @staticmethod
            async def run(agent, incoming, context=None, max_turns=None, hooks=None):
                captured["active_span_in_async"] = _ACTIVE_SPAN.get()
                return type("Result", (), {"final_output": "looks good"})()

        class RunHooks:
            pass

        sdk = types.SimpleNamespace(
            Agent=Agent,
            LitellmModel=LitellmModel,
            ModelSettings=ModelSettings,
            Runner=Runner,
            RunHooks=RunHooks,
            RunContextWrapper=object,
            function_tool=lambda fn=None, **kw: fn if callable(fn) else (lambda f: f),
            set_tracing_disabled=lambda _flag: None,
        )
        config = types.SimpleNamespace(
            github_repository="BerriAI/litellm",
            review_model="anthropic/claude-sonnet-4-5",
            review_max_steps=5,
            review_timeout_seconds=30,
            review_usage_cap=0,
            review_agent_tracing=False,
            discord_token="tok",
        )
        with mock.patch.dict(sys.modules, fake_mods):
            with mock.patch.dict(os.environ, {"OTEL_ENDPOINT": "https://lens.example.com/v1/traces"}, clear=True):
                with mock.patch("contributor_bot.review_agent._load_sdk", return_value=sdk):
                    with mock.patch("contributor_bot.review_agent._post_text"):
                        run_review_session(config, None, 4242, "999", mode="review")

        agent_spans = [s for s in state["spans"] if s.attributes.get(OPENINFERENCE_SPAN_KIND_ATTR) == OPENINFERENCE_SPAN_KIND_AGENT]
        chain_spans = [s for s in state["spans"] if s.attributes.get(OPENINFERENCE_SPAN_KIND_ATTR) == OPENINFERENCE_SPAN_KIND_CHAIN]
        self.assertEqual(len(agent_spans), 1)
        self.assertEqual(len(chain_spans), 1)
        root = agent_spans[0]
        session_span = chain_spans[0]
        self.assertEqual(root.name, AGENT_PR_REVIEW)
        self.assertEqual(root.attributes[OPENINFERENCE_AGENT_NAME_ATTR], AGENT_PR_REVIEW)
        self.assertEqual(root.attributes[OPENINFERENCE_SESSION_ID_ATTR], "pr_review_agent:BerriAI/litellm#4242")
        self.assertIn("4242", str(root.attributes.get(OPENINFERENCE_INPUT_VALUE_ATTR)))
        self.assertIn("looks good", str(root.attributes.get(OPENINFERENCE_OUTPUT_VALUE_ATTR)))
        self.assertEqual(session_span.name, "pr_review_session")
        self.assertIs(session_span.parent, root)
        # Async Runner.run inherited the nested pr_review_session -> pr_review_agent span!
        self.assertIs(captured["active_span_in_async"], session_span)
        # SDK metadata tags pr_review_agent cleanly.
        meta = captured["settings"]["metadata"]
        self.assertEqual(meta["agent_name"], AGENT_PR_REVIEW)
        self.assertEqual(meta["pr_number"], "4242")
        # Root AGENT span exit flushed the TracerProvider so PR reviews aren't lost.
        self.assertGreaterEqual(state["providers"][0].flush_called, 1)

    def test_only_three_top_level_agent_spans_exist_across_all_bot_tasks(self):
        fake_mods, state = _fake_otel_and_openinference_modules()
        fake_litellm = _fake_litellm_module(recorded_spans=state["spans"])
        fake_mods["litellm"] = fake_litellm

        config = types.SimpleNamespace(
            github_repository="BerriAI/litellm",
            review_classifier_model="anthropic/claude-haiku-4-5",
            review_model="anthropic/claude-sonnet-4-5",
            fast_model="anthropic/claude-haiku-4-5",
            support_agent_model="anthropic/claude-haiku-4-5",
            support_kb_refresh_hours=12,
            effective_github_token=None,
            review_classify=True,
            review_require_ask=False,
            discord_token="tok",
        )
        with mock.patch.dict(sys.modules, fake_mods):
            with mock.patch.dict(
                os.environ,
                {"OTEL_ENDPOINT": "https://lens.example.com/v1/traces", "ANTHROPIC_API_KEY": "test-key"},
                clear=True,
            ):
                # 1. Engagement agent -> weekly_summary child span
                summarize_pr_titles(config, [{"pulls": [{"number": 10, "title": "feat: add lens"}]}])
                # 2. Support agent (via backward-compatible questions.answer_question facade)
                with mock.patch("contributor_bot.questions.post_message"):
                    answer_question(
                        config,
                        "555666777",
                        "how do i run tests?",
                        complete=lambda _s, _b: (
                            fake_litellm.completion(
                                model="anthropic/claude-haiku-4-5",
                                messages=[{"role": "user", "content": "how do i run tests?"}],
                            )["choices"][0]["message"]["content"]
                        ),
                    )
                # 3. PR review agent -> pr_review_classifier child span
                message_decision(
                    config,
                    {
                        "id": "888999",
                        "channel_id": "1391835163780186154",
                        "author": {"id": "111", "bot": False},
                        "content": "can someone review https://github.com/BerriAI/litellm/pull/54805?",
                    },
                )
                # 4. Standalone decide() does NOT register a 4th top-level agent in Lens.
                decide(config, "how do I configure proxy timeout in litellm?", channel="general")

        agent_spans = [
            s for s in state["spans"]
            if s.attributes.get(OPENINFERENCE_SPAN_KIND_ATTR) == OPENINFERENCE_SPAN_KIND_AGENT
        ]
        agent_names = {s.attributes.get(OPENINFERENCE_AGENT_NAME_ATTR) for s in agent_spans}
        self.assertEqual(agent_names, TOP_LEVEL_AGENTS)
        self.assertEqual(agent_names, {AGENT_PR_REVIEW, AGENT_SUPPORT, AGENT_ENGAGEMENT})

        chain_names = {
            s.name for s in state["spans"]
            if s.attributes.get(OPENINFERENCE_SPAN_KIND_ATTR) == OPENINFERENCE_SPAN_KIND_CHAIN
        }
        self.assertIn("weekly_summary", chain_names)
        self.assertIn("pr_review_classifier", chain_names)
        self.assertNotIn("discord_qa", agent_names)
        self.assertNotIn("message_router", agent_names)
        self.assertNotIn("role_matcher", agent_names)
        self.assertNotIn("weekly_summary", agent_names)

        # Verify no "otel" callback was ever added to litellm.
        self.assertEqual(fake_litellm.callbacks, [])

    def test_role_matcher_single_call_tags_metadata(self):
        fake = _fake_litellm_module()
        profile = GitHubAuthorProfile(login="octocat", name="Mona Octocat")
        member = DiscordMemberInfo(discord_id="123456789012345678", username="octocat", global_name="Mona")
        candidate = CandidatePair(profile=profile, member=member, heuristic_score=0.9, signals=("exact_login",))
        config = types.SimpleNamespace(match_model="anthropic/claude-sonnet-4-5", review_model="anthropic/claude-sonnet-4-5")
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
                _select_with_llm(config, profile, [candidate])
        self.assertEqual(len(fake.calls), 1)
        _, kwargs, _ = fake.calls[0]
        self.assertEqual(kwargs["metadata"]["agent_name"], AGENT_ENGAGEMENT)
        self.assertEqual(kwargs["metadata"]["generation_name"], AGENT_ENGAGEMENT)
        self.assertEqual(kwargs["metadata"]["session_id"], "engagement_agent:role_matcher:octocat")

    def test_sdk_metadata_values_are_all_strings(self):
        meta = build_sdk_trace_metadata("pr_review", session_id="s1", extra={"pr_number": 99, "empty": ""})
        self.assertTrue(all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()))
        self.assertEqual(meta["pr_number"], "99")
        self.assertNotIn("empty", meta)


class RenderBlueprintAndDependencyTests(unittest.TestCase):
    def test_render_yaml_defines_only_lens_services_and_not_bot_worker(self):
        text = (Path(__file__).resolve().parents[1] / "render.yaml").read_text(encoding="utf-8")
        self.assertIn("name: aria-litellm-discord", text)
        self.assertIn("name: lens-clickhouse", text)
        self.assertIn("name: lens", text)
        self.assertIn("ghcr.io/berriai/lens:0.6.0", text)
        self.assertIn("healthCheckPath: /health/ready", text)
        self.assertIn("key: PORT", text)
        self.assertIn('value: "4318"', text)
        self.assertNotIn("type: worker", text)
        self.assertNotIn("srv-dc4a3215efls73dg9no0", text)
        self.assertNotIn("name: lol", text)
        self.assertNotIn("name: litellm-discord", text)

    def test_requirements_pins_openinference_compatible_with_litellm_and_matches_ci(self):
        root = Path(__file__).resolve().parents[1]
        req_lines = [
            line.strip()
            for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertIn("litellm==1.111.0", req_lines)
        self.assertIn("openinference-instrumentation-litellm==0.1.61", req_lines)
        self.assertIn("openinference-instrumentation==0.1.83", req_lines)
        self.assertIn("openinference-semantic-conventions==0.1.47", req_lines)
        self.assertIn("opentelemetry-instrumentation==0.66b2", req_lines)
        ci_text = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        for dep in req_lines:
            self.assertIn(f'"{dep}"', ci_text)


if __name__ == "__main__":
    unittest.main()
