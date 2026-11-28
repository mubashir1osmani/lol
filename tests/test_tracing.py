import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from contributor_bot.matcher import CandidatePair, DiscordMemberInfo, GitHubAuthorProfile, _select_with_llm
from contributor_bot.questions import answer_question
from contributor_bot.review_agent import ReviewContext, _run_sdk_session
from contributor_bot.review_llm import complete_litellm
from contributor_bot.tracing import (
    build_sdk_trace_metadata,
    build_trace_metadata,
    configure_litellm_tracing,
    tracing_endpoint,
)
from contributor_bot.weekly import summarize_pr_titles


def _fake_litellm_module(callbacks=None):
    mod = types.ModuleType("litellm")
    mod.callbacks = [] if callbacks is None else list(callbacks)
    mod.calls = []

    def completion(**kwargs):
        mod.calls.append((list(mod.callbacks), kwargs))
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    mod.completion = completion
    return mod


class TracingConfigTests(unittest.TestCase):
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

    def test_stays_off_without_importing_litellm(self):
        original = sys.modules.pop("litellm", None)
        try:
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertFalse(configure_litellm_tracing())
                self.assertNotIn("litellm", sys.modules)
        finally:
            if original is not None:
                sys.modules["litellm"] = original

    def test_turns_on_with_otel_endpoint_and_is_idempotent(self):
        fake = _fake_litellm_module(callbacks=["existing"])
        env = {
            "OTEL_ENDPOINT": "https://lens.example.com/v1/traces",
            "OTEL_HEADERS": "Authorization=Bearer secret-key",
            "OTEL_SERVICE_NAME": "litellm-contributor-bot",
        }
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            self.assertTrue(configure_litellm_tracing(env=env))
            self.assertEqual(fake.callbacks, ["existing", "otel"])
            self.assertTrue(configure_litellm_tracing(env=env))
            self.assertEqual(fake.callbacks, ["existing", "otel"])

    def test_turns_on_with_otel_exporter_otlp_endpoint(self):
        fake = _fake_litellm_module()
        env = {"OTEL_EXPORTER_OTLP_ENDPOINT": "https://lens.example.com/v1/traces"}
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            self.assertTrue(configure_litellm_tracing(env=env))
        self.assertEqual(fake.callbacks, ["otel"])
        self.assertEqual(
            tracing_endpoint({
                "OTEL_ENDPOINT": "https://fallback.example/v1/traces",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "https://preferred.example/v1/traces",
            }),
            "https://preferred.example/v1/traces",
        )

    def test_main_startup_enables_tracing_only_when_env_set(self):
        from contributor_bot import main as bot_main

        fake_off = _fake_litellm_module()
        with mock.patch.dict(sys.modules, {"litellm": fake_off}):
            with mock.patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "tok"}, clear=True):
                with mock.patch.object(bot_main, "load_dotenv"):
                    with mock.patch.object(bot_main, "send_test_message", return_value="https://discord.com/x"):
                        bot_main.main(["--test-channel-id", "123456789012345678"])
        self.assertEqual(fake_off.callbacks, [])

        fake_on = _fake_litellm_module()
        with mock.patch.dict(sys.modules, {"litellm": fake_on}):
            with mock.patch.dict(
                os.environ,
                {"DISCORD_BOT_TOKEN": "tok", "OTEL_ENDPOINT": "https://lens.example.com/v1/traces"},
                clear=True,
            ):
                with mock.patch.object(bot_main, "load_dotenv"):
                    with mock.patch.object(bot_main, "send_test_message", return_value="https://discord.com/x"):
                        bot_main.main(["--test-channel-id", "123456789012345678"])
        self.assertEqual(fake_on.callbacks, ["otel"])

    def test_complete_litellm_registers_otel_before_completion_and_forwards_metadata(self):
        fake = _fake_litellm_module()
        meta = build_trace_metadata("role_matcher", session_id="role_matcher:octocat", extra={"github_login": "octocat"})
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            with mock.patch.dict(os.environ, {"OTEL_ENDPOINT": "https://lens.example.com/v1/traces"}, clear=True):
                result = complete_litellm("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}], metadata=meta)
        self.assertEqual(result, '{"ok": true}')
        self.assertEqual(len(fake.calls), 1)
        callbacks_at_call, kwargs = fake.calls[0]
        self.assertEqual(callbacks_at_call, ["otel"])
        self.assertEqual(kwargs["metadata"]["generation_name"], "role_matcher")
        self.assertEqual(kwargs["metadata"]["agent_name"], "role_matcher")
        self.assertEqual(kwargs["metadata"]["session_id"], "role_matcher:octocat")
        self.assertEqual(kwargs["metadata"]["requester_metadata"]["github_login"], "octocat")


class TraceMetadataCallSiteTests(unittest.TestCase):
    def test_role_matcher_tags_litellm_call(self):
        fake = _fake_litellm_module()
        profile = GitHubAuthorProfile(login="octocat", name="Mona Octocat")
        member = DiscordMemberInfo(discord_id="123456789012345678", username="octocat", global_name="Mona")
        candidate = CandidatePair(profile=profile, member=member, heuristic_score=0.9, signals=("exact_login",))
        config = types.SimpleNamespace(match_model="anthropic/claude-sonnet-4-5", review_model="anthropic/claude-sonnet-4-5")
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
                _select_with_llm(config, profile, [candidate])
        self.assertEqual(len(fake.calls), 1)
        _, kwargs = fake.calls[0]
        self.assertEqual(kwargs["metadata"]["agent_name"], "role_matcher")
        self.assertEqual(kwargs["metadata"]["generation_name"], "role_matcher")
        self.assertEqual(kwargs["metadata"]["session_id"], "role_matcher:octocat")

    def test_pr_review_sdk_session_sets_pr_review_metadata(self):
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
        )
        ctx = ReviewContext(config=config, store=None, number=4242, thread_id="999")
        with mock.patch("contributor_bot.review_agent._load_sdk", return_value=sdk):
            with mock.patch.dict(os.environ, {}, clear=True):
                _run_sdk_session(ctx, "review pr", config)
        meta = captured["settings"]["metadata"]
        self.assertEqual(meta["agent_name"], "pr_review")
        self.assertEqual(meta["generation_name"], "pr_review")
        self.assertEqual(meta["session_id"], "pr_review:BerriAI/litellm#4242")
        self.assertEqual(meta["pr_number"], "4242")

    def test_weekly_summary_and_discord_qa_tag_metadata(self):
        fake = _fake_litellm_module()
        config = types.SimpleNamespace(
            review_classifier_model="anthropic/claude-haiku-4-5",
            review_model="anthropic/claude-sonnet-4-5",
            discord_token="tok",
        )
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
                summarize_pr_titles(config, [{"pulls": [{"number": 10, "title": "feat: add lens"}]}])
        self.assertEqual(fake.calls[0][1]["metadata"]["agent_name"], "weekly_summary")
        self.assertEqual(fake.calls[0][1]["metadata"]["session_id"], "weekly_summary:1prs")

        fake2 = _fake_litellm_module()
        with mock.patch.dict(sys.modules, {"litellm": fake2}):
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("contributor_bot.questions.post_message"):
                    answer_question(config, "555666777", "how do i run tests?")
        self.assertEqual(fake2.calls[0][1]["metadata"]["agent_name"], "discord_qa")
        self.assertEqual(fake2.calls[0][1]["metadata"]["session_id"], "discord_qa:555666777")

    def test_sdk_metadata_values_are_all_strings(self):
        meta = build_sdk_trace_metadata("pr_review", session_id="s1", extra={"pr_number": 99, "empty": ""})
        self.assertTrue(all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()))
        self.assertEqual(meta["pr_number"], "99")
        self.assertNotIn("empty", meta)


class RenderBlueprintTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
