"""Jev gate, trigger dedup, and the weekly community summary."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from contributor_bot.config import Config
from contributor_bot.routing import decide, obvious_skip, reset_decision_cache
from contributor_bot.store import Store
from contributor_bot.triggers import plan_triggers, parse_trigger_comments, DEFAULT_TRIGGERS
from contributor_bot.weekly import community_groups, render_weekly, week_key, weekly_is_due
from contributor_bot.reviews import BotGate, Gate


def _jev(noul, route, score=0.2, model="jev-1.14.0", tokens=180):
    def transport(_body):
        return {
            "model": model,
            "answers": {
                "should_reply": {"type": "noul", "noul": noul},
                "route": {"type": "choice", "choice": route, "confidence": 0.9, "probabilities": {route: 0.9}},
                "complexity": {"type": "score", "score": score, "confidence": 0.8, "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2}},
            },
            "usage": {"input_tokens": tokens, "output_tokens": 30},
        }
    return transport


class RouterCallTests(unittest.TestCase):
    def test_answer_uses_one_autorouter_completion(self):
        import sys
        import types
        from types import SimpleNamespace
        captured = {}

        class Router:
            def __init__(self, model_list, num_retries=0):
                captured["model_list"] = model_list

            async def acompletion(self, **kwargs):
                captured["call"] = kwargs
                message = SimpleNamespace(content="use the openai key in config.yaml")
                return SimpleNamespace(model="anthropic/claude-haiku-4-5", choices=[SimpleNamespace(message=message)])

        fake = types.ModuleType("litellm")
        fake.Router = Router
        fake.completion_cost = lambda **_kwargs: 0.0012
        config = Config("token", "123", "456", review_model="anthropic/claude-sonnet-4-5")
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            from contributor_bot.routing import complete_auto
            text = complete_auto(
                config,
                [{"role": "user", "content": "how do I add openai?"}],
                session_id="998877",
                tier="SIMPLE",
            )
        self.assertEqual(text, "use the openai key in config.yaml")
        self.assertEqual(captured["call"]["model"], "auto")
        self.assertEqual(captured["call"]["metadata"]["session_id"], "998877")
        self.assertEqual(captured["call"]["metadata"]["decision_tier"], "SIMPLE")
        names = [entry["model_name"] for entry in captured["model_list"]]
        self.assertEqual(names, ["fast", "capable", "auto"])
        auto = captured["model_list"][2]["litellm_params"]
        self.assertEqual(auto["model"], "auto_router/complexity_router")
        self.assertEqual(auto["complexity_router_default_model"], "capable")
        self.assertEqual(auto["complexity_router_config"]["classifier_type"], "custom")


class DecisionTests(unittest.TestCase):
    def tearDown(self):
        reset_decision_cache()

    def test_heuristics_skip_chatter_without_a_call(self):
        self.assertTrue(obvious_skip("lol"))
        self.assertTrue(obvious_skip("ok"))
        called = {"n": 0}

        def transport(_body):
            called["n"] += 1
            return {}

        config = Config("token", "123", "456", typesafe_api_key="ts")
        decision = decide(config, "thanks", transport=transport)
        self.assertFalse(decision.reply)
        self.assertEqual(decision.source, "heuristic")
        self.assertEqual(called["n"], 0)

    def test_threshold_and_maintainer_route(self):
        reset_decision_cache()
        config = Config("token", "123", "456", typesafe_api_key="ts", decision_reply_threshold=0.7)
        answer = decide(config, "how do I set a timeout in the proxy?", channel="help", transport=_jev(0.91, "simple_qa"))
        self.assertTrue(answer.reply)
        self.assertEqual(answer.route, "simple_qa")
        self.assertEqual(answer.tier, "SIMPLE")
        self.assertEqual(answer.model_id, "jev-1.14.0")
        self.assertAlmostEqual(answer.cost_usd, 180 * 4.2e-8)
        reset_decision_cache()
        quiet = decide(config, "should we merge this for the customer?", channel="help", transport=_jev(0.4, "maintainer_only"))
        self.assertFalse(quiet.reply)
        cached = decide(config, "should we merge this for the customer?", channel="help", transport=_jev(0.99, "simple_qa"))
        self.assertFalse(cached.reply)
        self.assertEqual(cached.source, "jev")

    def test_missing_key_uses_haiku_json(self):
        reset_decision_cache()
        config = Config("token", "123", "456", review_classifier_model="anthropic/claude-haiku-4-5")
        with mock.patch("contributor_bot.review_llm.complete_litellm", return_value='{"reply": true, "route": "bug_report", "reason": "traceback"}'):
            decision = decide(config, "the proxy crashes when I set num_retries, here is the traceback and the rest of the log")
        self.assertTrue(decision.reply)
        self.assertEqual(decision.route, "bug_report")
        self.assertEqual(decision.source, "haiku")


class TriggerTests(unittest.TestCase):
    def test_defaults_and_dedup_across_shas(self):
        self.assertEqual(DEFAULT_TRIGGERS, (("cursor[bot]", "bugbot run"), ("greptile-apps[bot]", "@greptileai")))
        self.assertNotIn("veria", " ".join(body for _login, body in DEFAULT_TRIGGERS))
        gate = Gate(False, ("missing",), (
            BotGate("cursor[bot]", False, False, ()),
            BotGate("greptile-apps[bot]", True, True, ()),
            BotGate("veria-ai[bot]", False, False, ()),
        ), "green")
        planned = plan_triggers(gate, DEFAULT_TRIGGERS, set())
        self.assertEqual(planned, [("cursor[bot]", "bugbot run")])
        again = plan_triggers(gate, DEFAULT_TRIGGERS, {"bugbot run"})
        self.assertEqual(again, [])
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        sha = "ab" * 20
        store.save_trigger("BerriAI/litellm", 12, sha, "bugbot run", "2026-11-26T00:00:00+00:00")
        self.assertTrue(store.trigger_posted("BerriAI/litellm", 12, sha, "bugbot run"))
        self.assertFalse(store.trigger_posted("BerriAI/litellm", 12, "cd" * 20, "bugbot run"))
        custom = parse_trigger_comments("bugbot run,@greptileai")
        self.assertEqual(custom[1], ("greptile-apps[bot]", "@greptileai"))


class WeeklyTests(unittest.TestCase):
    def test_groups_skip_maintainers_and_a_week_posts_once(self):
        groups = community_groups([
            {"login": "octocat", "type": "User", "user_id": 5, "number": 12, "title": "add a timeout", "url": "https://github.com/BerriAI/litellm/pull/12"},
            {"login": "ishaan-berri", "type": "User", "user_id": 9, "number": 13, "title": "internal", "url": "https://github.com/BerriAI/litellm/pull/13"},
            {"login": "cursor[bot]", "type": "Bot", "user_id": 1, "number": 14, "title": "bot", "url": "https://github.com/BerriAI/litellm/pull/14"},
        ], ("ishaan-berri",), {5: "111"})
        text = render_weekly(groups, since=datetime(2026, 11, 19, tzinfo=timezone.utc))
        self.assertIn("**octocat** <@111>", text)
        self.assertIn("add a timeout", text)
        self.assertNotIn("ishaan-berri", text)
        self.assertNotIn("cursor[bot]", text)
        from zoneinfo import ZoneInfo
        moment = datetime(2026, 11, 23, 15, 0, tzinfo=timezone.utc)  # Monday 10:00 in Toronto
        local = moment.astimezone(ZoneInfo("America/Toronto"))
        self.assertTrue(weekly_is_due(moment, 0, "10:00", "America/Toronto", None))
        self.assertFalse(weekly_is_due(moment, 0, "10:00", "America/Toronto", week_key(local)))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        self.assertTrue(store.mark_merged_week("BerriAI/litellm", "2026-W48", "2026-11-23T15:00:00+00:00"))
        self.assertFalse(store.mark_merged_week("BerriAI/litellm", "2026-W48", "2026-11-23T15:00:00+00:00"))

    def test_merged_this_week_defers_and_splits_long_summary_without_marking_posted(self):
        import asyncio
        from types import SimpleNamespace
        from contributor_bot.gateway import _dispatch_command, commands_for_guild
        from contributor_bot.interactions import handle_interaction
        from contributor_bot.weekly import weekly_chunks

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        config = Config(
            "token", "123", "456",
            discord_guild_id="333",
            discord_application_id="444",
            merged_weekly_enabled=True,
            oss_contributors_channel_id="",
        )
        self.assertIn("merged-this-week", [cmd["name"] for cmd in commands_for_guild(config)])

        # Build >4500 chars of merged PRs across many contributors to force multi-chunk output.
        items = [
            {
                "number": 1000 + i,
                "title": f"feat(provider-{i}): add streaming tool-call metrics and fallback handling for region {i}",
                "html_url": f"https://github.com/BerriAI/litellm/pull/{1000 + i}",
                "user": {"login": f"contrib-{i:02d}", "type": "User", "id": 500 + i},
            }
            for i in range(45)
        ]

        edits = []
        followups = []
        deferred = []

        class _InlineThread:
            def __init__(self, target=None, args=(), kwargs=None, **_rest):
                self._target = target
                self._args = args
                self._kwargs = kwargs or {}

            def start(self):
                self._target(*self._args, **self._kwargs)

        class _FakeResponse:
            def __init__(self):
                self.done = False

            def is_done(self):
                return self.done

            async def defer(self, *, ephemeral=False):
                self.done = True
                deferred.append(ephemeral)

            async def send_message(self, content, *, ephemeral=False):
                self.done = True
                raise AssertionError(f"expected defer, got immediate message: {content}")

        interaction = SimpleNamespace(
            guild_id=333,
            application_id=444,
            channel_id=123,
            token="tokentokentoken",
            user=SimpleNamespace(id=111),
            data={"name": "merged-this-week", "options": []},
            channel=SimpleNamespace(id=123, type=SimpleNamespace(value=0), parent_id=None),
            guild=None,
            message=None,
            extras={},
            response=_FakeResponse(),
            followup=SimpleNamespace(send=None),
        )

        with mock.patch("contributor_bot.interactions.threading.Thread", _InlineThread), \
                mock.patch("contributor_bot.weekly.github_get_json", return_value={"items": items}), \
                mock.patch("contributor_bot.interactions.edit_original_interaction", side_effect=lambda *a: edits.append(a)), \
                mock.patch("contributor_bot.interactions.post_interaction_followup", side_effect=lambda *a: followups.append(a)):
            asyncio.run(_dispatch_command(None, interaction, config, store))

        self.assertEqual(deferred, [True])
        self.assertEqual(len(edits), 1)
        self.assertGreaterEqual(len(followups), 2)
        all_chunks = [edits[0][2]["content"]] + [call[2]["content"] for call in followups]
        for chunk in all_chunks:
            self.assertLessEqual(len(chunk), 1900)
            # Never split inside a markdown link.
            self.assertEqual(chunk.count("["), chunk.count("]"))
        self.assertTrue(all(call[2].get("flags") == 64 for call in followups))
        # Preview must NOT mark the week posted in SQLite.
        self.assertFalse(store.merged_week_posted("BerriAI/litellm", week_key(datetime.now(timezone.utc))))

        # Direct handle_interaction returns type 5 immediately (defer).
        started = []
        with mock.patch("contributor_bot.interactions.threading.Thread") as thread_cls:
            response = handle_interaction(config, store, {
                "type": 2,
                "guild_id": "333",
                "application_id": "444",
                "channel_id": "123",
                "token": "tokentokentoken",
                "member": {"user": {"id": "111"}},
                "data": {"name": "merged-this-week", "options": []},
            })
            started.append(thread_cls.called)
        self.assertEqual(response, {"type": 5, "data": {"flags": 64}})
        self.assertEqual(started, [True])

        # Huge summary capped at max_chunks gets a clean footer under the limit.
        huge_groups = [
            {
                "login": f"dev-{i:03d}",
                "discord_id": None,
                "pulls": [{"number": i, "title": "fix " + ("x" * 90), "url": f"https://github.com/BerriAI/litellm/pull/{i}"}],
            }
            for i in range(120)
        ]
        capped = weekly_chunks(
            huge_groups,
            since=datetime(2026, 11, 19, tzinfo=timezone.utc),
            limit=1900,
            max_chunks=3,
            show_maintainer_hint=True,
        )
        self.assertEqual(len(capped), 3)
        for chunk in capped:
            self.assertLessEqual(len(chunk), 1900)
        self.assertIn("more contributors", capped[-1])
        self.assertIn("MAINTAINER_GITHUB_LOGINS", capped[-1])

    def test_merged_this_week_handles_github_rate_limit_and_optional_guild_config(self):
        from contributor_bot.api import TransientAPIError
        from contributor_bot.config import get_config
        from contributor_bot.interactions import _finish_merged_preview

        # Even with roles and PR review off, DISCORD_GUILD_ID / APPLICATION_ID are kept for /merged-this-week.
        cfg = get_config({
            "DISCORD_BOT_TOKEN": "token",
            "OSS_CONTRIBUTIONS_CHANNEL_ID": "123",
            "ANNOUNCEMENTS_CHANNEL_ID": "456",
            "DISCORD_GUILD_ID": "333",
            "DISCORD_APPLICATION_ID": "444",
        })
        self.assertEqual(cfg.discord_guild_id, "333")
        self.assertEqual(cfg.discord_application_id, "444")
        self.assertEqual(cfg.oss_contributors_channel_id, "123")

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)

        edits = []
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=TransientAPIError("rate limited")), \
                mock.patch("contributor_bot.interactions.edit_original_interaction", side_effect=lambda *a: edits.append(a)):
            _finish_merged_preview(cfg, store, "tokentokentoken", "444")
        self.assertEqual(len(edits), 1)
        self.assertIn("rate limit", edits[0][2]["content"].lower())
        self.assertIn("GITHUB_TOKEN", edits[0][2]["content"])

        # Rate limit on page 2 returns page 1 rows plus a note instead of failing.
        page1 = {
            "items": [
                {
                    "number": i,
                    "title": f"fix {i}",
                    "html_url": f"https://github.com/BerriAI/litellm/pull/{i}",
                    "user": {"login": "octocat", "type": "User", "id": 5},
                }
                for i in range(1, 101)
            ]
        }
        edits.clear()
        followups = []
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=[page1, TransientAPIError("rate limited")]), \
                mock.patch("contributor_bot.interactions.edit_original_interaction", side_effect=lambda *a: edits.append(a)), \
                mock.patch("contributor_bot.interactions.post_interaction_followup", side_effect=lambda *a: followups.append(a)):
            _finish_merged_preview(cfg, store, "tokentokentoken", "444")
        combined = "\n".join([edits[0][2]["content"]] + [call[2]["content"] for call in followups])
        self.assertIn("**octocat**", combined)
        self.assertIn("rate limit", combined.lower())


class SampleRoutingTests(unittest.TestCase):
    def test_sample_table_uses_the_mock(self):
        from scripts.dry_run_decisions import SAMPLES, table
        reset_decision_cache()
        rows = table(Config("token", "123", "456", typesafe_api_key="ts"))
        self.assertEqual(len(rows), len(SAMPLES))
        self.assertEqual(len(SAMPLES), 20)
        answered = [row for row in rows if row["action"] == "answer"]
        skipped = [row for row in rows if row["action"] == "skip"]
        self.assertGreaterEqual(len(answered), 5)
        self.assertGreaterEqual(len(skipped), 5)
        self.assertTrue(any(row["route"] == "maintainer_only" for row in skipped))
