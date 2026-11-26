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
