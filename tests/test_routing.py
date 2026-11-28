"""Jev gate, trigger dedup, and the weekly community summary."""

import json
import os
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
    def tearDown(self):
        from contributor_bot.weekly import reset_org_members_cache
        reset_org_members_cache()

    def test_groups_filter_org_members_staff_associations_and_bots(self):
        from contributor_bot.weekly import fetch_org_public_members, reset_org_members_cache
        reset_org_members_cache()

        groups = community_groups([
            {"login": "octocat", "type": "User", "user_id": 5, "number": 12, "title": "add a timeout", "url": "https://github.com/BerriAI/litellm/pull/12", "author_association": "CONTRIBUTOR"},
            {"login": "octocat", "type": "User", "user_id": 5, "number": 19, "title": "fix docs", "url": "https://github.com/BerriAI/litellm/pull/19", "author_association": "CONTRIBUTOR"},
            {"login": "newbie", "type": "User", "user_id": 6, "number": 18, "title": "first pr", "url": "https://github.com/BerriAI/litellm/pull/18", "author_association": "FIRST_TIME_CONTRIBUTOR"},
            {"login": "public-org-member", "type": "User", "user_id": 7, "number": 15, "title": "org", "url": "https://github.com/BerriAI/litellm/pull/15", "author_association": "CONTRIBUTOR"},
            {"login": "core-member", "type": "User", "user_id": 8, "number": 16, "title": "member", "url": "https://github.com/BerriAI/litellm/pull/16", "author_association": "MEMBER"},
            {"login": "repo-owner", "type": "User", "user_id": 10, "number": 20, "title": "owner", "url": "https://github.com/BerriAI/litellm/pull/20", "author_association": "OWNER"},
            {"login": "collab", "type": "User", "user_id": 11, "number": 21, "title": "collab", "url": "https://github.com/BerriAI/litellm/pull/21", "author_association": "COLLABORATOR"},
            {"login": "yuneng-berri", "type": "User", "user_id": 12, "number": 22, "title": "berri staff", "url": "https://github.com/BerriAI/litellm/pull/22", "author_association": "CONTRIBUTOR"},
            {"login": "ishaan", "type": "User", "user_id": 9, "number": 13, "title": "internal", "url": "https://github.com/BerriAI/litellm/pull/13", "author_association": "CONTRIBUTOR"},
            {"login": "cursor[bot]", "type": "Bot", "user_id": 1, "number": 14, "title": "bot", "url": "https://github.com/BerriAI/litellm/pull/14", "author_association": "CONTRIBUTOR"},
        ], ("ishaan",), {5: "111"}, org_members={"public-org-member"}, org="BerriAI")
        self.assertEqual([g["login"] for g in groups], ["octocat", "newbie"])

        text = render_weekly(
            groups,
            since=datetime(2026, 11, 19, tzinfo=timezone.utc),
        )
        self.assertTrue(text.startswith("thank you to this week's contributors 🙏\n\n"))
        self.assertIn(
            "<@111> - add a timeout (<https://github.com/BerriAI/litellm/pull/12>), "
            "fix docs (<https://github.com/BerriAI/litellm/pull/19>)",
            text,
        )
        self.assertTrue(text.endswith(
            "\n\nwant to be on next week's list? grab a good first issue: "
            "<https://github.com/BerriAI/litellm/issues>"
        ))
        self.assertNotIn("octocat", text)
        self.assertNotIn("newbie", text)
        for excluded in ("public-org-member", "core-member", "repo-owner", "collab", "yuneng-berri", "ishaan", "cursor[bot]"):
            self.assertNotIn(excluded, text)

        # Public org members lookup caches with TTL.
        with mock.patch("contributor_bot.weekly.github_get_json", return_value=[{"login": "AliceStaff"}, {"login": "BobStaff"}]) as getter:
            first = fetch_org_public_members("BerriAI", None)
            second = fetch_org_public_members("BerriAI", None)
        self.assertEqual(first, frozenset({"alicestaff", "bobstaff"}))
        self.assertEqual(second, first)
        self.assertEqual(getter.call_count, 1)

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

    def test_automatic_matcher_signals_jev_batch_haiku_fallback_and_sqlite_cache(self):
        from types import SimpleNamespace
        from contributor_bot.matcher import (
            extract_discord_handles_from_text,
            extract_discord_ids_from_text,
            normalize_base,
            normalize_strict,
        )
        from contributor_bot.weekly import build_weekly_report, reset_org_members_cache

        reset_org_members_cache()
        self.assertEqual(normalize_strict("Pavel.N_83"), "paveln83")
        self.assertEqual(normalize_base("Pavel.N_83"), "paveln")
        self.assertEqual(extract_discord_handles_from_text("Join discord.gg/litellm | Discord: @izzy_dev"), ["izzy_dev"])
        self.assertEqual(extract_discord_ids_from_text("https://discord.com/users/900000000000000009"), ["900000000000000009"])

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        # No /link-github entries needed at all; optional link_store still works if present.
        store.link_account("100000000000000001", "linkedDev", 101, "2026-11-20T00:00:00+00:00")

        gh_items = [
            {"number": 1, "title": "a", "html_url": "https://github.com/BerriAI/litellm/pull/1", "user": {"login": "linkedDev", "type": "User", "id": 101}, "author_association": "CONTRIBUTOR"},
            {"number": 2, "title": "b", "html_url": "https://github.com/BerriAI/litellm/pull/2", "user": {"login": "paveln83", "type": "User", "id": 102}, "author_association": "CONTRIBUTOR"},
            {"number": 3, "title": "c", "html_url": "https://github.com/BerriAI/litellm/pull/3", "user": {"login": "gh-bio-user", "type": "User", "id": 103}, "author_association": "CONTRIBUTOR"},
            {"number": 4, "title": "d", "html_url": "https://github.com/BerriAI/litellm/pull/4", "user": {"login": "gh-twit-user", "type": "User", "id": 104}, "author_association": "CONTRIBUTOR"},
            {"number": 5, "title": "e", "html_url": "https://github.com/BerriAI/litellm/pull/5", "user": {"login": "gh-name-user", "type": "User", "id": 105}, "author_association": "CONTRIBUTOR"},
            {"number": 6, "title": "f", "html_url": "https://github.com/BerriAI/litellm/pull/6", "user": {"login": "gh-fuzzy-win", "type": "User", "id": 106}, "author_association": "CONTRIBUTOR"},
            {"number": 7, "title": "g", "html_url": "https://github.com/BerriAI/litellm/pull/7", "user": {"login": "gh-fuzzy-low", "type": "User", "id": 107}, "author_association": "CONTRIBUTOR"},
            {"number": 8, "title": "h", "html_url": "https://github.com/BerriAI/litellm/pull/8", "user": {"login": "totally-unknown", "type": "User", "id": 108}, "author_association": "CONTRIBUTOR"},
        ]

        profiles = {
            "paveln83": {"login": "paveln83", "name": "Pavel", "bio": "", "blog": "", "twitter_username": None},
            "gh-bio-user": {"login": "gh-bio-user", "name": "", "bio": "ML eng | Discord: @real_bio_handle", "blog": "", "twitter_username": None},
            "gh-twit-user": {"login": "gh-twit-user", "name": "", "bio": "", "blog": "", "twitter_username": "cooltwitter"},
            "gh-name-user": {"login": "gh-name-user", "name": "Elena Rostova", "bio": "", "blog": "", "twitter_username": None},
            "gh-fuzzy-win": {"login": "gh-fuzzy-win", "name": "Marcus Vance", "bio": "working on proxies", "blog": "", "twitter_username": None},
            "gh-fuzzy-low": {"login": "gh-fuzzy-low", "name": "Jordan Kim", "bio": "", "blog": "", "twitter_username": None},
            "totally-unknown": {"login": "totally-unknown", "name": "Nobody Special", "bio": "", "blog": "", "twitter_username": None},
        }
        commits = {
            6: [{"author": {"login": "gh-fuzzy-win"}, "commit": {"author": {"name": "Marcus Vance", "email": "mvance@example.com"}}}],
            7: [{"author": {"login": "gh-fuzzy-low"}, "commit": {"author": {"name": "Jordan Kim", "email": "jkim@example.com"}}}],
        }
        github_calls = {"users": 0, "commits": 0}

        def fake_gh(url, _token, **_kwargs):
            if "/orgs/" in url:
                return []
            if "/search/issues" in url:
                return {"items": gh_items}
            if "/users/" in url:
                github_calls["users"] += 1
                login = url.rsplit("/users/", 1)[1]
                return profiles.get(login, {"login": login})
            if "/commits" in url:
                github_calls["commits"] += 1
                num = int(url.split("/pulls/")[1].split("/")[0])
                return commits.get(num, [])
            return {}

        guild_members = [
            SimpleNamespace(id=200000000000000002, name="pavel_n", global_name="Pavel N", nick=None, bot=False),
            SimpleNamespace(id=200000000000000003, name="real_bio_handle", global_name="Bio Person", nick=None, bot=False),
            SimpleNamespace(id=200000000000000004, name="someone_else", global_name="CoolTwitter", nick=None, bot=False),
            SimpleNamespace(id=200000000000000005, name="erostova_alt", global_name="Other", nick="Elena Rostova", bot=False),
            SimpleNamespace(id=200000000000000006, name="mvance", global_name="Marcus V", nick=None, bot=False),
            SimpleNamespace(id=200000000000000007, name="jkim", global_name="JK", nick=None, bot=False),
            SimpleNamespace(id=200000000000000099, name="totally-unknown", global_name=None, nick=None, bot=True),
        ]

        # Empty cache triggers guild.fetch_members().
        class _FetchGuild:
            members = []

            async def fetch_members(self, limit=None):
                for item in guild_members:
                    yield item

        llm_requests = []

        def fake_llm(model, messages):
            payload = json.loads(messages[1]["content"])
            llm_requests.append((model, payload))
            gh_login = payload["github_author"]["login"]
            cands = payload["candidates"]
            if gh_login == "gh-fuzzy-win":
                return json.dumps({
                    "discord_id": cands[0]["discord_id"],
                    "confidence": 0.91,
                    "reason": "commit name and email local match mvance",
                    "evidence": ["commit name Marcus Vance matches Marcus V", "email local mvance matches username"],
                })
            if gh_login == "gh-twit-user":
                return json.dumps({
                    "discord_id": cands[0]["discord_id"],
                    "confidence": 0.88,
                    "reason": "twitter handle and profile context match",
                    "evidence": ["twitter handle CoolTwitter matches global_name", "profile bio matches role"],
                })
            if gh_login == "gh-name-user":
                return json.dumps({
                    "discord_id": cands[0]["discord_id"],
                    "confidence": 0.87,
                    "reason": "full name and commit author match",
                    "evidence": ["GitHub name Elena Rostova matches member", "username erostova_alt matches initials"],
                })
            return json.dumps({
                "discord_id": cands[0]["discord_id"],
                "confidence": 0.62,
                "reason": "weak overlap only",
                "evidence": ["initials overlap"],
            })

        config = Config(
            "token", "123", "456",
            discord_guild_id="333",
            discord_application_id="444",
            match_model="anthropic/claude-sonnet-4-5",
            discord_members_intent=True,
        )

        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_gh), \
                self.assertLogs(level="INFO") as logs:
            report = build_weekly_report(
                config,
                store,
                now=datetime(2026, 11, 26, tzinfo=timezone.utc),
                guild=_FetchGuild(),
                llm_complete=fake_llm,
            )

        # List-wise LLM called once per non-deterministic candidate author (twitter/name/fuzzy); >=0.85 with >=2 signals accepted.
        self.assertEqual(len(llm_requests), 4)
        self.assertEqual({m for m, _p in llm_requests}, {"anthropic/claude-sonnet-4-5"})
        self.assertEqual(
            {p["github_author"]["login"] for _m, p in llm_requests},
            {"gh-twit-user", "gh-name-user", "gh-fuzzy-win", "gh-fuzzy-low"},
        )
        for _m, p in llm_requests:
            self.assertIn("none of these", p["options"])

        self.assertEqual(report.matched_count, 6)
        self.assertEqual(report.skipped_count, 2)
        methods = {d.github_login: (d.discord_id, d.method) for d in report.match_details}
        self.assertEqual(methods, {
            "linkedDev": ("100000000000000001", "link_store"),
            "paveln83": ("200000000000000002", "exact_login"),
            "gh-bio-user": ("200000000000000003", "bio_discord_handle"),
            "gh-twit-user": ("200000000000000004", "llm"),
            "gh-name-user": ("200000000000000005", "llm"),
            "gh-fuzzy-win": ("200000000000000006", "llm"),
        })
        body = "\n".join(report.chunks)
        for uid in ("100000000000000001", "200000000000000002", "200000000000000003", "200000000000000004", "200000000000000005", "200000000000000006"):
            self.assertIn(f"<@{uid}>", body)
        self.assertNotIn("200000000000000007", body)
        self.assertNotIn("gh-fuzzy-low", body)
        self.assertNotIn("totally-unknown", body)
        self.assertTrue(any("6 contributors matched in Discord, 2 unmatched skipped" in entry for entry in logs.output))

        # Second run reuses SQLite persisted matches for the 5 auto-matched users AND negative cache for skipped:
        # 0 GitHub profile calls AND 0 LLM calls on rerun!
        self.assertEqual(len(store.all_confirmed_matches()), 5)
        github_calls["users"] = 0
        github_calls["commits"] = 0
        llm_requests.clear()
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_gh):
            report2 = build_weekly_report(
                config,
                store,
                now=datetime(2026, 11, 26, tzinfo=timezone.utc),
                guild=SimpleNamespace(members=guild_members),
                llm_complete=fake_llm,
            )
        self.assertEqual(report2.matched_count, 6)
        self.assertEqual(report2.skipped_count, 2)
        self.assertEqual(github_calls["users"], 0)
        self.assertEqual(len(llm_requests), 0)
        methods2 = {d.github_login: d.method for d in report2.match_details}
        self.assertEqual(methods2["paveln83"], "sqlite:exact_login")
        self.assertEqual(methods2["gh-fuzzy-win"], "sqlite:llm")

        # Fallback to review_model when MATCH_MODEL is unset.
        store2 = Store(str(Path(directory.name) / "fallback.db"))
        self.addCleanup(store2.close)
        cfg_fallback = Config(
            "token", "123", "456",
            discord_guild_id="333",
            discord_application_id="444",
            review_model="anthropic/claude-sonnet-4-5",
            match_model="",
        )
        fallback_calls = []

        def fake_fallback(model, messages):
            payload = json.loads(messages[1]["content"])
            fallback_calls.append((model, payload["github_author"]["login"]))
            cands = payload["candidates"]
            if payload["github_author"]["login"] == "gh-fuzzy-win":
                return json.dumps({
                    "discord_id": cands[0]["discord_id"],
                    "confidence": 0.85,
                    "reason": "two signals match",
                    "evidence": ["commit name matches", "email local matches"],
                })
            return json.dumps({
                "discord_id": None,
                "confidence": 0.2,
                "reason": "none of these",
                "evidence": [],
            })

        with mock.patch.dict("os.environ", {"MATCH_MODEL": ""}, clear=False), \
                mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_gh):
            report_fb = build_weekly_report(
                cfg_fallback,
                store2,
                now=datetime(2026, 11, 26, tzinfo=timezone.utc),
                guild=SimpleNamespace(members=guild_members),
                llm_complete=fake_fallback,
            )
        self.assertEqual(len(fallback_calls), 4)
        self.assertEqual({m for m, _l in fallback_calls}, {"anthropic/claude-sonnet-4-5"})
        fb_methods = {d.github_login: d.method for d in report_fb.match_details}
        self.assertEqual(fb_methods["gh-fuzzy-win"], "llm")
        self.assertNotIn("gh-fuzzy-low", fb_methods)

    def test_ambiguous_collision_goes_to_jev_and_noreply_email_strips_id(self):
        from contributor_bot.matcher import (
            DiscordMemberInfo,
            GitHubAuthorProfile,
            _extract_email_local,
            match_community_contributors,
            score_author_against_members,
        )

        self.assertEqual(_extract_email_local("123456+AliceDev@users.noreply.github.com"), "AliceDev")
        self.assertIsNone(_extract_email_local("noreply@github.com"))

        # Two Discord members share the normalized name "alexsmith" -> collision is NOT strong, sent to Jev.
        members = [
            DiscordMemberInfo("300000000000000001", username="alex_smith", global_name="Alex Smith"),
            DiscordMemberInfo("300000000000000002", username="alexsmith99", global_name="Alex Smith"),
        ]
        profile = GitHubAuthorProfile(login="alexsmith", name="Alex Smith")
        strong, candidates = score_author_against_members(profile, members)
        self.assertIsNone(strong)
        self.assertEqual(len(candidates), 2)

        # If Jev scores both >=0.85 within 0.05 (ambiguous tie), neither is guessed.
        cfg = Config("token", "123", "456", typesafe_api_key="ts-key")
        groups = [{"login": "alexsmith", "user_id": 500, "pulls": [{"number": 1, "title": "x", "url": "https://github.com/BerriAI/litellm/pull/1"}]}]

        def tie_jev(body_bytes):
            payload = json.loads(body_bytes.decode("utf-8"))
            return {"answers": {k: {"type": "noul", "noul": 0.88} for k in payload["questions"]}}

        matched, skipped, _details = match_community_contributors(
            groups,
            cfg,
            None,
            members=members,
            skip_github_enrichment=True,
            jev_transport=tie_jev,
        )
        self.assertEqual(matched, [])
        self.assertEqual(skipped, 1)

    def test_pr_title_rewriting_first_time_badge_and_good_first_issue_label_check(self):
        from contributor_bot.weekly import (
            clean_pr_title,
            good_first_issue_url,
            reset_org_members_cache,
            summarize_pr_titles,
            weekly_chunks,
        )

        reset_org_members_cache()
        self.assertEqual(
            clean_pr_title("fix(gemini): tool call argument parsing (#50020)"),
            "fix gemini tool call argument parsing",
        )
        self.assertEqual(
            clean_pr_title("[LITELLM-42] feat(proxy): Log requests blocked by guardrails"),
            "log requests blocked by guardrails",
        )
        self.assertEqual(
            clean_pr_title("fix(gemini): return an error instead of an empty 200 when a Gemini image request gets no image"),
            "return an error instead of an empty 200",
        )

        # Batched fast-tier summarizer rewrites all PRs in one call and sanitizes output.
        groups = [
            {
                "login": "first_timer",
                "discord_id": "935715497901707324",
                "first_time": True,
                "pulls": [
                    {"number": 50020, "title": "fix(gemini): tool call argument parsing", "url": "https://github.com/BerriAI/litellm/pull/50020"},
                ],
            },
            {
                "login": "regular_dev",
                "discord_id": "409084113006493706",
                "first_time": False,
                "pulls": [
                    {"number": 51014, "title": "feat(guardrails): log requests blocked by guardrails", "url": "https://github.com/BerriAI/litellm/pull/51014"},
                    {"number": 51015, "title": "docs(proxy): clarify timeout config", "url": "https://github.com/BerriAI/litellm/pull/51015"},
                ],
            },
        ]
        calls = []

        def fake_summarizer(model, messages):
            calls.append((model, messages))
            return json.dumps({
                "50020": "Fix Gemini tool call argument parsing.",
                "51014": "log requests blocked by guardrails",
                "51015": "clarify proxy timeout config",
            })

        cfg = Config("token", "123", "456", fast_model="anthropic/claude-haiku-4-5")
        summaries = summarize_pr_titles(cfg, groups, summarizer=fake_summarizer)
        self.assertEqual(len(calls), 1)
        self.assertEqual(summaries[50020], "fix gemini tool call argument parsing")

        chunks = weekly_chunks(
            groups,
            since=datetime(2026, 11, 20, tzinfo=timezone.utc),
            summaries=summaries,
            issues_url=good_first_issue_url("BerriAI/litellm"),
        )
        self.assertEqual(len(chunks), 1)
        expected = (
            "thank you to this week's contributors 🙏\n\n"
            "<@935715497901707324> 🎉 - fix gemini tool call argument parsing (<https://github.com/BerriAI/litellm/pull/50020>)\n"
            "<@409084113006493706> - log requests blocked by guardrails (<https://github.com/BerriAI/litellm/pull/51014>), "
            "clarify proxy timeout config (<https://github.com/BerriAI/litellm/pull/51015>)\n\n"
            "want to be on next week's list? grab a good first issue: "
            "<https://github.com/BerriAI/litellm/issues>"
        )
        self.assertEqual(chunks[0], expected)
        self.assertEqual(good_first_issue_url("BerriAI/litellm"), "https://github.com/BerriAI/litellm/issues")

        # First-time contributor detection via prior merged PR count (when GitHub flips association to CONTRIBUTOR on merge).
        from contributor_bot.weekly import _mark_first_time_contributors
        post_merge_groups = [
            {"login": "brandnew", "discord_id": "111", "first_time": False, "pulls": [{"number": 1, "title": "fix x", "url": "u1"}]},
            {"login": "veteran", "discord_id": "222", "first_time": False, "pulls": [{"number": 2, "title": "fix y", "url": "u2"}]},
        ]

        def fake_author_count(url, _token, **_kwargs):
            return {"total_count": 1} if "author%3Abrandnew" in url else {"total_count": 7}

        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_author_count):
            marked = _mark_first_time_contributors(post_merge_groups, "BerriAI/litellm", None)
        self.assertTrue(marked[0]["first_time"])
        self.assertFalse(marked[1]["first_time"])

    def test_merged_this_week_defers_suppresses_embeds_and_splits_over_2000_chars(self):
        import asyncio
        from types import SimpleNamespace
        from contributor_bot.gateway import _dispatch_command, commands_for_guild
        from contributor_bot.interactions import handle_interaction
        from contributor_bot.weekly import SUPPRESS_EMBEDS_FLAG, chunk_mention_user_ids, post_weekly_if_due, reset_org_members_cache, weekly_chunks

        reset_org_members_cache()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        config = Config(
            "token", "123", "456",
            discord_guild_id="333",
            discord_application_id="444",
            merged_weekly_enabled=True,
            discord_members_intent=True,
        )
        self.assertIn("merged-this-week", [cmd["name"] for cmd in commands_for_guild(config)])

        # 22 matched external contributors (all included, splitting cleanly >1900 chars) + 3 unmatched + org/member.
        items = [
            {
                "number": 1000 + i,
                "title": f"feat(provider-{i}): add streaming provider support for endpoint {i}",
                "html_url": f"https://github.com/BerriAI/litellm/pull/{1000 + i}",
                "user": {"login": f"contrib-{i:02d}", "type": "User", "id": 500 + i},
                "author_association": "CONTRIBUTOR" if i % 2 == 0 else "FIRST_TIME_CONTRIBUTOR",
            }
            for i in range(22)
        ] + [
            {
                "number": 1500 + i,
                "title": f"unmatched pr {i}",
                "html_url": f"https://github.com/BerriAI/litellm/pull/{1500 + i}",
                "user": {"login": f"unmatched-{i:02d}", "type": "User", "id": 700 + i},
                "author_association": "CONTRIBUTOR",
            }
            for i in range(3)
        ] + [
            {
                "number": 2000,
                "title": "staff pr",
                "html_url": "https://github.com/BerriAI/litellm/pull/2000",
                "user": {"login": "staff-org", "type": "User", "id": 900},
                "author_association": "CONTRIBUTOR",
            },
            {
                "number": 2001,
                "title": "member pr",
                "html_url": "https://github.com/BerriAI/litellm/pull/2001",
                "user": {"login": "core-dev", "type": "User", "id": 901},
                "author_association": "MEMBER",
            },
        ]

        fake_guild = SimpleNamespace(
            members=[
                SimpleNamespace(
                    id=100000000000000000 + i,
                    name=f"contrib-{i:02d}",
                    global_name=None,
                    display_name=f"contrib-{i:02d}",
                    nick=None,
                    bot=False,
                )
                for i in range(22)
            ]
        )

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
            permissions=SimpleNamespace(value=1 << 13),  # manage_messages (moderator)
            data={"name": "merged-this-week", "options": []},
            channel=SimpleNamespace(id=123, type=SimpleNamespace(value=0), parent_id=None),
            guild=fake_guild,
            message=None,
            extras={},
            response=_FakeResponse(),
            followup=SimpleNamespace(send=None),
        )

        def fake_github(url, _token, **_kwargs):
            if "/orgs/BerriAI/public_members" in url:
                return [{"login": "staff-org"}]
            if "/labels/" in url:
                return {"name": "good first issue"}
            if "/users/" in url:
                return {"login": url.rsplit("/users/", 1)[1]}
            if "/commits" in url:
                return []
            return {"items": items}

        with mock.patch("contributor_bot.interactions.threading", SimpleNamespace(Thread=_InlineThread)), \
                mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_github), \
                mock.patch("contributor_bot.interactions.edit_original_interaction", side_effect=lambda *a: edits.append(a)), \
                mock.patch("contributor_bot.interactions.post_interaction_followup", side_effect=lambda *a: followups.append(a)):
            asyncio.run(_dispatch_command(None, interaction, config, store))

        self.assertEqual(deferred, [True])
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0][2].get("flags"), 64 | SUPPRESS_EMBEDS_FLAG)
        all_payloads = [edits[0][2]] + [call[2] for call in followups]
        all_chunks = [p["content"] for p in all_payloads]
        self.assertGreaterEqual(len(all_chunks), 2)
        for payload in all_payloads:
            chunk = payload["content"]
            self.assertLessEqual(len(chunk), 1900)
            self.assertEqual(payload.get("flags"), 64 | SUPPRESS_EMBEDS_FLAG)
            self.assertEqual(payload.get("allowed_mentions"), {"parse": [], "users": chunk_mention_user_ids(chunk)})
        combined = "\n".join(all_chunks)
        self.assertTrue(all_chunks[0].startswith("thank you to this week's contributors 🙏\n\n"))
        self.assertIn("(<https://github.com/BerriAI/litellm/pull/", combined)
        self.assertNotIn("contrib-", combined)
        self.assertNotIn("unmatched-", combined)
        self.assertNotIn("staff-org", combined)
        self.assertNotIn("core-dev", combined)
        contributor_lines = [line for line in combined.splitlines() if line.startswith("<@")]
        self.assertEqual(len(contributor_lines), 22)
        # Half of the 22 contributors are FIRST_TIME_CONTRIBUTOR and get the 🎉 badge.
        self.assertEqual(sum(1 for line in contributor_lines if " 🎉 - " in line), 11)
        self.assertIn(
            "want to be on next week's list? grab a good first issue: <https://github.com/BerriAI/litellm/issues>",
            all_chunks[-1],
        )
        # Preview must NOT mark the week posted in SQLite.
        self.assertFalse(store.merged_week_posted("BerriAI/litellm", week_key(datetime.now(timezone.utc))))

        # Scheduled weekly post fires Friday 10:00 AM America/Toronto to #announcements ("456"),
        # using the exact same format, SUPPRESS_EMBEDS flag, and allowed_mentions.
        posted = []
        monday_moment = datetime(2026, 11, 23, 15, 0, tzinfo=timezone.utc)
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_github):
            self.assertIsNone(post_weekly_if_due(
                config,
                store,
                monday_moment,
                post=lambda ch, _tok, msg: posted.append((ch, msg)),
                guild=fake_guild,
            ))
        self.assertEqual(posted, [])

        friday_moment = datetime(2026, 11, 27, 15, 0, tzinfo=timezone.utc)  # Friday 10:00 AM EST
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_github):
            post_weekly_if_due(
                config,
                store,
                friday_moment,
                post=lambda ch, _tok, msg: posted.append((ch, msg)),
                guild=fake_guild,
            )
        self.assertEqual([ch for ch, _msg in posted], ["456"] * len(all_chunks))
        self.assertEqual([msg["content"] for _ch, msg in posted], all_chunks)
        for _ch, msg in posted:
            self.assertEqual(msg.get("flags"), SUPPRESS_EMBEDS_FLAG)
            self.assertEqual(msg.get("allowed_mentions"), {"parse": [], "users": chunk_mention_user_ids(msg["content"])})

        # Direct handle_interaction returns type 5 immediately (defer) for moderators.
        started = []
        with mock.patch("contributor_bot.interactions.threading.Thread") as thread_cls:
            response = handle_interaction(config, store, {
                "type": 2,
                "guild_id": "333",
                "application_id": "444",
                "channel_id": "123",
                "token": "tokentokentoken",
                "member": {"user": {"id": "111"}, "permissions": str(1 << 13)},
                "data": {"name": "merged-this-week", "options": []},
            })
            started.append(thread_cls.called)
        self.assertEqual(response, {"type": 5, "data": {"flags": 64}})
        self.assertEqual(started, [True])

        # Tight limit splits cleanly on line boundaries with no contributor dropped.
        huge_groups = [
            {
                "login": f"dev-{i:03d}",
                "discord_id": str(200000000000000000 + i),
                "first_time": False,
                "pulls": [{"number": i, "title": "fix timeout handling", "url": f"https://github.com/BerriAI/litellm/pull/{i}"}],
            }
            for i in range(12)
        ]
        split = weekly_chunks(
            huge_groups,
            since=datetime(2026, 11, 19, tzinfo=timezone.utc),
            limit=500,
        )
        self.assertGreater(len(split), 1)
        for chunk in split:
            self.assertLessEqual(len(chunk), 500)
        self.assertEqual(sum(1 for line in "\n".join(split).splitlines() if line.startswith("<@")), 12)
        self.assertIn("want to be on next week's list?", split[-1])

    def test_merged_this_week_handles_github_rate_limit_and_optional_guild_config(self):
        from contributor_bot.api import TransientAPIError
        from contributor_bot.config import get_config
        from contributor_bot.interactions import _finish_merged_preview
        from contributor_bot.weekly import reset_org_members_cache

        reset_org_members_cache()
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
        self.assertFalse(hasattr(cfg, "oss_contributors_channel_id"))
        self.assertEqual(cfg.announcements_channel_id, "456")
        self.assertEqual(cfg.merged_weekday, 4)
        self.assertEqual(cfg.merged_time, "10:00")
        self.assertEqual(cfg.merged_timezone, "America/Toronto")

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        store.link_account("777777777777777777", "octocat", 5, "2026-11-20T00:00:00+00:00")

        def fail_search(url, _token, **_kwargs):
            if "/orgs/" in url:
                return []
            raise TransientAPIError("rate limited")

        edits = []
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fail_search), \
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
                    "author_association": "CONTRIBUTOR",
                }
                for i in range(1, 101)
            ]
        }
        edits.clear()
        followups = []
        search_calls = {"n": 0}

        def page2_rate_limit(url, _token, **_kwargs):
            if "/orgs/" in url:
                return []
            search_calls["n"] += 1
            if search_calls["n"] == 1:
                return page1
            raise TransientAPIError("rate limited")

        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=page2_rate_limit), \
                mock.patch("contributor_bot.interactions.edit_original_interaction", side_effect=lambda *a: edits.append(a)), \
                mock.patch("contributor_bot.interactions.post_interaction_followup", side_effect=lambda *a: followups.append(a)):
            _finish_merged_preview(cfg, store, "tokentokentoken", "444")
        combined = "\n".join([edits[0][2]["content"]] + [call[2]["content"] for call in followups])
        self.assertIn("<@777777777777777777>", combined)
        self.assertNotIn("@octocat", combined)
        self.assertEqual(edits[0][2].get("allowed_mentions"), {"parse": [], "users": ["777777777777777777"]})
        self.assertIn("rate limit", combined.lower())

    def test_matcher_no_jev_key_deterministic_fallback_and_display_name(self):
        from contributor_bot.matcher import (
            DiscordMemberInfo,
            match_community_contributors_with_diagnostics,
        )

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        cfg = Config(
            "token", "123", "456",
            discord_guild_id="333",
            discord_application_id="444",
            typesafe_api_key=None,
        )
        members = [
            DiscordMemberInfo("111111111111111111", username="octocat"),
            DiscordMemberInfo("222222222222222222", username="randomhandle", display_name="Jane Doe"),
            DiscordMemberInfo("333333333333333333", username="samsmith"),
            DiscordMemberInfo("444444444444444444", username="alexmuller"),
        ]
        groups = [
            {"login": "octocat", "user_id": 1, "first_time": False, "pulls": [{"number": 10, "title": "fix a", "url": "https://github.com/BerriAI/litellm/pull/10"}]},
            {"login": "gh-jane", "user_id": 2, "first_time": True, "pulls": [{"number": 11, "title": "fix b", "url": "https://github.com/BerriAI/litellm/pull/11"}]},
            {"login": "gh-sam-99", "user_id": 3, "first_time": False, "pulls": [{"number": 12, "title": "fix c", "url": "https://github.com/BerriAI/litellm/pull/12"}]},
            {"login": "alexmullr", "user_id": 4, "first_time": False, "pulls": [{"number": 13, "title": "fix d", "url": "https://github.com/BerriAI/litellm/pull/13"}]},
            {"login": "totalstranger", "user_id": 5, "first_time": False, "pulls": [{"number": 14, "title": "fix e", "url": "https://github.com/BerriAI/litellm/pull/14"}]},
        ]

        def fake_github(url, _token, **_kwargs):
            if url.endswith("/users/gh-jane"):
                return {"login": "gh-jane", "name": "Jane Doe"}
            if url.endswith("/users/gh-sam-99"):
                return {"login": "gh-sam-99", "name": ""}
            if url.endswith("/users/octocat"):
                return {"login": "octocat", "name": "Octo Cat"}
            if "/pulls/12/commits" in url:
                return [{"author": {"login": "gh-sam-99"}, "commit": {"author": {"name": "Sam Smith", "email": "samsmith@example.com"}}}]
            if url.endswith("/users/alexmullr"):
                return {"login": "alexmullr", "name": ""}
            if url.endswith("/users/totalstranger"):
                return {"login": "totalstranger", "name": "No Match"}
            return []

        cleared_env = {
            key: ""
            for key in (
                "TYPESAFE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                "OPENROUTER_API_KEY", "LITELLM_API_KEY", "GEMINI_API_KEY",
                "AZURE_API_KEY", "AWS_ACCESS_KEY_ID", "VERTEXAI_PROJECT",
            )
        }
        with mock.patch.dict(os.environ, cleared_env, clear=False):
            matched, skipped, _details, diag = match_community_contributors_with_diagnostics(
                groups,
                cfg,
                store,
                members=members,
                github_getter=fake_github,
            )

        by_login = {g["login"]: g for g in matched}
        self.assertEqual(by_login["octocat"]["discord_id"], "111111111111111111")
        self.assertEqual(by_login["octocat"]["match_method"], "exact_login")
        # Without LLM credentials, non-exact-identity candidates (exact_name, commit_name, fuzzy) are NOT guessed!
        self.assertNotIn("gh-jane", by_login)
        self.assertNotIn("gh-sam-99", by_login)
        self.assertNotIn("alexmullr", by_login)
        self.assertNotIn("totalstranger", by_login)
        self.assertEqual(skipped, 4)
        self.assertEqual(diag.contributor_count, 5)
        self.assertEqual(diag.member_count, 4)
        self.assertEqual(diag.matched_count, 1)
        self.assertEqual(diag.llm_source, "unavailable")
        skip_map = dict(diag.skip_reasons)
        self.assertEqual(skip_map.get("llm_unavailable"), 2)
        self.assertEqual(skip_map.get("no_signal_match"), 2)

    def test_matcher_empty_and_partial_member_cache_fallbacks(self):
        from types import SimpleNamespace
        from contributor_bot.matcher import (
            load_guild_members_with_source,
            match_community_contributors_with_diagnostics,
            reset_guild_members_cache,
        )

        reset_guild_members_cache()

        # 1) Partial cache (2 cached out of 5000, chunked=False) triggers guild.chunk(cache=True).
        chunk_called = {"n": 0}
        partial_guild = SimpleNamespace(
            id=333,
            member_count=5000,
            chunked=False,
            members=[
                SimpleNamespace(id=100000000000000001, name="someone", global_name=None, nick=None, display_name="someone", bot=False),
                SimpleNamespace(id=100000000000000002, name="other", global_name=None, nick=None, display_name="other", bot=False),
            ],
        )

        async def fake_chunk(cache=True):
            chunk_called["n"] += 1
            partial_guild.members.append(
                SimpleNamespace(id=100000000000000003, name="targetdev", global_name=None, nick=None, display_name="targetdev", bot=False)
            )
            partial_guild.chunked = True

        partial_guild.chunk = fake_chunk
        loaded, source = load_guild_members_with_source(partial_guild)
        self.assertEqual(chunk_called["n"], 1)
        self.assertEqual(source, "guild_chunk")
        self.assertIn("targetdev", [m.username for m in loaded])

        # 2) Empty cache + fetch_members raising ClientException (intents.members=False on Render)
        #    falls back cleanly to REST GET /guilds/{id}/members!
        def fail_fetch_members(limit=None):
            raise RuntimeError("Intents.members must be enabled to use this.")

        no_intent_guild = SimpleNamespace(
            id=333,
            member_count=8000,
            chunked=False,
            members=[],
            fetch_members=fail_fetch_members,
        )
        rest_payload = [
            {"user": {"id": "555555555555555555", "username": "alice_dev", "global_name": "Alice Dev", "bot": False}, "nick": None},
            {"user": {"id": "555555555555555556", "username": "helperbot", "bot": True}, "nick": None},
        ]
        loaded_rest, source_rest = load_guild_members_with_source(
            no_intent_guild,
            guild_id="333",
            discord_token="tok",
            rest_fetcher=lambda _gid, _tok: rest_payload,
        )
        self.assertEqual(source_rest, "rest_members")
        self.assertEqual([m.username for m in loaded_rest], ["alice_dev"])

        # 3) guild=None (HTTP interaction) also uses REST /members.
        loaded_http, source_http = load_guild_members_with_source(
            None,
            guild_id="333",
            discord_token="tok",
            rest_fetcher=lambda _gid, _tok: rest_payload,
        )
        self.assertEqual(source_http, "rest_members")
        self.assertEqual(len(loaded_http), 1)

        # 4) When REST /members returns 403, falls back to REST /members/search per contributor.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        cfg = Config("token", "123", "456", discord_guild_id="333", discord_application_id="444")

        def rest_403(_gid, _tok):
            raise RuntimeError("discord.com returned HTTP 403")

        def fake_search(_gid, query, _tok, limit=10):
            if query.lower() == "bobbuilder":
                return [{"user": {"id": "666666666666666666", "username": "bobbuilder", "bot": False}}]
            return []

        matched, skipped, _details, diag = match_community_contributors_with_diagnostics(
            [{"login": "bobbuilder", "user_id": 99, "first_time": False, "pulls": [{"number": 7, "title": "fix x", "url": "https://github.com/BerriAI/litellm/pull/7"}]}],
            cfg,
            store,
            guild=no_intent_guild,
            rest_fetcher=rest_403,
            rest_searcher=fake_search,
            skip_github_enrichment=True,
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual(skipped, 0)
        self.assertEqual(matched[0]["discord_id"], "666666666666666666")
        self.assertEqual(diag.members_source, "rest_search")

    def test_matcher_sqlite_ttl_and_no_negative_caching(self):
        from datetime import timedelta
        from contributor_bot.matcher import (
            DiscordMemberInfo,
            MATCH_CACHE_TTL_DAYS,
            match_community_contributors_with_diagnostics,
        )

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        cfg = Config("token", "123", "456", discord_guild_id="333", discord_application_id="444")
        t0 = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
        groups = [{"login": "cacheddev", "user_id": 42, "first_time": False, "pulls": [{"number": 1, "title": "fix", "url": "https://github.com/BerriAI/litellm/pull/1"}]}]

        # Run 1: empty member list -> 0 matches and NOTHING saved as a negative cache entry.
        matched0, skipped0, _d0, diag0 = match_community_contributors_with_diagnostics(
            groups, cfg, store, members=[], now=t0, skip_github_enrichment=True,
        )
        self.assertEqual(matched0, [])
        self.assertEqual(skipped0, 1)
        self.assertEqual(dict(diag0.skip_reasons), {"no_guild_members": 1})
        self.assertIsNone(store.get_confirmed_match("cacheddev"))

        # Run 2: members available -> matches and persists to SQLite.
        members = [DiscordMemberInfo("888888888888888888", username="cacheddev")]
        matched1, _s1, _d1, _diag1 = match_community_contributors_with_diagnostics(
            groups, cfg, store, members=members, now=t0, skip_github_enrichment=True,
        )
        self.assertEqual(matched1[0]["match_method"], "exact_login")
        self.assertIsNotNone(store.get_confirmed_match("cacheddev"))

        # Run 3: within TTL -> reuses SQLite cache.
        matched2, _s2, _d2, _diag2 = match_community_contributors_with_diagnostics(
            groups, cfg, store, members=members, now=t0 + timedelta(days=5), skip_github_enrichment=True,
        )
        self.assertEqual(matched2[0]["match_method"], "sqlite:exact_login")

        # Run 4: past TTL -> stale SQLite cache expires and recomputes fresh match.
        new_members = [DiscordMemberInfo("999999999999999999", username="cacheddev")]
        matched3, _s3, _d3, _diag3 = match_community_contributors_with_diagnostics(
            groups,
            cfg,
            store,
            members=new_members,
            now=t0 + timedelta(days=MATCH_CACHE_TTL_DAYS + 1),
            skip_github_enrichment=True,
        )
        self.assertEqual(matched3[0]["discord_id"], "999999999999999999")
        self.assertEqual(matched3[0]["match_method"], "exact_login")

    def test_merged_this_week_ephemeral_zero_match_diagnostic_and_post_format(self):
        from contributor_bot.interactions import _finish_merged_preview
        from contributor_bot.matcher import reset_guild_members_cache
        from contributor_bot.weekly import reset_org_members_cache

        reset_org_members_cache()
        reset_guild_members_cache()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        cfg = Config("token", "123", "456", discord_guild_id="333", discord_application_id="444")

        search_payload = {
            "items": [
                {
                    "number": 501,
                    "title": "feat: add gemini cache",
                    "html_url": "https://github.com/BerriAI/litellm/pull/501",
                    "user": {"login": "alice-dev", "type": "User", "id": 10},
                    "author_association": "FIRST_TIME_CONTRIBUTOR",
                },
                {
                    "number": 502,
                    "title": "fix: handle timeout",
                    "html_url": "https://github.com/BerriAI/litellm/pull/502",
                    "user": {"login": "bob-dev", "type": "User", "id": 11},
                    "author_association": "CONTRIBUTOR",
                },
            ]
        }

        def fake_github(url, _token, **_kwargs):
            if "/orgs/" in url:
                return []
            if "/users/" in url:
                return {"login": url.rsplit("/users/", 1)[1]}
            if "/search/issues" in url:
                return search_payload
            return []

        # Case 1: 0 members available -> ephemeral reply includes diagnostic line.
        edits = []
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_github), \
                mock.patch("contributor_bot.matcher.fetch_guild_members", side_effect=RuntimeError("HTTP 403")), \
                mock.patch("contributor_bot.matcher.search_guild_members", return_value=[]), \
                mock.patch("contributor_bot.interactions.edit_original_interaction", side_effect=lambda *a: edits.append(a)):
            _finish_merged_preview(cfg, store, "tokentokentoken", "444", guild=None)

        self.assertEqual(len(edits), 1)
        zero_text = edits[0][2]["content"]
        self.assertIn("no discord-matched community pull requests merged since", zero_text)
        self.assertIn("checked 2 contributors against 0 members", zero_text)
        self.assertIn("no_guild_members=2", zero_text)

        # Case 2: REST /members succeeds -> exact hand-made weekly post format, no diagnostic clutter.
        edits.clear()
        rest_members = [
            {"user": {"id": "111111111111111111", "username": "alice_dev", "bot": False}},
            {"user": {"id": "222222222222222222", "username": "bob_dev", "bot": False}},
        ]
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_github), \
                mock.patch("contributor_bot.matcher.fetch_guild_members", return_value=rest_members), \
                mock.patch("contributor_bot.interactions.edit_original_interaction", side_effect=lambda *a: edits.append(a)):
            _finish_merged_preview(cfg, store, "tokentokentoken", "444", guild=None)

        self.assertEqual(len(edits), 1)
        post_text = edits[0][2]["content"]
        self.assertTrue(post_text.startswith("thank you to this week's contributors 🙏\n\n"))
        self.assertIn("<@111111111111111111> 🎉 - add gemini cache (<https://github.com/BerriAI/litellm/pull/501>)", post_text)
        self.assertIn("<@222222222222222222> - handle timeout (<https://github.com/BerriAI/litellm/pull/502>)", post_text)
        self.assertIn("want to be on next week's list? grab a good first issue: <https://github.com/BerriAI/litellm/issues>", post_text)
        self.assertNotIn("checked 2 contributors", post_text)

    def test_friday_schedule_restart_resilience_channel_resolution_and_lookback(self):
        from types import SimpleNamespace
        from contributor_bot.weekly import (
            build_weekly_report,
            post_weekly_if_due,
            reset_org_members_cache,
            resolve_weekly_channel_id,
            weekly_lookback_since,
        )

        reset_org_members_cache()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        db_path = str(Path(directory.name) / "roles.db")
        store = Store(db_path)
        self.addCleanup(store.close)

        # 1) Channel resolution: ANNOUNCEMENTS_CHANNEL_ID standalone, else #announcements lookup.
        cfg_ann = Config("tok", "123", "456", discord_guild_id="333")
        self.assertEqual(resolve_weekly_channel_id(cfg_ann), "456")
        # Legacy OSS_CONTRIBUTORS_CHANNEL_ID env var is ignored and never overrides ANNOUNCEMENTS_CHANNEL_ID.
        from contributor_bot.config import get_config
        cfg_ignored = get_config({
            "DISCORD_BOT_TOKEN": "tok",
            "OSS_CONTRIBUTIONS_CHANNEL_ID": "123",
            "ANNOUNCEMENTS_CHANNEL_ID": "456",
            "OSS_CONTRIBUTORS_CHANNEL_ID": "789",
        })
        self.assertEqual(resolve_weekly_channel_id(cfg_ignored), "456")

        # ANNOUNCEMENTS_CHANNEL_ID unset -> resolves #announcements by name from guild cache.
        cfg_lookup = Config("tok", "123", "", discord_guild_id="333")
        guild_with_channels = SimpleNamespace(
            id=333,
            text_channels=[
                SimpleNamespace(id=9001, name="general", type=0),
                SimpleNamespace(id=9002, name="announcements", type=5),
            ],
            members=[
                SimpleNamespace(id=111111111111111111, name="alice-dev", global_name=None, nick=None, display_name="alice-dev", bot=False),
            ],
            chunked=True,
        )
        self.assertEqual(resolve_weekly_channel_id(cfg_lookup, guild=guild_with_channels), "9002")

        # Guild is None -> falls back to Discord REST /guilds/{id}/channels to find #announcements.
        rest_channels = [
            {"id": "9003", "name": "announcements", "type": 2},  # voice channel ignored
            {"id": "9004", "name": "Announcements", "type": 0},
        ]
        self.assertEqual(
            resolve_weekly_channel_id(cfg_lookup, guild=None, channels_fetcher=lambda _gid, _tok: rest_channels),
            "9004",
        )

        # 2) Friday-only 10:00 AM America/Toronto (EST = UTC-5 in Nov) + Render restart resilience.
        queries_seen = []

        def fake_github(url, _token, **_kwargs):
            if "/orgs/" in url:
                return []
            if "/users/" in url:
                return {"login": url.rsplit("/users/", 1)[1]}
            if "/search/issues" in url:
                queries_seen.append(url)
                return {
                    "items": [
                        {
                            "number": 801,
                            "title": "feat: add anthropic batch helper",
                            "html_url": "https://github.com/BerriAI/litellm/pull/801",
                            "user": {"login": "alice-dev", "type": "User", "id": 10},
                            "author_association": "FIRST_TIME_CONTRIBUTOR",
                        }
                    ]
                }
            return []

        posted = []
        send = lambda ch, _tok, msg: posted.append((ch, msg))

        # Thursday 10:00 AM Toronto -> must NOT fire.
        thu = datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc)
        # Friday 09:59 AM Toronto (e.g. Render restarted before 10am) -> must NOT fire yet.
        fri_before = datetime(2026, 11, 27, 14, 59, tzinfo=timezone.utc)
        # Friday 10:07 AM Toronto (e.g. Render restarted at 10:02am after missing 10:00 sharp) -> MUST fire!
        fri_after_restart = datetime(2026, 11, 27, 15, 7, tzinfo=timezone.utc)
        # Friday 10:15 AM Toronto (another Render restart after posting) -> MUST NOT double-post!
        fri_later = datetime(2026, 11, 27, 15, 15, tzinfo=timezone.utc)
        # Saturday 10:00 AM Toronto -> must NOT fire.
        sat = datetime(2026, 11, 28, 15, 0, tzinfo=timezone.utc)

        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_github):
            self.assertIsNone(post_weekly_if_due(cfg_lookup, store, thu, post=send, guild=guild_with_channels))
            self.assertIsNone(post_weekly_if_due(cfg_lookup, store, fri_before, post=send, guild=guild_with_channels))
            self.assertEqual(posted, [])

            result = post_weekly_if_due(cfg_lookup, store, fri_after_restart, post=send, guild=guild_with_channels)
            self.assertIsNotNone(result)
            self.assertEqual(len(posted), 1)
            self.assertEqual(posted[0][0], "9002")  # resolved #announcements channel!
            self.assertIn("thank you to this week's contributors 🙏", posted[0][1]["content"])
            self.assertIn("<@111111111111111111> 🎉 - add anthropic batch helper", posted[0][1]["content"])

            # Simulate a full Render process restart with a fresh Store reading the same SQLite DB.
            restarted_store = Store(db_path)
            self.addCleanup(restarted_store.close)
            self.assertIsNone(post_weekly_if_due(cfg_lookup, restarted_store, fri_later, post=send, guild=guild_with_channels))
            self.assertIsNone(post_weekly_if_due(cfg_lookup, restarted_store, sat, post=send, guild=guild_with_channels))
            self.assertEqual(len(posted), 1)

            # 3) Next Friday's lookback window starts at the exact previous Friday post timestamp!
            next_friday = datetime(2026, 12, 4, 15, 0, tzinfo=timezone.utc)
            since = weekly_lookback_since(restarted_store, "BerriAI/litellm", next_friday)
            self.assertEqual(since, fri_after_restart)
            queries_seen.clear()
            report_next = build_weekly_report(cfg_lookup, restarted_store, next_friday, guild=guild_with_channels)
            self.assertEqual(report_next.since, fri_after_restart)
            self.assertTrue(any("merged%3A2026-11-27..2026-11-27" in q for q in queries_seen))

    def test_dispatch_command_defers_first_and_handles_10062_gracefully(self):
        import asyncio
        import threading
        from types import SimpleNamespace
        from contributor_bot.gateway import _dispatch_command

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        config = Config("tok", "123", "456", discord_guild_id="333", discord_application_id="444")

        # 1) Defer must be the very first await, before handle_interaction, and handle_interaction runs off-loop.
        events = []
        main_thread_id = threading.get_ident()
        handler_thread_ids = []
        followups = []

        class _RecordingResponse:
            def __init__(self):
                self.done = False

            def is_done(self):
                return self.done

            async def defer(self, *, ephemeral=False):
                events.append(("defer", ephemeral))
                self.done = True

            async def send_message(self, content, *, ephemeral=False):
                events.append(("send_message", content))
                self.done = True

        async def record_followup(content, *, ephemeral=False):
            events.append(("followup", content, ephemeral))
            followups.append((content, ephemeral))

        def fake_handle(_cfg, _store, _payload):
            events.append(("handle_interaction",))
            handler_thread_ids.append(threading.get_ident())
            return {"type": 5, "data": {"flags": 64}}

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
            response=_RecordingResponse(),
            followup=SimpleNamespace(send=record_followup),
        )
        with mock.patch("contributor_bot.gateway.handle_interaction", side_effect=fake_handle):
            asyncio.run(_dispatch_command(None, interaction, config, store))

        self.assertEqual(events, [("defer", True), ("handle_interaction",)])
        self.assertEqual(len(handler_thread_ids), 1)
        self.assertNotEqual(handler_thread_ids[0], main_thread_id)

        # 2) Discord 10062 Unknown interaction on defer: logs once, no second send_message, no handler run.
        class _ExpiredError(RuntimeError):
            code = 10062

        send_calls = []

        class _ExpiredResponse:
            def is_done(self):
                return False

            async def defer(self, *, ephemeral=False):
                raise _ExpiredError("404 Not Found (error code: 10062): Unknown interaction")

            async def send_message(self, content, *, ephemeral=False):
                send_calls.append(content)

        expired_interaction = SimpleNamespace(
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
            response=_ExpiredResponse(),
            followup=SimpleNamespace(send=record_followup),
        )
        handler_mock = mock.Mock()
        with mock.patch("contributor_bot.gateway.handle_interaction", handler_mock), \
                self.assertLogs(level="WARNING") as logs:
            asyncio.run(_dispatch_command(None, expired_interaction, config, store))
        self.assertEqual(send_calls, [])
        self.assertEqual(followups, [])
        handler_mock.assert_not_called()
        self.assertEqual(sum("10062" in entry for entry in logs.output), 1)

    def test_github_token_fallback_403_retry_and_single_weekly_scheduler(self):
        import io
        import time
        from email.message import Message
        from urllib.error import HTTPError
        from contributor_bot import api
        from contributor_bot.config import get_config
        from contributor_bot.main import _run_loop
        from contributor_bot.weekly import post_weekly_if_due, reset_org_members_cache

        api.reset_github_rate_limit_state()
        reset_org_members_cache()

        # 1) GITHUB_ACTOR_TOKEN is used for GitHub reads when GITHUB_TOKEN is unset; GITHUB_TOKEN wins if both set.
        base_env = {
            "DISCORD_BOT_TOKEN": "token",
            "OSS_CONTRIBUTIONS_CHANNEL_ID": "123",
            "ANNOUNCEMENTS_CHANNEL_ID": "456",
        }
        cfg_actor = get_config(base_env | {"GITHUB_ACTOR_TOKEN": "ghp_actor_only"})
        self.assertEqual(cfg_actor.github_token, "ghp_actor_only")
        self.assertEqual(cfg_actor.effective_github_token, "ghp_actor_only")
        cfg_both = get_config(base_env | {"GITHUB_TOKEN": "ghp_primary", "GITHUB_ACTOR_TOKEN": "ghp_actor"})
        self.assertEqual(cfg_both.github_token, "ghp_primary")
        self.assertEqual(cfg_both.github_actor_token, "ghp_actor")

        # Verify Authorization: Bearer header is sent on GitHub reads and 403 rate-limit retries after reset.
        seen_auth = []
        sleeps = []

        class _FakeOK:
            status = 200
            headers = {"X-RateLimit-Remaining": "4999", "X-RateLimit-Reset": str(int(time.time()) + 3600)}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, _n=-1):
                return b'{"login": "octocat"}'

        call_count = {"n": 0}

        def fake_open(req, timeout=15):
            seen_auth.append(req.get_header("Authorization"))
            call_count["n"] += 1
            if call_count["n"] == 1:
                hdrs = Message()
                hdrs["X-RateLimit-Remaining"] = "0"
                hdrs["X-RateLimit-Reset"] = str(int(time.time()) + 1)
                raise HTTPError(req.full_url, 403, "rate limit exceeded", hdrs, io.BytesIO(b"{}"))
            return _FakeOK()

        with mock.patch.object(api, "_open_request", side_effect=fake_open), \
                mock.patch.object(api.time, "sleep", side_effect=lambda s: sleeps.append(s)):
            data = api.github_get_json("https://api.github.com/users/octocat", cfg_actor.effective_github_token)
        self.assertEqual(data, {"login": "octocat"})
        self.assertEqual(seen_auth, ["Bearer ghp_actor_only", "Bearer ghp_actor_only"])
        self.assertEqual(len(sleeps), 1)
        self.assertGreater(sleeps[0], 0.0)

        # 2) Long-reset 403 in post_weekly_if_due does NOT crash or mark the week posted (retries on next tick).
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        friday = datetime(2026, 11, 27, 15, 0, tzinfo=timezone.utc)

        def fail_403(url, _tok, **_kwargs):
            if "/orgs/" in url:
                return []
            raise api.TransientAPIError("api.github.com returned HTTP 403", retry_after=1800.0)

        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fail_403), \
                mock.patch("contributor_bot.weekly.fetch_recent_messages", return_value=[]):
            result = post_weekly_if_due(cfg_actor, store, friday)
        self.assertIsNone(result)
        self.assertFalse(store.merged_week_posted("BerriAI/litellm", "2026-W48"))

        # 3) Only one weekly scheduler: _run_loop never calls legacy poller.poll_once even if announcements_enabled=True.
        stop_event = mock.Mock()
        stop_event.is_set.return_value = True
        cfg_legacy_on = Config(
            "tok", "123", "456",
            announcements_enabled=True,
            merged_weekly_enabled=True,
            pr_review_enabled=False,
        )
        with mock.patch("contributor_bot.main.poll_once") as legacy_poll, \
                mock.patch("contributor_bot.weekly.fetch_recent_messages", return_value=[]), \
                self.assertLogs(level="INFO") as logs:
            _run_loop(cfg_legacy_on, stop_event, store=store, gateway=None)
        legacy_poll.assert_not_called()
        combined_logs = "\n".join(logs.output)
        self.assertIn("Weekly contributor summary: friday 10:00 America/Toronto, destination 456", combined_logs)
        self.assertNotIn("Weekly contributor roundup:", combined_logs)


class FridaySchedulerReliabilityTests(unittest.TestCase):
    def setUp(self):
        from contributor_bot.weekly import reset_org_members_cache, reset_weekly_scheduler_state
        reset_org_members_cache()
        reset_weekly_scheduler_state()

    def tearDown(self):
        from contributor_bot.weekly import reset_org_members_cache, reset_weekly_scheduler_state
        reset_org_members_cache()
        reset_weekly_scheduler_state()

    def _make_guild(self):
        from types import SimpleNamespace
        member = SimpleNamespace(
            id=111111111111111111,
            name="alice-dev",
            display_name="Alice Dev",
            global_name="Alice Dev",
            nick=None,
            bot=False,
        )
        return SimpleNamespace(
            id=333,
            members=[member],
            member_count=1,
            chunked=True,
            channels=[SimpleNamespace(id=456, name="announcements", type=0)],
        )

    def _make_fake_github(self, *, include_match: bool = True):
        def fake_github(url, _token=None, **_kwargs):
            if "/public_members" in url:
                return []
            if "/search/issues" in url:
                author = "alice-dev" if include_match else "unmatched-contributor"
                return {
                    "items": [
                        {
                            "number": 901,
                            "title": "feat: add anthropic prompt cache helper",
                            "html_url": "https://github.com/BerriAI/litellm/pull/901",
                            "user": {"login": author, "type": "User", "id": 9001},
                            "author_association": "FIRST_TIME_CONTRIBUTOR",
                        },
                    ]
                }
            if "/users/alice-dev" in url:
                return {"login": "alice-dev", "name": "Alice Dev", "bio": "", "blog": "", "twitter_username": ""}
            if "/users/unmatched-contributor" in url:
                return {"login": "unmatched-contributor", "name": "Nobody Here", "bio": "", "blog": "", "twitter_username": ""}
            return []

        return fake_github

    def test_manual_command_never_consumes_scheduled_friday_slot_and_late_friday_deploy_catches_up(self):
        import tempfile
        from datetime import datetime, timezone
        from pathlib import Path
        from contributor_bot.interactions import handle_interaction
        from contributor_bot.store import Store
        from contributor_bot.weekly import (
            build_weekly_chunks,
            post_weekly_if_due,
            week_key,
            weekly_schedule_status,
        )

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store = Store(str(Path(temp_dir.name) / "roles.db"))
        self.addCleanup(store.close)
        cfg = Config(
            "tok", "123", "456",
            discord_guild_id="333",
            discord_application_id="444",
            merged_weekly_enabled=True,
            merged_weekday=4,
            merged_time="10:00",
            merged_timezone="America/Toronto",
        )
        guild = self._make_guild()
        fake_gh = self._make_fake_github(include_match=True)

        # Thursday evening Nov 26, 2026: manual /merged-this-week runs and posts.
        thu_evening = datetime(2026, 11, 27, 1, 0, tzinfo=timezone.utc)  # Thu 8:00 PM Toronto
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_gh):
            chunks = build_weekly_chunks(cfg, store, thu_evening, guild=guild)
        self.assertEqual(len(chunks), 1)
        # Manual run must NOT touch merged_weeks in SQLite.
        self.assertFalse(store.merged_week_posted("BerriAI/litellm", "2026-W48"))

        # Even invoking handle_interaction for /merged-this-week as a mod does not mark merged_weeks.
        with mock.patch("contributor_bot.interactions.threading.Thread"):
            resp = handle_interaction(cfg, store, {
                "type": 2,
                "guild_id": "333",
                "application_id": "444",
                "channel_id": "123",
                "token": "tok",
                "member": {"user": {"id": "111"}, "permissions": str(1 << 13)},
                "data": {"name": "merged-this-week", "options": []},
            })
        self.assertEqual(resp, {"type": 5, "data": {"flags": 64}})
        self.assertFalse(store.merged_week_posted("BerriAI/litellm", "2026-W48"))

        # Friday afternoon Nov 27, 2026 2:35 PM Toronto (missed 10:00 AM window):
        # deploying now reports "due now (will catch up)" and immediately posts once.
        fri_afternoon = datetime(2026, 11, 27, 19, 35, tzinfo=timezone.utc)
        status = weekly_schedule_status(cfg, store, fri_afternoon)
        self.assertIn("week=2026-W48", status)
        self.assertIn("status=due now (will catch up)", status)

        posted = []
        send = lambda ch, _tok, msg: posted.append((ch, msg))
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_gh), \
                self.assertLogs(level="INFO") as post_logs:
            result = post_weekly_if_due(cfg, store, fri_afternoon, post=send, guild=guild)
        self.assertIsNotNone(result)
        self.assertEqual(len(posted), 1)
        self.assertEqual(posted[0][0], "456")
        self.assertTrue(store.merged_week_posted("BerriAI/litellm", week_key(fri_afternoon)))
        self.assertTrue(any("Posted weekly merge summary for 2026-W48" in line for line in post_logs.output))

        # Subsequent ticks on Friday skip cleanly, log once why, and never double-post.
        fri_next_min = datetime(2026, 11, 27, 19, 36, tzinfo=timezone.utc)
        fri_next_min2 = datetime(2026, 11, 27, 19, 37, tzinfo=timezone.utc)
        with self.assertLogs(level="INFO") as skip_logs:
            self.assertIsNone(post_weekly_if_due(cfg, store, fri_next_min, post=send, guild=guild))
            self.assertIsNone(post_weekly_if_due(cfg, store, fri_next_min2, post=send, guild=guild))
        already_lines = [line for line in skip_logs.output if "already posted at" in line]
        self.assertEqual(len(already_lines), 1)
        self.assertEqual(len(posted), 1)

    def test_zero_match_run_does_not_poison_sqlite_and_retries_after_backoff(self):
        import tempfile
        from datetime import datetime, timezone
        from pathlib import Path
        from contributor_bot.store import Store
        from contributor_bot.weekly import WEEKLY_ZERO_MATCH_RETRY_SECONDS, post_weekly_if_due

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store = Store(str(Path(temp_dir.name) / "roles.db"))
        self.addCleanup(store.close)
        cfg = Config(
            "tok", "123", "456",
            discord_guild_id="333",
            merged_weekly_enabled=True,
        )
        guild = self._make_guild()
        fri = datetime(2026, 11, 27, 15, 0, tzinfo=timezone.utc)  # Friday 10:00 AM Toronto
        now_mono = [1000.0]
        posted = []
        send = lambda ch, _tok, msg: posted.append((ch, msg))

        # Transient 0-match run with member_count > 0 must NOT mark merged_weeks in SQLite.
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=self._make_fake_github(include_match=False)):
            self.assertIsNone(post_weekly_if_due(cfg, store, fri, post=send, guild=guild, clock=lambda: now_mono[0]))
        self.assertFalse(store.merged_week_posted("BerriAI/litellm", "2026-W48"))
        self.assertEqual(posted, [])

        # Within the backoff window, next 60s tick defers without hitting GitHub.
        now_mono[0] += 60.0
        with mock.patch("contributor_bot.weekly.github_get_json") as gh_mock:
            self.assertIsNone(post_weekly_if_due(cfg, store, fri, post=send, guild=guild, clock=lambda: now_mono[0]))
        gh_mock.assert_not_called()

        # After backoff expires, a healthy run succeeds and posts!
        now_mono[0] += WEEKLY_ZERO_MATCH_RETRY_SECONDS + 1.0
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=self._make_fake_github(include_match=True)):
            result = post_weekly_if_due(cfg, store, fri, post=send, guild=guild, clock=lambda: now_mono[0])
        self.assertIsNotNone(result)
        self.assertEqual(len(posted), 1)
        self.assertTrue(store.merged_week_posted("BerriAI/litellm", "2026-W48"))

    def test_skip_logs_clear_reasons_for_wrong_weekday_and_before_time(self):
        import tempfile
        from datetime import datetime, timezone
        from pathlib import Path
        from contributor_bot.store import Store
        from contributor_bot.weekly import post_weekly_if_due

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store = Store(str(Path(temp_dir.name) / "roles.db"))
        self.addCleanup(store.close)
        cfg = Config("tok", "123", "456", discord_guild_id="333", merged_weekly_enabled=True)

        thu = datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc)
        fri_before = datetime(2026, 11, 27, 14, 58, tzinfo=timezone.utc)  # 09:58 Toronto
        with self.assertLogs(level="INFO") as logs:
            self.assertIsNone(post_weekly_if_due(cfg, store, thu))
            self.assertIsNone(post_weekly_if_due(cfg, store, thu))  # deduplicated
            self.assertIsNone(post_weekly_if_due(cfg, store, fri_before))
            self.assertIsNone(post_weekly_if_due(cfg, store, fri_before))  # deduplicated
        wrong_day = [line for line in logs.output if "today is thursday" in line]
        before_time = [line for line in logs.output if "is before friday 10:00 America/Toronto" in line]
        self.assertEqual(len(wrong_day), 1)
        self.assertEqual(len(before_time), 1)

    def test_main_run_loop_watchdog_runs_weekly_check_and_survives_exceptions(self):
        import tempfile
        from pathlib import Path
        from contributor_bot.main import _run_loop
        from contributor_bot.store import Store

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store = Store(str(Path(temp_dir.name) / "roles.db"))
        self.addCleanup(store.close)
        cfg = Config(
            "tok", "123", "456",
            merged_weekly_enabled=True,
            pr_review_enabled=False,
            poll_interval_seconds=300,
        )

        # Simulate 2 loop iterations: tick 1 raises RuntimeError, tick 2 succeeds, then stops.
        ticks = {"count": 0}
        stop_event = mock.Mock()
        stop_event.is_set.side_effect = lambda: ticks["count"] >= 2

        def fake_wait(_delay):
            ticks["count"] += 1
            return False

        stop_event.wait.side_effect = fake_wait
        calls = []

        def flaky_weekly(*_args, **_kwargs):
            calls.append(len(calls) + 1)
            if len(calls) == 1:
                raise RuntimeError("transient network error")
            return "posted"

        with mock.patch("contributor_bot.main.post_weekly_if_due", side_effect=flaky_weekly), \
                mock.patch("contributor_bot.weekly.fetch_recent_messages", return_value=[]), \
                self.assertLogs(level="INFO") as logs:
            _run_loop(cfg, stop_event, store=store, gateway=None)

        self.assertEqual(calls, [1, 2])
        self.assertTrue(any("Weekly merge summary watchdog check failed" in line for line in logs.output))
        # Delay must be capped at 60s when weekly watchdog is active.
        for call_args in stop_event.wait.call_args_list:
            self.assertLessEqual(call_args.args[0], 60.0)

    def test_gateway_run_reconnects_after_unexpected_exit(self):
        from contributor_bot.gateway import Gateway

        gw = Gateway()
        attempts = {"count": 0}

        async def fake_serve(_discord, _config, _store):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("gateway dropped overnight")
            gw._stop.set()
            return True

        gw._serve = fake_serve
        cfg = Config("tok", "123", "456")
        with mock.patch("contributor_bot.gateway._load_discord", return_value=object()), \
                mock.patch.object(gw._stop, "wait", return_value=False), \
                self.assertLogs(level="WARNING") as logs:
            gw._run(cfg, None)

        self.assertEqual(attempts["count"], 2)
        self.assertTrue(any("reconnecting in" in line for line in logs.output))

    def test_force_weekly_post_now_posts_and_marks_week(self):
        import tempfile
        from datetime import datetime, timezone
        from pathlib import Path
        from contributor_bot.store import Store
        from contributor_bot.weekly import post_weekly_if_due

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store = Store(str(Path(temp_dir.name) / "roles.db"))
        self.addCleanup(store.close)
        cfg = Config("tok", "123", "456", discord_guild_id="333", merged_weekly_enabled=True)
        guild = self._make_guild()
        posted = []
        send = lambda ch, _tok, msg: posted.append((ch, msg))
        sat = datetime(2026, 11, 28, 15, 0, tzinfo=timezone.utc)  # Saturday (outside Friday window)

        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=self._make_fake_github(include_match=True)):
            result = post_weekly_if_due(cfg, store, sat, post=send, guild=guild, force=True)
        self.assertIsNotNone(result)
        self.assertEqual(len(posted), 1)
        self.assertTrue(store.merged_week_posted("BerriAI/litellm", "2026-W48"))

    def test_merged_this_week_restricted_to_moderators(self):
        import asyncio
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from contributor_bot.gateway import _bind_command_tree, _dispatch_command
        from contributor_bot.interactions import MERGED_SLASH_COMMANDS, handle_interaction
        from contributor_bot.store import Store

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store = Store(str(Path(temp_dir.name) / "roles.db"))
        self.addCleanup(store.close)
        cfg = Config(
            "tok", "123", "456",
            discord_guild_id="333",
            discord_application_id="444",
            merged_weekly_enabled=True,
        )

        # 1) REST spec and discord.py CommandTree both set default_permissions to hide from regular members.
        self.assertEqual(MERGED_SLASH_COMMANDS[0].get("default_member_permissions"), str(1 << 13))

        registered = {}

        class _FakeCommandTree:
            def __init__(self, _client):
                pass

            def command(self, *, name, description, guild=None):
                def decorator(fn):
                    registered[name] = {
                        "fn": fn,
                        "description": description,
                        "guild": guild,
                        "default_permissions": getattr(fn, "_default_permissions", None),
                        "allowed_contexts": getattr(fn, "_allowed_contexts", None),
                    }
                    return fn
                return decorator

        def _fake_allowed_contexts(**kwargs):
            def decorator(fn):
                fn._allowed_contexts = kwargs
                return fn
            return decorator

        def _fake_default_permissions(**kwargs):
            def decorator(fn):
                fn._default_permissions = kwargs
                return fn
            return decorator

        fake_discord = SimpleNamespace(
            Object=lambda id: SimpleNamespace(id=id),
            app_commands=SimpleNamespace(
                CommandTree=_FakeCommandTree,
                allowed_contexts=_fake_allowed_contexts,
                default_permissions=_fake_default_permissions,
            ),
        )
        _bind_command_tree(fake_discord, object(), cfg, store)
        self.assertIn("merged-this-week", registered)
        self.assertEqual(registered["merged-this-week"]["default_permissions"], {"manage_messages": True})
        self.assertEqual(
            registered["merged-this-week"]["allowed_contexts"],
            {"guilds": True, "dms": False, "private_channels": False},
        )

        # Also verify against real discord.py when installed in the environment.
        try:
            import discord as real_discord
        except ImportError:
            real_discord = None
        if real_discord is not None:
            real_client = real_discord.Client(intents=real_discord.Intents.none())
            real_tree = _bind_command_tree(real_discord, real_client, cfg, store)
            real_cmd = real_tree.get_command("merged-this-week", guild=real_discord.Object(id=333))
            self.assertIsNotNone(real_cmd)
            self.assertEqual(real_cmd.default_permissions, real_discord.Permissions(manage_messages=True))

        # 2) Regular members (missing perms, 0, or only send_messages) get ephemeral "only mods can run this".
        for perms in (None, "0", str(1 << 11)):
            member = {"user": {"id": "111"}}
            if perms is not None:
                member["permissions"] = perms
            with mock.patch("contributor_bot.interactions.threading.Thread") as thread_cls:
                resp = handle_interaction(cfg, store, {
                    "type": 2,
                    "guild_id": "333",
                    "application_id": "444",
                    "channel_id": "123",
                    "token": "tok",
                    "member": member,
                    "data": {"name": "merged-this-week", "options": []},
                })
            self.assertEqual(resp["type"], 4)
            self.assertEqual(resp["data"]["content"], "only mods can run this")
            self.assertTrue(bool(resp["data"]["flags"] & 64))
            thread_cls.assert_not_called()

        # 3) Gateway _dispatch_command replies ephemerally with "only mods can run this" for non-mods.
        followups = []

        class _RecordingResponse:
            def __init__(self):
                self.done = False

            def is_done(self):
                return self.done

            async def defer(self, *, ephemeral=False):
                self.done = True

            async def send_message(self, content, *, ephemeral=False):
                self.done = True
                followups.append((content, ephemeral))

        async def record_followup(content, *, ephemeral=False):
            followups.append((content, ephemeral))

        non_mod_interaction = SimpleNamespace(
            guild_id=333,
            application_id=444,
            channel_id=123,
            token="tok",
            user=SimpleNamespace(id=111, guild_permissions=SimpleNamespace(value=1 << 11)),
            permissions=SimpleNamespace(value=1 << 11),
            data={"name": "merged-this-week", "options": []},
            channel=SimpleNamespace(id=123, type=SimpleNamespace(value=0), parent_id=None),
            guild=None,
            message=None,
            extras={},
            response=_RecordingResponse(),
            followup=SimpleNamespace(send=record_followup),
        )
        with mock.patch("contributor_bot.interactions._merged_preview") as preview_mock:
            asyncio.run(_dispatch_command(None, non_mod_interaction, cfg, store))
        self.assertEqual(followups, [("only mods can run this", True)])
        preview_mock.assert_not_called()

        # 4) Each moderator/admin permission bit is allowed to run /merged-this-week.
        for mod_bits in (
            1 << 3,   # administrator
            1 << 5,   # manage_guild
            1 << 13,  # manage_messages
            1 << 40,  # moderate_members
        ):
            with mock.patch("contributor_bot.interactions.threading.Thread") as thread_cls:
                resp = handle_interaction(cfg, store, {
                    "type": 2,
                    "guild_id": "333",
                    "application_id": "444",
                    "channel_id": "123",
                    "token": "tok",
                    "member": {"user": {"id": "111"}, "permissions": str(mod_bits)},
                    "data": {"name": "merged-this-week", "options": []},
                })
            self.assertEqual(resp, {"type": 5, "data": {"flags": 64}})
            thread_cls.assert_called_once()

    def test_weekly_report_fast_path_deadline_fallback_progress_logs_and_no_loop_deadlock(self):
        import asyncio
        import tempfile
        from datetime import datetime, timezone
        from pathlib import Path
        from types import SimpleNamespace
        from contributor_bot.matcher import (
            DiscordMemberInfo,
            GitHubAuthorProfile,
            _MemberIndex,
            _run_guild_coro,
            score_author_against_members,
        )
        from contributor_bot.store import Store
        from contributor_bot.weekly import build_weekly_report, post_weekly_if_due

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store = Store(str(Path(temp_dir.name) / "roles.db"))
        self.addCleanup(store.close)
        cfg = Config(
            "tok", "123", "456",
            discord_guild_id="333",
            merged_weekly_enabled=True,
            typesafe_api_key="ts-key",
        )
        guild = self._make_guild()
        fri = datetime(2026, 11, 27, 19, 56, tzinfo=timezone.utc)

        # 1) post_weekly_if_due uses pace=False everywhere (never sleeps 90s) and emits progress INFO logs.
        pace_seen = []
        fake_gh_inner = self._make_fake_github(include_match=True)

        def recording_github(url, token=None, **kwargs):
            pace_seen.append((url, kwargs.get("pace", False)))
            return fake_gh_inner(url, token, **kwargs)

        posted = []
        send = lambda ch, _tok, msg: posted.append((ch, msg))
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=recording_github), \
                self.assertLogs(level="INFO") as logs:
            result = post_weekly_if_due(cfg, store, fri, post=send, guild=guild)
        self.assertIsNotNone(result)
        self.assertEqual(len(posted), 1)
        self.assertTrue(all(pace is False for _url, pace in pace_seen), f"unexpected pace=True: {pace_seen}")
        joined = "\n".join(logs.output)
        self.assertIn("Weekly report: fetched", joined)
        self.assertIn("Weekly report: loaded", joined)
        self.assertIn("Weekly merge summary:", joined)
        self.assertIn("Weekly report: built", joined)

        # 2) Overall deadline expired falls back to deterministic matching + deterministic title cleanup (no LLM calls).
        jev_called = []
        haiku_called = []
        title_llm_called = []

        def fail_jev(_body):
            jev_called.append(True)
            raise AssertionError("Jev should not be called once deadline is reached")

        def fail_haiku(_model, _messages):
            haiku_called.append(True)
            raise AssertionError("Haiku should not be called once deadline is reached")

        def fail_summarizer(_model, _messages):
            title_llm_called.append(True)
            raise AssertionError("Title summarizer should not be called once deadline is reached")

        # Include one fuzzy-only author ("alice-devx") so candidate disambiguation is exercised under deadline.
        def fuzzy_github(url, _token=None, **_kwargs):
            if "/public_members" in url:
                return []
            if "/search/issues" in url:
                return {
                    "items": [
                        {
                            "number": 901,
                            "title": "feat: add anthropic prompt cache helper",
                            "html_url": "https://github.com/BerriAI/litellm/pull/901",
                            "user": {"login": "alice-dev", "type": "User", "id": 9001},
                            "author_association": "FIRST_TIME_CONTRIBUTOR",
                        },
                        {
                            "number": 902,
                            "title": "fix: fuzzy author pr",
                            "html_url": "https://github.com/BerriAI/litellm/pull/902",
                            "user": {"login": "alice-devx", "type": "User", "id": 9002},
                            "author_association": "CONTRIBUTOR",
                        },
                    ]
                }
            raise AssertionError(f"Unexpected enrichment call after deadline: {url}")

        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fuzzy_github):
            report = build_weekly_report(
                cfg,
                store,
                fri,
                guild=guild,
                pace=False,
                jev_transport=fail_jev,
                haiku_complete=fail_haiku,
                title_summarizer=fail_summarizer,
                deadline_seconds=0.0,
            )
        self.assertEqual(report.matched_count, 1)  # exact login still matched deterministically
        self.assertEqual(jev_called, [])
        self.assertEqual(haiku_called, [])
        self.assertEqual(title_llm_called, [])

        # 3) _run_guild_coro must never self-deadlock when called from the gateway event loop thread itself.
        async def check_no_self_deadlock():
            loop = asyncio.get_running_loop()

            async def noop():
                return []

            with self.assertRaises(RuntimeError):
                _run_guild_coro(noop(), loop)

        asyncio.run(check_no_self_deadlock())

        # 4) Bigram-indexed fuzzy matching finds close candidates quickly across thousands of members.
        big_members = [
            DiscordMemberInfo(discord_id=str(100000000000000000 + i), username=f"member_{i}", display_name=f"User {i}")
            for i in range(2000)
        ]
        big_members.append(DiscordMemberInfo(discord_id="999999999999999999", username="alex_rivera", display_name="Alex Rivera"))
        idx = _MemberIndex(big_members)
        strong, cands = score_author_against_members(
            GitHubAuthorProfile(login="alexrivera99"),
            big_members,
            member_index=idx,
        )
        # Either exact base match or top fuzzy candidate must find alex_rivera.
        found_ids = ([strong[0].discord_id] if strong else []) + [c.member.discord_id for c in cands]
        self.assertIn("999999999999999999", found_ids)

        # 5) Unexpected build failure in post_weekly_if_due logs an exception clearly.
        store.clear_merged_week("BerriAI/litellm", "2026-W48")
        with mock.patch("contributor_bot.weekly.build_weekly_report", side_effect=RuntimeError("boom")), \
                self.assertLogs(level="ERROR") as err_logs:
            with self.assertRaises(RuntimeError):
                post_weekly_if_due(cfg, store, fri, post=send, guild=guild)
        self.assertTrue(any("Weekly merge summary report build failed" in line for line in err_logs.output))

    def test_ephemeral_merged_preview_never_marks_week_and_boot_clears_bogus_w48_preview_mark(self):
        import sqlite3
        import tempfile
        from datetime import datetime, timezone
        from pathlib import Path
        from contributor_bot.interactions import _finish_merged_preview
        from contributor_bot.store import Store
        from contributor_bot.weekly import post_weekly_if_due

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        db_path = str(Path(temp_dir.name) / "roles.db")

        # 1) Simulate existing production SQLite DB (before `via` column) containing the bogus
        # ephemeral-preview row for 2026-W48 at 2026-11-27T20:34:06+00:00.
        raw_conn = sqlite3.connect(db_path)
        raw_conn.execute(
            "CREATE TABLE merged_weeks ("
            "repository TEXT NOT NULL, week_key TEXT NOT NULL, posted_at TEXT NOT NULL, "
            "PRIMARY KEY (repository, week_key))"
        )
        raw_conn.execute(
            "INSERT INTO merged_weeks (repository, week_key, posted_at) VALUES (?, ?, ?)",
            ("BerriAI/litellm", "2026-W48", "2026-11-27T20:34:06+00:00"),
        )
        raw_conn.commit()
        raw_conn.close()

        # Opening Store migrates `via` and clears the bogus 2026-W48 preview row once.
        with self.assertLogs(level="INFO") as boot_logs:
            store = Store(db_path)
        self.addCleanup(store.close)
        self.assertFalse(store.merged_week_posted("BerriAI/litellm", "2026-W48"))
        self.assertTrue(any("Cleared bogus ephemeral-preview merged_weeks entry" in l for l in boot_logs.output))

        cfg = Config(
            "tok", "123", "456",
            discord_guild_id="333",
            discord_application_id="444",
            merged_weekly_enabled=True,
            merged_weekday=4,
            merged_time="10:00",
            merged_timezone="America/Toronto",
        )
        guild = self._make_guild()
        fake_gh = self._make_fake_github(include_match=True)
        fri_after = datetime(2026, 11, 27, 20, 36, tzinfo=timezone.utc)  # Fri 3:36 PM Toronto

        # 2) Running ephemeral /merged-this-week preview in #announcements on Friday >= 10:00
        # must NEVER mark merged_weeks (nothing was posted publicly).
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_gh), \
                mock.patch("contributor_bot.interactions.edit_original_interaction"), \
                mock.patch("contributor_bot.interactions.post_interaction_followup"):
            _finish_merged_preview(cfg, store, "tok", "444", channel_id="456", guild=guild, now=fri_after)
        self.assertFalse(store.merged_week_posted("BerriAI/litellm", "2026-W48"))

        # 3) Scheduler catch-up posts publicly to #announcements and marks 2026-W48 via='scheduler'.
        posted = []
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_gh):
            res = post_weekly_if_due(
                cfg,
                store,
                fri_after,
                post=lambda ch, _t, m: posted.append((ch, m)),
                guild=guild,
            )
        self.assertIsNotNone(res)
        self.assertEqual(len(posted), 1)
        self.assertEqual(posted[0][0], "456")
        info = store.merged_week_info("BerriAI/litellm", "2026-W48")
        self.assertIsNotNone(info)
        self.assertEqual(info["via"], "scheduler")
        self.assertEqual(info["posted_at"], "2026-11-27T20:36:00+00:00")

        # 4) Reopening Store on a subsequent restart preserves the real scheduler mark and never double-posts.
        restarted = Store(db_path)
        self.addCleanup(restarted.close)
        self.assertTrue(restarted.merged_week_posted("BerriAI/litellm", "2026-W48"))
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_gh):
            again = post_weekly_if_due(
                cfg,
                restarted,
                datetime(2026, 11, 27, 20, 40, tzinfo=timezone.utc),
                post=lambda ch, _t, m: posted.append((ch, m)),
                guild=guild,
            )
        self.assertIsNone(again)
        self.assertEqual(len(posted), 1)


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
