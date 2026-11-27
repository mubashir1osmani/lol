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

        text = render_weekly(groups, since=datetime(2026, 11, 19, tzinfo=timezone.utc))
        self.assertIn("- <@111>: 2 PRs ([#12](<https://github.com/BerriAI/litellm/pull/12>), [#19](<https://github.com/BerriAI/litellm/pull/19>))", text)
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

    def test_discord_mapping_order_fallbacks_and_skipped_logging(self):
        import asyncio
        from types import SimpleNamespace
        from contributor_bot.weekly import build_weekly_report, resolve_discord_ids_for_groups

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        # /link-github store maps github user_id=101 -> discord_id="100000000000000001"
        store.link_account("100000000000000001", "linkedDev", 101, "2026-11-20T00:00:00+00:00")

        base_groups = [
            {"login": "linkedDev", "user_id": 101, "discord_id": None, "pulls": [{"number": 1, "title": "a", "url": "https://github.com/BerriAI/litellm/pull/1"}]},
            {"login": "NameMatch", "user_id": 102, "discord_id": None, "pulls": [{"number": 2, "title": "b", "url": "https://github.com/BerriAI/litellm/pull/2"}]},
            {"login": "GlobalMatch", "user_id": 103, "discord_id": None, "pulls": [{"number": 3, "title": "c", "url": "https://github.com/BerriAI/litellm/pull/3"}]},
            {"login": "DisplayMatch", "user_id": 104, "discord_id": None, "pulls": [{"number": 4, "title": "d", "url": "https://github.com/BerriAI/litellm/pull/4"}]},
            {"login": "ghostUser", "user_id": 105, "discord_id": None, "pulls": [{"number": 5, "title": "e", "url": "https://github.com/BerriAI/litellm/pull/5"}]},
        ]

        # 1. Store wins over guild member with colliding name; guild.members matches username, global_name, display_name case-insensitively and ignores bots.
        guild_cached = SimpleNamespace(members=[
            SimpleNamespace(id=999999999999999999, name="linkeddev", global_name=None, display_name="linkeddev", nick=None, bot=False),
            SimpleNamespace(id=200000000000000002, name="namematch", global_name="Someone", display_name="Someone", nick=None, bot=False),
            SimpleNamespace(id=300000000000000003, name="other_user", global_name="globalmatch", display_name="Other", nick=None, bot=False),
            SimpleNamespace(id=400000000000000004, name="nick_user", global_name="Nick", display_name="displaymatch", nick="displaymatch", bot=False),
            SimpleNamespace(id=500000000000000005, name="ghostuser", global_name=None, display_name="ghostuser", nick=None, bot=True),
        ])
        matched, skipped = resolve_discord_ids_for_groups(
            base_groups, store, guild=guild_cached, members_intent_enabled=True,
        )
        by_login = {g["login"]: g["discord_id"] for g in matched}
        self.assertEqual(by_login, {
            "linkedDev": "100000000000000001",
            "NameMatch": "200000000000000002",
            "GlobalMatch": "300000000000000003",
            "DisplayMatch": "400000000000000004",
        })
        self.assertEqual(skipped, 1)

        # 2. When members intent is off and guild.members is empty/partial, falls back to guild.fetch_members().
        class _FetchGuild:
            members = []

            async def fetch_members(self, limit=None):
                for item in guild_cached.members:
                    yield item

        matched_fetch, skipped_fetch = resolve_discord_ids_for_groups(
            base_groups, store, guild=_FetchGuild(), members_intent_enabled=False,
        )
        self.assertEqual(len(matched_fetch), 4)
        self.assertEqual(skipped_fetch, 1)

        # 3. When fetch_members fails (intent disabled on server), falls back to guild.search_members().
        class _SearchGuild:
            members = []

            async def fetch_members(self, limit=None):
                raise RuntimeError("missing members intent")
                yield  # pragma: no cover

            async def search_members(self, query, limit=10):
                q = query.casefold()
                return [m for m in guild_cached.members if q in (m.name or "").casefold() or q in (m.global_name or "").casefold() or q in (m.display_name or "").casefold()]

        matched_search, skipped_search = resolve_discord_ids_for_groups(
            base_groups, store, guild=_SearchGuild(), members_intent_enabled=False,
        )
        self.assertEqual(len(matched_search), 4)
        self.assertEqual(skipped_search, 1)

        # 4. When guild has neither cache nor methods, falls back to REST search_guild_members and logs matched/skipped counts.
        config = Config("token", "123", "456", discord_guild_id="333", discord_application_id="444")
        gh_items = [
            {"number": 1, "title": "a", "html_url": "https://github.com/BerriAI/litellm/pull/1", "user": {"login": "linkedDev", "type": "User", "id": 101}, "author_association": "CONTRIBUTOR"},
            {"number": 2, "title": "b", "html_url": "https://github.com/BerriAI/litellm/pull/2", "user": {"login": "restUser", "type": "User", "id": 106}, "author_association": "CONTRIBUTOR"},
            {"number": 3, "title": "c", "html_url": "https://github.com/BerriAI/litellm/pull/3", "user": {"login": "notInDiscord", "type": "User", "id": 107}, "author_association": "CONTRIBUTOR"},
        ]

        def fake_gh(url, _token, **_kwargs):
            return [] if "/orgs/" in url else {"items": gh_items}

        def fake_rest_search(_gid, query, _tok, **_kwargs):
            if query.casefold() == "restuser":
                return [{"user": {"id": "600000000000000006", "username": "restUser", "global_name": "Rest User", "bot": False}, "nick": None}]
            return []

        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_gh), \
                mock.patch("contributor_bot.weekly.search_guild_members", side_effect=fake_rest_search), \
                self.assertLogs(level="INFO") as logs:
            report = build_weekly_report(config, store, now=datetime(2026, 11, 26, tzinfo=timezone.utc))

        self.assertEqual(report.matched_count, 2)
        self.assertEqual(report.skipped_count, 1)
        body = "\n".join(report.chunks)
        self.assertIn("<@100000000000000001>", body)
        self.assertIn("<@600000000000000006>", body)
        self.assertNotIn("linkedDev", body)
        self.assertNotIn("restUser", body)
        self.assertNotIn("notInDiscord", body)
        self.assertTrue(any("2 contributors matched in Discord, 1 unmatched skipped" in entry for entry in logs.output))

    def test_merged_this_week_defers_suppresses_embeds_and_caps_at_15_lines(self):
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
            oss_contributors_channel_id="",
            discord_members_intent=True,
        )
        self.assertIn("merged-this-week", [cmd["name"] for cmd in commands_for_guild(config)])

        # 22 matched external contributors + 3 unmatched external contributors + org/member entries.
        items = [
            {
                "number": 1000 + i,
                "title": f"feat(provider-{i}): add support",
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
            return {"items": items}

        with mock.patch("contributor_bot.interactions.threading.Thread", _InlineThread), \
                mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_github), \
                mock.patch("contributor_bot.interactions.edit_original_interaction", side_effect=lambda *a: edits.append(a)), \
                mock.patch("contributor_bot.interactions.post_interaction_followup", side_effect=lambda *a: followups.append(a)):
            asyncio.run(_dispatch_command(None, interaction, config, store))

        self.assertEqual(deferred, [True])
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0][2].get("flags"), 64 | SUPPRESS_EMBEDS_FLAG)
        first_chunk = edits[0][2]["content"]
        self.assertEqual(
            edits[0][2].get("allowed_mentions"),
            {"parse": [], "users": chunk_mention_user_ids(first_chunk)},
        )
        self.assertEqual(len(edits[0][2]["allowed_mentions"]["users"]), 15)
        all_chunks = [first_chunk] + [call[2]["content"] for call in followups]
        combined = "\n".join(all_chunks)
        for chunk in all_chunks:
            self.assertLessEqual(len(chunk), 1900)
            self.assertEqual(chunk.count("["), chunk.count("]"))
        self.assertIn("](<https://github.com/BerriAI/litellm/pull/", combined)
        self.assertNotIn("](https://", combined)
        self.assertNotIn("contrib-", combined)
        self.assertNotIn("unmatched-", combined)
        self.assertNotIn("staff-org", combined)
        self.assertNotIn("core-dev", combined)
        contributor_lines = [line for line in combined.splitlines() if line.startswith("- <@")]
        self.assertEqual(len(contributor_lines), 15)
        self.assertIn("and 7 more contributors (7 PRs).", combined)
        # Preview must NOT mark the week posted in SQLite.
        self.assertFalse(store.merged_week_posted("BerriAI/litellm", week_key(datetime.now(timezone.utc))))

        # Scheduled weekly post also sets SUPPRESS_EMBEDS flag and allowed_mentions.
        posted = []
        moment = datetime(2026, 11, 23, 15, 0, tzinfo=timezone.utc)
        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_github):
            post_weekly_if_due(
                config,
                store,
                moment,
                post=lambda ch, _tok, msg: posted.append((ch, msg)),
                guild=fake_guild,
            )
        self.assertTrue(posted)
        self.assertEqual(posted[0][1].get("flags"), SUPPRESS_EMBEDS_FLAG)
        self.assertEqual(
            posted[0][1].get("allowed_mentions"),
            {"parse": [], "users": chunk_mention_user_ids(posted[0][1]["content"])},
        )
        self.assertEqual(len(posted[0][1]["allowed_mentions"]["users"]), 15)

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

        # Tight limit splits cleanly on line boundaries with overflow footer.
        huge_groups = [
            {
                "login": f"dev-{i:03d}",
                "discord_id": str(200000000000000000 + i),
                "pulls": [{"number": i, "title": "fix", "url": f"https://github.com/BerriAI/litellm/pull/{i}"}],
            }
            for i in range(30)
        ]
        capped = weekly_chunks(
            huge_groups,
            since=datetime(2026, 11, 19, tzinfo=timezone.utc),
            limit=500,
            max_chunks=2,
            max_contributors=15,
        )
        self.assertLessEqual(len(capped), 2)
        for chunk in capped:
            self.assertLessEqual(len(chunk), 500)
        self.assertIn("more contributors", capped[-1])

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
        self.assertEqual(cfg.oss_contributors_channel_id, "123")

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
