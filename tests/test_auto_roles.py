import os
import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from contributor_bot.api import TransientAPIError, fetch_author_merged_count
from contributor_bot.config import get_config
from contributor_bot.gateway import commands_for_guild
from contributor_bot.interactions import (
    ROLES_BACKFILL_SLASH_COMMANDS,
    handle_interaction,
    role_slash_commands,
)
from contributor_bot.matcher import DiscordMemberInfo
from contributor_bot.roles_sync import (
    RolesPollResult,
    format_backfill_result,
    has_pending_backfill,
    log_roles_daily_boot_status,
    poll_merged_prs_for_roles,
    reset_roles_daily_scheduler_state,
    roles_daily_schedule_status,
    run_daily_roles_if_due,
    run_roles_backfill,
    should_run_boot_backfill,
)
from contributor_bot.schedule import UTC
from contributor_bot.store import Store


def minimal_roles_env(directory: str, **extra: str) -> dict[str, str]:
    """Minimal env for contributor roles: NO OAuth, NO webhook secret, NO public key."""
    return {
        "DISCORD_BOT_TOKEN": "discord-secret",
        "OSS_CONTRIBUTIONS_CHANNEL_ID": "123",
        "ANNOUNCEMENTS_CHANNEL_ID": "456",
        "CONTRIBUTOR_ROLES_ENABLED": "true",
        "DISCORD_GUILD_ID": "333",
        "ROLE_CONTRIBUTOR_ID": "1001",
        "ROLE_OSS_CHAMPION_ID": "1002",
        "ROLE_AGENT_BUILDER_ID": "1003",
        "ROLE_PROVIDER_WIZARD_ID": "1004",
        "ROLE_ANNOUNCEMENTS_CHANNEL_ID": "789",
        "MAINTAINER_GITHUB_LOGINS": "maint-alice",
        "GITHUB_TOKEN": "github-secret",
        "ROLES_DB_PATH": str(Path(directory) / "roles.db"),
        "STATE_FILE": str(Path(directory) / "state.json"),
        **extra,
    }


def _member(discord_id: str, username: str, display_name: str = "", global_name: str = "") -> DiscordMemberInfo:
    return DiscordMemberInfo(
        discord_id=discord_id,
        username=username,
        global_name=global_name,
        display_name=display_name or global_name or username,
    )


class ZeroOAuthConfigAndCommandsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = get_config(minimal_roles_env(self.temp.name))
        self.store = Store(self.config.roles_db_path)
        self.addCleanup(self.store.close)

    def test_minimal_config_hides_link_commands_and_exposes_mod_backfill(self):
        self.assertTrue(self.config.contributor_roles_enabled)
        self.assertFalse(self.config.github_oauth_enabled)
        self.assertIsNone(self.config.discord_public_key)
        self.assertIsNone(self.config.github_webhook_secret)
        self.assertIsNone(self.config.discord_application_id)

        role_names = [cmd["name"] for cmd in role_slash_commands(self.config)]
        self.assertNotIn("link-github", role_names)
        self.assertNotIn("unlink-github", role_names)
        self.assertIn("my-prs", role_names)
        self.assertIn("roles-backfill", role_names)

        guild_names = [cmd["name"] for cmd in commands_for_guild(self.config)]
        self.assertNotIn("link-github", guild_names)
        self.assertNotIn("unlink-github", guild_names)
        self.assertIn("roles-backfill", guild_names)
        self.assertEqual(ROLES_BACKFILL_SLASH_COMMANDS[0]["default_member_permissions"], str(1 << 13))

        # Invoking /link-github or /unlink-github directly when OAuth is off replies cleanly.
        for name in ("link-github", "unlink-github"):
            resp = handle_interaction(
                self.config,
                self.store,
                {
                    "type": 2,
                    "guild_id": "333",
                    "application_id": "999999",
                    "member": {"user": {"id": "111"}, "permissions": "0"},
                    "data": {"name": name},
                },
            )
            self.assertEqual(resp["data"]["flags"], 64)
            self.assertIn("automatically", resp["data"]["content"])

    def test_oauth_configured_shows_link_commands(self):
        with_oauth = get_config(minimal_roles_env(
            self.temp.name,
            GITHUB_OAUTH_CLIENT_ID="Iv1.client123",
            GITHUB_OAUTH_CLIENT_SECRET="oauth-secret-value",
            GITHUB_OAUTH_REDIRECT_URI="https://bot.example/github/oauth/callback",
        ))
        self.assertTrue(with_oauth.github_oauth_enabled)
        names = [cmd["name"] for cmd in role_slash_commands(with_oauth)]
        self.assertIn("link-github", names)
        self.assertIn("unlink-github", names)
        self.assertIn("roles-backfill", names)

    def test_roles_backfill_command_is_mod_only_and_works_without_app_id_env(self):
        base = {
            "type": 2,
            "guild_id": "333",
            "application_id": "777888999",
            "token": "tok-abc",
            "data": {"name": "roles-backfill"},
        }
        non_mod = handle_interaction(
            self.config,
            self.store,
            {**base, "member": {"user": {"id": "111"}, "permissions": "0"}},
        )
        self.assertEqual(non_mod["data"]["flags"], 64)
        self.assertEqual(non_mod["data"]["content"], "only mods can run this")

        edits = []

        def inline_thread(target, args=(), name=None, daemon=None):
            class Started:
                def start(self_inner):
                    target(*args)
            return Started()

        with mock.patch("contributor_bot.interactions.threading.Thread", inline_thread), \
             mock.patch("contributor_bot.roles_sync.run_roles_backfill", return_value=None) as run_bf, \
             mock.patch("contributor_bot.interactions.edit_original_interaction", side_effect=lambda *a: edits.append(a)):
            mod_resp = handle_interaction(
                self.config,
                self.store,
                {**base, "member": {"user": {"id": "111"}, "permissions": str(1 << 13)}},
            )
        self.assertEqual(mod_resp["type"], 5)
        self.assertEqual(mod_resp["data"]["flags"], 64)
        run_bf.assert_called_once()
        self.assertEqual(edits[0][0], "777888999")
        self.assertIn("already running", edits[0][2]["content"])

    def test_my_prs_works_for_auto_matched_contributor_without_oauth_link(self):
        self.store.save_confirmed_match(
            "janedev",
            "222",
            method="exact_login",
            confidence=1.0,
            matched_at="2026-11-27T18:00:00+00:00",
            github_user_id=42,
        )
        self.store.upsert_pull(
            repository="BerriAI/litellm",
            number=101,
            github_user_id=42,
            github_login="janedev",
            merged_at="2026-11-27T12:00:00+00:00",
            title="add mcp server",
            labels=[],
            files=["litellm/proxy/mcp_tools.py"],
        )
        edits = []

        def inline_thread(target, args=(), name=None, daemon=None):
            class Started:
                def start(self_inner):
                    target(*args)
            return Started()

        with mock.patch("contributor_bot.interactions.threading.Thread", inline_thread), \
             mock.patch("contributor_bot.interactions.fetch_open_pulls_for_author", return_value=([], 0)), \
             mock.patch("contributor_bot.interactions.edit_original_interaction", side_effect=lambda *a: edits.append(a)):
            resp = handle_interaction(
                self.config,
                self.store,
                {
                    "type": 2,
                    "guild_id": "333",
                    "application_id": "777888999",
                    "token": "tok-myprs",
                    "member": {"user": {"id": "222"}, "permissions": "0"},
                    "data": {"name": "my-prs"},
                },
            )
        self.assertEqual(resp["type"], 5)
        self.assertIn("hey janedev", edits[0][2]["content"])
        self.assertIn("contributor, agent builder", edits[0][2]["content"])


class AutoRolesPollerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = get_config(minimal_roles_env(self.temp.name))
        self.store = Store(self.config.roles_db_path)
        self.addCleanup(self.store.close)

    def test_poller_stores_prs_auto_matches_skips_ambiguous_and_announces_new_roles(self):
        # Members:
        # - "janedev" is a unique exact_login match -> confident!
        # - "alex" has TWO guild members with identical display name "Alex Smith" -> ambiguous tie, must NOT match!
        members = [
            _member("2001", "janedev", "Jane Dev"),
            _member("2002", "alex_one", "Alex Smith"),
            _member("2003", "alex_two", "Alex Smith"),
        ]
        pulls = {
            # Jane merges 3 PRs in Nov 2026 (including an MCP PR) -> earns Contributor, OSS Champion, Agent Builder!
            10: {
                "number": 10, "merged": True, "merged_at": "2026-11-27T10:00:00Z", "title": "mcp fix",
                "user": {"login": "janedev", "id": 501, "type": "User"},
                "author_association": "FIRST_TIME_CONTRIBUTOR", "labels": [],
            },
            11: {
                "number": 11, "merged": True, "merged_at": "2026-11-27T11:00:00Z", "title": "second",
                "user": {"login": "janedev", "id": 501, "type": "User"},
                "author_association": "CONTRIBUTOR", "labels": [],
            },
            12: {
                "number": 12, "merged": True, "merged_at": "2026-11-27T12:00:00Z", "title": "third",
                "user": {"login": "janedev", "id": 501, "type": "User"},
                "author_association": "CONTRIBUTOR", "labels": [],
            },
            # Ambiguous contributor (GitHub profile name "Alex Smith" ties two Discord members).
            13: {
                "number": 13, "merged": True, "merged_at": "2026-11-27T12:30:00Z", "title": "ambiguous",
                "user": {"login": "asmith99", "id": 502, "type": "User"},
                "author_association": "FIRST_TIME_CONTRIBUTOR", "labels": [],
            },
            # Maintainer & bot -> must be excluded from storage & roles.
            14: {
                "number": 14, "merged": True, "merged_at": "2026-11-27T12:40:00Z", "title": "maint",
                "user": {"login": "maint-alice", "id": 503, "type": "User"},
                "author_association": "MEMBER", "labels": [],
            },
            15: {
                "number": 15, "merged": True, "merged_at": "2026-11-27T12:45:00Z", "title": "bot",
                "user": {"login": "dependabot[bot]", "id": 504, "type": "Bot"},
                "author_association": "NONE", "labels": [],
            },
        }
        files_map = {
            10: ["litellm/proxy/mcp_server.py"],
            11: ["README.md"],
            12: ["README.md"],
            13: ["README.md"],
        }

        def fake_github_get(url, token, **kwargs):
            if url.endswith("/users/asmith99"):
                return {"login": "asmith99", "name": "Alex Smith", "bio": "", "blog": "", "twitter_username": ""}
            return {}

        now = datetime(2026, 11, 27, 15, 0, tzinfo=UTC)
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role, \
             mock.patch("contributor_bot.rewards.post_message", return_value={"id": "900"}) as post_msg:
            res = poll_merged_prs_for_roles(
                self.config,
                self.store,
                now=now,
                search_merged_fn=lambda repo, start, end, tok: set(pulls.keys()),
                pull_fetcher=lambda repo, num, tok: pulls[num],
                files_fetcher=lambda repo, num, tok: files_map.get(num, []),
                count_fetcher=lambda repo, login, tok: 3,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
                github_getter=fake_github_get,
            )

        self.assertIsNotNone(res)
        self.assertEqual(res.discovered, 6)
        self.assertEqual(res.stored, 4)  # #10, #11, #12, #13 (excluded #14 maintainer & #15 bot)
        self.assertEqual(res.skipped_excluded, 2)
        self.assertEqual(res.authors_matched, 1)  # only janedev; asmith99 ambiguous tie skipped!
        self.assertEqual(res.roles_granted, 3)  # contributor, oss_champion, agent_builder

        # Only Jane (2001) got roles — ambiguous Alex members (2002, 2003) got NOTHING.
        granted_users = {call.args[1] for call in add_role.call_args_list}
        granted_roles = {call.args[2] for call in add_role.call_args_list}
        self.assertEqual(granted_users, {"2001"})
        self.assertEqual(granted_roles, {"1001", "1002", "1003"})

        # Exact short announcement format posted once for Jane!
        self.assertEqual(post_msg.call_count, 1)
        content = post_msg.call_args.args[2]["content"]
        self.assertEqual(
            content,
            "<@2001> just became a contributor, an oss champion, and an agent builder 🎉",
        )
        # Confirmed match cached in SQLite!
        cached = self.store.get_confirmed_match("janedev")
        self.assertIsNotNone(cached)
        self.assertEqual(cached["discord_id"], "2001")
        self.assertEqual(cached["method"], "exact_login")

        # Second poll with same PRs is a no-op (zero extra fetches, zero extra grants/announcements).
        add_role.reset_mock()
        post_msg.reset_mock()
        fetch_calls = []
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role2, \
             mock.patch("contributor_bot.rewards.post_message") as post_msg2:
            res2 = poll_merged_prs_for_roles(
                self.config,
                self.store,
                now=datetime(2026, 11, 27, 15, 10, tzinfo=UTC),
                search_merged_fn=lambda repo, start, end, tok: set(pulls.keys()),
                pull_fetcher=lambda repo, num, tok: (fetch_calls.append(num), pulls[num])[1],
                files_fetcher=lambda repo, num, tok: [],
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
            )
        self.assertEqual(res2.stored, 0)
        self.assertEqual(res2.skipped_existing, 4)
        self.assertEqual(res2.roles_granted, 0)
        add_role2.assert_not_called()
        post_msg2.assert_not_called()

    def test_poller_rate_limit_preserves_checkpoint(self):
        now = datetime(2026, 11, 27, 15, 0, tzinfo=UTC)
        res = poll_merged_prs_for_roles(
            self.config,
            self.store,
            now=now,
            search_merged_fn=mock.Mock(side_effect=TransientAPIError("rate limit")),
            org_members_fetcher=lambda org, tok: frozenset(),
        )
        self.assertTrue(res.rate_limited)
        self.assertIsNone(self.store.get_roles_meta("last_merged_checkpoint:berriai/litellm"))

    def test_lifetime_count_cached_incremented_and_files_fetched_only_when_needed(self):
        members = [_member("4001", "veteran-dev", "Veteran Dev")]
        count_calls = []

        def fake_count(repo, login, tok):
            count_calls.append((repo, login))
            return 9  # 9 lifetime merged PRs on first sight -> Contributor (not yet 10)

        files_calls = []

        def fake_files(repo, num, tok):
            files_calls.append(num)
            return ["litellm/llms/openai.py", "litellm/proxy/mcp_server.py"]

        # Seed 2 prior stored provider PRs so PR #401 (the 3rd) unlocks BOTH Provider Wizard + Agent Builder!
        for n in (398, 399):
            self.store.upsert_pull(
                repository="BerriAI/litellm",
                number=n,
                github_user_id=701,
                github_login="veteran-dev",
                merged_at=f"2026-0{n - 396}-01T00:00:00+00:00",
                title=f"old {n}",
                labels=[],
                files=["litellm/llms/anthropic.py"],
            )

        pr_401 = {
            "number": 401, "merged": True, "merged_at": "2026-11-27T14:00:00Z", "title": "third provider + mcp",
            "user": {"login": "veteran-dev", "id": 701, "type": "User"},
            "author_association": "CONTRIBUTOR", "labels": [],
        }
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role, \
             mock.patch("contributor_bot.rewards.post_message", return_value={"id": "901"}):
            res1 = poll_merged_prs_for_roles(
                self.config,
                self.store,
                now=datetime(2026, 11, 27, 15, 0, tzinfo=UTC),
                search_merged_fn=lambda repo, start, end, tok: {401},
                pull_fetcher=lambda repo, num, tok: pr_401,
                files_fetcher=fake_files,
                count_fetcher=fake_count,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
            )

        # Needed files for PR #401 (didn't have Agent Builder or Provider Wizard yet).
        self.assertEqual(files_calls, [401])
        self.assertEqual(res1.files_skipped, 0)
        # Lifetime count fetched ONCE and cached in SQLite as 9 -> earns Contributor, Agent Builder, Provider Wizard (not OSS Champion yet).
        self.assertEqual(count_calls, [("BerriAI/litellm", "veteran-dev")])
        cached = self.store.get_author_merged_count("BerriAI/litellm", "veteran-dev")
        self.assertEqual(cached["total_count"], 9)
        self.assertEqual(
            {call.args[2] for call in add_role.call_args_list},
            {"1001", "1003", "1004"},
        )

        # Next poll (10 min later): PR #402 merges AFTER cached fetched_at (15:00Z).
        # 1) Author ALREADY has both Agent Builder and Provider Wizard -> fetch_pull_files is SKIPPED!
        # 2) Cached lifetime count increments 9 -> 10 WITHOUT calling count_fetcher again -> earns OSS Champion!
        pr_402 = {
            "number": 402, "merged": True, "merged_at": "2026-11-27T15:05:00Z", "title": "tenth lifetime pr",
            "user": {"login": "veteran-dev", "id": 701, "type": "User"},
            "author_association": "CONTRIBUTOR", "labels": [],
        }
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role2, \
             mock.patch("contributor_bot.rewards.post_message", return_value={"id": "902"}) as post_msg2:
            res2 = poll_merged_prs_for_roles(
                self.config,
                self.store,
                now=datetime(2026, 11, 27, 15, 10, tzinfo=UTC),
                search_merged_fn=lambda repo, start, end, tok: {401, 402},
                pull_fetcher=lambda repo, num, tok: pr_402,
                files_fetcher=fake_files,
                count_fetcher=fake_count,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
            )

        self.assertEqual(files_calls, [401])  # ZERO file-list calls for PR #402!
        self.assertEqual(res2.files_skipped, 1)
        self.assertEqual(count_calls, [("BerriAI/litellm", "veteran-dev")])  # ZERO extra search calls!
        self.assertEqual(self.store.get_author_merged_count("BerriAI/litellm", "veteran-dev")["total_count"], 10)
        self.assertEqual([call.args[2] for call in add_role2.call_args_list], ["1002"])  # OSS Champion!
        self.assertEqual(post_msg2.call_args.args[2]["content"], "<@4001> just became an oss champion 🎉")

    def test_fetch_author_merged_count_search_query_and_validation(self):
        with mock.patch("contributor_bot.api._pace_github_search"), \
             mock.patch("contributor_bot.api._request_json", return_value={"total_count": 14, "incomplete_results": False, "items": []}) as req:
            count = fetch_author_merged_count("BerriAI/litellm", "octocat", "tok")
        self.assertEqual(count, 14)
        url = req.call_args.args[0].full_url
        self.assertIn("author%3Aoctocat", url)
        self.assertIn("is%3Apr+is%3Amerged", url)
        self.assertIn("per_page=1", url)
        with self.assertRaises(ValueError):
            fetch_author_merged_count("BerriAI/litellm", "bad login!", "tok")


class AutoRolesBackfillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = get_config(minimal_roles_env(
            self.temp.name,
            BACKFILL_SINCE="2026-11-01T00:00:00Z",
            ROLES_BACKFILL_ON_BOOT="true",
        ))
        self.store = Store(self.config.roles_db_path)
        self.addCleanup(self.store.close)

    def test_backfill_auto_matches_posts_at_most_one_summary_and_marks_boot_done(self):
        self.assertTrue(should_run_boot_backfill(self.config, self.store))
        members = [
            _member("3001", "contrib-one", "Contrib One"),
            _member("3002", "contrib-two", "Contrib Two"),
        ]
        pulls = {
            201: {
                "number": 201, "merged": True, "merged_at": "2026-11-05T10:00:00Z", "title": "one",
                "user": {"login": "contrib-one", "id": 601, "type": "User"},
                "author_association": "FIRST_TIME_CONTRIBUTOR", "labels": [],
            },
            202: {
                "number": 202, "merged": True, "merged_at": "2026-11-06T10:00:00Z", "title": "two",
                "user": {"login": "contrib-two", "id": 602, "type": "User"},
                "author_association": "FIRST_TIME_CONTRIBUTOR", "labels": ["agents"],
            },
        }
        now = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role, \
             mock.patch("contributor_bot.rewards.post_message") as reward_post, \
             mock.patch("contributor_bot.roles_sync.post_message", return_value={"id": "999"}) as summary_post:
            res = run_roles_backfill(
                self.config,
                self.store,
                now=now,
                search_merged_fn=lambda repo, start, end, tok: {201, 202},
                pull_fetcher=lambda repo, num, tok: pulls[num],
                files_fetcher=lambda repo, num, tok: ["README.md"],
                count_fetcher=lambda repo, login, tok: 1,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
            )

        self.assertTrue(res.completed)
        self.assertEqual(res.pulls_stored, 2)
        self.assertEqual(res.contributors_granted, 2)
        self.assertEqual(res.roles_granted, 3)  # contrib-one: contributor; contrib-two: contributor + agent_builder
        self.assertTrue(res.summary_announced)
        # Never posts one announcement per person!
        reward_post.assert_not_called()
        # Posts AT MOST ONE summary line (with no user pings) to ROLE_ANNOUNCEMENTS_CHANNEL_ID!
        summary_post.assert_called_once()
        summary_payload = summary_post.call_args.args[2]
        self.assertEqual(summary_payload["allowed_mentions"], {"parse": []})
        self.assertIn("2 contributors", summary_payload["content"])
        self.assertIn("3 roles", summary_payload["content"])
        self.assertEqual(add_role.call_count, 3)

        # Boot backfill flag is now marked complete so restarts don't re-run.
        self.assertFalse(should_run_boot_backfill(self.config, self.store))
        # Backfilled grants were marked announced so subsequent polls never spam them.
        self.assertEqual(self.store.unannounced_grants("3001"), [])
        self.assertEqual(self.store.unannounced_grants("3002"), [])

    def test_backfill_rate_limit_stops_gracefully_and_resumes_later(self):
        members = [_member("3001", "contrib-one", "Contrib One")]
        # Two 28-day windows: Oct 1..Oct 29, Oct 29..Nov 20
        cfg = get_config(minimal_roles_env(self.temp.name, BACKFILL_SINCE="2026-10-01T00:00:00Z"))
        window_calls = {"n": 0}

        def flaky_search(repo, start, end, tok):
            window_calls["n"] += 1
            if window_calls["n"] == 1:
                return {301}
            raise TransientAPIError("rate limit")

        pull_301 = {
            "number": 301, "merged": True, "merged_at": "2026-10-10T10:00:00Z", "title": "first",
            "user": {"login": "contrib-one", "id": 601, "type": "User"},
            "author_association": "FIRST_TIME_CONTRIBUTOR", "labels": [],
        }
        now = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role, \
             mock.patch("contributor_bot.roles_sync.post_message", return_value={"id": "999"}) as summary_post:
            res = run_roles_backfill(
                cfg,
                self.store,
                now=now,
                search_merged_fn=flaky_search,
                pull_fetcher=lambda repo, num, tok: pull_301,
                files_fetcher=lambda repo, num, tok: ["README.md"],
                count_fetcher=lambda repo, login, tok: 1,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
            )

        self.assertTrue(res.rate_limited)
        self.assertFalse(res.completed)
        self.assertEqual(res.windows_completed, 1)
        self.assertEqual(res.windows_remaining, 1)
        # Still granted role for the PR ingested in window 1!
        self.assertEqual(res.contributors_granted, 1)
        self.assertEqual(add_role.call_count, 1)
        self.assertEqual(summary_post.call_count, 1)

    def test_main_run_loop_runs_daily_roles_after_summary_and_independently_on_summary_failure(self):
        from contributor_bot.main import _run_loop

        stop_event = threading.Event()
        events = []

        def failing_weekly(cfg, st, **kwargs):
            events.append("weekly_summary_failed")
            raise RuntimeError("summary boom")

        def fake_daily_roles(cfg, st, **kwargs):
            events.append("daily_roles")
            stop_event.set()

        def fake_boot(cfg, st, guild=None, loop=None):
            events.append("boot_backfill")

        def inline_thread(target, args=(), name=None, daemon=None):
            class Started:
                def start(self_inner):
                    target(*args)
            return Started()

        cfg = get_config(minimal_roles_env(
            self.temp.name,
            ANNOUNCEMENTS_ENABLED="false",
            MERGED_WEEKLY_ENABLED="true",
            ROLES_BACKFILL_ON_BOOT="true",
        ))
        with mock.patch("contributor_bot.main.log_weekly_boot_status"), \
             mock.patch("contributor_bot.main.post_weekly_if_due", side_effect=failing_weekly), \
             mock.patch("contributor_bot.main.run_daily_roles_if_due", side_effect=fake_daily_roles), \
             mock.patch("contributor_bot.main._run_boot_backfill", side_effect=fake_boot), \
             mock.patch("contributor_bot.main.threading.Thread", inline_thread):
            _run_loop(cfg, stop_event, store=self.store, gateway=None)

        self.assertEqual(events, ["weekly_summary_failed", "boot_backfill", "daily_roles"])


class DailyRolesSchedulerTests(unittest.TestCase):
    def setUp(self):
        reset_roles_daily_scheduler_state()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = get_config(minimal_roles_env(self.temp.name))
        self.store = Store(self.config.roles_db_path)
        self.addCleanup(self.store.close)

    def test_daily_schedule_runs_once_per_local_day_with_restart_dedupe(self):
        # Thursday 09:59 EST (14:59 UTC) -> before 10:00, skip
        thu_before = datetime(2026, 11, 26, 14, 59, tzinfo=UTC)
        # Thursday 10:00 EST (15:00 UTC) -> due!
        thu_ten = datetime(2026, 11, 26, 15, 0, tzinfo=UTC)
        # Thursday 21:42 EST (2026-11-27 02:42 UTC) -> still 2026-11-26 local, must skip after restart!
        thu_night = datetime(2026, 11, 27, 2, 42, tzinfo=UTC)
        # Friday 09:59 EST (14:59 UTC) -> new local day (2026-11-27), before 10:00 -> skip
        fri_before = datetime(2026, 11, 27, 14, 59, tzinfo=UTC)
        # Friday 10:00 EST (15:00 UTC) -> due for 2026-11-27!
        fri_ten = datetime(2026, 11, 27, 15, 0, tzinfo=UTC)

        poll_calls = []

        def fake_poll(cfg, st, **kwargs):
            poll_calls.append(kwargs["now"])
            return RolesPollResult(
                discovered=7,
                stored=5,
                skipped_existing=1,
                skipped_excluded=1,
                authors_evaluated=3,
                authors_matched=2,
                roles_granted=3,
            )

        self.assertIsNone(run_daily_roles_if_due(self.config, self.store, thu_before, poll_fn=fake_poll))
        self.assertEqual(poll_calls, [])

        with self.assertLogs(level="INFO") as logs:
            res = run_daily_roles_if_due(self.config, self.store, thu_ten, poll_fn=fake_poll)
        self.assertIsNotNone(res)
        self.assertEqual(len(poll_calls), 1)
        self.assertTrue(
            any("Contributor roles daily run for 2026-11-26: 5 PRs, 3 new grants" in line for line in logs.output)
        )
        info = self.store.roles_day_info("BerriAI/litellm", "2026-11-26")
        self.assertIsNotNone(info)
        self.assertEqual(info["via"], "scheduler")
        self.assertEqual(info["roles_granted"], 3)

        # Simulate restart/redeploy later the same local day (2026-11-27 02:42 UTC = 2026-11-26 21:42 EST).
        self.store.close()
        reset_roles_daily_scheduler_state()
        reopened = Store(self.config.roles_db_path)
        self.store = reopened

        with self.assertLogs(level="INFO") as boot_logs:
            log_roles_daily_boot_status(self.config, reopened, thu_night)
        joined = "\n".join(boot_logs.output)
        self.assertIn(
            "Contributor roles daily run 2026-11-26 already completed at 2026-11-26T15:00:00+00:00 via scheduler; skipping",
            joined,
        )
        self.assertNotIn("due now", joined)

        with self.assertLogs(level="INFO") as skip_logs:
            again = run_daily_roles_if_due(self.config, reopened, thu_night, poll_fn=fake_poll)
        self.assertIsNone(again)
        self.assertEqual(len(poll_calls), 1)
        self.assertTrue(
            any("Contributor roles daily run 2026-11-26 already completed" in line for line in skip_logs.output)
        )

        # Next day (2026-11-27): waits until 10:00 EST, then runs fresh!
        self.assertIsNone(run_daily_roles_if_due(self.config, reopened, fri_before, poll_fn=fake_poll))
        self.assertEqual(len(poll_calls), 1)
        res_next = run_daily_roles_if_due(self.config, reopened, fri_ten, poll_fn=fake_poll)
        self.assertIsNotNone(res_next)
        self.assertEqual(len(poll_calls), 2)
        self.assertTrue(reopened.roles_day_completed("BerriAI/litellm", "2026-11-27"))

    def test_daily_catch_up_after_downtime_and_rate_limit_retries_without_marking(self):
        # Bot was down at 10:00 EST and boots at 16:30 EST (21:30 UTC).
        late_afternoon = datetime(2026, 11, 27, 21, 30, tzinfo=UTC)
        self.assertIn("due now (will catch up)", roles_daily_schedule_status(self.config, self.store, late_afternoon))

        # 1) First attempt hits GitHub rate limit -> day stays unmarked!
        now_clock = {"t": 100.0}
        rate_limited_result = RolesPollResult(
            discovered=0, stored=0, skipped_existing=0, skipped_excluded=0,
            authors_evaluated=0, authors_matched=0, roles_granted=0, rate_limited=True,
        )
        res1 = run_daily_roles_if_due(
            self.config,
            self.store,
            late_afternoon,
            clock=lambda: now_clock["t"],
            poll_fn=lambda *a, **k: rate_limited_result,
        )
        self.assertTrue(res1.rate_limited)
        self.assertFalse(self.store.roles_day_completed("BerriAI/litellm", "2026-11-27"))

        # Within backoff window -> skipped cleanly.
        self.assertIsNone(
            run_daily_roles_if_due(
                self.config,
                self.store,
                late_afternoon,
                clock=lambda: now_clock["t"] + 30.0,
                poll_fn=mock.Mock(side_effect=AssertionError("should respect backoff")),
            )
        )

        # After backoff expires -> catches up and marks 2026-11-27!
        now_clock["t"] += 400.0
        ok_result = RolesPollResult(
            discovered=2, stored=2, skipped_existing=0, skipped_excluded=0,
            authors_evaluated=1, authors_matched=1, roles_granted=1,
        )
        res2 = run_daily_roles_if_due(
            self.config,
            self.store,
            late_afternoon,
            clock=lambda: now_clock["t"],
            poll_fn=lambda *a, **k: ok_result,
        )
        self.assertEqual(res2.roles_granted, 1)
        self.assertTrue(self.store.roles_day_completed("BerriAI/litellm", "2026-11-27"))

    def test_daily_roles_batches_announcements_chunks_and_posts_nothing_when_zero_grants(self):
        members = [
            _member("5001", "alice-dev", "Alice Dev"),
            _member("5002", "bob-dev", "Bob Dev"),
            _member("5003", "carol-dev", "Carol Dev"),
        ]
        pulls = {
            601: {
                "number": 601, "merged": True, "merged_at": "2026-11-26T18:00:00Z", "title": "alice pr",
                "user": {"login": "alice-dev", "id": 801, "type": "User"},
                "author_association": "FIRST_TIME_CONTRIBUTOR", "labels": [],
            },
            602: {
                "number": 602, "merged": True, "merged_at": "2026-11-26T19:00:00Z", "title": "bob agent pr",
                "user": {"login": "bob-dev", "id": 802, "type": "User"},
                "author_association": "FIRST_TIME_CONTRIBUTOR", "labels": ["agents"],
            },
            603: {
                "number": 603, "merged": True, "merged_at": "2026-11-27T12:00:00Z", "title": "carol pr",
                "user": {"login": "carol-dev", "id": 803, "type": "User"},
                "author_association": "FIRST_TIME_CONTRIBUTOR", "labels": [],
            },
        }
        day1 = datetime(2026, 11, 27, 15, 0, tzinfo=UTC)

        # 1) All 3 contributors fit in ONE batched message instead of 3 separate messages!
        posted = []
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role, \
             mock.patch("contributor_bot.rewards.post_message", side_effect=lambda ch, tok, body: (posted.append(body), {"id": "1"})[1]):
            res = run_daily_roles_if_due(
                self.config,
                self.store,
                day1,
                search_merged_fn=lambda repo, start, end, tok: set(pulls.keys()),
                pull_fetcher=lambda repo, num, tok: pulls[num],
                files_fetcher=lambda repo, num, tok: ["README.md"],
                count_fetcher=lambda repo, login, tok: 1,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
            )

        self.assertEqual(res.stored, 3)
        self.assertEqual(res.roles_granted, 4)  # alice(1), bob(2: contributor + agent_builder), carol(1)
        self.assertEqual(add_role.call_count, 4)
        self.assertEqual(len(posted), 1)
        lines = posted[0]["content"].splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("<@5001> just became a contributor 🎉", lines)
        self.assertIn("<@5002> just became a contributor and an agent builder 🎉", lines)
        self.assertIn("<@5003> just became a contributor 🎉", lines)
        self.assertEqual(set(posted[0]["allowed_mentions"]["users"]), {"5001", "5002", "5003"})
        self.assertEqual(self.store.all_unannounced_grants(), [])

        # 2) Next day with zero new grants (no new PRs) posts NOTHING and marks the day complete!
        day2 = datetime(2026, 11, 28, 15, 0, tzinfo=UTC)
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role_zero, \
             mock.patch("contributor_bot.rewards.post_message") as post_zero, \
             self.assertLogs(level="INFO") as zero_logs:
            res_zero = run_daily_roles_if_due(
                self.config,
                self.store,
                day2,
                search_merged_fn=lambda repo, start, end, tok: set(),
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
            )
        self.assertEqual(res_zero.stored, 0)
        self.assertEqual(res_zero.roles_granted, 0)
        add_role_zero.assert_not_called()
        post_zero.assert_not_called()
        self.assertTrue(self.store.roles_day_completed("BerriAI/litellm", "2026-11-28"))
        self.assertTrue(
            any("Contributor roles daily run for 2026-11-28: 0 PRs, 0 new grants" in line for line in zero_logs.output)
        )

        # 3) Verify chunking when combined message exceeds limit.
        chunk_store = Store(str(Path(self.temp.name) / "chunk.db"))
        self.addCleanup(chunk_store.close)
        reset_roles_daily_scheduler_state()
        chunked_posts = []
        with mock.patch("contributor_bot.rewards.add_member_role"), \
             mock.patch("contributor_bot.rewards.post_message", side_effect=lambda ch, tok, body: (chunked_posts.append(body), {"id": "2"})[1]):
            run_daily_roles_if_due(
                self.config,
                chunk_store,
                day1,
                announcement_chunk_limit=80,
                search_merged_fn=lambda repo, start, end, tok: set(pulls.keys()),
                pull_fetcher=lambda repo, num, tok: pulls[num],
                files_fetcher=lambda repo, num, tok: ["README.md"],
                count_fetcher=lambda repo, login, tok: 1,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
            )
        self.assertEqual(len(chunked_posts), 3)
        self.assertEqual(chunk_store.all_unannounced_grants(), [])

    def test_backfill_processes_newest_first_and_resumes_only_via_backfill(self):
        # 3 windows: Sep 1..Sep 29, Sep 29..Oct 27, Oct 27..Nov 20 (newest-first!)
        cfg = get_config(minimal_roles_env(self.temp.name, BACKFILL_SINCE="2026-09-01T00:00:00Z"))
        members = [
            _member("6001", "recent-nov", "Recent Nov"),
            _member("6002", "mid-oct", "Mid Oct"),
            _member("6003", "old-sep", "Old Sep"),
        ]
        pulls = {
            903: {
                "number": 903, "merged": True, "merged_at": "2026-11-15T10:00:00Z", "title": "nov pr",
                "user": {"login": "recent-nov", "id": 9003, "type": "User"},
                "author_association": "FIRST_TIME_CONTRIBUTOR", "labels": [],
            },
            902: {
                "number": 902, "merged": True, "merged_at": "2026-10-15T10:00:00Z", "title": "oct pr",
                "user": {"login": "mid-oct", "id": 9002, "type": "User"},
                "author_association": "FIRST_TIME_CONTRIBUTOR", "labels": [],
            },
            901: {
                "number": 901, "merged": True, "merged_at": "2026-09-15T10:00:00Z", "title": "sep pr",
                "user": {"login": "old-sep", "id": 9001, "type": "User"},
                "author_association": "FIRST_TIME_CONTRIBUTOR", "labels": [],
            },
        }
        seen_windows = []
        stop_after = {"n": 1}

        def paged_search(repo, start, end, tok):
            seen_windows.append((start.date().isoformat(), end.date().isoformat()))
            if len(seen_windows) > stop_after["n"]:
                raise TransientAPIError("rate limit")
            if start.month == 10 and end.month == 11:
                return {903}
            if start.month == 9 and end.month == 10:
                return {902}
            return {901}

        now1 = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role1, \
             mock.patch("contributor_bot.roles_sync.post_message", return_value={"id": "1"}):
            r1 = run_roles_backfill(
                cfg,
                self.store,
                now=now1,
                search_merged_fn=paged_search,
                pull_fetcher=lambda repo, num, tok: pulls[num],
                files_fetcher=lambda repo, num, tok: ["README.md"],
                count_fetcher=lambda repo, login, tok: 1,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
            )

        # Window 1 MUST be the newest window (Oct 27..Nov 20), granting recent-nov immediately!
        self.assertEqual(seen_windows[0], ("2026-10-27", "2026-11-20"))
        self.assertEqual(r1.total_windows, 3)
        self.assertEqual(r1.windows_done_total, 1)
        self.assertEqual(r1.windows_remaining, 2)
        self.assertFalse(r1.completed)
        self.assertTrue(has_pending_backfill(cfg, self.store))
        self.assertEqual([c.args[1] for c in add_role1.call_args_list], ["6001"])
        reply1 = format_backfill_result(r1)
        self.assertIn("1/3 windows done, run /roles-backfill again to continue", reply1)

        # Run 2 (even at a later timestamp!) resumes at Oct window (skipping completed Nov window).
        seen_windows.clear()
        stop_after["n"] = 1
        now2 = datetime(2026, 11, 20, 14, 0, tzinfo=UTC)
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role2, \
             mock.patch("contributor_bot.roles_sync.post_message", return_value={"id": "2"}):
            r2 = run_roles_backfill(
                cfg,
                self.store,
                now=now2,
                search_merged_fn=paged_search,
                pull_fetcher=lambda repo, num, tok: pulls[num],
                files_fetcher=lambda repo, num, tok: ["README.md"],
                count_fetcher=lambda repo, login, tok: 1,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
            )
        self.assertEqual(seen_windows[0], ("2026-09-29", "2026-10-27"))
        self.assertEqual(r2.windows_done_total, 2)
        self.assertEqual(r2.windows_remaining, 1)
        self.assertFalse(r2.completed)
        self.assertEqual([c.args[1] for c in add_role2.call_args_list], ["6002"])
        self.assertIn("2/3 windows done", format_backfill_result(r2))

        # A daily run in between MUST NOT touch historical backfill windows (7-day rolling window only).
        daily_windows = []
        daily_day = datetime(2026, 11, 21, 15, 0, tzinfo=UTC)
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role_daily, \
             mock.patch("contributor_bot.rewards.post_message") as post_daily:
            run_daily_roles_if_due(
                cfg,
                self.store,
                daily_day,
                search_merged_fn=lambda repo, start, end, tok: (daily_windows.append((start, end)), set())[1],
                count_fetcher=lambda repo, login, tok: 1,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
            )
        self.assertEqual(len(daily_windows), 1)
        self.assertEqual(daily_windows[0][1] - daily_windows[0][0], timedelta(days=7))
        self.assertTrue(has_pending_backfill(cfg, self.store))
        add_role_daily.assert_not_called()
        post_daily.assert_not_called()

        # Run 3: Mod runs /roles-backfill again to finish the final Sep window until completed=True!
        seen_windows.clear()
        stop_after["n"] = 10
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role3, \
             mock.patch("contributor_bot.roles_sync.post_message", return_value={"id": "3"}):
            r3 = run_roles_backfill(
                cfg,
                self.store,
                now=daily_day,
                search_merged_fn=paged_search,
                pull_fetcher=lambda repo, num, tok: pulls[num],
                files_fetcher=lambda repo, num, tok: ["README.md"],
                count_fetcher=lambda repo, login, tok: 1,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
            )
        self.assertEqual(seen_windows, [("2026-09-01", "2026-09-29")])
        self.assertTrue(r3.completed)
        self.assertEqual(r3.windows_done_total, 3)
        self.assertFalse(has_pending_backfill(cfg, self.store))
        self.assertEqual([c.args[1] for c in add_role3.call_args_list], ["6003"])

    def test_daily_roles_only_considers_rolling_seven_day_window(self):
        now = datetime(2026, 11, 27, 15, 0, tzinfo=UTC)
        members = [
            _member("7101", "recent-dev", "Recent Dev"),
            _member("7102", "old-stored-dev", "Old Stored Dev"),
            _member("7103", "old-search-dev", "Old Search Dev"),
            _member("7104", "ancient-match-dev", "Ancient Match Dev"),
            _member("7105", "late-joiner", "Late Joiner"),
        ]
        # Pre-store an old PR from 14 days ago (must NOT be matched/graded by daily run).
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=1401, github_user_id=7202,
            github_login="old-stored-dev", merged_at="2026-11-13T10:00:00+00:00",
            title="old stored pr", labels=[], files=["README.md"],
        )
        # Pre-store an old confirmed match from 20 days ago (must NOT be graded by daily run).
        self.store.save_confirmed_match(
            "ancient-match-dev", "7104", method="exact_login", confidence=0.96,
            matched_at="2026-11-07T10:00:00+00:00", github_user_id=7204,
        )
        # Pre-store a PR merged 3 days ago (within 7d) whose author just joined Discord today!
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=1405, github_user_id=7205,
            github_login="late-joiner", merged_at="2026-11-24T10:00:00+00:00",
            title="merged 3 days ago", labels=[], files=["README.md"],
        )

        pulls = {
            1402: {
                "number": 1402, "merged": True, "merged_at": "2026-11-25T10:00:00Z", "title": "recent pr",
                "user": {"login": "recent-dev", "id": 7201, "type": "User"},
                "author_association": "FIRST_TIME_CONTRIBUTOR", "labels": [],
            },
            # Even if search returned a PR merged 10 days ago, it's outside the 7-day rolling window!
            1403: {
                "number": 1403, "merged": True, "merged_at": "2026-11-17T10:00:00Z", "title": "10d old pr",
                "user": {"login": "old-search-dev", "id": 7203, "type": "User"},
                "author_association": "FIRST_TIME_CONTRIBUTOR", "labels": [],
            },
        }
        search_windows = []

        def capture_search(repo, start, end, tok):
            search_windows.append((start, end))
            return {1402, 1403}

        with mock.patch("contributor_bot.rewards.add_member_role") as add_role, \
             mock.patch("contributor_bot.rewards.post_message", return_value={"id": "99"}):
            res = run_daily_roles_if_due(
                self.config,
                self.store,
                now,
                search_merged_fn=capture_search,
                pull_fetcher=lambda repo, num, tok: pulls[num],
                files_fetcher=lambda repo, num, tok: ["README.md"],
                count_fetcher=lambda repo, login, tok: 1,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
            )

        self.assertEqual(search_windows, [(now - timedelta(days=7), now)])
        # ONLY recent-dev (merged 2d ago) and late-joiner (merged 3d ago) are evaluated & granted!
        self.assertEqual(res.authors_evaluated, 2)
        self.assertEqual(res.authors_matched, 2)
        granted_users = {call.args[1] for call in add_role.call_args_list}
        self.assertEqual(granted_users, {"7101", "7105"})

    def test_weekly_summary_matches_are_persisted_and_reused_without_llm_by_daily_roles(self):
        from contributor_bot.matcher import MatchDetail
        from contributor_bot.weekly import WEEKLY_HEADER, WeeklyReport, post_weekly_if_due, reset_weekly_scheduler_state

        reset_weekly_scheduler_state()
        fri_ten = datetime(2026, 11, 27, 15, 0, tzinfo=UTC)

        fake_report = WeeklyReport(
            chunks=[f"{WEEKLY_HEADER}\n\n<@7001> 🎉 - cool fix (<https://github.com/BerriAI/litellm/pull/1234>)"],
            matched_count=1,
            skipped_count=0,
            since=datetime(2026, 11, 20, 15, 0, tzinfo=UTC),
            match_details=(
                MatchDetail(
                    github_login="friday-hero",
                    discord_id="7001",
                    method="jev",
                    confidence=0.95,
                    discord_username="fhero",
                    pr_count=1,
                    github_user_id=7701,
                ),
            ),
        )
        with mock.patch("contributor_bot.weekly.build_weekly_report", return_value=fake_report):
            post_weekly_if_due(
                self.config,
                self.store,
                fri_ten,
                post=lambda ch, tok, body: {"id": "1"},
                messages_fetcher=lambda *_a, **_k: [],
            )

        # Weekly post persisted the match into github_discord_matches with github_user_id!
        saved = self.store.get_confirmed_match("friday-hero")
        self.assertIsNotNone(saved)
        self.assertEqual(saved["discord_id"], "7001")
        self.assertEqual(saved["github_user_id"], 7701)

        # Daily roles run immediately after: even with 0 newly polled PRs and NO LLM/member access,
        # grants Contributor directly from github_discord_matches and announces once!
        posted = []
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role, \
             mock.patch("contributor_bot.rewards.post_message", side_effect=lambda ch, tok, body: (posted.append(body), {"id": "2"})[1]), \
             mock.patch("contributor_bot.roles_sync.match_community_contributors_with_diagnostics", side_effect=AssertionError("must reuse stored match without LLM")):
            res = run_daily_roles_if_due(
                self.config,
                self.store,
                fri_ten,
                search_merged_fn=lambda repo, start, end, tok: set(),
                count_fetcher=lambda repo, login, tok: 1,
                org_members_fetcher=lambda org, tok: frozenset(),
            )
        self.assertEqual(res.roles_granted, 1)
        self.assertEqual(add_role.call_args.args[1], "7001")
        self.assertEqual(add_role.call_args.args[2], "1001")  # Contributor role
        self.assertEqual(len(posted), 1)
        self.assertEqual(posted[0]["content"], "<@7001> just became a contributor 🎉")

    def test_enrichment_deadline_caches_progress_and_continues_pending_authors_next_run(self):
        cfg = get_config(minimal_roles_env(self.temp.name, BACKFILL_SINCE="2026-11-01T00:00:00Z"))
        members = [
            _member("8001", "zoe_discord", "Zoe Alpha"),
            _member("8002", "liam_discord", "Liam Beta"),
        ]
        # Pre-store 2 authors whose logins don't match Discord usernames (need GitHub profile name!).
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=801, github_user_id=8801,
            github_login="gh-alpha", merged_at="2026-11-18T10:00:00+00:00",
            title="pr 1", labels=[], files=["README.md"],
        )
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=802, github_user_id=8802,
            github_login="gh-beta", merged_at="2026-11-19T10:00:00+00:00",
            title="pr 2", labels=[], files=["README.md"],
        )

        fetched_users = []
        now = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)

        # Run 1: Simulate rate limit hitting on the 2nd uncached user after caching the 1st!
        def flaky_github_get(url, token, **kwargs):
            if url.endswith("/users/gh-beta"):
                fetched_users.append("gh-beta")
                return {"login": "gh-beta", "name": "Liam Beta", "bio": "", "blog": "", "twitter_username": ""}
            if url.endswith("/users/gh-alpha"):
                fetched_users.append("gh-alpha")
                raise TransientAPIError("rate limit")
            return {}

        with mock.patch("contributor_bot.rewards.add_member_role") as add_role1, \
             mock.patch("contributor_bot.roles_sync.post_message", return_value={"id": "1"}):
            r1 = run_roles_backfill(
                cfg,
                self.store,
                now=now,
                search_merged_fn=lambda repo, start, end, tok: set(),
                count_fetcher=lambda repo, login, tok: 1,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
                github_getter=flaky_github_get,
            )

        # Newest author (gh-beta, merged Nov 19) was enriched, cached, matched, and granted!
        # Older author (gh-alpha) was deferred -> completed=False so next run picks them up!
        self.assertEqual(fetched_users, ["gh-beta", "gh-alpha"])
        self.assertEqual(r1.enrichment_deferred, 1)
        self.assertFalse(r1.completed)
        self.assertEqual([c.args[1] for c in add_role1.call_args_list], ["8002"])
        self.assertIsNotNone(self.store.get_cached_github_user("gh-beta"))

        # Run 2: gh-beta is ALREADY matched & cached (0 GitHub calls for gh-beta!); gh-alpha now succeeds!
        fetched_users.clear()

        def healthy_github_get(url, token, **kwargs):
            if url.endswith("/users/gh-alpha"):
                fetched_users.append("gh-alpha")
                return {"login": "gh-alpha", "name": "Zoe Alpha", "bio": "", "blog": "", "twitter_username": ""}
            raise AssertionError(f"unexpected re-fetch: {url}")

        with mock.patch("contributor_bot.rewards.add_member_role") as add_role2, \
             mock.patch("contributor_bot.roles_sync.post_message", return_value={"id": "2"}):
            r2 = run_roles_backfill(
                cfg,
                self.store,
                now=now,
                search_merged_fn=lambda repo, start, end, tok: set(),
                count_fetcher=lambda repo, login, tok: 1,
                org_members_fetcher=lambda org, tok: frozenset(),
                members=members,
                github_getter=healthy_github_get,
            )

        self.assertEqual(fetched_users, ["gh-alpha"])
        self.assertEqual(r2.enrichment_deferred, 0)
        self.assertTrue(r2.completed)
        self.assertEqual([c.args[1] for c in add_role2.call_args_list], ["8001"])

    def test_priority_pool_matches_seven_day_joiners_before_full_fallback(self):
        from types import SimpleNamespace
        from contributor_bot.matcher import (
            match_community_contributors_with_diagnostics,
            member_from_raw,
            reset_guild_members_cache,
        )

        reset_guild_members_cache()
        self.addCleanup(reset_guild_members_cache)
        now = datetime(2026, 11, 27, 15, 0, tzinfo=UTC)

        # Verify 7-day member.joined_at cutoff from both REST dicts and discord.Member objects.
        recent_john = member_from_raw({
            "user": {"id": "9001", "username": "john_recent", "global_name": "John Smith"},
            "joined_at": "2026-11-24T12:00:00+00:00",  # 3 days ago -> IN 7-day priority pool
        })
        mid_john = member_from_raw(SimpleNamespace(
            id=9002,
            name="john_older",
            global_name="John Smith",
            nick=None,
            display_name="John Smith",
            bot=False,
            joined_at=datetime(2026, 11, 12, 12, 0, tzinfo=UTC),  # 15 days ago (>7d) -> full pool only!
        ))
        recent_mira = member_from_raw({
            "user": {"id": "9003", "username": "mira_codes", "global_name": "Mira K"},
            "joined_at": "2026-11-22T12:00:00+00:00",  # 5 days ago -> IN 7-day priority pool
        })
        old_lookalike = member_from_raw({
            "user": {"id": "9004", "username": "mira_coder", "global_name": "Mira L"},
            "joined_at": "2026-11-05T12:00:00+00:00",  # 22 days ago (>7d) -> full pool only
        })
        veteran = member_from_raw({
            "user": {"id": "9005", "username": "veteran_dev", "global_name": "Veteran Dev"},
            "joined_at": "2023-08-01T12:00:00+00:00",  # full pool only
        })
        members = [recent_john, mid_john, recent_mira, old_lookalike, veteran]

        cfg = get_config(minimal_roles_env(self.temp.name, TYPESAFE_API_KEY="jev-test-key"))

        groups = [
            # 1) "John Smith" would be an ambiguous tie in full pool, but uniquely matches recent_john in 7d pool (0 LLM calls!).
            {"login": "gh-jsmith", "user_id": 9101, "pulls": []},
            # 2) Fuzzy match to 5-day joiner 9003 -> Jev sees ONLY 9003 from 7d pool, never calls Jev on full pool!
            {"login": "miracode", "user_id": 9102, "pulls": []},
            # 3) No match in 7d pool -> falls back to full pool and matches veteran (9005)!
            {"login": "veteran-dev", "user_id": 9103, "pulls": []},
        ]

        profiles = {
            "gh-jsmith": {"login": "gh-jsmith", "name": "John Smith", "bio": "", "blog": "", "twitter_username": ""},
            "miracode": {"login": "miracode", "name": "Mira", "bio": "", "blog": "", "twitter_username": ""},
            "veteran-dev": {"login": "veteran-dev", "name": "Veteran Dev", "bio": "", "blog": "", "twitter_username": ""},
        }

        def fake_github_get(url, token, **kwargs):
            login = url.rsplit("/", 1)[-1]
            return profiles.get(login, {})

        jev_calls = []

        def fake_jev(body_bytes):
            import json as _json
            payload = _json.loads(body_bytes.decode("utf-8"))
            state = _json.loads(payload["state"])
            pairs = state["candidate_pairs"]
            jev_calls.append([(p["github"]["login"], p["discord"]["id"]) for p in pairs])
            answers = {}
            for p in pairs:
                score = 0.93 if (p["github"]["login"] == "miracode" and p["discord"]["id"] == "9003") else 0.1
                answers[p["pair"]] = {"type": "noul", "noul": score}
            return {"model": "jev-1.14.0", "answers": answers}

        with mock.patch.dict("os.environ", {"TYPESAFE_API_KEY": "jev-test-key"}, clear=False), \
             self.assertLogs(level="INFO") as logs:
            matched, skipped, _details, _diag = match_community_contributors_with_diagnostics(
                groups,
                cfg,
                self.store,
                now=now,
                members=members,
                github_getter=fake_github_get,
                jev_transport=fake_jev,
            )

        self.assertEqual(skipped, 0)
        by_login = {g["login"]: g["discord_id"] for g in matched}
        self.assertEqual(by_login, {
            "gh-jsmith": "9001",    # matched recent_john uniquely in 7-day priority pool!
            "miracode": "9003",     # matched 5-day joiner via priority-pool Jev!
            "veteran-dev": "9005",  # fell back to full member list!
        })
        # Exact pool-size log emitted (2 joined within 7d; 15d-old & 22d-old excluded from recent pool)!
        self.assertTrue(any("match pool: recent=2 full=5" in line for line in logs.output))
        # Jev was called ONLY ONCE (on the 7d recent pool for miracode -> 9003), NEVER on the full pool (9002 or 9004)!
        self.assertEqual(jev_calls, [[("miracode", "9003")]])


if __name__ == "__main__":
    unittest.main()
