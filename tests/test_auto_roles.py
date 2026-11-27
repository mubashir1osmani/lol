import os
import tempfile
import threading
import unittest
from datetime import datetime
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
    poll_merged_prs_for_roles,
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

    def test_main_run_loop_triggers_roles_poll_and_boot_backfill(self):
        from contributor_bot.main import _run_loop

        stop_event = threading.Event()
        poll_calls = []
        boot_calls = []

        def fake_poll(cfg, st, **kwargs):
            poll_calls.append(kwargs)
            stop_event.set()

        def fake_boot(cfg, st, guild=None, loop=None):
            boot_calls.append((cfg.github_repository, st))

        def inline_thread(target, args=(), name=None, daemon=None):
            class Started:
                def start(self_inner):
                    target(*args)
            return Started()

        cfg = get_config(minimal_roles_env(
            self.temp.name,
            ANNOUNCEMENTS_ENABLED="false",
            MERGED_WEEKLY_ENABLED="false",
            ROLES_BACKFILL_ON_BOOT="true",
        ))
        with mock.patch("contributor_bot.main.poll_merged_prs_for_roles", side_effect=fake_poll), \
             mock.patch("contributor_bot.main._run_boot_backfill", side_effect=fake_boot), \
             mock.patch("contributor_bot.main.threading.Thread", inline_thread):
            _run_loop(cfg, stop_event, store=self.store, gateway=None)

        self.assertEqual(len(poll_calls), 1)
        self.assertEqual(len(boot_calls), 1)


if __name__ == "__main__":
    unittest.main()
