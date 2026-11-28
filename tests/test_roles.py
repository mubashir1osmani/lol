import tempfile
import unittest
from pathlib import Path
from unittest import mock

from contributor_bot import backfill
from contributor_bot.api import TransientAPIError, fetch_merged_numbers, wait_for_github_budget
from contributor_bot.config import get_config
from contributor_bot.present import announcement_text, my_prs_text, review_status
from contributor_bot.rewards import apply_roles, assign_role
from contributor_bot.roles import MergedPull, earned_role_keys
from contributor_bot.schedule import parse_time
from contributor_bot.state import load_state, save_state
from contributor_bot.store import Store
from tests.test_weekly import THURSDAY, FRIDAY


def at(value):
    return parse_time(value)


def pull(number, when, files=(), labels=(), bot=False):
    return MergedPull(number, at(when), tuple(labels), tuple(files), bot)


class RoleRuleTests(unittest.TestCase):
    def test_thresholds_paths_labels_and_bots(self):
        january = ["2026-01-05T00:00:00Z", "2026-01-15T00:00:00Z", "2026-01-31T23:00:00Z"]
        humans = [pull(index + 1, when) for index, when in enumerate(january)]
        self.assertEqual(earned_role_keys([]), [])
        self.assertEqual(earned_role_keys(humans[:1]), ["contributor"])
        self.assertEqual(earned_role_keys(humans), ["contributor", "oss_champion"])
        # Two in January and one just after midnight UTC is the next month.
        split = humans[:2] + [pull(4, "2026-01-31T23:30:00-05:00")]
        self.assertEqual(earned_role_keys(split), ["contributor"])
        self.assertEqual(parse_time("2026-01-31T23:30:00-05:00").month, 2)
        spread = [pull(index + 1, f"2026-{(index % 9) + 1:02d}-02T00:00:00Z") for index in range(9)]
        self.assertNotIn("oss_champion", earned_role_keys(spread))
        self.assertIn("oss_champion", earned_role_keys(spread + [pull(10, "2025-12-01T00:00:00Z")]))
        bots = [pull(index + 1, january[index % 3], files=("litellm/llms/openai.py",), bot=True) for index in range(10)]
        self.assertEqual(earned_role_keys(bots), [])

        agent_files = [
            "litellm/proxy/mcp_server.py",
            "litellm/a2a_protocol/client.py",
            "docs/agents.md",
        ]
        for path in agent_files:
            with self.subTest(path=path):
                self.assertIn("agent_builder", earned_role_keys([pull(1, january[0], files=(path,))]))
        self.assertIn("agent_builder", earned_role_keys([pull(1, january[0], labels=("Agents",))]))
        self.assertNotIn("agent_builder", earned_role_keys([
            pull(1, january[0], files=("litellm/integrations/example.py",), labels=("agent",)),
        ]))
        self.assertNotIn("agent_builder", earned_role_keys([
            pull(1, january[0], files=("litellm/llms/openai/chat/gpt_transformation.py",)),
        ]))

        provider = [
            pull(1, january[0], files=("litellm/llms/openai.py", "litellm/llms/anthropic.py")),
            pull(2, january[1], files=("litellm/llms/gemini.py",)),
            pull(3, january[2], files=("README.md",)),
        ]
        self.assertNotIn("provider_wizard", earned_role_keys(provider))
        self.assertIn("provider_wizard", earned_role_keys(provider[:2] + [
            pull(3, january[2], files=("litellm/llms/cohere.py",)),
        ]))
        self.assertNotIn("provider_wizard", earned_role_keys([
            pull(index + 1, january[index], files=("litellm/llms_extra/foo.py",)) for index in range(3)
        ]))

        docs = [pull(index + 1, january[index], files=(f"docs/page-{index}.md",)) for index in range(3)]
        self.assertEqual(earned_role_keys(docs), ["contributor", "oss_champion"])
        # One pull request can earn more than one path role, and still counts once.
        both = pull(1, january[0], files=("docs/x.md", "litellm/llms/openai.py"), labels=("agents",))
        self.assertEqual(
            earned_role_keys([
                both, docs[1], docs[2], provider[1],
                pull(5, january[0], files=("litellm/llms/cohere.py",)),
            ]),
            ["contributor", "oss_champion", "agent_builder", "provider_wizard"],
        )


class CopyTests(unittest.TestCase):
    def test_announcements_are_lowercase_and_mention_only_the_user(self):
        self.assertEqual(
            announcement_text("111", ["OSS Champion"]),
            "<@111> just became an oss champion 🎉",
        )
        self.assertEqual(
            announcement_text("111", ["Contributor", "Agent Builder"], 42),
            "<@111> just shipped #42 and became a contributor and an agent builder 🎉",
        )
        self.assertEqual(announcement_text("111", [], 7), "<@111> just shipped #7 🎉")
        self.assertIn("roles: contributor, provider wizard", my_prs_text(
            "octocat",
            [{"number": 8, "title": "Fix @everyone\nnow", "review_status": "approved"}],
            1, 3, ["Contributor", "Provider Wizard"],
        ))
        self.assertNotIn("@everyone", my_prs_text(
            "octocat", [{"number": 8, "title": "@everyone", "review_status": "approved"}], 1, 1, [],
        ))

    def test_review_status_uses_the_latest_review_per_person(self):
        self.assertEqual(review_status({"draft": True}, []), "draft")
        reviews = [
            {"user": {"id": 1}, "state": "CHANGES_REQUESTED"},
            {"user": {"id": 1}, "state": "APPROVED"},
            {"user": {"id": 2}, "state": "COMMENTED"},
        ]
        self.assertEqual(review_status({}, reviews), "approved")
        reviews.append({"user": {"id": 3}, "state": "CHANGES_REQUESTED"})
        self.assertEqual(review_status({}, reviews), "changes requested")
        reviews.append({"user": {"id": 3}, "state": "DISMISSED"})
        self.assertEqual(review_status({}, reviews), "approved")
        self.assertEqual(review_status({}, []), "waiting on review")


def roles_env(directory, **extra):
    return {
        "DISCORD_BOT_TOKEN": "discord-secret",
        "OSS_CONTRIBUTIONS_CHANNEL_ID": "123",
        "ANNOUNCEMENTS_CHANNEL_ID": "456",
        "CONTRIBUTOR_ROLES_ENABLED": "true",
        "DISCORD_GUILD_ID": "333",
        "DISCORD_APPLICATION_ID": "444",
        "DISCORD_PUBLIC_KEY": "ab" * 32,
        "GITHUB_OAUTH_CLIENT_ID": "Iv1.client",
        "GITHUB_OAUTH_CLIENT_SECRET": "oauth-secret-value",
        "GITHUB_WEBHOOK_SECRET": "webhook-secret-value",
        "GITHUB_OAUTH_REDIRECT_URI": "https://bot.example/github/oauth/callback",
        "ROLE_CONTRIBUTOR_ID": "1001",
        "ROLE_OSS_CHAMPION_ID": "1002",
        "ROLE_AGENT_BUILDER_ID": "1003",
        "ROLE_PROVIDER_WIZARD_ID": "1004",
        "ROLE_ANNOUNCEMENTS_CHANNEL_ID": "789",
        "GITHUB_TOKEN": "github-secret",
        "ROLES_DB_PATH": str(Path(directory) / "roles.db"),
        "STATE_FILE": str(Path(directory) / "state.json"),
        **extra,
    }


class RoleConfigTests(unittest.TestCase):
    def test_roles_are_off_unless_enabled_and_do_not_move_weekly_state(self):
        base_env = {
            "DISCORD_BOT_TOKEN": "discord-secret",
            "OSS_CONTRIBUTIONS_CHANNEL_ID": "123",
            "ANNOUNCEMENTS_CHANNEL_ID": "456",
        }
        base = get_config(base_env)
        self.assertFalse(base.contributor_roles_enabled)
        self.assertEqual(base.role_ids, {})
        with tempfile.TemporaryDirectory() as directory:
            enabled = get_config(roles_env(directory, ROLE_OSS_CHAMPION_NAME="Ship Captain"))
        self.assertEqual(base.scope, enabled.scope)
        self.assertEqual(enabled.role_names["oss_champion"], "Ship Captain")
        self.assertEqual(enabled.role_ids["contributor"], "1001")
        text = repr(enabled)
        self.assertNotIn("discord-secret", text)
        self.assertNotIn("github-secret", text)
        self.assertNotIn("oauth-secret-value", text)
        self.assertNotIn("webhook-secret-value", text)
        self.assertNotIn(enabled.discord_public_key, text)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "state.json")
            save_state(path, load_state(get_config(base_env | {"STATE_FILE": path})))
            self.assertEqual(load_state(get_config(roles_env(directory))).get("version"), 2)

    def test_enabled_roles_require_their_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            env = roles_env(directory)
            for key in ("DISCORD_GUILD_ID", "ROLE_PROVIDER_WIZARD_ID", "GITHUB_OAUTH_REDIRECT_URI"):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    broken = dict(env)
                    broken.pop(key)
                    get_config(broken)
            # Without any OAuth, webhook secret, public key, or application id,
            # contributor roles still validate cleanly for auto-matched polling.
            minimal = dict(env)
            for optional_key in (
                "DISCORD_APPLICATION_ID",
                "DISCORD_PUBLIC_KEY",
                "GITHUB_WEBHOOK_SECRET",
                "GITHUB_OAUTH_CLIENT_ID",
                "GITHUB_OAUTH_CLIENT_SECRET",
                "GITHUB_OAUTH_REDIRECT_URI",
            ):
                minimal.pop(optional_key, None)
            minimal_cfg = get_config(minimal)
            self.assertTrue(minimal_cfg.contributor_roles_enabled)
            self.assertFalse(minimal_cfg.github_oauth_enabled)
            self.assertEqual(minimal_cfg.roles_schedule, "weekly")
            self.assertFalse(minimal_cfg.roles_backfill_on_boot)
            self.assertEqual(
                get_config(minimal | {"ROLES_SCHEDULE": "manual"}).roles_schedule,
                "manual",
            )
            with self.assertRaisesRegex(ValueError, "ROLES_SCHEDULE"):
                get_config(minimal | {"ROLES_SCHEDULE": "hourly"})
            with self.assertRaisesRegex(ValueError, "distinct"):
                get_config(env | {"ROLE_PROVIDER_WIZARD_ID": "1001"})
            with self.assertRaisesRegex(ValueError, "DISCORD_INVITE_URL"):
                get_config(env | {"PR_COMMENT_INVITES_ENABLED": "true"})
            invites = get_config(env | {
                "PR_COMMENT_INVITES_ENABLED": "true",
                "DISCORD_INVITE_URL": "https://discord.gg/litellm",
            })
            self.assertTrue(invites.pr_comment_invites_enabled)
            self.assertFalse(get_config(env | {"MERGE_SHOUTOUTS_ENABLED": "false"}).merge_shoutouts_enabled)

    def test_legacy_docs_hero_env_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            env = roles_env(directory) | {
                "ROLE_DOCS_HERO_ID": "not-a-snowflake",
                "ROLE_DOCS_HERO_NAME": "Docs Hero\n@everyone",
            }
            config = get_config(env)
        self.assertNotIn("docs_hero", config.role_ids)
        self.assertNotIn("docs_hero", config.role_names)
        self.assertEqual(
            set(config.role_ids),
            {"contributor", "oss_champion", "agent_builder", "provider_wizard"},
        )


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(str(Path(self.directory.name) / "roles.db"))
        self.addCleanup(self.store.close)

    def test_database_is_private_and_links_are_one_to_one(self):
        self.assertEqual(Path(self.store.path).stat().st_mode & 0o777, 0o600)
        self.assertIsNone(self.store.link_account("111", "OctoCat", 5, "2026-11-01T00:00:00+00:00"))
        self.assertIsNone(self.store.link_account("111", "octocat", 5, "2026-11-02T00:00:00+00:00"))
        self.assertEqual(self.store.link_for_discord("111")["github_login"], "octocat")
        self.assertIn("another discord", self.store.link_account("222", "octocat", 5, "2026-11-03T00:00:00+00:00"))
        self.assertIn("unlink", self.store.link_account("111", "other", 6, "2026-11-03T00:00:00+00:00"))
        self.assertTrue(self.store.unlink("111"))
        self.assertFalse(self.store.unlink("111"))

    def test_oauth_state_is_single_use_and_expires(self):
        state = "a" * 32
        self.store.put_oauth_state(state, "111", "2026-11-01T00:10:00+00:00", "2026-11-01T00:00:00+00:00")
        self.assertEqual(self.store.pop_oauth_state(state, "2026-11-01T00:05:00+00:00"), "111")
        self.assertIsNone(self.store.pop_oauth_state(state, "2026-11-01T00:05:00+00:00"))
        self.store.put_oauth_state(state, "111", "2026-11-01T00:10:00+00:00", "2026-11-01T00:00:00+00:00")
        self.assertIsNone(self.store.pop_oauth_state(state, "2026-11-01T00:10:00+00:00"))
        self.assertIsNone(self.store.pop_oauth_state("../etc/passwd", "2026-11-01T00:00:00+00:00"))


class GrantTests(unittest.TestCase):
    def test_missing_permissions_are_logged_and_not_announced(self):
        with tempfile.TemporaryDirectory() as directory:
            config = get_config(roles_env(directory))
            store = Store(config.roles_db_path)
            self.addCleanup(store.close)
            store.link_account("111", "octocat", 5, "2026-11-01T00:00:00+00:00")
            store.upsert_pull(
                repository="BerriAI/litellm", number=7, github_user_id=5, github_login="octocat",
                merged_at="2026-11-02T03:04:05+00:00", title="docs", labels=[], files=["docs/readme.md"],
            )
            with mock.patch("contributor_bot.rewards.add_member_role", side_effect=RuntimeError("discord.com returned HTTP 403")) as add, mock.patch("contributor_bot.rewards.post_message") as post, self.assertLogs(level="ERROR") as logs:
                apply_roles(config, store, "111", 5, pr_number=7, shoutout=True)
            add.assert_called_once()
            post.assert_not_called()
            self.assertEqual(store.grant_keys("111"), set())
            self.assertIn("Manage Roles", "\n".join(logs.output))
            self.assertIn("above", "\n".join(logs.output))
            with mock.patch("contributor_bot.rewards.add_member_role") as add, mock.patch("contributor_bot.rewards.post_message", return_value={"id": "1", "channel_id": "789"}) as post:
                apply_roles(config, store, "111", 5, pr_number=7, shoutout=True)
            self.assertEqual(add.call_args.args[2], "1001")
            self.assertIn("just shipped #7 and became a contributor", post.call_args.args[2]["content"])
            self.assertEqual(post.call_args.args[2]["allowed_mentions"]["users"], ["111"])
            with mock.patch("contributor_bot.rewards.post_message") as post:
                apply_roles(config, store, "111", 5, pr_number=7, shoutout=True)
            post.assert_not_called()

    def test_assign_role_reports_unknown_member(self):
        with tempfile.TemporaryDirectory() as directory:
            config = get_config(roles_env(directory))
            with mock.patch("contributor_bot.rewards.add_member_role", side_effect=RuntimeError("discord.com returned HTTP 404")), self.assertLogs(level="ERROR") as logs:
                self.assertFalse(assign_role(config, "111", "contributor"))
            self.assertIn("404", "\n".join(logs.output))


class BackfillTests(unittest.TestCase):
    def test_backfill_is_resumable_and_skips_bots(self):
        with tempfile.TemporaryDirectory() as directory:
            config = get_config(roles_env(directory, BACKFILL_SINCE="2026-11-01T00:00:00Z"))
            store = Store(config.roles_db_path)
            store.link_account("111", "octocat", 5, "2026-11-01T00:00:00+00:00")
            store.close()

            class Frozen:
                @staticmethod
                def now(tz=None):
                    return at("2026-11-20T15:00:00Z")

            def pull_response(repository, number, token):
                if number == 2:
                    return {"number": 2, "merged": True, "merged_at": "2026-11-03T00:00:00Z",
                            "title": "bot", "user": {"login": "dependabot[bot]", "id": 9, "type": "Bot"}, "labels": []}
                return {"number": number, "merged": True, "merged_at": "2026-11-02T00:00:00Z",
                        "title": "docs", "user": {"login": "octocat", "id": 5, "type": "User"},
                        "labels": [{"name": "docs"}]}

            with mock.patch.object(backfill, "datetime", Frozen), mock.patch.object(backfill, "fetch_merged_numbers", return_value={1, 2}) as search, mock.patch.object(backfill, "fetch_pull_request", side_effect=pull_response), mock.patch.object(backfill, "fetch_pull_files", return_value=["docs/readme.md"]), mock.patch.object(backfill.time, "sleep"), mock.patch("contributor_bot.rewards.add_member_role") as add, mock.patch("contributor_bot.rewards.post_message", return_value={"id": "1", "channel_id": "789"}) as post:
                backfill.backfill_merged_pulls(config)
                self.assertEqual(search.call_count, 1)
                backfill.backfill_merged_pulls(config)
                self.assertEqual(search.call_count, 1)
            self.assertEqual(add.call_count, 1)
            self.assertEqual(post.call_count, 1)
            saved = Store(config.roles_db_path)
            self.addCleanup(saved.close)
            self.assertEqual(saved.pull_count("BerriAI/litellm"), 1)
            self.assertEqual(saved.get_pull("BerriAI/litellm", 1)["files"], ["docs/readme.md"])
            self.assertIsNone(saved.get_pull("BerriAI/litellm", 2))
            self.assertEqual(saved.grant_keys("111"), {"contributor"})


class SearchPaceTests(unittest.TestCase):
    def test_author_filter_is_rejected_before_the_network_and_low_budget_waits(self):
        with mock.patch("contributor_bot.api._request_json") as request, self.assertRaises(ValueError):
            fetch_merged_numbers("BerriAI/litellm", THURSDAY, FRIDAY, "token", author="octo cat")
        request.assert_not_called()
        payload = {"total_count": 0, "incomplete_results": False, "items": []}
        with mock.patch("contributor_bot.api.time.sleep"), mock.patch("contributor_bot.api._request_json", return_value=payload) as request:
            self.assertEqual(fetch_merged_numbers("BerriAI/litellm", THURSDAY, FRIDAY, "token", author="octocat"), set())
        self.assertIn("author%3Aoctocat", request.call_args.args[0].full_url)
        slept = []
        wait_for_github_budget(
            {"X-RateLimit-Remaining": "1", "X-RateLimit-Reset": "200"},
            now=100, sleep=slept.append,
        )
        self.assertEqual(slept, [90.25])
        wait_for_github_budget({"X-RateLimit-Remaining": "40"}, now=100, sleep=slept.append)
        self.assertEqual(slept, [90.25])


if __name__ == "__main__":
    unittest.main()
