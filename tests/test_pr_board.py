import json
import os
import tempfile
import threading
import unittest
from datetime import datetime
from unittest import mock
from urllib.request import Request

from contributor_bot.api import _graphql_query_allowed, github_graphql
from contributor_bot.config import Config, get_config
from contributor_bot.gateway import commands_for_guild
from contributor_bot.interactions import PR_BOARD_SLASH_COMMANDS, handle_interaction
from contributor_bot.main import _run_loop
from contributor_bot.pr_board import (
    STATUS_APPROVED,
    STATUS_CHANGES_REQUESTED,
    STATUS_CLOSED,
    STATUS_IN_REVIEW,
    STATUS_MERGED,
    STATUS_OPEN,
    classify_pr_status,
    is_community_pr,
    reset_pr_board_cache,
    resolve_project_metadata,
    sync_pr_board,
)
from contributor_bot.schedule import UTC
from contributor_bot.store import Store


def _pr_node(
    number: int,
    *,
    author: str = "community-dev",
    author_type: str = "User",
    association: str = "FIRST_TIME_CONTRIBUTOR",
    state: str = "OPEN",
    review_decision: str | None = None,
    reviews: list[dict] | None = None,
    comments: list[dict] | None = None,
    updated_at: str = "2026-11-27T18:00:00Z",
) -> dict:
    return {
        "id": f"PR_node_{number}",
        "number": number,
        "title": f"Fix issue #{number}",
        "url": f"https://github.com/BerriAI/litellm/pull/{number}",
        "state": state,
        "updatedAt": updated_at,
        "authorAssociation": association,
        "reviewDecision": review_decision,
        "author": {"__typename": author_type, "login": author},
        "reviews": {"nodes": reviews or []},
        "comments": {"nodes": comments or []},
    }


def _project_payload(
    *,
    project_id: str = "PVT_kwDO12345",
    url: str = "https://github.com/orgs/BerriAI/projects/7",
    title: str = "Community PR Backlog",
    field_name: str = "Status",
) -> dict:
    return {
        "id": project_id,
        "title": title,
        "url": url,
        "fields": {
            "nodes": [
                {},
                {
                    "id": "PVTSSF_status_field",
                    "name": field_name,
                    "options": [
                        {"id": "opt_open", "name": "Open"},
                        {"id": "opt_in_review", "name": "In review"},
                        {"id": "opt_changes", "name": "Changes requested"},
                        {"id": "opt_approved", "name": "Approved"},
                        {"id": "opt_merged", "name": "Merged"},
                        {"id": "opt_closed", "name": "Closed"},
                    ],
                },
            ],
        },
    }


class StatusClassificationTests(unittest.TestCase):
    def test_open_without_reviews_or_bot_comments(self):
        pr = _pr_node(101, comments=[{"author": {"__typename": "User", "login": "community-dev"}}])
        self.assertEqual(classify_pr_status(pr), STATUS_OPEN)

    def test_in_review_from_human_commented_review(self):
        pr = _pr_node(
            102,
            reviews=[{"state": "COMMENTED", "author": {"__typename": "User", "login": "reviewer1"}}],
        )
        self.assertEqual(classify_pr_status(pr), STATUS_IN_REVIEW)

    def test_in_review_from_bugbot_or_greptile_comment(self):
        for bot_login in ("cursor[bot]", "greptile-apps[bot]", "veria-ai[bot]", "my-bugbot"):
            with self.subTest(bot=bot_login):
                pr = _pr_node(
                    103,
                    comments=[{"author": {"__typename": "Bot", "login": bot_login}}],
                )
                self.assertEqual(classify_pr_status(pr), STATUS_IN_REVIEW)

    def test_changes_requested_takes_precedence_over_approval(self):
        pr = _pr_node(
            104,
            reviews=[
                {"state": "APPROVED", "author": {"__typename": "User", "login": "alice"}},
                {"state": "CHANGES_REQUESTED", "author": {"__typename": "User", "login": "bob"}},
            ],
        )
        self.assertEqual(classify_pr_status(pr), STATUS_CHANGES_REQUESTED)

        pr_decision = _pr_node(105, review_decision="CHANGES_REQUESTED")
        self.assertEqual(classify_pr_status(pr_decision), STATUS_CHANGES_REQUESTED)

    def test_approved_from_review_or_decision(self):
        pr_decision = _pr_node(106, review_decision="APPROVED")
        self.assertEqual(classify_pr_status(pr_decision), STATUS_APPROVED)

        pr_review = _pr_node(
            107,
            reviews=[{"state": "APPROVED", "author": {"__typename": "User", "login": "maintainer1"}}],
        )
        self.assertEqual(classify_pr_status(pr_review), STATUS_APPROVED)

    def test_merged_and_closed_override_review_states(self):
        merged = _pr_node(108, state="MERGED", review_decision="APPROVED")
        closed = _pr_node(109, state="CLOSED", review_decision="CHANGES_REQUESTED")
        self.assertEqual(classify_pr_status(merged), STATUS_MERGED)
        self.assertEqual(classify_pr_status(closed), STATUS_CLOSED)


class CommunityFilterTests(unittest.TestCase):
    def test_excludes_maintainers_bots_org_members_and_staff(self):
        maintainers = ("AliceMaintainer", "bob-core")
        org_members = frozenset({"carol-org"})
        self.assertFalse(is_community_pr(_pr_node(1, author="alicemaintainer"), maintainers, org_members=org_members, org="BerriAI"))
        self.assertFalse(is_community_pr(_pr_node(2, author="carol-org"), maintainers, org_members=org_members, org="BerriAI"))
        self.assertFalse(is_community_pr(_pr_node(3, author="dependabot[bot]", author_type="Bot"), maintainers, org_members=org_members, org="BerriAI"))
        self.assertFalse(is_community_pr(_pr_node(4, author="alice-berri"), maintainers, org_members=org_members, org="BerriAI"))
        self.assertFalse(is_community_pr(_pr_node(5, author="outside-collab", association="MEMBER"), maintainers, org_members=org_members, org="BerriAI"))
        self.assertTrue(is_community_pr(_pr_node(6, author="helpful-contrib", association="FIRST_TIME_CONTRIBUTOR"), maintainers, org_members=org_members, org="BerriAI"))


class GraphQLAllowlistTests(unittest.TestCase):
    @staticmethod
    def _gql_req(query: str) -> Request:
        return Request(
            "https://api.github.com/graphql",
            data=json.dumps({"query": query}).encode("utf-8"),
            method="POST",
        )

    def test_project_v2_mutations_allowed_while_other_mutations_blocked(self):
        self.assertTrue(_graphql_query_allowed(self._gql_req("mutation ($p: ID!, $c: ID!) { addProjectV2ItemById(input: {projectId: $p, contentId: $c}) { item { id } } }")))
        self.assertTrue(_graphql_query_allowed(self._gql_req("mutation { updateProjectV2ItemFieldValue(input: {}) { projectV2Item { id } } }")))
        self.assertFalse(_graphql_query_allowed(self._gql_req("mutation { addReaction(input: {}) { clientMutationId } }")))
        self.assertFalse(_graphql_query_allowed(self._gql_req("mutation { mergePullRequest(input: {}) { clientMutationId } }")))
        self.assertFalse(_graphql_query_allowed(self._gql_req("mutation { deleteProjectV2Item(input: {}) { deletedItemId } }")))
        # Mixed mutation smuggling a forbidden field next to an allowlisted one must be rejected.
        self.assertFalse(_graphql_query_allowed(self._gql_req("mutation { addProjectV2ItemById(input: {}) { item { id } } mergePullRequest(input: {}) { clientMutationId } }")))
        self.assertFalse(_graphql_query_allowed(self._gql_req("subscription { onIssue { id } }")))
        with self.assertRaises(ValueError):
            github_graphql("token", "mutation { mergePullRequest(input: {}) { clientMutationId } }", {})


class ProjectResolutionAndConfigTests(unittest.TestCase):
    def setUp(self):
        reset_pr_board_cache()

    def test_config_disabled_by_default_and_validates_partial_owner_number(self):
        base_env = {"DISCORD_BOT_TOKEN": "tok", "OSS_CONTRIBUTIONS_CHANNEL_ID": "111", "ANNOUNCEMENTS_CHANNEL_ID": "222"}
        with mock.patch.dict(os.environ, base_env, clear=True):
            cfg = get_config()
            self.assertFalse(cfg.pr_board_enabled)
            self.assertEqual(cfg.pr_board_sync_seconds, 600)

        with mock.patch.dict(os.environ, {**base_env, "PR_BOARD_OWNER": "BerriAI"}, clear=True):
            with self.assertRaises(ValueError):
                get_config()

        with mock.patch.dict(os.environ, {**base_env, "PR_BOARD_OWNER": "BerriAI", "PR_BOARD_NUMBER": "7"}, clear=True):
            cfg = get_config()
            self.assertTrue(cfg.pr_board_enabled)
            self.assertEqual(cfg.pr_board_owner, "BerriAI")
            self.assertEqual(cfg.pr_board_number, 7)

    def test_resolve_project_by_owner_number_falls_back_to_user(self):
        cfg = Config(
            "token", "111", "222",
            pr_board_owner="mubashir1osmani",
            pr_board_number=3,
            github_actor_token="ghp_test",
        )
        calls = []

        def fake_graphql(token, query, variables):
            calls.append((query, variables))
            if "organization(" in query:
                return {"errors": [{"message": "Could not resolve to an Organization"}]}
            return {"data": {"user": {"projectV2": _project_payload(project_id="PVT_user3", url="https://github.com/users/mubashir1osmani/projects/3")}}}

        meta = resolve_project_metadata(cfg, "ghp_test", graphql=fake_graphql)
        self.assertEqual(meta.project_id, "PVT_user3")
        self.assertEqual(meta.url, "https://github.com/users/mubashir1osmani/projects/3")
        self.assertEqual(meta.status_options[STATUS_IN_REVIEW], "opt_in_review")
        self.assertEqual(len(calls), 2)


class IdempotentSyncAndCommandTests(unittest.TestCase):
    def setUp(self):
        reset_pr_board_cache()
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(os.path.join(self.temp.name, "roles.db"))
        self.config = Config(
            "discord-token",
            "111",
            "222",
            discord_guild_id="999",
            discord_application_id="888",
            github_repository="BerriAI/litellm",
            github_actor_token="ghp_actor_project_scope",
            maintainer_logins=("maint-alice",),
            pr_board_project_id="PVT_kwDO12345",
        )

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_sync_is_idempotent_caches_item_ids_and_updates_on_transition(self):
        open_prs = [
            _pr_node(201, author="contrib-a"),  # Open
            _pr_node(202, author="contrib-b", comments=[{"author": {"__typename": "Bot", "login": "cursor[bot]"}}]),  # In review
            _pr_node(203, author="maint-alice"),  # Maintainer -> excluded
            _pr_node(204, author="dependabot[bot]", author_type="Bot"),  # Bot -> excluded
        ]
        closed_prs = [
            _pr_node(205, author="contrib-c", state="MERGED", updated_at="2026-11-27T18:30:00Z"),
        ]

        calls: list[tuple[str, dict]] = []
        added_counter = {"n": 0}

        def fake_graphql(token, query, variables):
            self.assertEqual(token, "ghp_actor_project_scope")
            if "node(id: $projectId)" in query:
                calls.append(("project_meta", variables))
                return {"data": {"node": _project_payload(), "rateLimit": {"remaining": 4900, "resetAt": "2026-11-27T20:00:00Z"}}}
            if "pullRequests(" in query:
                states = variables.get("states")
                calls.append((f"pulls:{','.join(states)}", variables))
                nodes = open_prs if states == ["OPEN"] else closed_prs
                return {
                    "data": {
                        "repository": {
                            "pullRequests": {
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                                "nodes": nodes,
                            }
                        },
                        "rateLimit": {"remaining": 4899, "resetAt": "2026-11-27T20:00:00Z"},
                    }
                }
            if "addProjectV2ItemById" in query:
                added_counter["n"] += 1
                calls.append(("add_item", variables))
                return {"data": {"addProjectV2ItemById": {"item": {"id": f"PVTI_{added_counter['n']}"}}}}
            if "updateProjectV2ItemFieldValue" in query:
                calls.append(("update_status", variables))
                return {"data": {"updateProjectV2ItemFieldValue": {"projectV2Item": {"id": variables["itemId"]}}}}
            raise AssertionError(f"Unexpected query: {query}")

        now = datetime(2026, 11, 27, 19, 0, tzinfo=UTC)
        res1 = sync_pr_board(
            self.config,
            self.store,
            graphql=fake_graphql,
            org_members_fetcher=lambda _org, _tok: frozenset(),
            now=now,
        )
        self.assertIsNotNone(res1)
        self.assertEqual(res1.scanned, 5)
        self.assertEqual(res1.community, 3)  # #201, #202, #205 (excluded #203 maintainer & #204 bot)
        self.assertEqual(res1.added, 3)
        self.assertEqual(res1.updated, 0)
        self.assertEqual(res1.unchanged, 0)

        # Verify item IDs and statuses are cached in SQLite.
        cached_201 = self.store.get_pr_board_item("PVT_kwDO12345", "BerriAI/litellm", 201)
        cached_202 = self.store.get_pr_board_item("PVT_kwDO12345", "BerriAI/litellm", 202)
        cached_205 = self.store.get_pr_board_item("PVT_kwDO12345", "BerriAI/litellm", 205)
        self.assertEqual(cached_201["status"], STATUS_OPEN)
        self.assertEqual(cached_202["status"], STATUS_IN_REVIEW)
        self.assertEqual(cached_205["status"], STATUS_MERGED)
        self.assertIsNone(self.store.get_pr_board_item("PVT_kwDO12345", "BerriAI/litellm", 203))

        # Run 2: identical PR state -> ZERO mutations!
        calls.clear()
        res2 = sync_pr_board(
            self.config,
            self.store,
            graphql=fake_graphql,
            org_members_fetcher=lambda _org, _tok: frozenset(),
            now=now,
        )
        self.assertEqual(res2.added, 0)
        self.assertEqual(res2.updated, 0)
        self.assertEqual(res2.unchanged, 3)
        mutation_ops = [op for op, _ in calls if op in {"add_item", "update_status"}]
        self.assertEqual(mutation_ops, [])

        # Run 3: #201 gets approved (status transition), #206 opens (new PR).
        open_prs[0] = _pr_node(201, author="contrib-a", review_decision="APPROVED", updated_at="2026-11-27T19:05:00Z")
        open_prs.append(_pr_node(206, author="contrib-d", updated_at="2026-11-27T19:06:00Z"))
        calls.clear()
        res3 = sync_pr_board(
            self.config,
            self.store,
            graphql=fake_graphql,
            org_members_fetcher=lambda _org, _tok: frozenset(),
            now=now,
        )
        self.assertEqual(res3.added, 1)  # only #206
        self.assertEqual(res3.updated, 1)  # only #201
        self.assertEqual(res3.unchanged, 2)  # #202 and #205
        add_calls = [v for op, v in calls if op == "add_item"]
        update_calls = [v for op, v in calls if op == "update_status"]
        self.assertEqual(len(add_calls), 1)
        self.assertEqual(add_calls[0]["contentId"], "PR_node_206")
        self.assertEqual(len(update_calls), 2)
        # #201 reused its cached item ID without re-adding.
        update_201 = next(v for v in update_calls if v["itemId"] == cached_201["item_id"])
        self.assertEqual(update_201["optionId"], "opt_approved")

    def test_pr_board_slash_command_is_mod_only_and_ephemeral(self):
        self.assertEqual(PR_BOARD_SLASH_COMMANDS[0]["name"], "pr-board")
        self.assertEqual(PR_BOARD_SLASH_COMMANDS[0]["default_member_permissions"], str(1 << 13))
        self.assertIn("pr-board", [c["name"] for c in commands_for_guild(self.config)])

        # Seed cached board metadata & items.
        self.store.set_pr_board_meta("project_id", "PVT_kwDO12345", "2026-11-27T19:00:00Z")
        self.store.set_pr_board_meta("project_url", "https://github.com/orgs/BerriAI/projects/7", "2026-11-27T19:00:00Z")
        self.store.set_pr_board_meta("last_synced_at", "2026-11-27T19:00:00Z", "2026-11-27T19:00:00Z")
        self.store.upsert_pr_board_item("PVT_kwDO12345", "BerriAI/litellm", 10, "PR_10", "PVTI_10", STATUS_OPEN, "2026-11-27T18:00:00Z", "2026-11-27T19:00:00Z")
        self.store.upsert_pr_board_item("PVT_kwDO12345", "BerriAI/litellm", 11, "PR_11", "PVTI_11", STATUS_IN_REVIEW, "2026-11-27T18:00:00Z", "2026-11-27T19:00:00Z")
        self.store.upsert_pr_board_item("PVT_kwDO12345", "BerriAI/litellm", 12, "PR_12", "PVTI_12", STATUS_IN_REVIEW, "2026-11-27T18:00:00Z", "2026-11-27T19:00:00Z")

        base_payload = {
            "type": 2,
            "guild_id": "999",
            "application_id": "888",
            "data": {"name": "pr-board"},
        }
        non_mod = {**base_payload, "member": {"user": {"id": "123456789012345678"}, "permissions": "0"}}
        resp_non_mod = handle_interaction(self.config, self.store, non_mod)
        self.assertEqual(resp_non_mod["data"]["flags"], 64)
        self.assertEqual(resp_non_mod["data"]["content"], "only mods can run this")

        mod = {**base_payload, "member": {"user": {"id": "123456789012345678"}, "permissions": str(1 << 13)}}
        resp_mod = handle_interaction(self.config, self.store, mod)
        self.assertEqual(resp_mod["data"]["flags"], 64)
        content = resp_mod["data"]["content"]
        self.assertIn("https://github.com/orgs/BerriAI/projects/7", content)
        self.assertIn("- Open: **1**", content)
        self.assertIn("- In review: **2**", content)
        self.assertIn("- Changes requested: **0**", content)
        self.assertIn("- Approved: **0**", content)
        self.assertIn("- Merged: **0**", content)
        self.assertIn("- Closed: **0**", content)
        self.assertIn("Total tracked community PRs: **3**", content)

    def test_disabled_pr_board_emits_single_info_log(self):
        disabled_cfg = Config("token", "111", "222", announcements_enabled=False, merged_weekly_enabled=False)
        stop_event = threading.Event()
        stop_event.set()
        with self.assertLogs(level="INFO") as cm, mock.patch("contributor_bot.main.sync_pr_board") as sync_mock:
            _run_loop(disabled_cfg, stop_event, store=self.store, gateway=None)
        board_logs = [line for line in cm.output if "Contributor PR board is disabled" in line]
        self.assertEqual(len(board_logs), 1)
        sync_mock.assert_not_called()

    def test_sync_skips_cleanly_when_actor_token_unset_and_stops_on_low_rate_limit(self):
        no_token_cfg = Config(
            "discord-token", "111", "222",
            pr_board_project_id="PVT_kwDO12345",
            github_actor_token=None,
        )
        with self.assertLogs(level="WARNING") as cm:
            self.assertIsNone(sync_pr_board(no_token_cfg, self.store))
        self.assertTrue(any("GITHUB_ACTOR_TOKEN" in line for line in cm.output))

        pages_fetched = []

        def low_rl_graphql(token, query, variables):
            if "node(id: $projectId)" in query:
                return {"data": {"node": _project_payload(), "rateLimit": {"remaining": 40, "resetAt": "2026-11-27T20:00:00Z"}}}
            if "pullRequests(" in query:
                pages_fetched.append((variables.get("states"), variables.get("after")))
                return {
                    "data": {
                        "repository": {
                            "pullRequests": {
                                "pageInfo": {"hasNextPage": True, "endCursor": "cursor_page_2"},
                                "nodes": [_pr_node(301, author="contrib-rl")],
                            }
                        },
                        "rateLimit": {"remaining": 10, "resetAt": "2026-11-27T20:00:00Z"},
                    }
                }
            if "addProjectV2ItemById" in query:
                return {"data": {"addProjectV2ItemById": {"item": {"id": "PVTI_rl_1"}}}}
            if "updateProjectV2ItemFieldValue" in query:
                return {"data": {"updateProjectV2ItemFieldValue": {"projectV2Item": {"id": "PVTI_rl_1"}}}}
            raise AssertionError(f"Unexpected query: {query}")

        res = sync_pr_board(
            self.config,
            self.store,
            graphql=low_rl_graphql,
            org_members_fetcher=lambda _org, _tok: frozenset(),
            now=datetime(2026, 11, 27, 19, 0, tzinfo=UTC),
        )
        self.assertTrue(res.rate_limited)
        # Stopped after the single low-rate-limit open page and skipped closed pages.
        self.assertEqual(pages_fetched, [(["OPEN"], None)])


if __name__ == "__main__":
    unittest.main()
