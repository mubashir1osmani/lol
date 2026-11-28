from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from contributor_bot.config import Config, get_config
from contributor_bot.interactions import handle_interaction, role_slash_commands
from contributor_bot.matcher import (
    DiscordMemberInfo,
    GitHubAuthorProfile,
    PRLinkPosterEvidence,
    match_community_contributors_with_diagnostics,
    multi_author_poster_ids,
    score_author_against_members,
)
from contributor_bot.rewards import announce_pending_role_grants, apply_roles
from contributor_bot.roles_sync import (
    _grant_roles_for_stored_matches,
    execute_roles_revoke,
    poll_merged_prs_for_roles,
    revalidate_pr_link_poster_matches_on_boot,
    run_daily_roles_if_due,
    run_roles_backfill,
    should_run_pr_link_poster_revalidation,
)
from contributor_bot.store import Store


def _make_config(**overrides) -> Config:
    defaults = dict(
        discord_guild_id="333333333333333333",
        discord_application_id="444444444444444444",
        github_repository="BerriAI/litellm",
        contributor_roles_enabled=True,
        role_ids={
            "contributor": "900000000000000001",
            "oss_champion": "900000000000000002",
            "agent_builder": "900000000000000003",
            "provider_wizard": "900000000000000004",
        },
        role_announcements_channel_id="800000000000000001",
    )
    defaults.update(overrides)
    return Config("tok", "123", "456", **defaults)


def _seed_pull(
    store: Store,
    repo: str,
    number: int,
    uid: int,
    login: str,
    merged_at: str,
    *,
    title: str = "feat: change",
    labels: list[str] | None = None,
    files: list[str] | None = None,
) -> None:
    store.upsert_pull(
        repository=repo,
        number=number,
        github_user_id=uid,
        github_login=login,
        merged_at=merged_at,
        title=title,
        labels=labels or [],
        files=files or [],
    )


class TestPRLinkPosterHardening(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(str(Path(self.tmp.name) / "roles.db"))
        self.addCleanup(self.store.close)
        self.cfg = _make_config()
        self.now = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)

    def test_reviewer_posting_links_by_multiple_authors_is_rejected_completely(self):
        """Production bug repro: community member 'edvn' links PRs by other authors."""
        repo = self.cfg.github_repository
        _seed_pull(
            self.store, repo, 1001, 501, "real-prolific-author", "2026-11-20T10:00:00+00:00",
            labels=["agents"], files=["litellm/llms/openai/chat.py", "litellm/proxy/agent.py"],
        )
        _seed_pull(
            self.store, repo, 1002, 501, "real-prolific-author", "2026-11-21T10:00:00+00:00",
            files=["litellm/llms/anthropic/chat.py"],
        )
        _seed_pull(
            self.store, repo, 2001, 502, "another-author", "2026-11-22T10:00:00+00:00",
            files=["litellm/router.py"],
        )

        # Discord member 'edvn' links PRs from BOTH real-prolific-author and another-author.
        edvn_id = "700000000000000001"
        self.store.record_pr_link_poster(repo, 1001, edvn_id, "800000000000000011", "2026-11-20T11:00:00+00:00", message_text="taking a look at this PR")
        self.store.record_pr_link_poster(repo, 1002, edvn_id, "800000000000000012", "2026-11-21T11:00:00+00:00", message_text="my pr")  # even with ownership phrase!
        self.store.record_pr_link_poster(repo, 2001, edvn_id, "800000000000000013", "2026-11-22T11:00:00+00:00", message_text="also reviewing #2001")

        # Verify multi_author_poster_ids catches edvn.
        all_rows = self.store.all_pr_link_posters(repo)
        pr_authors = self.store.pr_authors_for_numbers(repo, [1001, 1002, 2001])
        multi = multi_author_poster_ids(all_rows, pr_authors)
        self.assertIn(edvn_id, multi)

        members = [DiscordMemberInfo(edvn_id, username="edvn", global_name="Edvn")]
        groups = [
            {
                "login": "real-prolific-author",
                "user_id": 501,
                "first_time": False,
                "pulls": [
                    {"number": 1001, "title": "feat: a", "url": "https://github.com/BerriAI/litellm/pull/1001"},
                    {"number": 1002, "title": "feat: b", "url": "https://github.com/BerriAI/litellm/pull/1002"},
                ],
            },
        ]

        llm_called = []

        def fake_llm(prompt, model):
            llm_called.append(prompt)
            return '{"discord_id": "700000000000000001", "confidence": 0.99, "evidence": ["x"], "reason": "forced"}'

        matched, skipped, _details, _diag = match_community_contributors_with_diagnostics(
            groups,
            self.cfg,
            self.store,
            members=members,
            now=self.now,
            skip_github_enrichment=True,
            llm_complete=fake_llm,
        )
        # edvn must NOT be matched deterministically AND must NOT even be sent to LLM!
        self.assertEqual(matched, [])
        self.assertEqual(skipped, 1)
        self.assertEqual(llm_called, [])
        self.assertIsNone(self.store.get_confirmed_match("real-prolific-author"))

    def test_two_links_without_ownership_or_similarity_never_auto_accepts(self):
        """Linking >=2 PRs by one author without ownership text or similarity >= 0.5 is LLM-only."""
        prof = GitHubAuthorProfile(login="kiran-dev-99", name="Kiran Patel")
        members = [DiscordMemberInfo("700000000000000002", username="edvn", global_name="Edvn")]
        strong, cands = score_author_against_members(
            prof,
            members,
            pr_link_posters=[
                PRLinkPosterEvidence(
                    "700000000000000002",
                    (101, 102, 103),
                    ("check out #101", "please review #102", "bump #103"),
                )
            ],
        )
        # Must NOT auto-accept (not even with "please review" or 3 PRs!).
        self.assertIsNone(strong)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0].member.discord_id, "700000000000000002")
        self.assertIn("pr_link_poster:3_prs", cands[0].signals)

    def test_ownership_phrases_and_name_similarity_auto_accept_for_single_author_poster(self):
        prof = GitHubAuthorProfile(login="unusual-gh-author", name="Carlos Vega")
        unrelated_member = DiscordMemberInfo("700000000000000002", username="edvn", global_name="Edvn")
        similar_member = DiscordMemberInfo("700000000000000003", username="carlos_v_dev", global_name="Carlos V")
        for phrase in (
            "here is my pr for streaming",
            "I opened https://github.com/BerriAI/litellm/pull/55",
            "I made this fix yesterday",
            "I raised a PR for gemini",
            "I submitted a patch",
        ):
            strong, _ = score_author_against_members(
                prof,
                [unrelated_member],
                pr_link_posters=[PRLinkPosterEvidence("700000000000000002", (55,), (phrase,))],
            )
            self.assertIsNotNone(strong, f"Expected auto-accept for phrase: {phrase!r}")
            self.assertEqual(strong[0].discord_id, "700000000000000002")
            self.assertEqual(strong[1], "pr_link_poster")

        # Similarity >= 0.50 against global_name/username without ownership text auto-accepts.
        strong_sim, _ = score_author_against_members(
            prof,
            [similar_member],
            pr_link_posters=[PRLinkPosterEvidence("700000000000000003", (55,), ("https://github.com/BerriAI/litellm/pull/55",))],
        )
        self.assertIsNotNone(strong_sim)
        self.assertEqual(strong_sim[0].discord_id, "700000000000000003")
        self.assertEqual(strong_sim[1], "pr_link_poster")

        # "please review" alone (and short token overlap like "edvn" vs "Zed") must NOT auto-accept.
        strong_pr, _ = score_author_against_members(
            prof,
            [unrelated_member],
            pr_link_posters=[PRLinkPosterEvidence("700000000000000002", (55,), ("please review https://github.com/BerriAI/litellm/pull/55",))],
        )
        self.assertIsNone(strong_pr)


class TestExactLoginNoNick(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(str(Path(self.tmp.name) / "roles.db"))
        self.addCleanup(self.store.close)
        self.cfg = _make_config()
        self.now = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)

    def test_exact_login_ignores_nick_and_display_name(self):
        # Member sets server nick to someone else's GitHub login!
        impostor = DiscordMemberInfo(
            "710000000000000001",
            username="edvn",
            global_name="Edvn",
            nick="prolific-gh-dev",
            display_name="prolific-gh-dev",
        )
        prof = GitHubAuthorProfile(login="prolific-gh-dev", name="Alex Rivera")
        strong, _cands = score_author_against_members(prof, [impostor])
        self.assertIsNone(strong)

        # Also check pre-enrichment fast path in match_community_contributors_with_diagnostics.
        groups = [
            {
                "login": "prolific-gh-dev",
                "user_id": 801,
                "first_time": False,
                "pulls": [{"number": 9001, "title": "fix", "url": "https://github.com/BerriAI/litellm/pull/9001"}],
            }
        ]
        matched, skipped, _details, _diag = match_community_contributors_with_diagnostics(
            groups,
            self.cfg,
            self.store,
            members=[impostor],
            now=self.now,
            skip_github_enrichment=True,
        )
        self.assertEqual(matched, [])
        self.assertEqual(skipped, 1)

    def test_exact_login_matches_username_and_global_name(self):
        by_user = DiscordMemberInfo("710000000000000002", username="prolific-gh-dev", global_name="Other")
        by_global = DiscordMemberInfo("710000000000000003", username="other-handle", global_name="prolific-gh-dev")
        prof = GitHubAuthorProfile(login="prolific-gh-dev")

        s1, _ = score_author_against_members(prof, [by_user])
        self.assertIsNotNone(s1)
        self.assertEqual((s1[0].discord_id, s1[1]), ("710000000000000002", "exact_login"))

        s2, _ = score_author_against_members(prof, [by_global])
        self.assertIsNotNone(s2)
        self.assertEqual((s2[0].discord_id, s2[1]), ("710000000000000003", "exact_login"))


class TestStrictOneToOneBijection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(str(Path(self.tmp.name) / "roles.db"))
        self.addCleanup(self.store.close)
        self.cfg = _make_config()
        self.now = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)

    def test_intra_batch_conflict_grants_nothing_and_logs(self):
        # Two GitHub logins both match the SAME Discord user (one via username, one via global_name).
        member = DiscordMemberInfo("720000000000000001", username="alice-gh", global_name="bob-gh")
        groups = [
            {"login": "alice-gh", "user_id": 1, "first_time": False, "pulls": [{"number": 1, "title": "a", "url": "u"}]},
            {"login": "bob-gh", "user_id": 2, "first_time": False, "pulls": [{"number": 2, "title": "b", "url": "u"}]},
        ]
        with self.assertLogs(level="WARNING") as logs:
            matched, skipped, _details, diag = match_community_contributors_with_diagnostics(
                groups,
                self.cfg,
                self.store,
                members=[member],
                now=self.now,
                skip_github_enrichment=True,
            )
        self.assertEqual(matched, [])
        self.assertEqual(skipped, 2)
        self.assertEqual(dict(diag.skip_reasons).get("discord_id_conflict"), 2)
        self.assertTrue(any("1-to-1 match conflict" in msg for msg in logs.output))
        self.assertIsNone(self.store.get_confirmed_match("alice-gh"))
        self.assertIsNone(self.store.get_confirmed_match("bob-gh"))

    def test_cross_run_conflict_against_stored_match_grants_nothing(self):
        # Stored match already binds discord_id to existing-author.
        self.store.save_confirmed_match(
            "existing-author",
            "720000000000000002",
            method="exact_login",
            confidence=0.98,
            matched_at="2026-11-27T00:00:00+00:00",
        )
        member = DiscordMemberInfo("720000000000000002", username="second-author", global_name="Second")
        groups = [
            {"login": "second-author", "user_id": 3, "first_time": False, "pulls": [{"number": 3, "title": "c", "url": "u"}]},
        ]
        with self.assertLogs(level="WARNING") as logs:
            matched, skipped, _details, _diag = match_community_contributors_with_diagnostics(
                groups,
                self.cfg,
                self.store,
                members=[member],
                now=self.now,
                skip_github_enrichment=True,
            )
        self.assertEqual(matched, [])
        self.assertEqual(skipped, 1)
        self.assertTrue(any("1-to-1 match conflict" in msg for msg in logs.output))
        self.assertIsNone(self.store.get_confirmed_match("second-author"))

    def test_roles_sync_stored_matches_blocks_conflicting_discord_id(self):
        # Simulate corrupt DB state where two GitHub logins point to the same discord_id.
        self.store.save_confirmed_match(
            "author-one", "720000000000000003", method="exact_login", confidence=0.98, matched_at="2026-11-27T00:00:00+00:00",
        )
        self.store.save_confirmed_match(
            "author-two", "720000000000000003", method="pr_link_poster", confidence=0.92, matched_at="2026-11-27T00:00:00+00:00",
        )
        _seed_pull(self.store, self.cfg.github_repository, 10, 100, "author-one", "2026-11-27T00:00:00+00:00")
        _seed_pull(self.store, self.cfg.github_repository, 11, 101, "author-two", "2026-11-27T00:00:00+00:00")

        with patch("contributor_bot.rewards.add_member_role") as mock_add, self.assertLogs(level="WARNING") as logs:
            granted, _rl = _grant_roles_for_stored_matches(
                self.cfg,
                self.store,
                now=self.now,
                org="BerriAI",
                org_members=frozenset(),
                count_fetcher=lambda *_a, **_kw: 5,
            )
        self.assertEqual(granted, 0)
        mock_add.assert_not_called()
        self.assertTrue(any("1-to-1 stored-match conflict" in msg for msg in logs.output))


class TestRolesWhyAndRevokeCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(str(Path(self.tmp.name) / "roles.db"))
        self.addCleanup(self.store.close)
        self.cfg = _make_config()
        self.now = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)
        self.edvn_id = "730000000000000001"
        self.mod_id = "730000000000000099"

        # Seed edvn with a bad match, 4 role grants, and PRs for the wrongly attached author.
        repo = self.cfg.github_repository
        self.store.save_confirmed_match(
            "wrong-author",
            self.edvn_id,
            method="pr_link_poster",
            confidence=0.92,
            matched_at="2026-11-25T10:00:00+00:00",
            github_user_id=901,
            evidence="pr_link_poster:2_prs",
        )
        self.store.record_grant(self.edvn_id, "contributor", "Contributor", "2026-11-25T10:00:00+00:00")
        self.store.record_grant(self.edvn_id, "oss_champion", "OSS Champion", "2026-11-25T10:00:00+00:00")
        self.store.record_grant(self.edvn_id, "agent_builder", "Agent Builder", "2026-11-25T10:00:00+00:00")
        self.store.record_grant(self.edvn_id, "provider_wizard", "Provider Wizard", "2026-11-25T10:00:00+00:00")
        _seed_pull(
            self.store, repo, 5001, 901, "wrong-author", "2026-11-20T00:00:00+00:00",
            labels=["agents"], files=["litellm/llms/openai/a.py", "litellm/proxy/mcp.py"],
        )
        _seed_pull(
            self.store, repo, 5002, 901, "wrong-author", "2026-11-21T00:00:00+00:00",
            files=["litellm/llms/anthropic/b.py"],
        )
        _seed_pull(
            self.store, repo, 5003, 901, "wrong-author", "2026-11-22T00:00:00+00:00",
            files=["litellm/llms/gemini/c.py"],
        )
        self.store.record_pr_link_poster(repo, 5001, self.edvn_id, "800000000000000021", "2026-11-20T01:00:00+00:00", message_text="reviewing")

    def test_commands_registered_when_roles_enabled(self):
        names = {cmd["name"] for cmd in role_slash_commands(self.cfg)}
        self.assertIn("roles-why", names)
        self.assertIn("roles-revoke", names)

    def test_roles_why_requires_mod_and_shows_full_audit(self):
        non_mod_payload = {
            "type": 2,
            "application_id": self.cfg.discord_application_id,
            "guild_id": self.cfg.discord_guild_id,
            "data": {"name": "roles-why", "options": [{"name": "member", "value": self.edvn_id}]},
            "member": {"user": {"id": "730000000000000050"}, "permissions": "0"},
        }
        resp = handle_interaction(self.cfg, self.store, non_mod_payload)
        self.assertEqual(resp["data"]["flags"], 64)
        self.assertIn("only mods can run this", resp["data"]["content"])

        mod_payload = {
            "type": 2,
            "application_id": self.cfg.discord_application_id,
            "guild_id": self.cfg.discord_guild_id,
            "data": {"name": "roles-why", "options": [{"name": "member", "value": self.edvn_id}]},
            "member": {"user": {"id": self.mod_id}, "permissions": str(1 << 13)},
        }
        resp_mod = handle_interaction(self.cfg, self.store, mod_payload)
        self.assertEqual(resp_mod["data"]["flags"], 64)
        content = resp_mod["data"]["content"]
        self.assertIn("wrong-author", content)
        self.assertIn("pr_link_poster", content)
        self.assertIn("0.92", content)
        self.assertIn("pr_link_poster:2_prs", content)
        self.assertIn("Contributor", content)
        self.assertIn("OSS Champion", content)
        self.assertIn("Agent Builder", content)
        self.assertIn("Provider Wizard", content)
        self.assertIn("#5001", content)

    def test_roles_revoke_removes_roles_deletes_grants_and_blocks_rematch(self):
        non_mod_payload = {
            "type": 2,
            "application_id": self.cfg.discord_application_id,
            "guild_id": self.cfg.discord_guild_id,
            "data": {"name": "roles-revoke", "options": [{"name": "member", "value": self.edvn_id}]},
            "member": {"user": {"id": "730000000000000050"}, "permissions": "0"},
        }
        resp_denied = handle_interaction(self.cfg, self.store, non_mod_payload)
        self.assertIn("only mods can run this", resp_denied["data"]["content"])

        mod_payload = {
            "type": 2,
            "application_id": self.cfg.discord_application_id,
            "guild_id": self.cfg.discord_guild_id,
            "data": {"name": "roles-revoke", "options": [{"name": "member", "value": self.edvn_id}]},
            "member": {"user": {"id": self.mod_id}, "permissions": str(1 << 13)},
        }
        removed_calls = []
        with patch(
            "contributor_bot.rewards.remove_member_role",
            side_effect=lambda gid, uid, rid, tok: removed_calls.append((gid, uid, rid)),
        ):
            resp = handle_interaction(self.cfg, self.store, mod_payload)

        self.assertEqual(resp["data"]["flags"], 64)
        self.assertIn("wrong-author", resp["data"]["content"])
        # All 4 contributor roles removed in Discord.
        self.assertEqual(len(removed_calls), 4)
        self.assertEqual(
            {rid for _, _, rid in removed_calls},
            set(self.cfg.role_ids.values()),
        )
        # All role_grants and github_discord_matches deleted; rejection recorded.
        self.assertEqual(self.store.grants_for_discord(self.edvn_id), [])
        self.assertIsNone(self.store.get_confirmed_match("wrong-author"))
        self.assertTrue(self.store.is_match_rejected("wrong-author", self.edvn_id))

        # Even if member later matches wrong-author by exact_login or pr_link_poster, rejection blocks rematch!
        members = [DiscordMemberInfo(self.edvn_id, username="wrong-author", global_name="wrong-author")]
        groups = [{"login": "wrong-author", "user_id": 901, "first_time": False, "pulls": [{"number": 5001, "title": "x", "url": "u"}]}]
        matched, skipped, _details, _diag = match_community_contributors_with_diagnostics(
            groups,
            self.cfg,
            self.store,
            members=members,
            now=self.now,
            skip_github_enrichment=True,
        )
        self.assertEqual(matched, [])
        self.assertEqual(skipped, 1)

    def test_roles_revoke_with_optional_login_sets_mod_override_and_checks_conflict(self):
        with patch("contributor_bot.rewards.remove_member_role"):
            msg = execute_roles_revoke(
                self.cfg,
                self.store,
                self.edvn_id,
                moderator_id=self.mod_id,
                replacement_login="real-edvn-gh",
                now=self.now,
            )
        self.assertIn("Replacement mapping saved", msg)
        row = self.store.get_confirmed_match("real-edvn-gh")
        self.assertIsNotNone(row)
        self.assertEqual(row["discord_id"], self.edvn_id)
        self.assertEqual(row["method"], "mod_override")
        self.assertTrue(self.store.is_match_rejected("wrong-author", self.edvn_id))

        # Attempting to assign real-edvn-gh to a second Discord member must be blocked (1-to-1 conflict).
        other_id = "730000000000000002"
        with patch("contributor_bot.rewards.remove_member_role"):
            msg_conflict = execute_roles_revoke(
                self.cfg,
                self.store,
                other_id,
                moderator_id=self.mod_id,
                replacement_login="real-edvn-gh",
                now=self.now,
            )
        self.assertIn("Override skipped", msg_conflict)
        self.assertIn("1-to-1 conflict", msg_conflict)
        self.assertEqual(self.store.get_confirmed_match("real-edvn-gh")["discord_id"], self.edvn_id)


class TestBootRevalidationAndGrantLogging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(str(Path(self.tmp.name) / "roles.db"))
        self.addCleanup(self.store.close)
        self.cfg = _make_config()
        self.now = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)

    def test_boot_revalidation_revokes_bad_pr_link_poster_keeps_valid_and_runs_once(self):
        repo = self.cfg.github_repository
        # 1) Bad match A: reviewer who posted links for 2 different authors (edvn scenario).
        edvn_id = "740000000000000001"
        _seed_pull(self.store, repo, 101, 1, "author-a", "2026-11-20T00:00:00+00:00")
        _seed_pull(self.store, repo, 102, 1, "author-a", "2026-11-21T00:00:00+00:00")
        _seed_pull(self.store, repo, 103, 2, "author-b", "2026-11-22T00:00:00+00:00")
        self.store.record_pr_link_poster(repo, 101, edvn_id, "800000000000000031", "2026-11-20T01:00:00+00:00", message_text="my pr")
        self.store.record_pr_link_poster(repo, 102, edvn_id, "800000000000000032", "2026-11-21T01:00:00+00:00", message_text="link")
        self.store.record_pr_link_poster(repo, 103, edvn_id, "800000000000000033", "2026-11-22T01:00:00+00:00", message_text="reviewing")
        self.store.save_confirmed_match(
            "author-a", edvn_id, method="pr_link_poster", confidence=0.92, matched_at="2026-11-22T02:00:00+00:00", github_user_id=1,
        )
        self.store.record_grant(edvn_id, "contributor", "Contributor", "2026-11-22T02:00:00+00:00")

        # 2) Bad match B: posted 2 links by one author but NO ownership text and NO name similarity.
        triager_id = "740000000000000002"
        _seed_pull(self.store, repo, 201, 3, "author-c", "2026-11-20T00:00:00+00:00")
        _seed_pull(self.store, repo, 202, 3, "author-c", "2026-11-21T00:00:00+00:00")
        self.store.record_pr_link_poster(repo, 201, triager_id, "800000000000000034", "2026-11-20T01:00:00+00:00", message_text="please review")
        self.store.record_pr_link_poster(repo, 202, triager_id, "800000000000000035", "2026-11-21T01:00:00+00:00", message_text="bump")
        self.store.save_confirmed_match(
            "author-c", triager_id, method="pr_link_poster", confidence=0.92, matched_at="2026-11-22T02:00:00+00:00", github_user_id=3,
        )
        self.store.record_grant(triager_id, "contributor", "Contributor", "2026-11-22T02:00:00+00:00")

        # 3) Good match C: sole poster of 1 author's PR with explicit ownership phrase.
        real_id = "740000000000000003"
        _seed_pull(self.store, repo, 301, 4, "author-good", "2026-11-20T00:00:00+00:00")
        self.store.record_pr_link_poster(repo, 301, real_id, "800000000000000036", "2026-11-20T01:00:00+00:00", message_text="Hey, I opened this PR!")
        self.store.save_confirmed_match(
            "author-good", real_id, method="pr_link_poster", confidence=0.92, matched_at="2026-11-22T02:00:00+00:00", github_user_id=4,
        )
        self.store.record_grant(real_id, "contributor", "Contributor", "2026-11-22T02:00:00+00:00")

        # 4) Unrelated exact_login match: must not be touched.
        exact_id = "740000000000000004"
        self.store.save_confirmed_match(
            "exact-dev", exact_id, method="exact_login", confidence=0.98, matched_at="2026-11-22T02:00:00+00:00",
        )
        self.store.record_grant(exact_id, "contributor", "Contributor", "2026-11-22T02:00:00+00:00")

        members = [
            DiscordMemberInfo(edvn_id, username="edvn", global_name="Edvn"),
            DiscordMemberInfo(triager_id, username="helpful-triager", global_name="Triager"),
            DiscordMemberInfo(real_id, username="some-handle", global_name="Some Handle"),
            DiscordMemberInfo(exact_id, username="exact-dev", global_name="Exact Dev"),
        ]

        self.assertTrue(should_run_pr_link_poster_revalidation(self.cfg, self.store))

        removed = []
        with patch(
            "contributor_bot.rewards.remove_member_role",
            side_effect=lambda gid, uid, rid, tok: removed.append((uid, rid)),
        ), patch("contributor_bot.rewards.post_message") as mock_announce, self.assertLogs(level="INFO") as logs:
            res = revalidate_pr_link_poster_matches_on_boot(
                self.cfg,
                self.store,
                members=members,
                now=self.now,
                org_members_fetcher=lambda _o, _t: [],
            )

        self.assertIsNotNone(res)
        self.assertEqual(res.checked, 3)
        self.assertEqual(res.kept, 1)
        self.assertEqual(res.revoked, 2)
        # No announcements posted!
        mock_announce.assert_not_called()

        # edvn and triager revoked + rejected; good and exact kept.
        self.assertIsNone(self.store.get_confirmed_match("author-a"))
        self.assertIsNone(self.store.get_confirmed_match("author-c"))
        self.assertEqual(self.store.grants_for_discord(edvn_id), [])
        self.assertEqual(self.store.grants_for_discord(triager_id), [])
        self.assertTrue(self.store.is_match_rejected("author-a", edvn_id))
        self.assertTrue(self.store.is_match_rejected("author-c", triager_id))
        self.assertIn(edvn_id, {uid for uid, _ in removed})
        self.assertIn(triager_id, {uid for uid, _ in removed})
        self.assertNotIn(real_id, {uid for uid, _ in removed})
        self.assertNotIn(exact_id, {uid for uid, _ in removed})

        self.assertIsNotNone(self.store.get_confirmed_match("author-good"))
        self.assertEqual(len(self.store.grants_for_discord(real_id)), 1)
        self.assertIsNotNone(self.store.get_confirmed_match("exact-dev"))
        self.assertEqual(len(self.store.grants_for_discord(exact_id)), 1)

        # Each revocation logged.
        self.assertTrue(any("REVOKED discord_id=740000000000000001 github_login=author-a" in m for m in logs.output))
        self.assertTrue(any("REVOKED discord_id=740000000000000002 github_login=author-c" in m for m in logs.output))

        # One-time marker set: second run is a no-op!
        self.assertFalse(should_run_pr_link_poster_revalidation(self.cfg, self.store))
        res2 = revalidate_pr_link_poster_matches_on_boot(
            self.cfg, self.store, members=members, now=self.now, org_members_fetcher=lambda _o, _t: [],
        )
        self.assertIsNone(res2)

    def test_role_grant_logs_discord_id_login_method_confidence(self):
        uid = "750000000000000001"
        _seed_pull(self.store, self.cfg.github_repository, 901, 55, "dev-alice", "2026-11-27T00:00:00+00:00")
        self.store.save_confirmed_match(
            "dev-alice", uid, method="exact_login", confidence=0.98, matched_at="2026-11-27T00:00:00+00:00", github_user_id=55,
        )
        with patch("contributor_bot.rewards.add_member_role"), self.assertLogs(level="INFO") as logs:
            granted = apply_roles(
                self.cfg,
                self.store,
                uid,
                55,
                shoutout=False,
                announce=False,
                github_login="dev-alice",
                match_method="exact_login",
                match_confidence=0.98,
            )
        self.assertEqual(granted, ["contributor"])
        self.assertTrue(
            any(
                "Granted Discord role Contributor" in m
                and f"discord_id={uid}" in m
                and "github_login=dev-alice" in m
                and "method=exact_login" in m
                and "confidence=0.98" in m
                for m in logs.output
            ),
            f"Expected structured role grant log, got: {logs.output}",
        )


class TestRolesAnnounceDisabledByDefault(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(str(Path(self.tmp.name) / "roles.db"))
        self.addCleanup(self.store.close)
        self.now = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)

    def test_config_defaults_roles_announce_enabled_to_false(self):
        cfg = get_config({
            "DISCORD_BOT_TOKEN": "tok",
            "OSS_CONTRIBUTIONS_CHANNEL_ID": "123",
            "ANNOUNCEMENTS_CHANNEL_ID": "456",
            "CONTRIBUTOR_ROLES_ENABLED": "true",
            "DISCORD_GUILD_ID": "333333333333333333",
            "ROLE_CONTRIBUTOR_ID": "900000000000000001",
            "ROLE_OSS_CHAMPION_ID": "900000000000000002",
            "ROLE_AGENT_BUILDER_ID": "900000000000000003",
            "ROLE_PROVIDER_WIZARD_ID": "900000000000000004",
            "ROLE_ANNOUNCEMENTS_CHANNEL_ID": "800000000000000001",
            "ROLES_DB_PATH": str(Path(self.tmp.name) / "cfg.db"),
        })
        self.assertFalse(cfg.roles_announce_enabled)

    def test_apply_roles_silent_no_shoutout_and_no_backlog_when_enabled_later(self):
        cfg_off = _make_config(roles_announce_enabled=False)
        uid = "760000000000000001"
        _seed_pull(
            self.store, cfg_off.github_repository, 701, 601, "silent-dev", "2026-11-27T10:00:00+00:00",
            labels=["agents"],
        )
        with patch("contributor_bot.rewards.add_member_role"), patch(
            "contributor_bot.rewards.post_message"
        ) as mock_post, self.assertLogs(level="INFO") as logs:
            granted = apply_roles(
                cfg_off,
                self.store,
                uid,
                601,
                pr_number=701,
                shoutout=True,
                announce=True,
                github_login="silent-dev",
                match_method="exact_login",
                match_confidence=0.98,
            )
        self.assertEqual(set(granted), {"contributor", "agent_builder"})
        mock_post.assert_not_called()
        # Marked announced_at + shoutout_at silently so turning it on later never backlogs!
        self.assertEqual(self.store.all_unannounced_grants(), [])
        self.assertIsNotNone(self.store.get_pull(cfg_off.github_repository, 701)["shoutout_at"])
        self.assertTrue(
            any("Role announcements disabled (ROLES_ANNOUNCE_ENABLED=false); granted 2 roles silently" in m for m in logs.output),
            f"Expected silent grant log, got: {logs.output}",
        )

        # Turning ROLES_ANNOUNCE_ENABLED=true later must post ZERO backlog!
        cfg_on = _make_config(roles_announce_enabled=True)
        with patch("contributor_bot.rewards.post_message") as mock_post_later:
            posted_chunks = announce_pending_role_grants(cfg_on, self.store, now=self.now)
        self.assertEqual(posted_chunks, 0)
        mock_post_later.assert_not_called()

    def test_daily_poll_and_backfill_post_nothing_and_log_silent_when_disabled(self):
        cfg_off = _make_config(roles_announce_enabled=False, backfill_since="2026-11-01T00:00:00+00:00")
        uid = "760000000000000002"
        members = [DiscordMemberInfo(uid, username="quiet-dev", global_name="Quiet Dev")]

        def fake_pull(_repo, num, _tok):
            return {
                "number": num,
                "merged": True,
                "merged_at": "2026-11-26T10:00:00Z",
                "title": "feat: quiet",
                "author_association": "CONTRIBUTOR",
                "user": {"login": "quiet-dev", "id": 602, "type": "User"},
                "labels": [],
            }

        with patch("contributor_bot.rewards.add_member_role"), patch(
            "contributor_bot.rewards.post_message"
        ) as mock_rewards_post, patch(
            "contributor_bot.roles_sync.post_message"
        ) as mock_sync_post, self.assertLogs(level="INFO") as logs:
            poll_res = poll_merged_prs_for_roles(
                cfg_off,
                self.store,
                now=self.now,
                members=members,
                search_merged_fn=lambda _r, _s, _e, _t: {801},
                pull_fetcher=fake_pull,
                files_fetcher=lambda _r, _n, _t: ["litellm/router.py"],
                count_fetcher=lambda _r, _l, _t: 1,
                org_members_fetcher=lambda _o, _t: [],
            )
        self.assertIsNotNone(poll_res)
        self.assertEqual(poll_res.roles_granted, 1)
        mock_rewards_post.assert_not_called()
        mock_sync_post.assert_not_called()
        self.assertEqual(self.store.all_unannounced_grants(), [])
        self.assertTrue(
            any("Role announcements disabled (ROLES_ANNOUNCE_ENABLED=false); granted 1 roles silently" in m for m in logs.output),
            f"Expected silent grant log in poll, got: {logs.output}",
        )

        # Backfill summary is also silenced when ROLES_ANNOUNCE_ENABLED=false.
        uid2 = "760000000000000003"
        members2 = members + [DiscordMemberInfo(uid2, username="backfill-quiet", global_name="Backfill Quiet")]

        def fake_pull_bf(_repo, num, _tok):
            return {
                "number": num,
                "merged": True,
                "merged_at": "2026-11-10T10:00:00Z",
                "title": "feat: bf",
                "author_association": "CONTRIBUTOR",
                "user": {"login": "backfill-quiet", "id": 603, "type": "User"},
                "labels": [],
            }

        with patch("contributor_bot.rewards.add_member_role"), patch(
            "contributor_bot.rewards.post_message"
        ) as mock_rewards_post2, patch(
            "contributor_bot.roles_sync.post_message"
        ) as mock_sync_post2, self.assertLogs(level="INFO") as logs_bf:
            bf_res = run_roles_backfill(
                cfg_off,
                self.store,
                now=self.now,
                members=members2,
                search_merged_fn=lambda _r, _s, _e, _t: [802],
                pull_fetcher=fake_pull_bf,
                files_fetcher=lambda _r, _n, _t: ["litellm/router.py"],
                count_fetcher=lambda _r, _l, _t: 1,
                org_members_fetcher=lambda _o, _t: [],
                announce_summary=True,
            )
        self.assertIsNotNone(bf_res)
        self.assertEqual(bf_res.roles_granted, 1)
        self.assertFalse(bf_res.summary_announced)
        mock_rewards_post2.assert_not_called()
        mock_sync_post2.assert_not_called()
        self.assertEqual(self.store.all_unannounced_grants(), [])
        self.assertTrue(
            any("Role announcements disabled (ROLES_ANNOUNCE_ENABLED=false); granted 1 roles silently" in m for m in logs_bf.output),
            f"Expected silent grant log in backfill, got: {logs_bf.output}",
        )


class TestLLMEnrichmentDeferredGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(str(Path(self.tmp.name) / "roles.db"))
        self.addCleanup(self.store.close)
        self.cfg = _make_config(github_token="gh-tok")
        self.now = datetime(2026, 11, 28, 4, 13, tzinfo=timezone.utc)

    def test_llm_never_accepts_when_enrichment_deferred_and_retries_next_run(self):
        """Repro 04:13 UTC prod bug: enrichment_deferred > 0 must block LLM and treat as llm_unavailable."""
        edvn_id = "1466417354824224830"
        members = [DiscordMemberInfo(edvn_id, username="edvn", global_name="edvn")]
        groups = [
            {
                "login": "edvn-unrelated-dev",
                "user_id": 777,
                "first_time": False,
                "pulls": [{"number": 4201, "title": "feat", "url": "https://github.com/BerriAI/litellm/pull/4201"}],
            }
        ]
        llm_calls = []

        def aggressive_llm(model, messages):
            llm_calls.append(messages)
            return json.dumps({
                "discord_id": edvn_id,
                "confidence": 0.96,
                "reason": "similar handle prefix",
                "evidence": ["handle prefix", "handle"],
            })

        # Simulate GraphQL rate-limit deferral (enrichment_deferred=1, zero profile data).
        def rate_limited_gql(_query, _vars, _tok, **_kw):
            raise RuntimeError("api.github.com returned HTTP 403 rate limit exceeded")

        matched, skipped, _details, diag = match_community_contributors_with_diagnostics(
            groups,
            self.cfg,
            self.store,
            members=members,
            now=self.now,
            graphql_caller=rate_limited_gql,
            llm_complete=aggressive_llm,
        )
        self.assertEqual(matched, [])
        self.assertEqual(skipped, 1)
        self.assertEqual(diag.enrichment_deferred, 1)
        self.assertEqual(dict(diag.skip_reasons), {"llm_unavailable": 1})
        self.assertEqual(llm_calls, [])
        self.assertIsNone(self.store.get_confirmed_match("edvn-unrelated-dev"))
        # Must NOT write 7-day negative cache so next run retries!
        self.assertIsNone(self.store.get_match_attempt("edvn-unrelated-dev"))

    def test_llm_never_accepts_when_enrichment_returns_empty_profile(self):
        """Even without rate limit, an author with zero profile/commit fields must not be LLM-accepted."""
        edvn_id = "1466417354824224830"
        members = [DiscordMemberInfo(edvn_id, username="edvn", global_name="edvn")]
        groups = [
            {
                "login": "edvn-empty-gh",
                "user_id": 778,
                "first_time": False,
                "pulls": [],
            }
        ]
        llm_calls = []

        def aggressive_llm(model, messages):
            llm_calls.append(messages)
            return json.dumps({
                "discord_id": edvn_id,
                "confidence": 0.95,
                "reason": "handle overlap",
                "evidence": ["handle prefix", "handle"],
            })

        # GraphQL returns a user object with ALL enrichment fields empty.
        def empty_profile_gql(_query, _vars, _tok, **_kw):
            return {
                "data": {
                    "u0": {
                        "login": "edvn-empty-gh",
                        "name": None,
                        "bio": None,
                        "company": None,
                        "email": None,
                        "twitterUsername": None,
                        "websiteUrl": None,
                        "socialAccounts": {"nodes": []},
                        "readme": None,
                    }
                }
            }

        matched, skipped, _details, diag = match_community_contributors_with_diagnostics(
            groups,
            self.cfg,
            self.store,
            members=members,
            now=self.now,
            graphql_caller=empty_profile_gql,
            llm_complete=aggressive_llm,
        )
        self.assertEqual(matched, [])
        self.assertEqual(skipped, 1)
        self.assertEqual(dict(diag.skip_reasons), {"llm_unavailable": 1})
        self.assertEqual(llm_calls, [])
        self.assertIsNone(self.store.get_match_attempt("edvn-empty-gh"))


class TestDBBackedRunLeaseLock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "shared.db")
        self.store_a = Store(self.db_path)
        self.store_b = Store(self.db_path)
        self.addCleanup(self.store_a.close)
        self.addCleanup(self.store_b.close)
        self.now = datetime(2026, 11, 28, 16, 0, tzinfo=timezone.utc)

    def test_two_instances_cannot_hold_lease_simultaneously_and_expired_lease_recovers(self):
        # Instance A acquires lease.
        self.assertTrue(self.store_a.acquire_lease("roles_daily:berriai/litellm", "inst-a", 300, now=self.now))
        # Instance B (separate connection to the same SQLite DB) is blocked while lease is active.
        self.assertFalse(self.store_b.acquire_lease("roles_daily:berriai/litellm", "inst-b", 300, now=self.now + timedelta(seconds=60)))
        # Non-owner cannot renew or release Instance A's lease.
        self.assertFalse(self.store_b.renew_lease("roles_daily:berriai/litellm", "inst-b", 300, now=self.now + timedelta(seconds=60)))
        self.store_b.release_lease("roles_daily:berriai/litellm", "inst-b")
        self.assertEqual(self.store_a.get_lease("roles_daily:berriai/litellm")["owner_id"], "inst-a")

        # Owner can renew lease.
        self.assertTrue(self.store_a.renew_lease("roles_daily:berriai/litellm", "inst-a", 300, now=self.now + timedelta(seconds=100)))
        self.assertFalse(self.store_b.acquire_lease("roles_daily:berriai/litellm", "inst-b", 300, now=self.now + timedelta(seconds=350)))

        # Once lease expires (e.g. Instance A crashed during deploy), Instance B takes over cleanly.
        self.assertTrue(self.store_b.acquire_lease("roles_daily:berriai/litellm", "inst-b", 300, now=self.now + timedelta(seconds=401)))
        self.assertEqual(self.store_b.get_lease("roles_daily:berriai/litellm")["owner_id"], "inst-b")

        # Release lets Instance A acquire immediately.
        self.store_b.release_lease("roles_daily:berriai/litellm", "inst-b")
        self.assertIsNone(self.store_a.get_lease("roles_daily:berriai/litellm"))
        self.assertTrue(self.store_a.acquire_lease("roles_daily:berriai/litellm", "inst-a", 300, now=self.now + timedelta(seconds=402)))

    def test_daily_roles_skips_when_other_instance_holds_db_lease(self):
        cfg = _make_config(roles_daily_time="10:00", merged_timezone="America/Toronto")
        # Simulate another Render instance holding the DB lease for roles_daily.
        self.assertTrue(self.store_a.acquire_lease("roles_daily:berriai/litellm", "other-instance", 600, now=self.now))

        poll_called = []

        def fake_poll(*_args, **_kwargs):
            poll_called.append(True)
            return None

        res = run_daily_roles_if_due(
            cfg,
            self.store_b,
            now=self.now,
            force=True,
            poll_fn=fake_poll,
        )
        self.assertIsNone(res)
        self.assertEqual(poll_called, [])



if __name__ == "__main__":
    unittest.main()
