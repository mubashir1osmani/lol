from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from contributor_bot.config import Config
from contributor_bot.interactions import handle_interaction, role_slash_commands
from contributor_bot.matcher import (
    DiscordMemberInfo,
    GitHubAuthorProfile,
    PRLinkPosterEvidence,
    match_community_contributors_with_diagnostics,
    multi_author_poster_ids,
    score_author_against_members,
)
from contributor_bot.rewards import apply_roles
from contributor_bot.roles_sync import (
    _grant_roles_for_stored_matches,
    execute_roles_revoke,
    revalidate_pr_link_poster_matches_on_boot,
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


if __name__ == "__main__":
    unittest.main()
