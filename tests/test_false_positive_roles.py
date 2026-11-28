from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from contributor_bot.api import TransientAPIError
from contributor_bot.config import Config, get_config
from contributor_bot.interactions import role_slash_commands
from contributor_bot.matcher import (
    DiscordMemberInfo,
    GitHubAuthorProfile,
    PRLinkPosterEvidence,
    _cached_github_profile,
    match_community_contributors_with_diagnostics,
    score_author_against_members,
    validate_pr_link_poster_match,
)
from contributor_bot.rewards import announce_pending_role_grants, apply_roles
from contributor_bot.roles_sync import (
    _grant_roles_for_stored_matches,
    poll_merged_prs_for_roles,
    revalidate_all_grants_on_boot,
    run_daily_roles_if_due,
    run_roles_backfill,
    should_run_grants_revalidation,
)
from contributor_bot.schedule import UTC, iso
from contributor_bot.store import Store


def _roles_env(directory: str, **extra: str) -> dict[str, str]:
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


def _member(
    discord_id: str,
    username: str,
    global_name: str = "",
    nick: str = "",
) -> DiscordMemberInfo:
    return DiscordMemberInfo(
        discord_id=discord_id,
        username=username,
        global_name=global_name,
        nick=nick,
        display_name=nick or global_name or username,
    )


def _gql_user_node(
    login: str,
    *,
    name: str = "",
    bio: str = "",
    company: str = "",
    email: str = "",
    twitter: str = "",
    website: str = "",
    socials: list[dict] | None = None,
    readme: str = "",
) -> dict:
    return {
        "login": login,
        "name": name,
        "bio": bio,
        "company": company,
        "email": email,
        "twitterUsername": twitter,
        "websiteUrl": website,
        "socialAccounts": {"nodes": socials or []},
        "profileRepo": {"object": {"text": readme}} if readme else None,
    }


def _stub_gql_from_nodes(nodes_by_login: dict[str, dict]):
    def _caller(_token, query, variables):
        if "BatchPRCommits" in query:
            return {"data": {"repository": {}}}
        data = {}
        for var_key, login in (variables or {}).items():
            idx = var_key.removeprefix("l")
            node = nodes_by_login.get(login.lower())
            if node is not None:
                data[f"u{idx}"] = node
            else:
                data[f"u{idx}"] = _gql_user_node(login)
        return {"data": data}

    return _caller


class NoExtraModCommandsTests(unittest.TestCase):
    def test_roles_why_and_roles_revoke_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = get_config(_roles_env(tmp))
            cmd_names = {cmd["name"] for cmd in role_slash_commands(cfg)}
            self.assertNotIn("roles-why", cmd_names)
            self.assertNotIn("roles-revoke", cmd_names)
            self.assertIn("my-prs", cmd_names)
            self.assertIn("roles-backfill", cmd_names)


class FourTierMatchingOrderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = get_config(_roles_env(self.temp.name))
        self.store = Store(self.config.roles_db_path)
        self.addCleanup(self.store.close)
        self.now = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)

    def test_tier_a_direct_link_socials_bio_readme_and_never_matches_nick(self):
        members = [
            _member("111111111111111111", "real_social_user", "Real Social"),
            _member("222222222222222222", "bio_handle_user", "Bio Handle"),
            _member("333333333333333333", "readme_user", "Readme Person"),
            _member("444444444444444444", "spoof_acct", "Unrelated", nick="target_handle"),
        ]
        groups = [
            {"login": "gh-social", "user_id": 101, "pulls": [{"number": 1}]},
            {"login": "gh-bio", "user_id": 102, "pulls": [{"number": 2}]},
            {"login": "gh-readme", "user_id": 103, "pulls": [{"number": 3}]},
            {"login": "gh-nick-spoof", "user_id": 104, "pulls": [{"number": 4}]},
        ]
        gql = _stub_gql_from_nodes({
            "gh-social": _gql_user_node(
                "gh-social",
                socials=[{"provider": "GENERIC", "displayName": "Discord", "url": "https://discord.com/users/111111111111111111"}],
            ),
            "gh-bio": _gql_user_node("gh-bio", bio="Reach me on Discord: bio_handle_user"),
            "gh-readme": _gql_user_node("gh-readme", readme="Find me on discord: readme_user"),
            "gh-nick-spoof": _gql_user_node("gh-nick-spoof", bio="Discord: target_handle"),
        })

        matched, _skipped, details, _diag = match_community_contributors_with_diagnostics(
            groups,
            self.config,
            self.store,
            now=self.now,
            members=members,
            graphql_caller=gql,
        )
        by_login = {g["login"]: g for g in matched}
        self.assertEqual(by_login["gh-social"]["discord_id"], "111111111111111111")
        self.assertEqual(by_login["gh-social"]["match_method"], "social_discord_id")
        self.assertEqual(by_login["gh-social"]["match_confidence"], 0.99)
        self.assertTrue(any("direct_link:social_discord_id" in e for e in by_login["gh-social"]["match_evidence"]))

        self.assertEqual(by_login["gh-bio"]["discord_id"], "222222222222222222")
        self.assertEqual(by_login["gh-bio"]["match_method"], "bio_discord_handle")
        self.assertEqual(by_login["gh-bio"]["match_confidence"], 0.98)

        self.assertEqual(by_login["gh-readme"]["discord_id"], "333333333333333333")
        self.assertEqual(by_login["gh-readme"]["match_method"], "readme_discord_handle")

        # Server nickname alone must NOT satisfy Tier (a) direct-link handle match.
        self.assertNotIn("gh-nick-spoof", by_login)
        self.assertEqual(len(details), 3)

    def test_tier_b_exact_identity_matches_username_or_global_name_only_not_nick_or_name_alone(self):
        members = [
            _member("111111111111111111", "alice_dev", "Alice"),
            _member("222222222222222222", "other_user", "Bob-Coder"),
            _member("333333333333333333", "imposter99", "Imposter", nick="carol_dev"),
            _member("444444444444444444", "john_smith", "John Smith"),
            _member("555555555555555555", "cooltweet", "Cool"),
        ]
        groups = [
            {"login": "Alice-Dev", "user_id": 201, "pulls": [{"number": 10}]},
            {"login": "bob-coder", "user_id": 202, "pulls": [{"number": 11}]},
            {"login": "carol-dev", "user_id": 203, "pulls": [{"number": 12}]},
            # Full-name only and twitter-only must NOT auto-accept in Tier (b).
            {"login": "gh-unrelated-1", "user_id": 204, "pulls": [{"number": 13}]},
            {"login": "gh-unrelated-2", "user_id": 205, "pulls": [{"number": 14}]},
        ]
        gql = _stub_gql_from_nodes({
            "alice-dev": _gql_user_node("Alice-Dev", name="Alice"),
            "bob-coder": _gql_user_node("bob-coder", name="Bob"),
            "carol-dev": _gql_user_node("carol-dev", name="Carol"),
            "gh-unrelated-1": _gql_user_node("gh-unrelated-1", name="John Smith"),
            "gh-unrelated-2": _gql_user_node("gh-unrelated-2", twitter="cooltweet"),
        })
        cleared_env = {
            k: ""
            for k in ("TYPESAFE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "LITELLM_API_KEY")
        }
        with mock.patch.dict("os.environ", cleared_env, clear=False):
            matched, _skipped, _details, _diag = match_community_contributors_with_diagnostics(
                groups,
                self.config,
                self.store,
                now=self.now,
                members=members,
                graphql_caller=gql,
            )

        by_login = {g["login"]: g for g in matched}
        self.assertEqual(by_login["Alice-Dev"]["discord_id"], "111111111111111111")
        self.assertEqual(by_login["Alice-Dev"]["match_method"], "exact_login")
        self.assertTrue(any("exact_identity:" in e for e in by_login["Alice-Dev"]["match_evidence"]))
        self.assertEqual(by_login["bob-coder"]["discord_id"], "222222222222222222")
        self.assertNotIn("carol-dev", by_login)
        self.assertNotIn("gh-unrelated-1", by_login)
        self.assertNotIn("gh-unrelated-2", by_login)

    def test_tier_c_pr_link_poster_corroboration_signals_and_multi_author_rejection(self):
        # 1) Same X handle corroborates PR-link poster -> ACCEPTED!
        prof_x = GitHubAuthorProfile(login="gh-z1", name="Zed", twitter_username="zed_builds")
        member_x = _member("100000000000000001", "zed_builds", "Z")
        ev_x = PRLinkPosterEvidence("100000000000000001", (501,), ("check out https://github.com/BerriAI/litellm/pull/501",))
        ok, reason = validate_pr_link_poster_match(prof_x, member_x, ev_x, eligible_posters=[ev_x])
        self.assertTrue(ok)
        self.assertIn("same_x:zed_builds", reason)

        # 2) Same website / email domain corroborates PR-link poster -> ACCEPTED!
        prof_web = GitHubAuthorProfile(login="gh-w1", name="W", blog="https://wrenworks.dev", email_domain="wrenworks.dev")
        member_web = _member("100000000000000002", "random_u", "Random")
        ev_web = PRLinkPosterEvidence(
            "100000000000000002",
            (502,),
            ("linked PR https://github.com/BerriAI/litellm/pull/502 docs on https://wrenworks.dev/post",),
        )
        ok_web, reason_web = validate_pr_link_poster_match(prof_web, member_web, ev_web, eligible_posters=[ev_web])
        self.assertTrue(ok_web)
        self.assertIn("same_website:wrenworks.dev", reason_web)

        # 3) Ownership language ("my PR") -> ACCEPTED; "please review" alone -> REJECTED!
        prof_own = GitHubAuthorProfile(login="gh-own", name="Quiet Author")
        member_own = _member("100000000000000003", "totally_diff", "Diff")
        ev_own = PRLinkPosterEvidence("100000000000000003", (503,), ("Here is my PR https://github.com/BerriAI/litellm/pull/503",))
        ok_own, _ = validate_pr_link_poster_match(prof_own, member_own, ev_own, eligible_posters=[ev_own])
        self.assertTrue(ok_own)

        ev_review = PRLinkPosterEvidence("100000000000000003", (503, 504, 505), ("please review https://github.com/BerriAI/litellm/pull/503",))
        ok_rev, _ = validate_pr_link_poster_match(prof_own, member_own, ev_review, eligible_posters=[ev_review])
        self.assertFalse(ok_rev)

        # 4) Name similarity >= 0.50 -> ACCEPTED!
        prof_sim = GitHubAuthorProfile(login="alexchen99", name="Alex Chen")
        member_sim = _member("100000000000000004", "alexchen_ai", "Alex")
        ev_sim = PRLinkPosterEvidence("100000000000000004", (506,), ("https://github.com/BerriAI/litellm/pull/506",))
        ok_sim, reason_sim = validate_pr_link_poster_match(prof_sim, member_sim, ev_sim, eligible_posters=[ev_sim])
        self.assertTrue(ok_sim)
        self.assertIn("name_sim:", reason_sim)

        # 5) Multi-author poster (linked PRs by >= 2 authors) -> REJECTED from Tier (c) AND Tier (d)!
        self.store.record_pr_link_poster(
            "BerriAI/litellm", 601, "100000000000000009", "900000000000000001",
            iso(self.now), message_text="my PR https://github.com/BerriAI/litellm/pull/601",
        )
        self.store.record_pr_link_poster(
            "BerriAI/litellm", 602, "100000000000000009", "900000000000000002",
            iso(self.now), message_text="also check https://github.com/BerriAI/litellm/pull/602",
        )
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=601, github_user_id=6010,
            github_login="author-one", merged_at=iso(self.now), title="p1", labels=[], files=[],
        )
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=602, github_user_id=6020,
            github_login="author-two", merged_at=iso(self.now), title="p2", labels=[], files=[],
        )
        reviewer = _member("100000000000000009", "triager_bot_human", "Triager")
        llm_seen = []
        matched, _skipped, _details, _diag = match_community_contributors_with_diagnostics(
            [{"login": "author-one", "user_id": 6010, "pulls": [{"number": 601}]}],
            self.config,
            self.store,
            now=self.now,
            members=[reviewer],
            graphql_caller=_stub_gql_from_nodes({"author-one": _gql_user_node("author-one", name="Author One")}),
            llm_complete=lambda m, msgs: (llm_seen.append(msgs), json.dumps({"discord_id": None, "confidence": 0.0, "reason": "none", "evidence": []}))[1],
        )
        self.assertEqual(matched, [])
        self.assertEqual(llm_seen, [])

    def test_tier_d_llm_requires_full_enrichment_top5_two_signals_and_085(self):
        members = [
            _member(f"20000000000000000{i}", f"kpatel_{i}", f"Kiran Patel {i}")
            for i in range(1, 8)
        ]
        group = [{"login": "kpatel-gh", "user_id": 701, "pulls": [{"number": 701}]}]
        gql_rich = _stub_gql_from_nodes({
            "kpatel-gh": _gql_user_node(
                "kpatel-gh",
                name="Kiran Patel 1",
                company="Acme",
                email="kiran@acme.io",
            )
        })

        # Captures payload to verify top-5 cap + email_domain present.
        seen_payloads = []

        def llm_reject_single(model, messages):
            seen_payloads.append(json.loads(messages[1]["content"]))
            return json.dumps({
                "discord_id": seen_payloads[-1]["candidates"][0]["discord_id"],
                "confidence": 0.99,
                "reason": "only one signal",
                "evidence": ["single handle overlap"],
            })

        matched_single, _sk, _d, diag_single = match_community_contributors_with_diagnostics(
            group, self.config, None, now=self.now, members=members, graphql_caller=gql_rich, llm_complete=llm_reject_single,
        )
        self.assertEqual(matched_single, [])
        self.assertEqual(dict(diag_single.skip_reasons), {"llm_below_threshold": 1})
        self.assertLessEqual(len(seen_payloads[0]["candidates"]), 5)
        self.assertEqual(seen_payloads[0]["github_author"]["email_domain"], "acme.io")

        # 2 signals but confidence 0.84 (< 0.85) -> REJECTED!
        def llm_below_85(model, messages):
            payload = json.loads(messages[1]["content"])
            return json.dumps({
                "discord_id": payload["candidates"][0]["discord_id"],
                "confidence": 0.84,
                "reason": "below 0.85",
                "evidence": ["sig 1", "sig 2"],
            })

        matched_low, _sk, _d, diag_low = match_community_contributors_with_diagnostics(
            group, self.config, None, now=self.now, members=members, graphql_caller=gql_rich, llm_complete=llm_below_85,
        )
        self.assertEqual(matched_low, [])
        self.assertEqual(dict(diag_low.skip_reasons), {"llm_below_threshold": 1})

        # 2 independent signals AND confidence >= 0.85 -> ACCEPTED with evidence!
        def llm_accept(model, messages):
            payload = json.loads(messages[1]["content"])
            return json.dumps({
                "discord_id": payload["candidates"][0]["discord_id"],
                "confidence": 0.87,
                "reason": "name and login match",
                "evidence": ["GitHub name matches global_name", "login prefix matches username"],
            })

        matched_ok, _sk, details_ok, _diag_ok = match_community_contributors_with_diagnostics(
            group, self.config, self.store, now=self.now, members=members, graphql_caller=gql_rich, llm_complete=llm_accept,
        )
        self.assertEqual(len(matched_ok), 1)
        self.assertEqual(matched_ok[0]["match_method"], "llm")
        self.assertGreaterEqual(len(details_ok[0].evidence), 2)

        # Empty GitHub enrichment (no name/bio/socials/company/email/readme/commits) must NOT call LLM!
        llm_called_on_empty = []
        gql_empty = _stub_gql_from_nodes({"kpatel-empty": _gql_user_node("kpatel-empty")})
        matched_empty, _sk, _d, _diag = match_community_contributors_with_diagnostics(
            [{"login": "kpatel-empty", "user_id": 702, "pulls": []}],
            self.config,
            None,
            now=self.now,
            members=members,
            graphql_caller=gql_empty,
            llm_complete=lambda m, msgs: (llm_called_on_empty.append(True), "{}")[1],
        )
        self.assertEqual(matched_empty, [])
        self.assertEqual(llm_called_on_empty, [])


class NoSkipEnrichmentAndNegativeCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = get_config(_roles_env(self.temp.name))
        self.store = Store(self.config.roles_db_path)
        self.addCleanup(self.store.close)
        self.now = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)

    def test_exact_login_never_skips_github_profile_lookup_and_rate_limit_backs_off_then_succeeds(self):
        members = [_member("300000000000000001", "exact_hero", "Exact Hero")]
        groups = [{"login": "exact-hero", "user_id": 301, "pulls": []}]
        attempts = {"n": 0}
        sleeps: list[float] = []

        def flaky_gql(_token, query, variables):
            if "BatchPRCommits" in query:
                return {"data": {"repository": {}}}
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise TransientAPIError("GitHub secondary rate limit", retry_after=3.0)
            return {"data": {"u0": _gql_user_node("exact-hero", name="Exact Hero", email="hero@acme.io")}}

        matched, skipped, _details, diag = match_community_contributors_with_diagnostics(
            groups,
            self.config,
            self.store,
            now=self.now,
            members=members,
            graphql_caller=flaky_gql,
            sleep_fn=sleeps.append,
        )
        self.assertEqual(attempts["n"], 3)  # Proves lookup ran AND retried on rate limit!
        self.assertEqual(len(sleeps), 2)
        self.assertEqual(len(matched), 1)
        self.assertEqual(skipped, 0)
        self.assertEqual(diag.enrichment_deferred, 0)
        cached_prof = _cached_github_profile(self.store, "exact-hero", self.now)
        self.assertIsNotNone(cached_prof)
        self.assertEqual(cached_prof.email_domain, "acme.io")

    def test_enrichment_failure_blocks_matching_without_guessing_and_persists_completed_batches(self):
        members = [
            _member("310000000000000001", "batch_one_user", "Batch One"),
            _member("310000000000000002", "batch_two_user", "Batch Two"),
        ]
        groups = [
            {"login": "batch-one-user", "user_id": 311, "pulls": []},
            {"login": "batch-two-user", "user_id": 312, "pulls": []},
        ]
        sleeps: list[float] = []
        called_logins: list[str] = []

        # Force 1 login per GraphQL batch so batch 1 succeeds and persists while batch 2 fails after retries.
        def partial_gql(_token, query, variables):
            if "BatchPRCommits" in query:
                return {"data": {"repository": {}}}
            login = variables["l0"]
            called_logins.append(login)
            if login == "batch-one-user":
                return {"data": {"u0": _gql_user_node("batch-one-user", name="Batch One")}}
            raise TransientAPIError("403 rate limit exceeded")

        with mock.patch("contributor_bot.matcher.GRAPHQL_PROFILE_BATCH_SIZE", 1):
            matched1, skipped1, _d1, diag1 = match_community_contributors_with_diagnostics(
                groups,
                self.config,
                self.store,
                now=self.now,
                members=members,
                graphql_caller=partial_gql,
                sleep_fn=sleeps.append,
            )

        # Batch 1 was persisted and matched; batch 2 failed enrichment -> NOT matched even though login matches!
        self.assertEqual([g["login"] for g in matched1], ["batch-one-user"])
        self.assertEqual(skipped1, 1)
        self.assertEqual(dict(diag1.skip_reasons), {"enrichment_failed": 1})
        self.assertIsNotNone(self.store.get_cached_github_user("batch-one-user"))
        # Must NOT negatively cache enrichment_failed!
        self.assertIsNone(self.store.get_match_attempt("batch-two-user"))

        # Next run: batch 1 comes from confirmed cache / SQLite (0 GraphQL calls for batch-one-user),
        # batch 2 now succeeds and matches!
        called_logins.clear()

        def recovered_gql(_token, query, variables):
            if "BatchPRCommits" in query:
                return {"data": {"repository": {}}}
            login = variables["l0"]
            called_logins.append(login)
            return {"data": {"u0": _gql_user_node(login, name="Batch Two")}}

        with mock.patch("contributor_bot.matcher.GRAPHQL_PROFILE_BATCH_SIZE", 1):
            matched2, skipped2, _d2, _diag2 = match_community_contributors_with_diagnostics(
                groups,
                self.config,
                self.store,
                now=self.now + timedelta(minutes=10),
                members=members,
                graphql_caller=recovered_gql,
                sleep_fn=sleeps.append,
            )
        self.assertEqual(called_logins, ["batch-two-user"])
        self.assertEqual({g["login"] for g in matched2}, {"batch-one-user", "batch-two-user"})
        self.assertEqual(skipped2, 0)

    def test_negative_cache_expires_after_7_days_and_picks_up_new_evidence(self):
        members = [_member("320000000000000001", "newbie_handle", "Newbie")]
        group = [{"login": "gh-stranger", "user_id": 321, "pulls": []}]

        gql_calls = {"n": 0}

        def gql_phase1(_token, query, variables):
            if "BatchPRCommits" in query:
                return {"data": {"repository": {}}}
            gql_calls["n"] += 1
            return {"data": {"u0": _gql_user_node("gh-stranger", name="Unrelated Person")}}

        # Day 0: no match -> recorded in match_attempts negative cache.
        m0, _s0, _d0, diag0 = match_community_contributors_with_diagnostics(
            group, self.config, self.store, now=self.now, members=members, graphql_caller=gql_phase1,
        )
        self.assertEqual(m0, [])
        self.assertEqual(dict(diag0.skip_reasons), {"no_signal_match": 1})
        self.assertIsNotNone(self.store.get_match_attempt("gh-stranger"))

        # Day 3: cached_negative hit within 7 days!
        m3, _s3, _d3, diag3 = match_community_contributors_with_diagnostics(
            group, self.config, self.store, now=self.now + timedelta(days=3), members=members, graphql_caller=gql_phase1,
        )
        self.assertEqual(m3, [])
        self.assertEqual(dict(diag3.skip_reasons), {"cached_negative": 1})

        # Day 8 (> 7 days): negative cache expired AND profile cache expired -> picks up new Discord bio link!
        def gql_phase2(_token, query, variables):
            if "BatchPRCommits" in query:
                return {"data": {"repository": {}}}
            return {"data": {"u0": _gql_user_node("gh-stranger", name="Unrelated Person", bio="Discord: newbie_handle")}}

        m8, _s8, _d8, _diag8 = match_community_contributors_with_diagnostics(
            group, self.config, self.store, now=self.now + timedelta(days=8), members=members, graphql_caller=gql_phase2,
        )
        self.assertEqual(len(m8), 1)
        self.assertEqual(m8[0]["discord_id"], "320000000000000001")
        self.assertEqual(m8[0]["match_method"], "bio_discord_handle")


class OneToOneGuildPresenceAndGitHubTruthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = get_config(_roles_env(self.temp.name))
        self.store = Store(self.config.roles_db_path)
        self.addCleanup(self.store.close)
        self.now = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)

    def test_one_to_one_conflict_grants_nothing(self):
        shared_member = _member("400000000000000001", "shared_user", "Shared User")
        groups = [
            {"login": "gh-first", "user_id": 401, "pulls": [{"number": 41}]},
            {"login": "gh-second", "user_id": 402, "pulls": [{"number": 42}]},
        ]
        gql = _stub_gql_from_nodes({
            "gh-first": _gql_user_node("gh-first", bio="Discord: shared_user"),
            "gh-second": _gql_user_node("gh-second", bio="Discord: shared_user"),
        })
        with self.assertLogs(level="WARNING") as logs:
            matched, skipped, _details, diag = match_community_contributors_with_diagnostics(
                groups, self.config, self.store, now=self.now, members=[shared_member], graphql_caller=gql,
            )
        self.assertEqual(matched, [])
        self.assertEqual(skipped, 2)
        self.assertEqual(dict(diag.skip_reasons), {"discord_id_conflict": 2})
        self.assertTrue(any("1-to-1 match conflict" in line for line in logs.output))

    def test_guild_presence_verified_github_only_truth_and_grant_log_includes_evidence(self):
        # 1) Member NOT in guild -> apply_roles refuses to grant.
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=501, github_user_id=501,
            github_login="departed-dev", merged_at=iso(self.now), title="fix", labels=[], files=[],
        )
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role:
            granted = apply_roles(
                self.config,
                self.store,
                "500000000000000001",
                501,
                github_login="departed-dev",
                match_method="exact_login",
                match_confidence=0.96,
                match_evidence="exact_identity:departed-dev->departed_dev",
                present_member_ids=frozenset({"999999999999999999"}),
            )
        self.assertEqual(granted, [])
        add_role.assert_not_called()

        # 2) Stored match with ZERO GitHub PRs (count_fetcher returns 0 or raises) -> NO role granted (no fabrication!).
        self.store.save_confirmed_match(
            "ghost-author", "500000000000000002", method="exact_login", confidence=0.96,
            matched_at=iso(self.now), github_user_id=502, evidence="exact_identity:ghost-author",
        )
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role_ghost:
            granted_ghost, _rl = _grant_roles_for_stored_matches(
                self.config,
                self.store,
                now=self.now,
                org="BerriAI",
                org_members=frozenset(),
                count_fetcher=lambda repo, login, tok: 0,
                present_member_ids=frozenset({"500000000000000002"}),
            )
        self.assertEqual(granted_ghost, 0)
        add_role_ghost.assert_not_called()

        # 3) Present guild member with real GitHub PR -> granted AND log includes discord_id, github_login, method, confidence, evidence!
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=503, github_user_id=503,
            github_login="present-dev", merged_at=iso(self.now), title="feat", labels=[], files=[],
        )
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role_ok, \
             self.assertLogs(level="INFO") as logs:
            granted_ok = apply_roles(
                self.config,
                self.store,
                "500000000000000003",
                503,
                github_login="present-dev",
                match_method="exact_login",
                match_confidence=0.96,
                match_evidence="exact_identity:present-dev->present_dev",
                present_member_ids=frozenset({"500000000000000003"}),
            )
        self.assertEqual(granted_ok, ["contributor"])
        add_role_ok.assert_called_once()
        self.assertTrue(
            any(
                "discord_id=500000000000000003" in line
                and "github_login=present-dev" in line
                and "method=exact_login" in line
                and "confidence=0.96" in line
                and "evidence=exact_identity:present-dev->present_dev" in line
                for line in logs.output
            )
        )


class SilentAnnouncementsLeaseLockAndBootRevalidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = get_config(_roles_env(self.temp.name))
        self.store = Store(self.config.roles_db_path)
        self.addCleanup(self.store.close)
        self.now = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)

    def test_roles_announce_disabled_by_default_and_no_backlog_later(self):
        self.assertFalse(self.config.roles_announce_enabled)
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=601, github_user_id=601,
            github_login="quiet-dev", merged_at=iso(self.now), title="fix", labels=[], files=[],
        )
        members = [_member("600000000000000001", "quiet-dev", "Quiet Dev")]
        with mock.patch("contributor_bot.rewards.add_member_role") as add_role, \
             mock.patch("contributor_bot.rewards.post_message") as post_msg, \
             self.assertLogs(level="INFO") as logs:
            res = poll_merged_prs_for_roles(
                self.config,
                self.store,
                now=self.now,
                search_merged_fn=lambda r, s, e, t: set(),
                count_fetcher=lambda r, l, t: 1,
                org_members_fetcher=lambda o, t: frozenset(),
                members=members,
                graphql_caller=_stub_gql_from_nodes({"quiet-dev": _gql_user_node("quiet-dev", name="Quiet Dev")}),
            )
        self.assertEqual(res.roles_granted, 1)
        add_role.assert_called_once()
        post_msg.assert_not_called()
        self.assertEqual(self.store.all_unannounced_grants(), [])
        self.assertTrue(
            any("Role announcements disabled (ROLES_ANNOUNCE_ENABLED=false); granted 1 roles silently" in l for l in logs.output)
        )

        # Turning ROLES_ANNOUNCE_ENABLED=true later must NOT post any backlog!
        cfg_on = get_config(_roles_env(self.temp.name, ROLES_ANNOUNCE_ENABLED="true"))
        with mock.patch("contributor_bot.rewards.post_message") as post_later:
            chunks = announce_pending_role_grants(cfg_on, self.store, now=self.now)
        self.assertEqual(chunks, 0)
        post_later.assert_not_called()

    def test_db_lease_lock_blocks_concurrent_instance_and_allows_expired_takeover(self):
        store2 = Store(self.config.roles_db_path)
        self.addCleanup(store2.close)

        self.assertTrue(self.store.acquire_lease("roles_poll:berriai/litellm", "inst-1", 300.0, now=self.now))
        self.assertFalse(store2.acquire_lease("roles_poll:berriai/litellm", "inst-2", 300.0, now=self.now + timedelta(seconds=10)))

        # Second instance calling poll_merged_prs_for_roles skips while lease is held!
        res_skipped = poll_merged_prs_for_roles(self.config, store2, now=self.now + timedelta(seconds=10))
        self.assertIsNone(res_skipped)

        # Once expired, inst-2 acquires the lease cleanly.
        self.assertTrue(store2.acquire_lease("roles_poll:berriai/litellm", "inst-2", 300.0, now=self.now + timedelta(seconds=301)))
        store2.release_lease("roles_poll:berriai/litellm", "inst-2")

    def test_boot_all_grants_revalidation_catches_edvn_and_checks_1155909469373468692(self):
        # Set up realistic production-like grants:
        # 1) edvn (1466417354824224830): bad old LLM/pr_link_poster grant that does NOT pass new rules -> REVOKED!
        # 2) 1155909469373468692: real contributor with exact_login + merged PR -> KEPT!
        # 3) 700000000000000003: old single-signal exact_name match (no 2nd signal) -> REVOKED!
        # 4) 700000000000000004: orphan role_grant with NO GitHub mapping -> REVOKED!
        # 5) 700000000000000005: member who left the Discord guild -> REVOKED!
        # 6) 700000000000000006: exact_login match but 0 GitHub merged PRs -> REVOKED!
        now_iso = iso(self.now)

        # PRs
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=901, github_user_id=901,
            github_login="edvn-unrelated-author", merged_at=now_iso, title="fix", labels=[], files=[],
        )
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=902, github_user_id=902,
            github_login="real-contrib-115", merged_at=now_iso, title="feat", labels=[], files=[],
        )
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=903, github_user_id=903,
            github_login="gh-single-name", merged_at=now_iso, title="chore", labels=[], files=[],
        )
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=905, github_user_id=905,
            github_login="departed-contrib", merged_at=now_iso, title="docs", labels=[], files=[],
        )

        # Existing grants & confirmed matches
        self.store.save_confirmed_match(
            "edvn-unrelated-author", "1466417354824224830", method="llm", confidence=0.81,
            matched_at=now_iso, github_user_id=901, evidence="weak name guess",
        )
        self.store.record_grant("1466417354824224830", "contributor", "Contributor", now_iso)

        self.store.save_confirmed_match(
            "real-contrib-115", "1155909469373468692", method="exact_login", confidence=0.96,
            matched_at=now_iso, github_user_id=902, evidence="exact_identity:real-contrib-115",
        )
        self.store.record_grant("1155909469373468692", "contributor", "Contributor", now_iso)

        self.store.save_confirmed_match(
            "gh-single-name", "700000000000000003", method="exact_name", confidence=0.93,
            matched_at=now_iso, github_user_id=903, evidence="exact_name:Sam",
        )
        self.store.record_grant("700000000000000003", "contributor", "Contributor", now_iso)

        self.store.record_grant("700000000000000004", "contributor", "Contributor", now_iso)

        self.store.save_confirmed_match(
            "departed-contrib", "700000000000000005", method="exact_login", confidence=0.96,
            matched_at=now_iso, github_user_id=905, evidence="exact_identity:departed-contrib",
        )
        self.store.record_grant("700000000000000005", "contributor", "Contributor", now_iso)

        self.store.save_confirmed_match(
            "zero-pr-user", "700000000000000006", method="exact_login", confidence=0.96,
            matched_at=now_iso, github_user_id=906, evidence="exact_identity:zero-pr-user",
        )
        self.store.record_grant("700000000000000006", "contributor", "Contributor", now_iso)

        guild_members = [
            _member("1466417354824224830", "edvn", "Edvn"),
            _member("1155909469373468692", "real_contrib_115", "Real Contrib"),
            _member("700000000000000003", "totally_other", "Sam"),
            _member("700000000000000004", "orphan_member", "Orphan"),
            _member("700000000000000006", "zero_pr_user", "Zero PR"),
            # Note: 700000000000000005 is NOT in guild_members (departed)!
        ]
        gql = _stub_gql_from_nodes({
            "edvn-unrelated-author": _gql_user_node("edvn-unrelated-author", name="Completely Different"),
            "real-contrib-115": _gql_user_node("real-contrib-115", name="Real Contrib"),
            "gh-single-name": _gql_user_node("gh-single-name", name="Sam"),
            "zero-pr-user": _gql_user_node("zero-pr-user", name="Zero PR"),
        })

        self.assertTrue(should_run_grants_revalidation(self.config, self.store))
        cleared_env = {
            k: ""
            for k in ("TYPESAFE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "LITELLM_API_KEY")
        }
        removed_calls: list[tuple] = []
        with mock.patch.dict("os.environ", cleared_env, clear=False), \
             mock.patch("contributor_bot.rewards.remove_member_role", side_effect=lambda *a: removed_calls.append(a)), \
             mock.patch("contributor_bot.rewards.post_message") as post_msg:
            res = revalidate_all_grants_on_boot(
                self.config,
                self.store,
                members=guild_members,
                now=self.now,
                org_members_fetcher=lambda o, t: frozenset(),
                graphql_caller=gql,
                count_fetcher=lambda repo, login, tok: 0 if login == "zero-pr-user" else 1,
            )

        self.assertIsNotNone(res)
        self.assertEqual(res.checked, 6)
        self.assertEqual(res.kept, 1)      # Only 1155909469373468692 kept!
        self.assertEqual(res.revoked, 5)   # edvn + single-name + orphan + departed + zero-pr all revoked!
        self.assertFalse(res.rate_limited)
        post_msg.assert_not_called()       # Completely silent!

        # edvn (1466417354824224830) revoked, match deleted, and rejected so never rematched!
        self.assertEqual(self.store.grants_for_discord("1466417354824224830"), [])
        self.assertIsNone(self.store.get_confirmed_match("edvn-unrelated-author"))
        self.assertTrue(self.store.is_match_rejected("edvn-unrelated-author", "1466417354824224830"))

        # 1155909469373468692 kept intact!
        self.assertEqual(self.store.grant_keys("1155909469373468692"), {"contributor"})
        self.assertIsNotNone(self.store.get_confirmed_match("real-contrib-115"))

        # Others all revoked cleanly!
        for bad_uid in (
            "700000000000000003",
            "700000000000000004",
            "700000000000000005",
            "700000000000000006",
        ):
            self.assertEqual(self.store.grants_for_discord(bad_uid), [])

        # Runs only once!
        self.assertFalse(should_run_grants_revalidation(self.config, self.store))
        self.assertIsNone(revalidate_all_grants_on_boot(self.config, self.store, members=guild_members, now=self.now))


if __name__ == "__main__":
    unittest.main()
