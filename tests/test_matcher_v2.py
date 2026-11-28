import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from contributor_bot.api import search_guild_messages
from contributor_bot.config import Config, get_config
from contributor_bot.interactions import handle_interaction
from contributor_bot.links_backfill import (
    GUILD_SEARCH_MAX_OFFSET,
    backfill_pr_link_posters_from_review_threads,
    extract_pr_links_from_text,
    record_pr_links_from_message,
    run_links_backfill,
)
from contributor_bot.matcher import (
    DEFAULT_MATCH_MODEL,
    MAX_CANDIDATES_PER_AUTHOR,
    PROFILE_README_MAX_BYTES,
    DiscordMemberInfo,
    GitHubAuthorProfile,
    PRLinkPosterEvidence,
    _excluded_staff_discord_ids,
    _profile_source_norms,
    _resolve_match_model,
    fetch_github_profiles_graphql,
    match_community_contributors_with_diagnostics,
    score_author_against_members,
)
from contributor_bot.roles_sync import _search_merged_numbers_per_day
from contributor_bot.store import Store
from contributor_bot.weekly import _search_merged_pages


class TestCompletePRFetch(unittest.TestCase):
    def test_weekly_per_day_windows_exceeds_500_dedupes_and_warns_at_1000(self):
        since = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)
        until = datetime(2026, 11, 22, 23, 59, tzinfo=timezone.utc)
        queries_seen = []

        def fake_github(url, token, pace=False, **kwargs):
            queries_seen.append(url)
            # Extract day and page from URL.
            import urllib.parse as up
            qs = up.parse_qs(up.urlparse(url).query)
            q = qs["q"][0]
            page = int(qs["page"][0])
            if "merged:2026-11-20..2026-11-20" in q:
                # 6 full pages = 600 PRs (exceeds old max_pages=5 cap!)
                if page <= 6:
                    base = (page - 1) * 100 + 1
                    items = [
                        {
                            "number": n,
                            "title": f"PR {n}",
                            "html_url": f"https://github.com/BerriAI/litellm/pull/{n}",
                            "user": {"login": f"dev-{n}", "type": "User", "id": n},
                            "pull_request": {"merged_at": "2026-11-20T12:00:00Z"},
                        }
                        for n in range(base, base + 100)
                    ]
                    return {"total_count": 600, "items": items}
                return {"total_count": 600, "items": []}

            if "merged:2026-11-21..2026-11-21" in q:
                # Simulated 1050-result day: hits 1000 warning and caps at page 10.
                # Also overlaps PRs 591..600 to verify deduplication by PR number!
                base = 1000 + (page - 1) * 100
                items = []
                for idx in range(100):
                    num = 591 + idx if (page == 1 and idx < 10) else base + idx
                    items.append({
                        "number": num,
                        "title": f"PR {num}",
                        "html_url": f"https://github.com/BerriAI/litellm/pull/{num}",
                        "user": {"login": f"dev-{num}", "type": "User", "id": num},
                        "pull_request": {"merged_at": "2026-11-21T15:00:00Z"},
                    })
                return {"total_count": 1050, "items": items}

            if "merged:2026-11-22..2026-11-22" in q:
                return {
                    "total_count": 2,
                    "items": [
                        {
                            "number": 9001,
                            "title": "PR 9001",
                            "html_url": "https://github.com/BerriAI/litellm/pull/9001",
                            "user": {"login": "dev-9001", "type": "User", "id": 9001},
                            "pull_request": {"merged_at": "2026-11-22T09:00:00Z"},
                        },
                        {
                            "number": 9002,
                            "title": "PR 9002",
                            "html_url": "https://github.com/BerriAI/litellm/pull/9002",
                            "user": {"login": "dev-9002", "type": "User", "id": 9002},
                            "pull_request": {"merged_at": "2026-11-22T18:00:00Z"},
                        },
                    ],
                }
            return {"total_count": 0, "items": []}

        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_github), \
                self.assertLogs(level="WARNING") as logs:
            rows, rate_limited = _search_merged_pages(
                "BerriAI/litellm", since, "gh-tok", until=until
            )

        self.assertFalse(rate_limited)
        # 600 (day 1) + 990 unique (day 2, 10 dupes dropped) + 2 (day 3) = 1592 unique PRs!
        self.assertEqual(len(rows), 1592)
        self.assertEqual(len({r["number"] for r in rows}), 1592)
        # Day 2 capped at 10 pages (never requests page 11).
        self.assertFalse(any("page=11" in u for u in queries_seen))
        self.assertTrue(any("2026-11-20..2026-11-20" in u for u in queries_seen))
        self.assertTrue(any("2026-11-21..2026-11-21" in u for u in queries_seen))
        self.assertTrue(any("2026-11-22..2026-11-22" in u for u in queries_seen))
        self.assertTrue(any("hit 1050 results (>=1000" in line for line in logs.output))

    def test_weekly_filters_out_items_outside_exact_timestamp_window(self):
        since = datetime(2026, 11, 20, 15, 0, tzinfo=timezone.utc)
        until = datetime(2026, 11, 21, 10, 0, tzinfo=timezone.utc)

        def fake_github(url, token, pace=False, **kwargs):
            if "2026-11-20..2026-11-20" in url:
                return {
                    "total_count": 2,
                    "items": [
                        {"number": 1, "title": "before", "user": {"login": "a", "type": "User", "id": 1}, "pull_request": {"merged_at": "2026-11-20T14:59:59Z"}},
                        {"number": 2, "title": "inside", "user": {"login": "b", "type": "User", "id": 2}, "pull_request": {"merged_at": "2026-11-20T15:00:00Z"}},
                    ],
                }
            return {
                "total_count": 2,
                "items": [
                    {"number": 3, "title": "inside2", "user": {"login": "c", "type": "User", "id": 3}, "pull_request": {"merged_at": "2026-11-21T10:00:00Z"}},
                    {"number": 4, "title": "after", "user": {"login": "d", "type": "User", "id": 4}, "pull_request": {"merged_at": "2026-11-21T10:00:01Z"}},
                ],
            }

        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_github):
            rows, _rl = _search_merged_pages("BerriAI/litellm", since, "tok", until=until)

        self.assertEqual([r["number"] for r in rows], [2, 3])

    def test_roles_sync_per_day_search_dedupes_and_warns(self):
        since = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)
        until = datetime(2026, 11, 21, 0, 0, tzinfo=timezone.utc)

        def fake_github(url, token, pace=False, **kwargs):
            if "2026-11-20" in url:
                return {
                    "total_count": 1005,
                    "items": [
                        {"number": 10, "title": "x", "user": {"login": "a", "type": "User", "id": 1}},
                        {"number": 11, "title": "y", "user": {"login": "b", "type": "User", "id": 2}},
                    ],
                }
            return {
                "total_count": 2,
                "items": [
                    {"number": 11, "title": "y-dup", "user": {"login": "b", "type": "User", "id": 2}},
                    {"number": 12, "title": "z", "user": {"login": "c", "type": "User", "id": 3}},
                ],
            }

        with mock.patch("contributor_bot.weekly.github_get_json", side_effect=fake_github), \
                self.assertLogs(level="WARNING") as logs:
            nums = _search_merged_numbers_per_day("BerriAI/litellm", since, until, "tok")

        self.assertEqual(nums, {10, 11, 12})
        self.assertTrue(any("1005" in line for line in logs.output))


class TestPRLinkPosterSignal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(str(Path(self.tmp.name) / "bot.db"))
        self.addCleanup(self.store.close)

    def test_store_pr_link_posters_table_index_and_crud(self):
        indexes = self.store._conn.execute("PRAGMA index_list('pr_link_posters')").fetchall()
        index_names = {row["name"] for row in indexes}
        self.assertIn("pr_link_posters_repo_pr", index_names)

        inserted1 = self.store.record_pr_link_poster(
            "BerriAI/litellm", 42, "100000000000000001", "900000000000000001",
            "2026-11-25T10:00:00+00:00", message_text="please review my PR",
        )
        inserted_dup = self.store.record_pr_link_poster(
            "BerriAI/litellm", 42, "100000000000000001", "900000000000000001",
            "2026-11-25T10:00:00+00:00", message_text="please review my PR",
        )
        self.assertTrue(inserted1)
        self.assertFalse(inserted_dup)
        self.assertTrue(self.store.has_pr_link_poster_message("BerriAI/litellm", 42, "900000000000000001"))
        self.assertEqual(self.store.pr_link_posters_count("BerriAI/litellm"), 1)

        rows = self.store.pr_link_posters_for_pulls("BerriAI/litellm", [42, 99])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["discord_id"], "100000000000000001")
        self.assertEqual(rows[0]["message_text"], "please review my PR")

    def test_extract_and_record_pr_links_from_message(self):
        text = (
            "Hey team, I opened https://github.com/BerriAI/litellm/pull/101 and "
            "github.com/berriai/litellm/pull/202 (also see https://github.com/other/repo/pull/999)"
        )
        self.assertEqual(extract_pr_links_from_text(text, "BerriAI/litellm"), [101, 202])

        count = record_pr_links_from_message(
            self.store,
            "BerriAI/litellm",
            discord_id="100000000000000001",
            message_id="800000000000000001",
            content=text,
            ts=datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(count, 2)
        rows = self.store.pr_link_posters_for_pulls("BerriAI/litellm", [101, 202])
        self.assertEqual([r["pr_number"] for r in rows], [101, 202])

    def test_derive_posters_from_review_threads(self):
        self.store.save_review_thread(
            repository="BerriAI/litellm",
            number=501,
            thread_id="700000000000000001",
            channel_id="600000000000000001",
            message_id="500000000000000001",
            created_at="2026-11-24T10:00:00+00:00",
        )
        self.store.save_review_thread(
            repository="BerriAI/litellm",
            number=502,
            thread_id="700000000000000002",
            channel_id="600000000000000001",
            message_id="500000000000000002",
            created_at="2026-11-24T11:00:00+00:00",
        )
        cfg = Config("discord-tok", "123", "456", discord_guild_id="333")
        fetched = []

        def fake_fetch(channel_id, message_id, token):
            fetched.append(message_id)
            if message_id == "500000000000000001":
                return {
                    "id": message_id,
                    "content": "please review https://github.com/BerriAI/litellm/pull/501",
                    "timestamp": "2026-11-24T10:00:00Z",
                    "author": {"id": "100000000000000009", "bot": False},
                }
            # Bot message should be skipped.
            return {
                "id": message_id,
                "content": "bot post https://github.com/BerriAI/litellm/pull/502",
                "timestamp": "2026-11-24T11:00:00Z",
                "author": {"id": "100000000000000099", "bot": True},
            }

        scanned, recorded, rate_limited = backfill_pr_link_posters_from_review_threads(
            cfg, self.store, message_fetcher=fake_fetch
        )
        self.assertEqual((scanned, recorded, rate_limited), (2, 1, False))
        rows = self.store.pr_link_posters_for_pulls("BerriAI/litellm", [501, 502])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["discord_id"], "100000000000000009")

        # Second call skips already-recorded PR 501 message!
        fetched.clear()
        backfill_pr_link_posters_from_review_threads(cfg, self.store, message_fetcher=fake_fetch)
        self.assertEqual(fetched, ["500000000000000002"])

    def test_search_guild_messages_handles_202_index_building_and_retry_after(self):
        class _FakeResp:
            def __init__(self, status, body):
                self.status = status
                self._body = json.dumps(body).encode("utf-8")
                self.headers = {}

            def read(self, n=-1):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        responses = [
            _FakeResp(202, {"message": "Index not yet available", "code": 110000, "retry_after": 1.25}),
            _FakeResp(202, {"message": "Index not yet available", "code": 110000, "retry_after": 2.0}),
            _FakeResp(200, {"total_results": 1, "messages": [[{"id": "999", "hit": True, "content": "github.com/BerriAI/litellm/pull/77"}]]}),
        ]
        sleeps = []

        with mock.patch("contributor_bot.api._open_request", side_effect=responses):
            payload = search_guild_messages(
                "333", "github.com/BerriAI/litellm/pull", "tok", offset=0, sleep_fn=sleeps.append
            )

        self.assertEqual(sleeps, [1.25, 2.0])
        self.assertEqual(payload["total_results"], 1)
        with self.assertRaises(ValueError):
            search_guild_messages("333", "github.com/BerriAI/litellm/pull", "tok", offset=9976)

    def test_links_backfill_mod_only_and_pages_to_9975(self):
        cfg = Config("tok", "123", "456", discord_guild_id="333", discord_application_id="444")

        # Non-moderator is rejected!
        non_mod_interaction = {
            "type": 2,
            "application_id": "444",
            "guild_id": "333",
            "data": {"name": "links-backfill"},
            "member": {"permissions": "0", "user": {"id": "100000000000000001"}},
        }
        reply = handle_interaction(cfg, self.store, non_mod_interaction)
        self.assertEqual(reply["data"]["flags"], 64)
        self.assertIn("only mods", reply["data"]["content"].lower())

        # Moderator interaction defers (type 5) and triggers run_links_backfill in background!
        import threading as _threading
        done_evt = _threading.Event()
        mod_interaction = {
            "type": 2,
            "application_id": "444",
            "guild_id": "333",
            "token": "interaction-tok",
            "data": {"name": "links-backfill"},
            "member": {"permissions": str(1 << 13), "user": {"id": "100000000000000002"}},
        }

        def _fake_bf(*args, **kwargs):
            done_evt.set()
            return None

        with mock.patch("contributor_bot.links_backfill.run_links_backfill", side_effect=_fake_bf) as mocked_bf, \
                mock.patch("contributor_bot.interactions.edit_original_interaction"):
            mod_reply = handle_interaction(cfg, self.store, mod_interaction)
            self.assertTrue(done_evt.wait(timeout=2.0))
        self.assertEqual(mod_reply["type"], 5)
        self.assertEqual(mod_reply["data"]["flags"], 64)
        mocked_bf.assert_called_once()

        # Moderator can run backfill and paging stops at offset 9975 (never 10000).
        offsets_seen = []

        def fake_search(guild_id, content, token, *, offset=0, sleep_fn=None):
            offsets_seen.append(offset)
            # Simulate a large guild with >10,000 results to confirm hard cap at offset=9975.
            msgs = []
            for i in range(25):
                pr_num = offset + i + 1
                msgs.append([{
                    "id": str(800000000000000000 + pr_num),
                    "hit": True,
                    "content": f"my pr https://github.com/BerriAI/litellm/pull/{pr_num}",
                    "timestamp": "2026-11-25T10:00:00Z",
                    "author": {"id": "200000000000000001", "bot": (i == 0)},
                }])
            return {"total_results": 12000, "messages": msgs}

        # Jump right near the upper cap by overriding GUILD_SEARCH_MAX_OFFSET in a quick test
        # AND verify the constant itself is 9975.
        self.assertEqual(GUILD_SEARCH_MAX_OFFSET, 9975)
        with mock.patch("contributor_bot.links_backfill.GUILD_SEARCH_MAX_OFFSET", 50):
            res = run_links_backfill(cfg, self.store, search_fn=fake_search, sleep_fn=lambda _: None)

        self.assertIsNotNone(res)
        self.assertEqual(offsets_seen, [0, 25, 50])
        self.assertEqual(res.search_pages, 3)
        # 24 non-bot messages per page * 3 pages = 72 recorded links!
        self.assertEqual(res.links_recorded, 72)

    def test_pr_link_poster_tier_rules_and_staff_exclusion(self):
        members = [
            DiscordMemberInfo("300000000000000001", username="staff_berri", global_name="Staff", role_ids=("999999999999999999",)),
            DiscordMemberInfo("300000000000000002", username="totally_diff_handle", global_name="Diff Handle"),
            DiscordMemberInfo("300000000000000003", username="samantha_r", global_name="Sam Rivera"),
            DiscordMemberInfo("300000000000000004", username="other_member", global_name="Other Member"),
        ]
        cfg = Config("tok", "123", "456", review_maintainer_role_id="999999999999999999")
        excluded = _excluded_staff_discord_ids(cfg, self.store, members, org_members=frozenset({"org-staff"}), org="BerriAI")
        self.assertIn("300000000000000001", excluded)

        # 1) Staff poster with 3 PRs is EXCLUDED (never matched).
        prof = GitHubAuthorProfile(login="unusual_gh_author", name="Zed Q")
        strong, cands = score_author_against_members(
            prof,
            members,
            pr_link_posters=[PRLinkPosterEvidence("300000000000000001", (1, 2, 3), ("my pr",))],
            excluded_discord_ids=excluded,
        )
        self.assertIsNone(strong)
        self.assertEqual(cands, [])

        # 2) One non-staff poster linked >=2 distinct PRs WITHOUT ownership text or name similarity >= 0.5
        #    must NEVER auto-accept on its own — it is only passed as an LLM signal!
        strong, cands = score_author_against_members(
            prof,
            members,
            pr_link_posters=[
                PRLinkPosterEvidence("300000000000000001", (1, 2), ("triage",)),  # staff ignored
                PRLinkPosterEvidence("300000000000000002", (1, 2), ("link",)),
            ],
            excluded_discord_ids=excluded,
        )
        self.assertIsNone(strong)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0].member.discord_id, "300000000000000002")
        self.assertIn("pr_link_poster:2_prs", cands[0].signals)

        # 2b) Multi-author poster (reviewer/triager who linked PRs by >=2 different authors)
        #     is rejected completely (not even passed to LLM).
        strong, cands = score_author_against_members(
            prof,
            members,
            pr_link_posters=[
                PRLinkPosterEvidence("300000000000000002", (1, 2), ("my pr",)),
            ],
            excluded_discord_ids=excluded,
            multi_author_ids=frozenset({"300000000000000002"}),
        )
        self.assertIsNone(strong)
        self.assertEqual(cands, [])

        # 3) One non-staff poster linked 1 PR + ownership phrase -> accepted at 0.92!
        strong, _cands = score_author_against_members(
            prof,
            members,
            pr_link_posters=[PRLinkPosterEvidence("300000000000000002", (10,), ("Hi, I opened this PR, please review",))],
            excluded_discord_ids=excluded,
        )
        self.assertIsNotNone(strong)
        self.assertEqual((strong[0].discord_id, strong[1], strong[2]), ("300000000000000002", "pr_link_poster", 0.92))

        # 4) One non-staff poster linked 1 PR + name similarity >= 0.5 -> accepted at 0.92!
        prof_sam = GitHubAuthorProfile(login="sam_riv_dev", name="Samantha Rivera")
        strong, _cands = score_author_against_members(
            prof_sam,
            members,
            pr_link_posters=[PRLinkPosterEvidence("300000000000000003", (11,), ("https://github.com/BerriAI/litellm/pull/11",))],
            excluded_discord_ids=excluded,
        )
        self.assertIsNotNone(strong)
        self.assertEqual((strong[0].discord_id, strong[1], strong[2]), ("300000000000000003", "pr_link_poster", 0.92))

        # 5) One non-staff poster linked 1 PR WITHOUT similarity or ownership phrase -> passed to LLM!
        strong, cands = score_author_against_members(
            prof,
            members,
            pr_link_posters=[PRLinkPosterEvidence("300000000000000002", (12,), ("check out https://github.com/BerriAI/litellm/pull/12",))],
            excluded_discord_ids=excluded,
        )
        self.assertIsNone(strong)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0].member.discord_id, "300000000000000002")
        self.assertIn("pr_link_poster:1_prs", cands[0].signals)

        # 6) Two different non-staff posters linked author's PRs -> NOT deterministic, both passed to LLM!
        strong, cands = score_author_against_members(
            prof,
            members,
            pr_link_posters=[
                PRLinkPosterEvidence("300000000000000002", (12, 13), ("my pr",)),
                PRLinkPosterEvidence("300000000000000004", (14,), ("nice fix",)),
            ],
            excluded_discord_ids=excluded,
        )
        self.assertIsNone(strong)
        self.assertEqual({c.member.discord_id for c in cands}, {"300000000000000002", "300000000000000004"})


class TestGraphQLBatchEnrichment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(str(Path(self.tmp.name) / "bot.db"))
        self.addCleanup(self.store.close)

    def test_fetch_github_profiles_graphql_batches_50_logins_and_caps_readme_4kb(self):
        logins = [f"user{i}" for i in range(65)]
        batch_sizes = []
        big_readme = "A" * 6000

        def fake_gql(token, query, variables):
            batch_sizes.append(len(variables))
            data = {}
            for key, login in variables.items():
                idx = key[1:]
                data[f"u{idx}"] = {
                    "login": login,
                    "name": f"Name {login}",
                    "bio": f"Bio {login}",
                    "company": "@AcmeLabs",
                    "email": f"{login}@acme.dev",
                    "twitterUsername": f"tw_{login}",
                    "websiteUrl": f"https://{login}.io/blog",
                    "socialAccounts": {"nodes": [{"provider": "GENERIC", "displayName": f"soc_{login}", "url": "https://discord.com/users/123456789012345678"}]},
                    "profileRepo": {"object": {"text": big_readme}},
                }
            return {"data": data}

        profiles, rate_limited = fetch_github_profiles_graphql(logins, "gh-tok", graphql_caller=fake_gql)
        self.assertFalse(rate_limited)
        self.assertEqual(batch_sizes, [50, 15])
        self.assertEqual(len(profiles), 65)
        p0 = profiles["user0"]
        self.assertEqual(p0.company, "@AcmeLabs")
        self.assertEqual(p0.email, "user0@acme.dev")
        self.assertEqual(p0.email_locals, ("user0",))
        self.assertEqual(len(p0.readme.encode("utf-8")), PROFILE_README_MAX_BYTES)
        self.assertEqual(len(p0.social_accounts), 1)

    def test_readme_and_social_discord_extraction_and_profile_norms(self):
        members = [
            DiscordMemberInfo("400000000000000001", username="readme_handle_user", global_name="RHU"),
            DiscordMemberInfo("400000000000000002", username="social_discord_user", global_name="SDU"),
            DiscordMemberInfo("400000000000000003", username="someone_else", global_name="SE"),
        ]
        # Discord handle in profile README -> deterministic match!
        prof_readme = GitHubAuthorProfile(
            login="gh-readme-dev",
            readme="Hi! Reach me on Discord: @readme_handle_user",
        )
        strong, _ = score_author_against_members(prof_readme, members)
        self.assertIsNotNone(strong)
        self.assertEqual((strong[0].discord_id, strong[1]), ("400000000000000001", "readme_discord_handle"))

        # Discord handle in socialAccounts -> deterministic match!
        prof_social = GitHubAuthorProfile(
            login="gh-social-dev",
            social_accounts=({"provider": "DISCORD", "displayName": "social_discord_user", "url": ""},),
        )
        strong, _ = score_author_against_members(prof_social, members)
        self.assertIsNotNone(strong)
        self.assertEqual((strong[0].discord_id, strong[1]), ("400000000000000002", "social_discord_handle"))

        # Discord user ID in socialAccounts URL -> deterministic match!
        prof_social_id = GitHubAuthorProfile(
            login="gh-social-id",
            social_accounts=({"provider": "GENERIC", "displayName": "Discord", "url": "https://discord.com/users/400000000000000003"},),
        )
        strong, _ = score_author_against_members(prof_social_id, members)
        self.assertIsNotNone(strong)
        self.assertEqual((strong[0].discord_id, strong[1]), ("400000000000000003", "social_discord_id"))

        # Website domain, social displayName, and company added to profile norms!
        prof_norms = GitHubAuthorProfile(
            login="gh-x",
            blog="https://www.hyperion-labs.dev/about",
            company="@NebulaSystems Inc.",
            social_accounts=({"provider": "LINKEDIN", "displayName": "kiran_patel", "url": "https://linkedin.com/in/kiran"},),
        )
        norms = _profile_source_norms(prof_norms)
        labels = {label for label, _val in norms}
        values = {val for _label, val in norms}
        self.assertIn("website_domain", labels)
        self.assertIn("social_display", labels)
        self.assertIn("company", labels)
        self.assertIn("hyperionlabs", values)
        self.assertIn("kiranpatel", values)
        self.assertIn("nebulasystems", values)

    def test_full_profile_7_day_sqlite_cache_includes_commits(self):
        cfg = Config("tok", "123", "456", github_token="gh-tok")
        now = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)
        gql_calls = []

        def fake_gql(token, query, variables):
            gql_calls.append(query)
            if "BatchProfiles" in query:
                return {
                    "data": {
                        "u0": {
                            "login": "gh-cache-user",
                            "name": "Different Name",
                            "bio": "",
                            "company": "@CacheCo",
                            "email": "",
                            "twitterUsername": "",
                            "websiteUrl": "",
                            "socialAccounts": {"nodes": []},
                            "profileRepo": None,
                        },
                    }
                }
            # BatchPRCommits
            return {
                "data": {
                    "repository": {
                        "p0": {
                            "commits": {
                                "nodes": [
                                    {"commit": {"author": {"name": "Nadia K", "email": "nadia_cached@example.com"}}},
                                ]
                            }
                        }
                    }
                }
            }

        members = [DiscordMemberInfo("400000000000000010", username="nadia_cached", global_name="Nadia K")]
        groups = [{"login": "gh-cache-user", "user_id": 77, "pulls": [{"number": 1001, "title": "fix", "url": "https://github.com/BerriAI/litellm/pull/1001"}]}]

        # Run 1: GraphQL fetches profile + commits, persists full profile to github_users_cache.
        def reject_llm(model, messages):
            return json.dumps({"discord_id": None, "confidence": 0.1, "reason": "none", "evidence": []})

        match_community_contributors_with_diagnostics(
            groups, cfg, self.store, now=now, members=members,
            graphql_caller=fake_gql, llm_complete=reject_llm,
        )
        self.assertEqual(len(gql_calls), 2)
        cached = self.store.get_cached_github_user("gh-cache-user")
        self.assertIsNotNone(cached)
        self.assertIn("nadia_cached", cached["profile_json"])
        self.assertIn("Nadia K", cached["profile_json"])

        # Run 2 (day +3, new store attempt / clear negative cache to force re-read of profile cache):
        # MUST make 0 GraphQL calls and still have cached commit email/name!
        self.store.delete_match_attempt("gh-cache-user")
        gql_calls.clear()
        seen_profiles = []

        def inspect_llm(model, messages):
            payload = json.loads(messages[1]["content"])
            seen_profiles.append(payload["github_author"])
            return json.dumps({"discord_id": None, "confidence": 0.1, "reason": "none", "evidence": []})

        match_community_contributors_with_diagnostics(
            groups, cfg, self.store, now=now + timedelta(days=3), members=members,
            graphql_caller=fake_gql, llm_complete=inspect_llm,
        )
        self.assertEqual(len(gql_calls), 0)
        self.assertEqual(len(seen_profiles), 1)
        self.assertIn("Nadia K", seen_profiles[0]["commit_names"])
        self.assertIn("nadia_cached", seen_profiles[0]["email_locals"])
        self.assertEqual(seen_profiles[0]["company"], "@CacheCo")

        # Run 3 (day +8): cache expired -> re-fetches via GraphQL!
        self.store.delete_match_attempt("gh-cache-user")
        match_community_contributors_with_diagnostics(
            groups, cfg, self.store, now=now + timedelta(days=8), members=members,
            graphql_caller=fake_gql, llm_complete=inspect_llm,
        )
        self.assertEqual(len(gql_calls), 2)


class TestListWiseLLMSelection(unittest.TestCase):
    def test_select_with_llm_thresholds_and_none_of_these(self):
        cfg = Config("tok", "123", "456", match_model="anthropic/claude-sonnet-4-5")
        members = [
            DiscordMemberInfo("500000000000000001", username="jdoe99", global_name="Jordan D"),
            DiscordMemberInfo("500000000000000002", username="jdoe_alt", global_name="J Doe"),
        ]
        groups = [{"login": "jdoe_dev", "user_id": 88, "pulls": []}]

        # 1) >=0.85 with >=2 corroborating signals -> ACCEPTED!
        def accept_multi(model, messages):
            return json.dumps({
                "discord_id": "500000000000000001",
                "confidence": 0.86,
                "reason": "matches login prefix and full name",
                "evidence": ["login prefix jdoe matches jdoe99", "name Jordan Doe matches Jordan D"],
            })

        matched, skipped, _details, _diag = match_community_contributors_with_diagnostics(
            groups, cfg, None, members=members, skip_github_enrichment=True, llm_complete=accept_multi
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["discord_id"], "500000000000000001")
        self.assertEqual(skipped, 0)

        # 2) <0.85 even with 2 signals -> REJECTED!
        def reject_below_85(model, messages):
            return json.dumps({
                "discord_id": "500000000000000001",
                "confidence": 0.82,
                "reason": "confidence below 0.85",
                "evidence": ["login prefix match", "name match"],
            })

        matched, skipped, _details, diag = match_community_contributors_with_diagnostics(
            groups, cfg, None, members=members, skip_github_enrichment=True, llm_complete=reject_below_85
        )
        self.assertEqual(matched, [])
        self.assertEqual(skipped, 1)
        self.assertEqual(dict(diag.skip_reasons), {"llm_below_threshold": 1})

        # 3) Single signal (even at confidence 0.95) -> REJECTED (no single-signal accepts)!
        def reject_solo_95(model, messages):
            return json.dumps({
                "discord_id": "500000000000000002",
                "confidence": 0.95,
                "reason": "single signal only",
                "evidence": ["unique handle"],
            })

        matched, skipped, _details, diag = match_community_contributors_with_diagnostics(
            groups, cfg, None, members=members, skip_github_enrichment=True, llm_complete=reject_solo_95
        )
        self.assertEqual(matched, [])
        self.assertEqual(skipped, 1)
        self.assertEqual(dict(diag.skip_reasons), {"llm_below_threshold": 1})

        # 4) Explicit 'none of these' (discord_id: null) -> REJECTED with llm_none_of_these!
        def choose_none(model, messages):
            return json.dumps({
                "discord_id": None,
                "confidence": 0.95,
                "reason": "none of these match closely enough",
                "evidence": [],
            })

        matched, skipped, _details, diag = match_community_contributors_with_diagnostics(
            groups, cfg, None, members=members, skip_github_enrichment=True, llm_complete=choose_none
        )
        self.assertEqual(matched, [])
        self.assertEqual(dict(diag.skip_reasons), {"llm_none_of_these": 1})

    def test_match_model_resolution_and_max_5_candidates_and_logging(self):
        self.assertEqual(MAX_CANDIDATES_PER_AUTHOR, 5)
        cfg_explicit = Config("tok", "123", "456", match_model="openai/gpt-5", review_model="anthropic/claude-sonnet-4-5")
        self.assertEqual(_resolve_match_model(cfg_explicit), "openai/gpt-5")

        cfg_fallback = Config("tok", "123", "456", match_model="", review_model="anthropic/claude-sonnet-4-5")
        with mock.patch.dict(os.environ, {"MATCH_MODEL": ""}, clear=False):
            self.assertEqual(_resolve_match_model(cfg_fallback), "anthropic/claude-sonnet-4-5")

        cfg_default = Config("tok", "123", "456", match_model="", review_model="")
        with mock.patch.dict(os.environ, {"MATCH_MODEL": ""}, clear=False):
            self.assertEqual(_resolve_match_model(cfg_default), DEFAULT_MATCH_MODEL)

        parsed_cfg = get_config({
            "DISCORD_BOT_TOKEN": "tok",
            "OSS_CONTRIBUTIONS_CHANNEL_ID": "123",
            "ANNOUNCEMENTS_CHANNEL_ID": "456",
            "MATCH_MODEL": "anthropic/claude-sonnet-4-5",
        })
        self.assertEqual(parsed_cfg.match_model, "anthropic/claude-sonnet-4-5")

        # 8 fuzzy members -> capped at MAX_CANDIDATES_PER_AUTHOR = 5 in LLM prompt!
        members = [
            DiscordMemberInfo(f"60000000000000000{i}", username=f"alexdev_{i}", global_name=f"Alex Dev {i}")
            for i in range(1, 9)
        ] + [
            DiscordMemberInfo("600000000000000099", username="exactwinner", global_name="Exact Winner"),
        ]
        groups = [
            {"login": "exactwinner", "user_id": 1, "pulls": []},
            {"login": "alexdev_x", "user_id": 2, "pulls": []},
            {"login": "zzz_no_match", "user_id": 3, "pulls": []},
        ]
        candidate_counts = []

        def fake_llm(model, messages):
            payload = json.loads(messages[1]["content"])
            candidate_counts.append(len(payload["candidates"]))
            return json.dumps({"discord_id": None, "confidence": 0.25, "reason": "ambiguous", "evidence": []})

        with self.assertLogs(level="INFO") as logs:
            matched, skipped, _details, diag = match_community_contributors_with_diagnostics(
                groups, cfg_fallback, None, members=members, skip_github_enrichment=True, llm_complete=fake_llm
            )

        self.assertEqual(candidate_counts, [5])
        self.assertEqual(len(matched), 1)
        self.assertEqual(skipped, 2)
        self.assertEqual(dict(diag.method_counts), {"exact_login": 1})
        # One INFO log per skipped author with top_score + reason, AND matched-per-method log!
        self.assertTrue(any("Match skipped for alexdev_x: reason=llm_none_of_these top_score=" in line for line in logs.output))
        self.assertTrue(any("Match skipped for zzz_no_match: reason=no_signal_match top_score=" in line for line in logs.output))
        self.assertTrue(any("Match summary by method: {'exact_login': 1}" in line for line in logs.output))


class TestNegativeCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(str(Path(self.tmp.name) / "bot.db"))
        self.addCleanup(self.store.close)

    def test_negative_cache_skips_7d_and_invalidates_on_pr_link_or_signature_or_expiry(self):
        cfg = Config("tok", "123", "456", github_repository="BerriAI/litellm")
        now = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
        members = [
            DiscordMemberInfo("700000000000000001", username="dev_mira_a", global_name="Mira A"),
            DiscordMemberInfo("700000000000000002", username="other_person", global_name="Other Person"),
        ]
        groups = [
            {
                "login": "dev-mira-x",
                "user_id": 55,
                "pulls": [{"number": 701, "title": "fix", "url": "https://github.com/BerriAI/litellm/pull/701"}],
            }
        ]
        llm_calls = []

        def fake_llm(model, messages):
            llm_calls.append(messages)
            return json.dumps({"discord_id": None, "confidence": 0.2, "reason": "none of these", "evidence": []})

        # Run 1: LLM returns none_of_these -> persisted to match_attempts.
        matched1, skipped1, _d1, diag1 = match_community_contributors_with_diagnostics(
            groups, cfg, self.store, now=now, members=members,
            skip_github_enrichment=True, llm_complete=fake_llm,
        )
        self.assertEqual((len(matched1), skipped1), (0, 1))
        self.assertEqual(len(llm_calls), 1)
        self.assertEqual(dict(diag1.skip_reasons), {"llm_none_of_these": 1})
        attempt = self.store.get_match_attempt("dev-mira-x")
        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["outcome"], "no_match")

        # Run 2 (day +3, identical inputs): skipped via cached_negative with 0 LLM calls!
        llm_calls.clear()
        matched2, skipped2, _d2, diag2 = match_community_contributors_with_diagnostics(
            groups, cfg, self.store, now=now + timedelta(days=3), members=members,
            skip_github_enrichment=True, llm_complete=fake_llm,
        )
        self.assertEqual((len(matched2), skipped2), (0, 1))
        self.assertEqual(len(llm_calls), 0)
        self.assertEqual(dict(diag2.skip_reasons), {"cached_negative": 1})

        # Run 3 (day +8, > 7 days): negative cache expired -> LLM is called again!
        matched3, skipped3, _d3, diag3 = match_community_contributors_with_diagnostics(
            groups, cfg, self.store, now=now + timedelta(days=8), members=members,
            skip_github_enrichment=True, llm_complete=fake_llm,
        )
        self.assertEqual(len(llm_calls), 1)
        self.assertEqual(dict(diag3.skip_reasons), {"llm_none_of_these": 1})

        # Run 4 (day +9, new member joins guild changing candidate signature): cache invalidated!
        llm_calls.clear()
        members_expanded = members + [
            DiscordMemberInfo("700000000000000003", username="dev_mira_b", global_name="Mira B"),
        ]
        match_community_contributors_with_diagnostics(
            groups, cfg, self.store, now=now + timedelta(days=9), members=members_expanded,
            skip_github_enrichment=True, llm_complete=fake_llm,
        )
        self.assertEqual(len(llm_calls), 1)

        # Run 5 (day +10, new pr_link_posters row added for PR #701 with ownership text):
        # cache invalidated AND pr_link_poster matches at 0.92 AND match_attempts row cleared!
        llm_calls.clear()
        self.store.record_pr_link_poster(
            "BerriAI/litellm", 701, "700000000000000002", "900000000000000701",
            "2026-11-30T10:00:00+00:00", message_text="please review my PR https://github.com/BerriAI/litellm/pull/701",
        )
        matched5, skipped5, _d5, _diag5 = match_community_contributors_with_diagnostics(
            groups, cfg, self.store, now=now + timedelta(days=10), members=members_expanded,
            skip_github_enrichment=True, llm_complete=fake_llm,
        )
        self.assertEqual((len(matched5), skipped5), (1, 0))
        self.assertEqual(matched5[0]["discord_id"], "700000000000000002")
        self.assertEqual(matched5[0]["match_method"], "pr_link_poster")
        self.assertEqual(len(llm_calls), 0)
        self.assertIsNone(self.store.get_match_attempt("dev-mira-x"))

    def test_transient_llm_unavailable_is_not_negative_cached(self):
        cfg = Config("tok", "123", "456")
        now = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
        members = [DiscordMemberInfo("700000000000000001", username="dev_mira_a", global_name="Mira A")]
        groups = [{"login": "dev_mira_x", "user_id": 55, "pulls": []}]

        def boom_llm(model, messages):
            raise RuntimeError("proxy timeout")

        matched, skipped, _d, diag = match_community_contributors_with_diagnostics(
            groups, cfg, self.store, now=now, members=members,
            skip_github_enrichment=True, llm_complete=boom_llm,
        )
        self.assertEqual((len(matched), skipped), (0, 1))
        self.assertEqual(dict(diag.skip_reasons), {"llm_unavailable": 1})
        self.assertIsNone(self.store.get_match_attempt("dev_mira_x"))


if __name__ == "__main__":
    unittest.main()
