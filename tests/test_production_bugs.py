from __future__ import annotations

import io
import json
import logging
import sqlite3
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request

from contributor_bot import api, review_job
from contributor_bot.api import TransientAPIError
from contributor_bot.config import Config
from contributor_bot.matcher import (
    DiscordMemberInfo,
    fetch_github_profiles_graphql,
    match_community_contributors_with_diagnostics,
)
from contributor_bot.review_agent import AgentResult
from contributor_bot.review_job import (
    handle_channel_message,
    handle_thread_message,
    perform_review,
    poll_review_channels,
    recover_pending_reviews,
)
from contributor_bot.roles_sync import poll_merged_prs_for_roles
from contributor_bot.schedule import UTC, iso
from contributor_bot.store import Store


def _make_config(state_path: Path, **overrides) -> Config:
    base = Config(
        discord_token="test-discord-token",
        contributions_channel_id="100000000000000000",
        announcements_channel_id="100000000000000001",
        discord_guild_id="300000000000000001",
        github_repository="BerriAI/litellm",
        github_token="ghp_test_token",
        roles_db_path=str(state_path),
        pr_review_enabled=True,
        review_channel_ids=("1391835163780186154",),
        review_classify=False,
        review_require_ask=False,
        review_cooldown_seconds=600,
        review_daily_run_budget=100,
        contributor_roles_enabled=True,
        role_ids={"contributor": "400000000000000001"},
        roles_announce_enabled=False,
    )
    return replace(base, **overrides)


class MultiPrReviewAndMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "state.db"
        self.config = _make_config(self.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_legacy_unique_thread_id_schema_migrates_and_allows_shared_thread(self) -> None:
        # Build a legacy SQLite database matching production before this fix (UNIQUE thread_id).
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE review_threads (
                repository TEXT NOT NULL,
                number INTEGER NOT NULL,
                channel_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                thread_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                last_reviewed_at TEXT,
                head_sha TEXT,
                last_followup_at TEXT,
                PRIMARY KEY (repository, number)
            );
            INSERT INTO review_threads (
                repository, number, channel_id, message_id, thread_id, created_at
            ) VALUES (
                'BerriAI/litellm', 54852, '1391835163780186154', '1576292666428104857',
                '999000111222333444', '2026-11-28T17:44:37+00:00'
            );
            """
        )
        conn.commit()
        conn.close()

        store = Store(self.db_path)
        # Existing row is preserved.
        existing = store.review_thread("BerriAI/litellm", 54852)
        self.assertIsNotNone(existing)
        self.assertEqual(existing["thread_id"], "999000111222333444")

        # Second PR from the same message reuses the exact same thread_id with zero IntegrityError.
        store.save_review_thread(
            repository="BerriAI/litellm",
            number=54805,
            channel_id="1391835163780186154",
            message_id="1576292666428104857",
            thread_id="999000111222333444",
            created_at="2026-11-28T17:44:38+00:00",
        )
        threads = store.review_threads_by_discord("999000111222333444")
        self.assertEqual([t["number"] for t in threads], [54852, 54805])
        by_msg = store.review_thread_by_message("1391835163780186154", "1576292666428104857")
        self.assertIsNotNone(by_msg)
        self.assertEqual(by_msg["thread_id"], "999000111222333444")

    def test_concurrent_two_prs_in_one_message_share_single_thread_and_isolate_failures(self) -> None:
        store = Store(self.db_path)
        thread_creates: list[tuple[str, str, str]] = []
        reviewed_numbers: list[int] = []
        lock = threading.Lock()

        def fake_start_thread(channel_id: str, message_id: str, token: str, name: str) -> dict:
            # Delay slightly to maximize concurrency overlap between #54805 and #54852.
            time.sleep(0.05)
            with lock:
                thread_creates.append((channel_id, message_id, name))
            return {"id": "777000000000000001"}

        def fake_session(config, st, number: int, thread_id: str, *, mode: str, history: str) -> AgentResult:
            with lock:
                reviewed_numbers.append(number)
            if number == 54805:
                raise RuntimeError("simulated crash for 54805")
            return AgentResult("looks good", verdict_posted=True, ready=True, head_sha="a" * 40)

        message = {
            "id": "1576292666428104857",
            "author": {"id": "222000000000000001", "bot": False},
            "content": (
                "please review https://github.com/BerriAI/litellm/pull/54805 "
                "and https://github.com/BerriAI/litellm/pull/54852"
            ),
        }

        with (
            mock.patch.object(review_job, "start_public_thread", side_effect=fake_start_thread),
            mock.patch.object(review_job, "fetch_recent_messages", return_value=[]),
            mock.patch.object(review_job, "run_review_session", side_effect=fake_session),
            mock.patch.object(review_job, "acknowledge_message"),
            self.assertLogs(level="ERROR") as logs,
        ):
            handle_channel_message(self.config, store, "1391835163780186154", message)
            for thread in list(threading.enumerate()):
                if thread.name == "pr-review":
                    thread.join(timeout=5.0)

        # Thread creation was serialized: Discord only received ONE start_public_thread call.
        self.assertEqual(len(thread_creates), 1)
        # Both PRs ran and share the same thread_id in SQLite.
        self.assertEqual(sorted(reviewed_numbers), [54805, 54852])
        rec_a = store.review_thread("BerriAI/litellm", 54805)
        rec_b = store.review_thread("BerriAI/litellm", 54852)
        self.assertIsNotNone(rec_a)
        self.assertIsNotNone(rec_b)
        self.assertEqual(rec_a["thread_id"], "777000000000000001")
        self.assertEqual(rec_b["thread_id"], "777000000000000001")
        # #54852 succeeded despite #54805 raising an error.
        self.assertIsNotNone(rec_b["last_reviewed_at"])
        self.assertIsNone(rec_a["last_reviewed_at"])
        # Durable review_requests records exact per-PR outcomes.
        req_a = store.get_review_request("BerriAI/litellm", 54805, "1576292666428104857")
        req_b = store.get_review_request("BerriAI/litellm", 54852, "1576292666428104857")
        self.assertEqual(req_a["status"], "failed")
        self.assertEqual(req_b["status"], "done")
        # Error log includes the failing PR number.
        self.assertTrue(any("#54805" in entry for entry in logs.output))

    def test_shared_thread_recheck_targets_named_pr_or_all(self) -> None:
        store = Store(self.db_path)
        store.save_review_thread(
            repository="BerriAI/litellm",
            number=54805,
            channel_id="1391835163780186154",
            message_id="1576292666428104857",
            thread_id="777000000000000001",
            created_at="2026-11-28T17:44:37+00:00",
        )
        store.save_review_thread(
            repository="BerriAI/litellm",
            number=54852,
            channel_id="1391835163780186154",
            message_id="1576292666428104857",
            thread_id="777000000000000001",
            created_at="2026-11-28T17:44:37+00:00",
        )
        called: list[int] = []
        with (
            mock.patch.object(review_job, "perform_review", side_effect=lambda cfg, st, num, **kw: called.append(num)),
            mock.patch.object(review_job, "acknowledge_message"),
        ):
            handle_thread_message(
                self.config,
                store,
                {
                    "id": "888000000000000001",
                    "channel_id": "777000000000000001",
                    "author": {"id": "222000000000000001", "bot": False},
                    "content": "fixed #54852, please recheck",
                },
            )
        self.assertEqual(called, [54852])


class BootRecoveryAndTransientPollTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "state.db"
        self.config = _make_config(self.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_boot_recovery_retries_pending_failed_and_24h_unreviewed_without_duplicates(self) -> None:
        store = Store(self.db_path)
        now = datetime(2026, 11, 28, 18, 0, 0, tzinfo=UTC)

        # 1. #54700 already completed in DB -> must NOT be retried.
        store.save_review_thread(
            repository="BerriAI/litellm",
            number=54700,
            channel_id="1391835163780186154",
            message_id="1576290000000000001",
            thread_id="700000000000000001",
            created_at=iso(now - timedelta(hours=3)),
        )
        store.mark_review_done("BerriAI/litellm", 54700, iso(now - timedelta(hours=3)), "1" * 40)

        # 2. #54852 started before restart, created thread, never finished -> MUST be retried.
        store.save_review_thread(
            repository="BerriAI/litellm",
            number=54852,
            channel_id="1391835163780186154",
            message_id="1576292666428104857",
            thread_id="700000000000000002",
            created_at=iso(now - timedelta(minutes=15)),
        )

        # 3. Durable failed request within 24h (attempts=1) -> MUST be retried.
        store.upsert_review_request(
            repository="BerriAI/litellm",
            number=54860,
            channel_id="1391835163780186154",
            message_id="1576292666428104900",
            status="pending",
            source="new",
            now_iso=iso(now - timedelta(hours=1)),
        )
        store.mark_review_request(
            "BerriAI/litellm", 54860, "1576292666428104900", "failed",
            error="timeout", updated_at=iso(now - timedelta(minutes=50)), increment_attempts=True,
        )

        # 4. Exhausted failed request (attempts=3) -> MUST NOT loop forever.
        store.upsert_review_request(
            repository="BerriAI/litellm",
            number=54861,
            channel_id="1391835163780186154",
            message_id="1576292666428104901",
            status="pending",
            source="new",
            now_iso=iso(now - timedelta(hours=1)),
        )
        for _ in range(3):
            store.mark_review_request(
                "BerriAI/litellm", 54861, "1576292666428104901", "failed",
                error="permanent", updated_at=iso(now - timedelta(minutes=30)), increment_attempts=True,
            )

        # Recent Discord messages in the review channel:
        # - 30h old (#54000): outside 24h window -> skip
        # - 3h old (#54700): already reviewed in DB -> skip
        # - 2h old (#54750): not in DB, but its Discord thread already has a bot review -> skip
        # - 15m old (#54805 + #54852): #54805 crashed before DB save, #54852 incomplete -> BOTH recover!
        channel_messages = [
            {
                "id": "1576200000000000001",
                "timestamp": iso(now - timedelta(hours=30)),
                "author": {"id": "222000000000000001", "bot": False},
                "content": "please review https://github.com/BerriAI/litellm/pull/54000",
            },
            {
                "id": "1576290000000000001",
                "timestamp": iso(now - timedelta(hours=3)),
                "author": {"id": "222000000000000001", "bot": False},
                "content": "please review https://github.com/BerriAI/litellm/pull/54700",
            },
            {
                "id": "1576291000000000001",
                "timestamp": iso(now - timedelta(hours=2)),
                "author": {"id": "222000000000000001", "bot": False},
                "content": "please review https://github.com/BerriAI/litellm/pull/54750",
                "thread": {"id": "700000000000000099"},
            },
            {
                "id": "1576292666428104857",
                "timestamp": iso(now - timedelta(minutes=15)),
                "author": {"id": "222000000000000001", "bot": False},
                "content": (
                    "please review https://github.com/BerriAI/litellm/pull/54805 "
                    "and https://github.com/BerriAI/litellm/pull/54852"
                ),
                "thread": {"id": "700000000000000002"},
            },
        ]

        def fake_fetch_recent(channel_or_thread_id: str, token: str, limit: int = 50) -> list[dict]:
            if channel_or_thread_id == "1391835163780186154":
                return list(channel_messages)
            if channel_or_thread_id == "700000000000000099":
                return [{"id": "1", "author": {"id": "bot", "bot": True}, "content": "ready for maintainer review!"}]
            return []

        scheduled_tasks: list = []
        with (
            mock.patch.object(review_job, "fetch_recent_messages", side_effect=fake_fetch_recent),
            mock.patch.object(review_job, "_spawn", side_effect=lambda fn: scheduled_tasks.append(fn)),
        ):
            queued = recover_pending_reviews(self.config, store, now=now)

        self.assertEqual(sorted(queued), [54805, 54852, 54860])
        self.assertEqual(len(scheduled_tasks), 3)

        # Execute recovered tasks and verify completion prevents any duplicate on next boot.
        with (
            mock.patch.object(
                review_job, "start_public_thread",
                side_effect=lambda ch, msg, tok, name: {"id": f"800{msg[-6:]}"},
            ),
            mock.patch.object(review_job, "fetch_recent_messages", return_value=[]),
            mock.patch.object(
                review_job, "run_review_session",
                return_value=AgentResult("verdict", verdict_posted=True, ready=True, head_sha="b" * 40),
            ),
        ):
            for task in scheduled_tasks:
                task()

        # Verify #54805 reused #54852's existing thread on the same message!
        self.assertEqual(
            store.review_thread("BerriAI/litellm", 54805)["thread_id"],
            "700000000000000002",
        )
        self.assertIsNotNone(store.review_thread("BerriAI/litellm", 54805)["last_reviewed_at"])
        self.assertIsNotNone(store.review_thread("BerriAI/litellm", 54852)["last_reviewed_at"])
        self.assertIsNotNone(store.review_thread("BerriAI/litellm", 54860)["last_reviewed_at"])

        # Second restart: zero duplicates!
        scheduled_again: list = []
        with (
            mock.patch.object(review_job, "fetch_recent_messages", side_effect=fake_fetch_recent),
            mock.patch.object(review_job, "_spawn", side_effect=lambda fn: scheduled_again.append(fn)),
        ):
            queued_again = recover_pending_reviews(self.config, store, now=now + timedelta(minutes=5))
        self.assertEqual(queued_again, [])
        self.assertEqual(scheduled_again, [])

    def test_poll_review_channels_handles_transient_500_without_crashing_other_channels(self) -> None:
        store = Store(self.db_path)
        cfg = replace(
            self.config,
            review_channel_ids=("1391835163780186154", "1391835163780186999"),
        )
        store.set_review_cursor("1391835163780186154", "100")
        store.set_review_cursor("1391835163780186999", "200")

        scheduled: list = []

        def fake_fetch_channel(channel_id: str, token: str, after: str):
            if channel_id == "1391835163780186154":
                raise TransientAPIError("discord.com returned HTTP 500")
            return [
                {
                    "id": "205",
                    "author": {"id": "222000000000000001", "bot": False},
                    "content": "please review https://github.com/BerriAI/litellm/pull/54999",
                }
            ]

        with (
            mock.patch.object(review_job, "fetch_channel_messages", side_effect=fake_fetch_channel),
            mock.patch.object(review_job, "acknowledge_message"),
            mock.patch.object(review_job, "_spawn", side_effect=lambda fn: scheduled.append(fn)),
            self.assertLogs(level="WARNING") as logs,
        ):
            poll_review_channels(cfg, store)

        # Channel 1 logged a clean warning and kept its cursor; Channel 2 still succeeded.
        self.assertEqual(store.review_cursor("1391835163780186154"), "100")
        self.assertEqual(store.review_cursor("1391835163780186999"), "205")
        self.assertEqual(len(scheduled), 1)
        self.assertTrue(any("transient Discord error" in line for line in logs.output))

    def test_http_5xx_retries_with_backoff_and_attaches_safe_github_error_detail(self) -> None:
        sleeps: list[float] = []
        url = "https://api.github.com/graphql"

        def make_500(*_args, **_kwargs):
            raise HTTPError(
                url,
                502,
                "Bad Gateway",
                {},
                io.BytesIO(json.dumps({"message": "Server Error", "errors": [{"type": "INTERNAL", "message": "upstream timeout"}]}).encode()),
            )

        with (
            mock.patch.object(api, "_open_request", side_effect=make_500),
            mock.patch.object(api.time, "sleep", side_effect=sleeps.append),
        ):
            with self.assertRaises(TransientAPIError) as raised:
                api._http(Request(url, data=b'{"query":"{ viewer { login } }"}', method="POST"), attempts=3)

        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertEqual(str(raised.exception), "api.github.com returned HTTP 502")
        self.assertIn("Server Error", getattr(raised.exception, "github_error_detail", ""))
        self.assertIn("upstream timeout", getattr(raised.exception, "github_error_detail", ""))


class GitHubProfileEnrichmentRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "state.db"
        self.config = _make_config(self.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_graphql_realistic_response_with_bytechoreographer_and_missing_profile_repo(self) -> None:
        """Realistic GitHub GraphQL response for Bytechoreographer (no profile README repo)."""
        realistic_response = {
            "data": {
                "u0": {
                    "login": "Bytechoreographer",
                    "name": "Byte Choreographer",
                    "bio": "Building LLM tooling",
                    "company": "@Acme",
                    "email": "byte@acme.dev",
                    "twitterUsername": "bytechoreo",
                    "websiteUrl": "https://byte.dev",
                    "socialAccounts": {
                        "nodes": [
                            {"provider": "LINKEDIN", "displayName": "byte", "url": "https://linkedin.com/in/byte"},
                        ]
                    },
                    "profileRepo": None,
                },
                "u1": {
                    "login": "OtherDev",
                    "name": "Other Developer",
                    "bio": "",
                    "company": "",
                    "email": "",
                    "twitterUsername": None,
                    "websiteUrl": None,
                    "socialAccounts": {"nodes": []},
                    "profileRepo": {
                        "object": {"text": "Hi, find me on Discord as otherdev_dc!"},
                    },
                },
            },
            "errors": [
                {
                    "type": "NOT_FOUND",
                    "path": ["u0", "profileRepo"],
                    "locations": [{"line": 1, "column": 54}],
                    "message": "Could not resolve to a Repository with the name 'Bytechoreographer/Bytechoreographer'.",
                }
            ],
        }

        def fake_graphql(token, query, variables):
            self.assertEqual(variables, {"l0": "Bytechoreographer", "l1": "otherdev"})
            return realistic_response

        profiles, rate_limited = fetch_github_profiles_graphql(
            ["Bytechoreographer", "otherdev"],
            "ghp_test",
            graphql_caller=fake_graphql,
        )
        self.assertFalse(rate_limited)
        self.assertIn("bytechoreographer", profiles)
        self.assertIn("otherdev", profiles)
        self.assertEqual(profiles["bytechoreographer"].login, "Bytechoreographer")
        self.assertEqual(profiles["bytechoreographer"].name, "Byte Choreographer")
        self.assertEqual(profiles["bytechoreographer"].email_domain, "acme.dev")
        self.assertIn("otherdev_dc", profiles["otherdev"].readme)

    def test_graphql_single_bad_user_does_not_fail_batch_and_logs_error_body(self) -> None:
        # Case A: Partial GraphQL response where one user is NOT_FOUND (`u1: null`)
        partial_response = {
            "data": {
                "u0": {
                    "login": "Bytechoreographer",
                    "name": "Byte",
                    "bio": "",
                    "company": "",
                    "email": "",
                    "twitterUsername": None,
                    "websiteUrl": None,
                    "socialAccounts": {"nodes": []},
                    "profileRepo": None,
                },
                "u1": None,
                "u2": {
                    "login": "gooddev",
                    "name": "Good Dev",
                    "bio": "",
                    "company": "",
                    "email": "",
                    "twitterUsername": None,
                    "websiteUrl": None,
                    "socialAccounts": {"nodes": []},
                    "profileRepo": None,
                },
            },
            "errors": [
                {
                    "type": "NOT_FOUND",
                    "path": ["u1"],
                    "message": "Could not resolve to a User with the login of 'ghost-deleted-user'.",
                }
            ],
        }

        with self.assertLogs(level="WARNING") as logs:
            profiles, rate_limited = fetch_github_profiles_graphql(
                ["Bytechoreographer", "ghost-deleted-user", "gooddev"],
                "ghp_test",
                graphql_caller=lambda tok, q, v: partial_response,
            )
        self.assertFalse(rate_limited)
        self.assertIn("bytechoreographer", profiles)
        self.assertIn("gooddev", profiles)
        self.assertNotIn("ghost-deleted-user", profiles)
        self.assertTrue(any("ghost-deleted-user" in line for line in logs.output))

        # Case B: One poison user causes the whole multi-login GraphQL call to raise HTTP error ->
        # bisection isolates the bad user and still enriches all valid users, logging error detail.
        def poison_graphql(tok, query, variables):
            vals = list(variables.values())
            if "poison-user" in vals:
                err = RuntimeError("api.github.com returned HTTP 422")
                setattr(err, "github_error_detail", "Unprocessable Entity: invalid login poison-user")
                raise err
            return {
                "data": {
                    f"u{idx}": {
                        "login": login,
                        "name": f"Name {login}",
                        "bio": "",
                        "company": "",
                        "email": "",
                        "twitterUsername": None,
                        "websiteUrl": None,
                        "socialAccounts": {"nodes": []},
                        "profileRepo": None,
                    }
                    for idx, login in enumerate(vals)
                }
            }

        with self.assertLogs(level="WARNING") as bisect_logs:
            profiles_b, rl_b = fetch_github_profiles_graphql(
                ["Bytechoreographer", "poison-user", "gooddev"],
                "ghp_test",
                graphql_caller=poison_graphql,
            )
        self.assertFalse(rl_b)
        self.assertEqual(sorted(profiles_b.keys()), ["bytechoreographer", "gooddev"])
        self.assertTrue(any("Unprocessable Entity: invalid login poison-user" in line for line in bisect_logs.output))

        # Case C: GraphQL-level RATE_LIMITED error retries with backoff and succeeds.
        calls = {"n": 0}
        sleeps: list[float] = []

        def rate_limited_once(tok, query, variables):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "data": None,
                    "errors": [{"type": "RATE_LIMITED", "message": "API rate limit already exceeded"}],
                }
            return {
                "data": {
                    "u0": {
                        "login": "Bytechoreographer",
                        "name": "Byte",
                        "bio": "",
                        "company": "",
                        "email": "",
                        "twitterUsername": None,
                        "websiteUrl": None,
                        "socialAccounts": {"nodes": []},
                        "profileRepo": None,
                    }
                }
            }

        profiles_c, rl_c = fetch_github_profiles_graphql(
            ["Bytechoreographer"],
            "ghp_test",
            graphql_caller=rate_limited_once,
            sleep_fn=sleeps.append,
        )
        self.assertFalse(rl_c)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(sleeps), 1)
        self.assertIn("bytechoreographer", profiles_c)

    def test_short_remaining_deadline_does_not_starve_enrichment(self) -> None:
        """Reproduces the exact 18s-left condition that previously set enrich_deadline = -2s."""
        store = Store(self.db_path)
        gql_calls = {"count": 0}

        def fake_graphql(token, query, variables):
            gql_calls["count"] += 1
            if "BatchProfiles" in query:
                return {
                    "data": {
                        "u0": {
                            "login": "Bytechoreographer",
                            "name": "Byte Choreographer",
                            "bio": "",
                            "company": "",
                            "email": "",
                            "twitterUsername": None,
                            "websiteUrl": None,
                            "socialAccounts": {"nodes": []},
                            "profileRepo": None,
                        }
                    },
                    "errors": [
                        {
                            "type": "NOT_FOUND",
                            "path": ["u0", "profileRepo"],
                            "message": "Could not resolve to a Repository",
                        }
                    ],
                }
            return {"data": {"repository": {}}}

        members = [
            DiscordMemberInfo(
                discord_id="900000000000000001",
                username="bytechoreographer",
                global_name="Byte Choreographer",
                nick="",
            )
        ]
        groups = [
            {
                "login": "Bytechoreographer",
                "user_id": 12345,
                "pulls": [{"number": 54800, "title": "fix something"}],
            }
        ]
        matched_groups, skipped_count, _details, diag = match_community_contributors_with_diagnostics(
            groups,
            self.config,
            store,
            members=members,
            graphql_caller=fake_graphql,
            deadline=time.monotonic() + 18.0,
        )
        self.assertGreaterEqual(gql_calls["count"], 1)
        self.assertEqual(diag.enrichment_deferred, 0)
        self.assertEqual(skipped_count, 0)
        self.assertEqual(diag.matched_count, 1)
        self.assertEqual(matched_groups[0]["discord_id"], "900000000000000001")

    def test_roles_poll_skips_excluded_search_rows_without_rest_fetch(self) -> None:
        """Excluded authors from per-day search rows must not trigger per-PR REST fetches."""
        store = Store(self.db_path)
        now = datetime(2026, 11, 28, 18, 0, 0, tzinfo=UTC)
        fetched_pulls: list[int] = []

        def fake_search_rows(repo, start, end, token):
            return [
                {"number": 54901, "login": "dependabot[bot]", "type": "Bot", "author_association": "NONE"},
                {"number": 54902, "login": "ishaan-jaff", "type": "User", "author_association": "MEMBER"},
                {"number": 54903, "login": "Bytechoreographer", "type": "User", "author_association": "CONTRIBUTOR"},
            ]

        def fake_pull_fetcher(repo, number, token):
            fetched_pulls.append(number)
            return {
                "number": number,
                "title": "Community PR",
                "merged": True,
                "merged_at": iso(now - timedelta(hours=2)),
                "draft": False,
                "author_association": "CONTRIBUTOR",
                "user": {"login": "Bytechoreographer", "id": 55501, "type": "User"},
            }

        def fake_graphql(token, query, variables):
            if "BatchProfiles" in query:
                return {
                    "data": {
                        "u0": {
                            "login": "Bytechoreographer",
                            "name": "Byte",
                            "bio": "",
                            "company": "",
                            "email": "",
                            "twitterUsername": None,
                            "websiteUrl": None,
                            "socialAccounts": {"nodes": []},
                            "profileRepo": None,
                        }
                    }
                }
            return {"data": {"repository": {}}}

        members = [
            DiscordMemberInfo(
                discord_id="900000000000000001",
                username="bytechoreographer",
                global_name="Byte",
                nick="",
            )
        ]

        with mock.patch("contributor_bot.rewards.add_member_role"):
            res = poll_merged_prs_for_roles(
                self.config,
                store,
                now=now,
                search_merged_fn=fake_search_rows,
                pull_fetcher=fake_pull_fetcher,
                files_fetcher=lambda repo, num, tok: [],
                org_members_fetcher=lambda org, tok: frozenset({"ishaan-jaff"}),
                members=members,
                graphql_caller=fake_graphql,
                count_fetcher=lambda repo, login, tok: 1,
            )
        self.assertIsNotNone(res)
        # Only the community contributor PR (#54903) triggered a REST pull fetch!
        self.assertEqual(fetched_pulls, [54903])
        self.assertEqual(res.skipped_excluded, 2)
        self.assertEqual(res.stored, 1)
        self.assertEqual(res.authors_matched, 1)


if __name__ == "__main__":
    unittest.main()
