import base64
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from contributor_bot.config import Config
from contributor_bot.store import Store
from contributor_bot.weekly import (
    WEEKLY_HEADER,
    _bot_user_id_from_token,
    log_weekly_boot_status,
    post_weekly_if_due,
    reconcile_weekly_from_channel,
    reset_org_members_cache,
    reset_weekly_scheduler_state,
    weekly_schedule_status,
)


def _bot_token_for(user_id: str) -> str:
    encoded = base64.b64encode(user_id.encode("ascii")).decode("ascii").rstrip("=")
    return f"{encoded}.fake_ts.fake_hmac"


class WeeklyRepostPreventionTests(unittest.TestCase):
    def setUp(self):
        reset_org_members_cache()
        reset_weekly_scheduler_state()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = str(Path(self.temp.name) / "roles.db")
        self.bot_id = "123456789012345678"
        self.channel_id = "1175143623420035123"
        self.config = Config(
            _bot_token_for(self.bot_id),
            "123456789012345600",
            self.channel_id,
            discord_guild_id="333444555666777888",
            discord_application_id=self.bot_id,
            merged_weekly_enabled=True,
            merged_weekday=4,
            merged_time="10:00",
            merged_timezone="America/Toronto",
        )

    def test_one_time_cleanup_preserves_legit_rows_and_clears_only_preview_rows(self):
        # Seed a pre-migration DB with a legitimate W48 scheduler post (different timestamp).
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE merged_weeks ("
            "repository TEXT NOT NULL, week_key TEXT NOT NULL, posted_at TEXT NOT NULL, "
            "via TEXT NOT NULL DEFAULT 'scheduler', PRIMARY KEY (repository, week_key))"
        )
        conn.execute(
            "INSERT INTO merged_weeks (repository, week_key, posted_at, via) VALUES (?, ?, ?, ?)",
            ("BerriAI/litellm", "2026-W48", "2026-11-27T21:15:00+00:00", "scheduler"),
        )
        conn.commit()
        conn.close()

        # Reopening Store multiple times must NEVER delete the legitimate W48 row.
        for _ in range(3):
            store = Store(self.db_path)
            info = store.merged_week_info("BerriAI/litellm", "2026-W48")
            self.assertIsNotNone(info)
            self.assertEqual(info["posted_at"], "2026-11-27T21:15:00+00:00")
            self.assertEqual(info["via"], "scheduler")
            # Case-insensitive repository lookup also finds it.
            self.assertIsNotNone(store.merged_week_info("berriai/litellm", "2026-W48"))
            store.close()

        # Separately verify bogus preview row (via='preview') IS cleared on first migration only.
        bogus_db = str(Path(self.temp.name) / "bogus.db")
        conn = sqlite3.connect(bogus_db)
        conn.execute(
            "CREATE TABLE merged_weeks ("
            "repository TEXT NOT NULL, week_key TEXT NOT NULL, posted_at TEXT NOT NULL, "
            "via TEXT NOT NULL DEFAULT 'scheduler', PRIMARY KEY (repository, week_key))"
        )
        conn.execute(
            "INSERT INTO merged_weeks (repository, week_key, posted_at, via) VALUES (?, ?, ?, ?)",
            ("BerriAI/litellm", "2026-W48", "2026-11-27T20:00:00+00:00", "preview"),
        )
        conn.commit()
        conn.close()

        store2 = Store(bogus_db)
        self.assertIsNone(store2.merged_week_info("BerriAI/litellm", "2026-W48"))
        # Record a real post and reopen; one-time migration has already run and never deletes it.
        self.assertTrue(store2.mark_merged_week("BerriAI/litellm", "2026-W48", "2026-11-27T21:20:00+00:00", via="cli"))
        store2.close()

        store3 = Store(bogus_db)
        self.addCleanup(store3.close)
        info3 = store3.merged_week_info("BerriAI/litellm", "2026-W48")
        self.assertIsNotNone(info3)
        self.assertEqual(info3["via"], "cli")

    def test_restart_after_real_post_does_not_repost_and_logs_boot_skip(self):
        store = Store(self.db_path)
        posted_chunks = []

        def fake_report(*_args, **_kwargs):
            from contributor_bot.weekly import WeeklyReport
            return WeeklyReport(
                chunks=(f"{WEEKLY_HEADER}\n\n- <@999> (`alice`) — fix (#1)",),
                matched_count=1,
                skipped_count=0,
                since=datetime(2026, 11, 20, 15, 0, tzinfo=timezone.utc),
                rate_limited=False,
            )

        fri_afternoon = datetime(2026, 11, 27, 21, 15, tzinfo=timezone.utc)  # Fri 4:15 PM EST
        with mock.patch("contributor_bot.weekly.build_weekly_report", side_effect=fake_report):
            result = post_weekly_if_due(
                self.config,
                store,
                fri_afternoon,
                post=lambda ch, _t, msg: posted_chunks.append((ch, msg)),
                messages_fetcher=lambda *_a, **_k: [],
            )
        self.assertIsNotNone(result)
        self.assertEqual(len(posted_chunks), 1)
        store.close()

        # Simulate deploy/restart at 2026-11-28T02:42 UTC (Fri 2026-11-27 21:42 EST -> still 2026-W48).
        reset_weekly_scheduler_state()
        restarted_store = Store(self.db_path)
        self.addCleanup(restarted_store.close)
        restart_time = datetime(2026, 11, 28, 2, 42, tzinfo=timezone.utc)

        with self.assertLogs(level="INFO") as boot_logs:
            log_weekly_boot_status(
                self.config,
                restarted_store,
                restart_time,
                messages_fetcher=mock.Mock(side_effect=AssertionError("should not need Discord scan when DB mark exists")),
            )
        joined_boot = "\n".join(boot_logs.output)
        self.assertIn(
            "Weekly merge summary 2026-W48 already posted at 2026-11-27T21:15:00+00:00 via scheduler; skipping",
            joined_boot,
        )
        self.assertNotIn("due now (will catch up)", joined_boot)

        with mock.patch("contributor_bot.weekly.build_weekly_report", side_effect=AssertionError("must not rebuild")):
            again = post_weekly_if_due(
                self.config,
                restarted_store,
                restart_time,
                post=lambda ch, _t, msg: posted_chunks.append((ch, msg)),
            )
        self.assertIsNone(again)
        self.assertEqual(len(posted_chunks), 1)

    def test_channel_history_self_heal_prevents_repost_on_empty_db(self):
        self.assertEqual(_bot_user_id_from_token(self.config.discord_token), self.bot_id)
        restart_time = datetime(2026, 11, 28, 2, 42, tzinfo=timezone.utc)  # Fri 21:42 EST (2026-W48)

        # Noise messages that must NOT trigger a false self-heal:
        noise_messages = [
            # Last week's post (2026-W47) from our bot
            {
                "id": "10001",
                "author": {"id": self.bot_id, "bot": True},
                "content": f"{WEEKLY_HEADER}\n\n- <@111> (`old`)",
                "timestamp": "2026-11-20T15:00:00+00:00",
            },
            # A human quoting the header this week
            {
                "id": "10002",
                "author": {"id": "999999999999999999", "bot": False},
                "content": f"{WEEKLY_HEADER}\n\nquoted",
                "timestamp": "2026-11-27T20:00:00+00:00",
            },
            # Another bot posting the header this week
            {
                "id": "10003",
                "author": {"id": "888888888888888888", "bot": True},
                "content": f"{WEEKLY_HEADER}\n\nother bot",
                "timestamp": "2026-11-27T20:05:00+00:00",
            },
            # A webhook posting the header this week
            {
                "id": "10004",
                "webhook_id": "777777777777777777",
                "author": {"id": self.bot_id, "bot": True},
                "content": f"{WEEKLY_HEADER}\n\nwebhook",
                "timestamp": "2026-11-27T20:10:00+00:00",
            },
        ]

        empty_store = Store(self.db_path)
        self.addCleanup(empty_store.close)
        self.assertIsNone(
            reconcile_weekly_from_channel(
                self.config,
                empty_store,
                restart_time,
                messages_fetcher=lambda _ch, _tok, limit=50: noise_messages,
            )
        )
        self.assertFalse(empty_store.merged_week_posted("BerriAI/litellm", "2026-W48"))

        # Now include our bot's actual 2026-W48 public summary in #announcements.
        with_real_post = [
            {
                "id": "10005",
                "author": {"id": self.bot_id, "bot": True},
                "content": f"{WEEKLY_HEADER}\n\n- <@222> (`contrib`) — great fix (#999)",
                "timestamp": "2026-11-27T21:05:00+00:00",
            },
            *noise_messages,
        ]

        # 1) Boot self-heal writes the mark and logs the exact skip message.
        with self.assertLogs(level="INFO") as boot_logs:
            healed = log_weekly_boot_status(
                self.config,
                empty_store,
                restart_time,
                messages_fetcher=lambda _ch, _tok, limit=50: with_real_post,
            )
        self.assertIsNotNone(healed)
        self.assertEqual(healed["via"], "channel_self_heal")
        self.assertEqual(healed["posted_at"], "2026-11-27T21:05:00+00:00")
        joined = "\n".join(boot_logs.output)
        self.assertIn(
            "Weekly merge summary 2026-W48 already posted at 2026-11-27T21:05:00+00:00 via channel_self_heal; skipping",
            joined,
        )
        self.assertIn("already posted at 2026-11-27T21:05:00+00:00 via channel_self_heal", weekly_schedule_status(self.config, empty_store, restart_time))

        # 2) Even if post_weekly_if_due is called directly on a brand-new empty DB without boot self-heal,
        #    it scans #announcements BEFORE catching up, self-heals SQLite, and NEVER reposts.
        fresh_store = Store(str(Path(self.temp.name) / "fresh_ephemeral.db"))
        self.addCleanup(fresh_store.close)
        reset_weekly_scheduler_state()
        posted = []
        with mock.patch("contributor_bot.weekly.build_weekly_report", side_effect=AssertionError("must not build report when channel already has post")), \
                self.assertLogs(level="INFO") as catchup_logs:
            res = post_weekly_if_due(
                self.config,
                fresh_store,
                restart_time,
                post=lambda ch, _t, msg: posted.append((ch, msg)),
                messages_fetcher=lambda _ch, _tok, limit=50: with_real_post,
            )
        self.assertIsNone(res)
        self.assertEqual(posted, [])
        saved = fresh_store.merged_week_info("BerriAI/litellm", "2026-W48")
        self.assertIsNotNone(saved)
        self.assertEqual(saved["via"], "channel_self_heal")
        self.assertTrue(
            any("Weekly merge summary 2026-W48 already posted at 2026-11-27T21:05:00+00:00 via channel_self_heal; skipping" in line for line in catchup_logs.output)
        )

    def test_cli_force_post_marks_week_and_catchup_only_runs_when_unposted(self):
        from contributor_bot.weekly import WeeklyReport

        def fake_report(*_args, **_kwargs):
            return WeeklyReport(
                chunks=(f"{WEEKLY_HEADER}\n\n- <@333> (`dev`) — ship (#42)",),
                matched_count=1,
                skipped_count=0,
                since=datetime(2026, 11, 20, 15, 0, tzinfo=timezone.utc),
                rate_limited=False,
            )

        store = Store(self.db_path)
        self.addCleanup(store.close)
        sent = []
        cli_time = datetime(2026, 11, 27, 21, 0, tzinfo=timezone.utc)
        with mock.patch("contributor_bot.weekly.build_weekly_report", side_effect=fake_report):
            res = post_weekly_if_due(
                self.config,
                store,
                cli_time,
                force=True,
                via="cli",
                post=lambda ch, _t, msg: sent.append((ch, msg)),
            )
        self.assertIsNotNone(res)
        self.assertEqual(len(sent), 1)
        info = store.merged_week_info("BerriAI/litellm", "2026-W48")
        self.assertIsNotNone(info)
        self.assertEqual(info["via"], "cli")

        # Restart later Friday night: must NOT repost.
        restart_time = datetime(2026, 11, 28, 2, 42, tzinfo=timezone.utc)
        with self.assertLogs(level="INFO") as logs:
            log_weekly_boot_status(self.config, store, restart_time)
        self.assertTrue(
            any("Weekly merge summary 2026-W48 already posted at 2026-11-27T21:00:00+00:00 via cli; skipping" in line for line in logs.output)
        )
        again = post_weekly_if_due(
            self.config,
            store,
            restart_time,
            post=lambda ch, _t, msg: sent.append((ch, msg)),
        )
        self.assertIsNone(again)
        self.assertEqual(len(sent), 1)

        # Fresh store + empty channel history -> catch-up after Friday 10:00 DOES post once.
        empty_store = Store(str(Path(self.temp.name) / "catchup_ok.db"))
        self.addCleanup(empty_store.close)
        reset_weekly_scheduler_state()
        catchup_sent = []
        with mock.patch("contributor_bot.weekly.build_weekly_report", side_effect=fake_report):
            res2 = post_weekly_if_due(
                self.config,
                empty_store,
                restart_time,
                post=lambda ch, _t, msg: catchup_sent.append((ch, msg)),
                messages_fetcher=lambda _ch, _tok, limit=50: [],
            )
        self.assertIsNotNone(res2)
        self.assertEqual(len(catchup_sent), 1)
        self.assertTrue(empty_store.merged_week_posted("BerriAI/litellm", "2026-W48"))


if __name__ == "__main__":
    unittest.main()
