"""SQLite persistence for GitHub links, merged PRs, and granted roles.

The weekly thank-you ledger stays in the JSON state file. This database is
separate so role history cannot reset announcement dedupe.

The SQL is ordinary (explicit columns, no SQLite-only types) so a Postgres
backend can replace this module later: keep the method names, switch
placeholders from ``?`` to ``%s``, and translate ``INSERT OR IGNORE`` to
``INSERT ... ON CONFLICT DO NOTHING``. Callers only use these methods.
"""

import json
import logging
import os
import re
import sqlite3
import threading
from pathlib import Path

from .config import is_discord_id

_LOGIN = re.compile(r"[A-Za-z0-9-]+(?:\[bot\])?")
_INVITE_PENDING = "pending"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS github_links (
    discord_id TEXT PRIMARY KEY,
    github_login TEXT NOT NULL,
    github_user_id INTEGER NOT NULL UNIQUE,
    linked_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS github_links_login_lower
    ON github_links (lower(github_login));

CREATE TABLE IF NOT EXISTS merged_pull_requests (
    repository TEXT NOT NULL,
    number INTEGER NOT NULL,
    github_user_id INTEGER NOT NULL,
    github_login TEXT NOT NULL,
    merged_at TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    labels_json TEXT NOT NULL,
    files_json TEXT NOT NULL,
    shoutout_at TEXT,
    invite_commented_at TEXT,
    PRIMARY KEY (repository, number)
);
CREATE INDEX IF NOT EXISTS merged_pull_requests_author
    ON merged_pull_requests (repository, github_user_id);

CREATE TABLE IF NOT EXISTS role_grants (
    discord_id TEXT NOT NULL,
    role_key TEXT NOT NULL,
    role_name TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    announced_at TEXT,
    PRIMARY KEY (discord_id, role_key)
);

CREATE TABLE IF NOT EXISTS oauth_states (
    state TEXT PRIMARY KEY,
    discord_id TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backfill_windows (
    repository TEXT NOT NULL,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    PRIMARY KEY (repository, window_start, window_end)
);

CREATE TABLE IF NOT EXISTS review_threads (
    repository TEXT NOT NULL,
    number INTEGER NOT NULL,
    channel_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    thread_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    last_reviewed_at TEXT,
    last_followup_at TEXT,
    head_sha TEXT,
    PRIMARY KEY (repository, number)
);

CREATE TABLE IF NOT EXISTS review_cursors (
    channel_id TEXT PRIMARY KEY,
    last_message_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_budget (
    day TEXT PRIMARY KEY,
    runs INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS review_triggers (
    repository TEXT NOT NULL,
    number INTEGER NOT NULL,
    head_sha TEXT NOT NULL,
    body TEXT NOT NULL,
    posted_at TEXT NOT NULL,
    PRIMARY KEY (repository, number, head_sha, body)
);

CREATE TABLE IF NOT EXISTS merged_weeks (
    repository TEXT NOT NULL,
    week_key TEXT NOT NULL,
    posted_at TEXT NOT NULL,
    via TEXT NOT NULL DEFAULT 'scheduler',
    PRIMARY KEY (repository, week_key)
);

CREATE TABLE IF NOT EXISTS github_discord_matches (
    github_login TEXT PRIMARY KEY,
    github_user_id INTEGER,
    discord_id TEXT NOT NULL,
    method TEXT NOT NULL,
    confidence REAL NOT NULL,
    matched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS github_users_cache (
    github_login TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    blog TEXT NOT NULL DEFAULT '',
    twitter_username TEXT NOT NULL DEFAULT '',
    bio TEXT NOT NULL DEFAULT '',
    email_local TEXT NOT NULL DEFAULT '',
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pr_board_items (
    project_id TEXT NOT NULL,
    repository TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    pr_node_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT '',
    pr_updated_at TEXT NOT NULL DEFAULT '',
    synced_at TEXT NOT NULL,
    PRIMARY KEY (project_id, repository, pr_number)
);

CREATE TABLE IF NOT EXISTS pr_board_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS roles_poller_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _strings(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("Expected a list of strings")
    return list(value)


class Store:
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            parent = Path(path).parent
            if parent == Path("/"):
                raise ValueError("ROLES_DB_PATH must not live in the filesystem root")
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(review_threads)")}
            if columns and "last_followup_at" not in columns:
                self._conn.execute("ALTER TABLE review_threads ADD COLUMN last_followup_at TEXT")
            mw_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(merged_weeks)")}
            if mw_columns and "via" not in mw_columns:
                self._conn.execute("ALTER TABLE merged_weeks ADD COLUMN via TEXT NOT NULL DEFAULT 'scheduler'")
            # One-time recovery for the bogus 2026-W48 mark created by an ephemeral /merged-this-week preview.
            bad_rows = self._conn.execute(
                "SELECT repository, week_key, posted_at FROM merged_weeks "
                "WHERE week_key = '2026-W48' AND (posted_at LIKE '2026-11-27T20:34:06%' OR via = 'manual_preview')",
            ).fetchall()
            if bad_rows:
                self._conn.execute(
                    "DELETE FROM merged_weeks "
                    "WHERE week_key = '2026-W48' AND (posted_at LIKE '2026-11-27T20:34:06%' OR via = 'manual_preview')",
                )
                for row in bad_rows:
                    logging.info(
                        "Cleared bogus ephemeral-preview merged_weeks entry for %s %s (posted_at=%s) "
                        "so scheduled post can run",
                        row["repository"],
                        row["week_key"],
                        row["posted_at"],
                    )
            self._conn.commit()
        if path != ":memory:":
            os.chmod(path, 0o600)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def link_account(self, discord_id: str, github_login: str, github_user_id: int, linked_at: str) -> str | None:
        """Link the accounts. Return an error message, or None on success.

        Linking the same pair again refreshes the GitHub login. A Discord
        account cannot hold two GitHub accounts, and a GitHub account cannot
        be claimed by a second Discord account.
        """
        if not is_discord_id(discord_id) or not _LOGIN.fullmatch(github_login):
            raise ValueError("Invalid Discord id or GitHub login")
        if not isinstance(github_user_id, int) or github_user_id <= 0:
            raise ValueError("Invalid GitHub user id")
        with self._lock:
            by_discord = self._conn.execute(
                "SELECT github_login, github_user_id FROM github_links WHERE discord_id = ?",
                (discord_id,),
            ).fetchone()
            by_github = self._conn.execute(
                "SELECT discord_id, github_login FROM github_links WHERE github_user_id = ?",
                (github_user_id,),
            ).fetchone()
            if by_discord and int(by_discord["github_user_id"]) != github_user_id:
                return f"your discord account is already linked to {by_discord['github_login']}. use /unlink-github first."
            if by_github and by_github["discord_id"] != discord_id:
                return "that github account is already linked to another discord user."
            try:
                if by_discord:
                    self._conn.execute(
                        "UPDATE github_links SET github_login = ?, linked_at = ? WHERE discord_id = ?",
                        (github_login, linked_at, discord_id),
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO github_links (discord_id, github_login, github_user_id, linked_at) VALUES (?, ?, ?, ?)",
                        (discord_id, github_login, github_user_id, linked_at),
                    )
                self._conn.commit()
            except sqlite3.IntegrityError:
                self._conn.rollback()
                return "that github account is already linked to another discord user."
        return None

    def unlink(self, discord_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM github_links WHERE discord_id = ?", (discord_id,))
            self._conn.commit()
            return cursor.rowcount == 1

    def link_for_discord(self, discord_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT discord_id, github_login, github_user_id, linked_at FROM github_links WHERE discord_id = ?",
                (discord_id,),
            ).fetchone()
        return dict(row) if row else None

    def link_for_github_user(self, github_user_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT discord_id, github_login, github_user_id, linked_at FROM github_links WHERE github_user_id = ?",
                (github_user_id,),
            ).fetchone()
        return dict(row) if row else None

    def all_links(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT discord_id, github_login, github_user_id, linked_at FROM github_links ORDER BY linked_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def put_oauth_state(self, state: str, discord_id: str, expires_at: str, now: str) -> None:
        if not is_discord_id(discord_id) or not re.fullmatch(r"[A-Za-z0-9_-]{20,200}", state):
            raise ValueError("Invalid OAuth state")
        with self._lock:
            self._conn.execute("DELETE FROM oauth_states WHERE expires_at <= ?", (now,))
            self._conn.execute(
                """
                INSERT INTO oauth_states (state, discord_id, expires_at) VALUES (?, ?, ?)
                ON CONFLICT(state) DO UPDATE SET
                    discord_id = excluded.discord_id,
                    expires_at = excluded.expires_at
                """,
                (state, discord_id, expires_at),
            )
            self._conn.commit()

    def pop_oauth_state(self, state: str, now: str) -> str | None:
        """Return the Discord id and delete the state. Expired states return None."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{20,200}", state):
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT discord_id, expires_at FROM oauth_states WHERE state = ?",
                (state,),
            ).fetchone()
            self._conn.execute("DELETE FROM oauth_states WHERE state = ?", (state,))
            self._conn.commit()
        if row is None or row["expires_at"] <= now:
            return None
        return row["discord_id"]

    def upsert_pull(
        self,
        *,
        repository: str,
        number: int,
        github_user_id: int,
        github_login: str,
        merged_at: str,
        title: str,
        labels: list[str],
        files: list[str],
    ) -> None:
        if not _LOGIN.fullmatch(github_login):
            raise ValueError("Invalid GitHub login")
        labels = _strings(labels)
        files = _strings(files)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO merged_pull_requests (
                    repository, number, github_user_id, github_login, merged_at, title, labels_json, files_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repository, number) DO UPDATE SET
                    github_user_id = excluded.github_user_id,
                    github_login = excluded.github_login,
                    merged_at = excluded.merged_at,
                    title = excluded.title,
                    labels_json = excluded.labels_json,
                    files_json = excluded.files_json
                """,
                (
                    repository,
                    number,
                    github_user_id,
                    github_login,
                    merged_at,
                    title[:300],
                    json.dumps(labels),
                    json.dumps(files),
                ),
            )
            self._conn.commit()

    def get_pull(self, repository: str, number: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM merged_pull_requests WHERE repository = ? AND number = ?",
                (repository, number),
            ).fetchone()
        return _pull_dict(row) if row else None

    def pulls_for_author(self, repository: str, github_user_id: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM merged_pull_requests
                WHERE repository = ? AND github_user_id = ?
                ORDER BY merged_at, number
                """,
                (repository, github_user_id),
            ).fetchall()
        return [_pull_dict(row) for row in rows]

    def pulls_for_author_login(self, repository: str, github_login: str) -> list[dict]:
        key = (github_login or "").strip().lower()
        if not key:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM merged_pull_requests
                WHERE repository = ? AND lower(github_login) = ?
                ORDER BY merged_at, number
                """,
                (repository, key),
            ).fetchall()
        return [_pull_dict(row) for row in rows]

    def all_pull_authors(self, repository: str) -> list[dict]:
        """Return distinct PR authors stored for ``repository``, with latest login and PR count."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    github_user_id,
                    (
                        SELECT p2.github_login
                        FROM merged_pull_requests p2
                        WHERE p2.repository = p1.repository
                          AND p2.github_user_id = p1.github_user_id
                        ORDER BY p2.merged_at DESC, p2.number DESC
                        LIMIT 1
                    ) AS github_login,
                    COUNT(*) AS pull_count
                FROM merged_pull_requests p1
                WHERE repository = ?
                GROUP BY github_user_id
                ORDER BY github_user_id
                """,
                (repository,),
            ).fetchall()
        return [
            {
                "github_user_id": int(row["github_user_id"]),
                "github_login": str(row["github_login"]),
                "pull_count": int(row["pull_count"]),
            }
            for row in rows
        ]

    def pull_count(self, repository: str | None = None) -> int:
        with self._lock:
            if repository is None:
                row = self._conn.execute("SELECT COUNT(*) AS n FROM merged_pull_requests").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM merged_pull_requests WHERE repository = ?",
                    (repository,),
                ).fetchone()
        return int(row["n"])

    def grant_keys(self, discord_id: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role_key FROM role_grants WHERE discord_id = ?",
                (discord_id,),
            ).fetchall()
        return {row["role_key"] for row in rows}

    def record_grant(self, discord_id: str, role_key: str, role_name: str, granted_at: str) -> bool:
        """Insert a grant. Return False when this Discord user already has the role key."""
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO role_grants (discord_id, role_key, role_name, granted_at, announced_at)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (discord_id, role_key, role_name, granted_at),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def unannounced_grants(self, discord_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT role_key, role_name FROM role_grants
                WHERE discord_id = ? AND announced_at IS NULL
                ORDER BY granted_at, role_key
                """,
                (discord_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_grants_announced(self, discord_id: str, role_keys: list[str], announced_at: str) -> None:
        if not role_keys:
            return
        with self._lock:
            self._conn.executemany(
                """
                UPDATE role_grants SET announced_at = ?
                WHERE discord_id = ? AND role_key = ? AND announced_at IS NULL
                """,
                [(announced_at, discord_id, key) for key in role_keys],
            )
            self._conn.commit()

    def mark_shoutout(self, repository: str, number: int, shoutout_at: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE merged_pull_requests SET shoutout_at = ?
                WHERE repository = ? AND number = ? AND shoutout_at IS NULL
                """,
                (shoutout_at, repository, number),
            )
            self._conn.commit()

    def claim_invite(self, repository: str, number: int) -> bool:
        """Reserve the one invite comment for this PR. False if one was already sent or claimed."""
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE merged_pull_requests SET invite_commented_at = ?
                WHERE repository = ? AND number = ? AND invite_commented_at IS NULL
                """,
                (_INVITE_PENDING, repository, number),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def finish_invite(self, repository: str, number: int, commented_at: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE merged_pull_requests SET invite_commented_at = ?
                WHERE repository = ? AND number = ? AND invite_commented_at = ?
                """,
                (commented_at, repository, number, _INVITE_PENDING),
            )
            self._conn.commit()

    def release_invite(self, repository: str, number: int) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE merged_pull_requests SET invite_commented_at = NULL
                WHERE repository = ? AND number = ? AND invite_commented_at = ?
                """,
                (repository, number, _INVITE_PENDING),
            )
            self._conn.commit()

    def backfill_done(self, repository: str, window_start: str, window_end: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT 1 FROM backfill_windows
                WHERE repository = ? AND window_start = ? AND window_end = ?
                """,
                (repository, window_start, window_end),
            ).fetchone()
        return row is not None

    def mark_backfill_done(self, repository: str, window_start: str, window_end: str, completed_at: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO backfill_windows (repository, window_start, window_end, completed_at)
                VALUES (?, ?, ?, ?)
                """,
                (repository, window_start, window_end, completed_at),
            )
            self._conn.commit()

    def review_cursor(self, channel_id: str) -> str | None:
        if not is_discord_id(channel_id):
            raise ValueError("Invalid Discord channel id")
        with self._lock:
            row = self._conn.execute(
                "SELECT last_message_id FROM review_cursors WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
        return row["last_message_id"] if row else None

    def set_review_cursor(self, channel_id: str, message_id: str) -> None:
        if not is_discord_id(channel_id) or not is_discord_id(message_id):
            raise ValueError("Invalid Discord message id")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO review_cursors (channel_id, last_message_id) VALUES (?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET last_message_id = excluded.last_message_id
                """,
                (channel_id, message_id),
            )
            self._conn.commit()

    def review_thread(self, repository: str, number: int) -> dict | None:
        self._review_key(repository, number)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM review_threads WHERE repository = ? AND number = ?",
                (repository, number),
            ).fetchone()
        return dict(row) if row else None

    def review_thread_by_discord(self, thread_id: str) -> dict | None:
        if not is_discord_id(thread_id):
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM review_threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return dict(row) if row else None

    def save_review_thread(
        self,
        *,
        repository: str,
        number: int,
        channel_id: str,
        message_id: str,
        thread_id: str,
        created_at: str,
    ) -> None:
        self._review_key(repository, number)
        if not all(is_discord_id(value) for value in (channel_id, message_id, thread_id)):
            raise ValueError("Invalid Discord id for a review thread")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO review_threads (
                    repository, number, channel_id, message_id, thread_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repository, number) DO UPDATE SET
                    channel_id = excluded.channel_id,
                    message_id = excluded.message_id,
                    thread_id = excluded.thread_id
                """,
                (repository, number, channel_id, message_id, thread_id, created_at),
            )
            self._conn.commit()

    def mark_followup(self, repository: str, number: int, followed_at: str) -> None:
        self._review_key(repository, number)
        with self._lock:
            self._conn.execute(
                "UPDATE review_threads SET last_followup_at = ? WHERE repository = ? AND number = ?",
                (followed_at, repository, number),
            )
            self._conn.commit()

    def claim_daily_run(self, day: str, limit: int) -> bool:
        """Reserve one agent run for a UTC day. ``limit`` 0 means unlimited."""
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Invalid review budget")
        if limit == 0:
            return True
        with self._lock:
            self._conn.execute(
                "INSERT INTO review_budget (day, runs) VALUES (?, 0) ON CONFLICT(day) DO NOTHING",
                (day,),
            )
            cursor = self._conn.execute(
                "UPDATE review_budget SET runs = runs + 1 WHERE day = ? AND runs < ?",
                (day, limit),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def trigger_posted(self, repository: str, number: int, head_sha: str, body: str) -> bool:
        self._review_key(repository, number)
        with self._lock:
            row = self._conn.execute(
                """
                SELECT 1 FROM review_triggers
                WHERE repository = ? AND number = ? AND head_sha = ? AND body = ?
                """,
                (repository, number, head_sha, body),
            ).fetchone()
        return row is not None

    def triggers_since(self, repository: str, number: int, since: str) -> int:
        self._review_key(repository, number)
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS n FROM review_triggers
                WHERE repository = ? AND number = ? AND posted_at >= ?
                """,
                (repository, number, since),
            ).fetchone()
        return int(row["n"])

    def save_trigger(self, repository: str, number: int, head_sha: str, body: str, posted_at: str) -> None:
        self._review_key(repository, number)
        if not re.fullmatch(r"[0-9a-f]{40}", head_sha or ""):
            raise ValueError("Invalid head sha")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO review_triggers (repository, number, head_sha, body, posted_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(repository, number, head_sha, body) DO NOTHING
                """,
                (repository, number, head_sha, body, posted_at),
            )
            self._conn.commit()

    def merged_week_posted(self, repository: str, week_key: str) -> bool:
        return self.merged_week_info(repository, week_key) is not None

    def merged_week_info(self, repository: str, week_key: str) -> dict | None:
        """Return ``{"week_key", "posted_at", "via"}`` if this week was already recorded."""
        with self._lock:
            row = self._conn.execute(
                "SELECT week_key, posted_at, via FROM merged_weeks WHERE repository = ? AND week_key = ?",
                (repository, week_key),
            ).fetchone()
        return dict(row) if row else None

    def clear_merged_week(self, repository: str, week_key: str) -> bool:
        """Delete a recorded week entry so a forced manual catch-up can repost cleanly."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM merged_weeks WHERE repository = ? AND week_key = ?",
                (repository, week_key),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def last_merged_week(self, repository: str) -> dict | None:
        """Return the most recent posted weekly summary for ``repository``."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT week_key, posted_at, via
                FROM merged_weeks
                WHERE repository = ?
                ORDER BY posted_at DESC
                LIMIT 1
                """,
                (repository,),
            ).fetchone()
        return dict(row) if row else None

    def mark_merged_week(
        self,
        repository: str,
        week_key: str,
        posted_at: str,
        *,
        via: str = "scheduler",
    ) -> bool:
        """Return False when this week was already recorded."""
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO merged_weeks (repository, week_key, posted_at, via)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(repository, week_key) DO NOTHING
                """,
                (repository, week_key, posted_at, via),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def get_confirmed_match(self, github_login: str) -> dict | None:
        key = (github_login or "").strip().lower()
        if not key:
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT github_login, github_user_id, discord_id, method, confidence, matched_at
                FROM github_discord_matches
                WHERE github_login = ?
                """,
                (key,),
            ).fetchone()
        return dict(row) if row else None

    def save_confirmed_match(
        self,
        github_login: str,
        discord_id: str,
        *,
        method: str,
        confidence: float,
        matched_at: str,
        github_user_id: int | None = None,
    ) -> None:
        key = (github_login or "").strip().lower()
        if not key or not _LOGIN.fullmatch(key) or not is_discord_id(discord_id):
            raise ValueError("Invalid GitHub login or Discord id")
        uid = github_user_id if isinstance(github_user_id, int) and github_user_id > 0 else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO github_discord_matches (
                    github_login, github_user_id, discord_id, method, confidence, matched_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(github_login) DO UPDATE SET
                    github_user_id = COALESCE(excluded.github_user_id, github_discord_matches.github_user_id),
                    discord_id = excluded.discord_id,
                    method = excluded.method,
                    confidence = excluded.confidence,
                    matched_at = excluded.matched_at
                """,
                (key, uid, discord_id, str(method), float(confidence), matched_at),
            )
            self._conn.commit()

    def all_confirmed_matches(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT github_login, github_user_id, discord_id, method, confidence, matched_at
                FROM github_discord_matches
                ORDER BY github_login
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def confirmed_match_for_discord(self, discord_id: str) -> dict | None:
        if not is_discord_id(discord_id):
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT github_login, github_user_id, discord_id, method, confidence, matched_at
                FROM github_discord_matches
                WHERE discord_id = ?
                ORDER BY matched_at DESC
                LIMIT 1
                """,
                (discord_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_cached_github_user(self, github_login: str) -> dict | None:
        key = (github_login or "").strip().lower()
        if not key:
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT github_login, name, blog, twitter_username, bio, email_local, fetched_at
                FROM github_users_cache
                WHERE github_login = ?
                """,
                (key,),
            ).fetchone()
        return dict(row) if row else None

    def save_cached_github_user(
        self,
        github_login: str,
        *,
        name: str = "",
        blog: str = "",
        twitter_username: str = "",
        bio: str = "",
        email_local: str = "",
        fetched_at: str,
    ) -> None:
        key = (github_login or "").strip().lower()
        if not key or not _LOGIN.fullmatch(key):
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO github_users_cache (
                    github_login, name, blog, twitter_username, bio, email_local, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(github_login) DO UPDATE SET
                    name = excluded.name,
                    blog = excluded.blog,
                    twitter_username = excluded.twitter_username,
                    bio = excluded.bio,
                    email_local = excluded.email_local,
                    fetched_at = excluded.fetched_at
                """,
                (
                    key,
                    (name or "")[:200],
                    (blog or "")[:300],
                    (twitter_username or "")[:100],
                    (bio or "")[:400],
                    (email_local or "")[:120],
                    fetched_at,
                ),
            )
            self._conn.commit()

    def mark_review_done(self, repository: str, number: int, reviewed_at: str, head_sha: str) -> None:
        self._review_key(repository, number)
        if not re.fullmatch(r"[0-9a-f]{40}", head_sha or ""):
            head_sha = ""
        with self._lock:
            self._conn.execute(
                """
                UPDATE review_threads SET last_reviewed_at = ?, head_sha = ?
                WHERE repository = ? AND number = ?
                """,
                (reviewed_at, head_sha or None, repository, number),
            )
            self._conn.commit()

    def get_pr_board_item(self, project_id: str, repository: str, pr_number: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT project_id, repository, pr_number, pr_node_id, item_id, status, pr_updated_at, synced_at
                FROM pr_board_items
                WHERE project_id = ? AND repository = ? AND pr_number = ?
                """,
                (project_id, repository, pr_number),
            ).fetchone()
        return dict(row) if row else None

    def upsert_pr_board_item(
        self,
        project_id: str,
        repository: str,
        pr_number: int,
        pr_node_id: str,
        item_id: str,
        status: str,
        pr_updated_at: str,
        synced_at: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO pr_board_items (
                    project_id, repository, pr_number, pr_node_id, item_id, status, pr_updated_at, synced_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, repository, pr_number) DO UPDATE SET
                    pr_node_id = excluded.pr_node_id,
                    item_id = excluded.item_id,
                    status = excluded.status,
                    pr_updated_at = excluded.pr_updated_at,
                    synced_at = excluded.synced_at
                """,
                (project_id, repository, pr_number, pr_node_id, item_id, status, pr_updated_at, synced_at),
            )
            self._conn.commit()

    def pr_board_status_counts(self, project_id: str, repository: str) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM pr_board_items
                WHERE project_id = ? AND repository = ?
                GROUP BY status
                """,
                (project_id, repository),
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def get_pr_board_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM pr_board_meta WHERE key = ?",
                (key,),
            ).fetchone()
        return str(row["value"]) if row else None

    def set_pr_board_meta(self, key: str, value: str, updated_at: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO pr_board_meta (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, updated_at),
            )
            self._conn.commit()

    def get_roles_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM roles_poller_meta WHERE key = ?",
                (key,),
            ).fetchone()
        return str(row["value"]) if row else None

    def set_roles_meta(self, key: str, value: str, updated_at: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO roles_poller_meta (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, updated_at),
            )
            self._conn.commit()

    @staticmethod
    def _review_key(repository: str, number: int) -> None:
        if not _REPO.fullmatch(repository) or not isinstance(number, int) or number <= 0:
            raise ValueError("Invalid pull request")


_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def _pull_dict(row: sqlite3.Row) -> dict:
    return {
        "repository": row["repository"],
        "number": row["number"],
        "github_user_id": row["github_user_id"],
        "github_login": row["github_login"],
        "merged_at": row["merged_at"],
        "title": row["title"],
        "labels": json.loads(row["labels_json"]),
        "files": json.loads(row["files_json"]),
        "shoutout_at": row["shoutout_at"],
        "invite_commented_at": row["invite_commented_at"],
    }
