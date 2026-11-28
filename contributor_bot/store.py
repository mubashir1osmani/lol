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
from datetime import datetime, timedelta
from pathlib import Path

from .config import is_discord_id
from .schedule import UTC, iso

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
    thread_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_reviewed_at TEXT,
    last_followup_at TEXT,
    head_sha TEXT,
    PRIMARY KEY (repository, number)
);
CREATE INDEX IF NOT EXISTS review_threads_thread_id
    ON review_threads (thread_id);
CREATE INDEX IF NOT EXISTS review_threads_message
    ON review_threads (channel_id, message_id);

CREATE TABLE IF NOT EXISTS review_requests (
    repository TEXT NOT NULL,
    number INTEGER NOT NULL,
    channel_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    status TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'new',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    PRIMARY KEY (repository, number, message_id)
);
CREATE INDEX IF NOT EXISTS review_requests_lookup
    ON review_requests (repository, status, updated_at);

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
    matched_at TEXT NOT NULL,
    evidence TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS match_rejections (
    github_login TEXT NOT NULL,
    discord_id TEXT NOT NULL,
    rejected_at TEXT NOT NULL,
    rejected_by TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (github_login, discord_id)
);
CREATE INDEX IF NOT EXISTS match_rejections_discord
    ON match_rejections (discord_id);

CREATE TABLE IF NOT EXISTS github_users_cache (
    github_login TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    blog TEXT NOT NULL DEFAULT '',
    twitter_username TEXT NOT NULL DEFAULT '',
    bio TEXT NOT NULL DEFAULT '',
    email_local TEXT NOT NULL DEFAULT '',
    commit_name TEXT NOT NULL DEFAULT '',
    profile_json TEXT NOT NULL DEFAULT '',
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pr_link_posters (
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    discord_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    message_text TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (repo, pr_number, message_id)
);
CREATE INDEX IF NOT EXISTS pr_link_posters_repo_pr
    ON pr_link_posters (repo, pr_number);

CREATE TABLE IF NOT EXISTS match_attempts (
    login TEXT PRIMARY KEY,
    attempted_at TEXT NOT NULL,
    signature TEXT NOT NULL,
    outcome TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS author_merged_counts (
    repository TEXT NOT NULL,
    github_login TEXT NOT NULL,
    github_user_id INTEGER,
    total_count INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (repository, github_login)
);

CREATE TABLE IF NOT EXISTS roles_daily_runs (
    repository TEXT NOT NULL,
    run_date TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    via TEXT NOT NULL DEFAULT 'scheduler',
    prs_discovered INTEGER NOT NULL DEFAULT 0,
    roles_granted INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (repository, run_date)
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_locks (
    lock_name TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS member_identity_notes (
    discord_id TEXT PRIMARY KEY,
    inferred_github_login TEXT,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    message_links_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.0,
    last_message_id TEXT NOT NULL DEFAULT '',
    last_seen_message_id TEXT NOT NULL DEFAULT '',
    messages_scanned INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_member_identity_notes_login
    ON member_identity_notes(lower(inferred_github_login));
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
        self._conn.execute("PRAGMA busy_timeout=30000")
        if path != ":memory:":
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass
        with self._lock:
            self._conn.executescript(_SCHEMA)
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(review_threads)")}
            if columns and "last_followup_at" not in columns:
                self._conn.execute("ALTER TABLE review_threads ADD COLUMN last_followup_at TEXT")
            self._migrate_review_threads_drop_unique_thread_id()
            mw_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(merged_weeks)")}
            if mw_columns and "via" not in mw_columns:
                self._conn.execute("ALTER TABLE merged_weeks ADD COLUMN via TEXT NOT NULL DEFAULT 'scheduler'")
            guc_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(github_users_cache)")}
            if guc_columns and "commit_name" not in guc_columns:
                self._conn.execute("ALTER TABLE github_users_cache ADD COLUMN commit_name TEXT NOT NULL DEFAULT ''")
            if guc_columns and "profile_json" not in guc_columns:
                self._conn.execute("ALTER TABLE github_users_cache ADD COLUMN profile_json TEXT NOT NULL DEFAULT ''")
            plp_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(pr_link_posters)")}
            if plp_columns and "message_text" not in plp_columns:
                self._conn.execute("ALTER TABLE pr_link_posters ADD COLUMN message_text TEXT NOT NULL DEFAULT ''")
            gdm_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(github_discord_matches)")}
            if gdm_columns and "evidence" not in gdm_columns:
                self._conn.execute("ALTER TABLE github_discord_matches ADD COLUMN evidence TEXT NOT NULL DEFAULT ''")
            # One-time migration for the bogus 2026-W48 mark created by an ephemeral /merged-this-week preview.
            # Guarded by schema_migrations so it runs at most once and never touches legitimate public posts.
            migration_name = "clear_bogus_2026_w48_preview_v1"
            already_migrated = self._conn.execute(
                "SELECT 1 FROM schema_migrations WHERE name = ?",
                (migration_name,),
            ).fetchone()
            if already_migrated is None:
                bad_rows = self._conn.execute(
                    "SELECT repository, week_key, posted_at, via FROM merged_weeks "
                    "WHERE week_key = '2026-W48' "
                    "AND (posted_at LIKE '2026-11-27T20:34:06%' OR via IN ('preview', 'manual_preview'))",
                ).fetchall()
                if bad_rows:
                    self._conn.execute(
                        "DELETE FROM merged_weeks "
                        "WHERE week_key = '2026-W48' "
                        "AND (posted_at LIKE '2026-11-27T20:34:06%' OR via IN ('preview', 'manual_preview'))",
                    )
                    for row in bad_rows:
                        logging.info(
                            "Cleared bogus ephemeral-preview merged_weeks entry for %s %s "
                            "(posted_at=%s, via=%s) so scheduled post can run",
                            row["repository"],
                            row["week_key"],
                            row["posted_at"],
                            row["via"],
                        )
                self._conn.execute(
                    "INSERT OR IGNORE INTO schema_migrations (name, applied_at) VALUES (?, datetime('now'))",
                    (migration_name,),
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

    def all_pull_authors(self, repository: str, *, merged_since: str | None = None) -> list[dict]:
        """Return distinct PR authors stored for ``repository``, with latest login and PR count."""
        with self._lock:
            if merged_since:
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
                        COUNT(*) AS pull_count,
                        MAX(merged_at) AS last_merged_at
                    FROM merged_pull_requests p1
                    WHERE repository = ? AND merged_at >= ?
                    GROUP BY github_user_id
                    ORDER BY last_merged_at DESC, github_user_id DESC
                    """,
                    (repository, merged_since),
                ).fetchall()
            else:
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
                        COUNT(*) AS pull_count,
                        MAX(merged_at) AS last_merged_at
                    FROM merged_pull_requests p1
                    WHERE repository = ?
                    GROUP BY github_user_id
                    ORDER BY last_merged_at DESC, github_user_id DESC
                    """,
                    (repository,),
                ).fetchall()
        return [
            {
                "github_user_id": int(row["github_user_id"]),
                "github_login": str(row["github_login"]),
                "pull_count": int(row["pull_count"]),
                "last_merged_at": str(row["last_merged_at"] or ""),
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

    def grants_for_discord(self, discord_id: str) -> list[dict]:
        if not is_discord_id(discord_id):
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT discord_id, role_key, role_name, granted_at, announced_at
                FROM role_grants
                WHERE discord_id = ?
                ORDER BY granted_at ASC, role_key ASC
                """,
                (discord_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_grants_for_discord(self, discord_id: str) -> list[str]:
        """Delete all role_grants rows for ``discord_id`` and return the deleted role keys."""
        if not is_discord_id(discord_id):
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT role_key FROM role_grants WHERE discord_id = ?",
                (discord_id,),
            ).fetchall()
            keys = [row["role_key"] for row in rows]
            if keys:
                self._conn.execute("DELETE FROM role_grants WHERE discord_id = ?", (discord_id,))
                self._conn.commit()
        return keys

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

    def all_unannounced_grants(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT discord_id, role_key, role_name, granted_at FROM role_grants
                WHERE announced_at IS NULL
                ORDER BY granted_at, discord_id, role_key
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def all_granted_discord_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT discord_id FROM role_grants ORDER BY discord_id"
            ).fetchall()
        return [str(row["discord_id"]) for row in rows]

    def delete_grant(self, discord_id: str, role_key: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM role_grants WHERE discord_id = ? AND role_key = ?",
                (discord_id, role_key),
            )
            self._conn.commit()
        return cur.rowcount > 0

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

    def advance_review_cursor(self, channel_id: str, message_id: str) -> None:
        """Advance a channel's review cursor forward only (never backwards)."""
        if not is_discord_id(channel_id) or not is_discord_id(message_id):
            return
        with self._lock:
            row = self._conn.execute(
                "SELECT last_message_id FROM review_cursors WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
            if row is not None and is_discord_id(row["last_message_id"]):
                if int(message_id) <= int(row["last_message_id"]):
                    return
            self._conn.execute(
                """
                INSERT INTO review_cursors (channel_id, last_message_id) VALUES (?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET last_message_id = excluded.last_message_id
                """,
                (channel_id, message_id),
            )
            self._conn.commit()

    def _migrate_review_threads_drop_unique_thread_id(self) -> None:
        """Allow multiple PR rows to share one Discord thread (e.g. a message linking 2+ PRs)."""
        has_unique_thread_id = False
        for idx in self._conn.execute("PRAGMA index_list(review_threads)").fetchall():
            if not idx["unique"]:
                continue
            cols = [
                col["name"]
                for col in self._conn.execute(f'PRAGMA index_info("{idx["name"]}")').fetchall()
            ]
            if cols == ["thread_id"]:
                has_unique_thread_id = True
                break
        if not has_unique_thread_id:
            return
        logging.info("Migrating review_threads table to allow multiple PRs per Discord thread")
        existing_cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(review_threads)").fetchall()
        }
        target_cols = (
            "repository",
            "number",
            "channel_id",
            "message_id",
            "thread_id",
            "created_at",
            "last_reviewed_at",
            "last_followup_at",
            "head_sha",
        )
        select_exprs = ", ".join(
            col if col in existing_cols else f"NULL AS {col}"
            for col in target_cols
        )
        self._conn.execute(
            """
            CREATE TABLE review_threads_new (
                repository TEXT NOT NULL,
                number INTEGER NOT NULL,
                channel_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_reviewed_at TEXT,
                last_followup_at TEXT,
                head_sha TEXT,
                PRIMARY KEY (repository, number)
            )
            """
        )
        self._conn.execute(
            f"""
            INSERT INTO review_threads_new ({", ".join(target_cols)})
            SELECT {select_exprs}
            FROM review_threads
            """
        )
        self._conn.execute("DROP TABLE review_threads")
        self._conn.execute("ALTER TABLE review_threads_new RENAME TO review_threads")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS review_threads_thread_id ON review_threads (thread_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS review_threads_message ON review_threads (channel_id, message_id)"
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

    def review_threads_by_discord(self, thread_id: str) -> list[dict]:
        if not is_discord_id(thread_id):
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM review_threads WHERE thread_id = ? ORDER BY created_at ASC, number ASC",
                (thread_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def review_thread_by_discord(self, thread_id: str) -> dict | None:
        rows = self.review_threads_by_discord(thread_id)
        return rows[0] if rows else None

    def review_thread_by_message(self, channel_id: str, message_id: str) -> dict | None:
        if not is_discord_id(channel_id) or not is_discord_id(message_id):
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM review_threads
                WHERE channel_id = ? AND message_id = ?
                ORDER BY created_at ASC, number ASC
                LIMIT 1
                """,
                (channel_id, message_id),
            ).fetchone()
        return dict(row) if row else None

    def incomplete_review_threads(self, repository: str, *, since_iso: str) -> list[dict]:
        """Return review_threads created since ``since_iso`` that never finished a review."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM review_threads
                WHERE lower(repository) = lower(?)
                  AND last_reviewed_at IS NULL
                  AND created_at >= ?
                ORDER BY created_at ASC, number ASC
                """,
                (repository, since_iso),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_review_request(
        self,
        *,
        repository: str,
        number: int,
        channel_id: str,
        message_id: str,
        status: str = "pending",
        source: str = "new",
        now_iso: str,
    ) -> None:
        self._review_key(repository, number)
        if not is_discord_id(channel_id) or not is_discord_id(message_id):
            raise ValueError("Invalid Discord id for review request")
        if status not in {"pending", "done", "failed"}:
            raise ValueError("Invalid review request status")
        completed_at = now_iso if status == "done" else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO review_requests (
                    repository, number, channel_id, message_id, status, source,
                    attempts, last_error, created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, '', ?, ?, ?)
                ON CONFLICT(repository, number, message_id) DO UPDATE SET
                    status = CASE
                        WHEN review_requests.status = 'done' THEN 'done'
                        ELSE excluded.status
                    END,
                    source = CASE
                        WHEN review_requests.status = 'done' THEN review_requests.source
                        ELSE excluded.source
                    END,
                    updated_at = excluded.updated_at,
                    completed_at = COALESCE(review_requests.completed_at, excluded.completed_at)
                """,
                (repository, number, channel_id, message_id, status, source, now_iso, now_iso, completed_at),
            )
            self._conn.commit()

    def get_review_request(self, repository: str, number: int, message_id: str) -> dict | None:
        self._review_key(repository, number)
        if not is_discord_id(message_id):
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM review_requests
                WHERE repository = ? AND number = ? AND message_id = ?
                """,
                (repository, number, message_id),
            ).fetchone()
        return dict(row) if row else None

    def mark_review_request(
        self,
        repository: str,
        number: int,
        message_id: str,
        status: str,
        *,
        error: str = "",
        updated_at: str,
        increment_attempts: bool = False,
    ) -> None:
        self._review_key(repository, number)
        if not is_discord_id(message_id):
            return
        if status not in {"pending", "done", "failed"}:
            raise ValueError("Invalid review request status")
        clean_error = str(error or "")[:500]
        completed_at = updated_at if status == "done" else None
        inc = 1 if increment_attempts else 0
        with self._lock:
            self._conn.execute(
                """
                UPDATE review_requests
                SET status = ?,
                    last_error = ?,
                    updated_at = ?,
                    attempts = attempts + ?,
                    completed_at = CASE
                        WHEN ? = 'done' THEN COALESCE(completed_at, ?)
                        ELSE completed_at
                    END
                WHERE repository = ? AND number = ? AND message_id = ?
                """,
                (status, clean_error, updated_at, inc, status, completed_at, repository, number, message_id),
            )
            self._conn.commit()

    def pending_review_requests(
        self,
        repository: str,
        *,
        since_iso: str,
        max_attempts: int = 3,
    ) -> list[dict]:
        """Return pending or failed review requests from ``since_iso`` with attempts below ``max_attempts``."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM review_requests
                WHERE lower(repository) = lower(?)
                  AND status IN ('pending', 'failed')
                  AND updated_at >= ?
                  AND attempts < ?
                ORDER BY created_at ASC, number ASC
                """,
                (repository, since_iso, max(1, int(max_attempts))),
            ).fetchall()
        return [dict(row) for row in rows]

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
                """
                SELECT week_key, posted_at, via
                FROM merged_weeks
                WHERE lower(repository) = lower(?) AND week_key = ?
                ORDER BY posted_at DESC
                LIMIT 1
                """,
                (repository, week_key),
            ).fetchone()
        return dict(row) if row else None

    def clear_merged_week(self, repository: str, week_key: str) -> bool:
        """Delete a recorded week entry so a forced manual catch-up can repost cleanly."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM merged_weeks WHERE lower(repository) = lower(?) AND week_key = ?",
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
                WHERE lower(repository) = lower(?)
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
            existing = self._conn.execute(
                "SELECT 1 FROM merged_weeks WHERE lower(repository) = lower(?) AND week_key = ?",
                (repository, week_key),
            ).fetchone()
            if existing is not None:
                return False
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
                SELECT github_login, github_user_id, discord_id, method, confidence, matched_at, evidence
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
        evidence: str | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        key = (github_login or "").strip().lower()
        if not key or not _LOGIN.fullmatch(key) or not is_discord_id(discord_id):
            raise ValueError("Invalid GitHub login or Discord id")
        uid = github_user_id if isinstance(github_user_id, int) and github_user_id > 0 else None
        if isinstance(evidence, (list, tuple)):
            ev_text = json.dumps([str(item) for item in evidence if str(item).strip()])
        else:
            ev_text = str(evidence or "")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO github_discord_matches (
                    github_login, github_user_id, discord_id, method, confidence, matched_at, evidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(github_login) DO UPDATE SET
                    github_user_id = COALESCE(excluded.github_user_id, github_discord_matches.github_user_id),
                    discord_id = excluded.discord_id,
                    method = excluded.method,
                    confidence = excluded.confidence,
                    matched_at = excluded.matched_at,
                    evidence = CASE WHEN excluded.evidence != '' THEN excluded.evidence ELSE github_discord_matches.evidence END
                """,
                (key, uid, discord_id, str(method), float(confidence), matched_at, ev_text),
            )
            self._conn.commit()

    def all_confirmed_matches(self, *, matched_since: str | None = None) -> list[dict]:
        with self._lock:
            if matched_since:
                rows = self._conn.execute(
                    """
                    SELECT github_login, github_user_id, discord_id, method, confidence, matched_at, evidence
                    FROM github_discord_matches
                    WHERE matched_at >= ?
                    ORDER BY github_login
                    """,
                    (matched_since,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT github_login, github_user_id, discord_id, method, confidence, matched_at, evidence
                    FROM github_discord_matches
                    ORDER BY github_login
                    """
                ).fetchall()
        return [dict(row) for row in rows]

    def confirmed_matches_by_method(self, method: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT github_login, github_user_id, discord_id, method, confidence, matched_at, evidence
                FROM github_discord_matches
                WHERE method = ?
                ORDER BY github_login
                """,
                (str(method),),
            ).fetchall()
        return [dict(row) for row in rows]

    def confirmed_match_for_discord(self, discord_id: str) -> dict | None:
        if not is_discord_id(discord_id):
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT github_login, github_user_id, discord_id, method, confidence, matched_at, evidence
                FROM github_discord_matches
                WHERE discord_id = ?
                ORDER BY matched_at DESC
                LIMIT 1
                """,
                (discord_id,),
            ).fetchone()
        return dict(row) if row else None

    def confirmed_matches_for_discord(self, discord_id: str) -> list[dict]:
        if not is_discord_id(discord_id):
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT github_login, github_user_id, discord_id, method, confidence, matched_at, evidence
                FROM github_discord_matches
                WHERE discord_id = ?
                ORDER BY matched_at DESC, github_login ASC
                """,
                (discord_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_confirmed_match(self, github_login: str) -> bool:
        key = (github_login or "").strip().lower()
        if not key:
            return False
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM github_discord_matches WHERE github_login = ?",
                (key,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def delete_confirmed_matches_for_discord(self, discord_id: str) -> list[dict]:
        """Delete all github_discord_matches rows for ``discord_id`` and return the deleted rows."""
        if not is_discord_id(discord_id):
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT github_login, github_user_id, discord_id, method, confidence, matched_at, evidence
                FROM github_discord_matches
                WHERE discord_id = ?
                ORDER BY github_login ASC
                """,
                (discord_id,),
            ).fetchall()
            deleted = [dict(row) for row in rows]
            if deleted:
                self._conn.execute(
                    "DELETE FROM github_discord_matches WHERE discord_id = ?",
                    (discord_id,),
                )
                self._conn.commit()
        return deleted

    def add_match_rejection(
        self,
        github_login: str,
        discord_id: str,
        rejected_at: str,
        *,
        rejected_by: str = "",
        reason: str = "",
    ) -> None:
        key = (github_login or "").strip().lower()
        if not key or not is_discord_id(discord_id):
            raise ValueError("Invalid GitHub login or Discord id")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO match_rejections (github_login, discord_id, rejected_at, rejected_by, reason)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(github_login, discord_id) DO UPDATE SET
                    rejected_at = excluded.rejected_at,
                    rejected_by = excluded.rejected_by,
                    reason = excluded.reason
                """,
                (key, discord_id, str(rejected_at), str(rejected_by or ""), str(reason or "")),
            )
            self._conn.commit()

    def remove_match_rejection(self, github_login: str, discord_id: str) -> bool:
        key = (github_login or "").strip().lower()
        if not key or not is_discord_id(discord_id):
            return False
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM match_rejections WHERE github_login = ? AND discord_id = ?",
                (key, discord_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def is_match_rejected(self, github_login: str, discord_id: str) -> bool:
        key = (github_login or "").strip().lower()
        if not key or not is_discord_id(discord_id):
            return False
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM match_rejections WHERE github_login = ? AND discord_id = ?",
                (key, discord_id),
            ).fetchone()
        return row is not None

    def all_match_rejections(self) -> set[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT github_login, discord_id FROM match_rejections"
            ).fetchall()
        return {(str(row["github_login"]).lower(), str(row["discord_id"])) for row in rows}

    def match_rejections_for_discord(self, discord_id: str) -> list[dict]:
        if not is_discord_id(discord_id):
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT github_login, discord_id, rejected_at, rejected_by, reason
                FROM match_rejections
                WHERE discord_id = ?
                ORDER BY rejected_at DESC, github_login ASC
                """,
                (discord_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def all_links(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT discord_id, github_user_id, github_login, linked_at FROM github_links"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_cached_github_user(self, github_login: str) -> dict | None:
        key = (github_login or "").strip().lower()
        if not key:
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT github_login, name, blog, twitter_username, bio, email_local,
                       commit_name, profile_json, fetched_at
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
        commit_name: str = "",
        profile_json: str = "",
        fetched_at: str,
    ) -> None:
        key = (github_login or "").strip().lower()
        if not key or not _LOGIN.fullmatch(key):
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO github_users_cache (
                    github_login, name, blog, twitter_username, bio, email_local,
                    commit_name, profile_json, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(github_login) DO UPDATE SET
                    name = excluded.name,
                    blog = excluded.blog,
                    twitter_username = excluded.twitter_username,
                    bio = excluded.bio,
                    email_local = excluded.email_local,
                    commit_name = CASE
                        WHEN excluded.commit_name != '' THEN excluded.commit_name
                        ELSE github_users_cache.commit_name
                    END,
                    profile_json = CASE
                        WHEN excluded.profile_json != '' THEN excluded.profile_json
                        ELSE github_users_cache.profile_json
                    END,
                    fetched_at = excluded.fetched_at
                """,
                (
                    key,
                    (name or "")[:200],
                    (blog or "")[:300],
                    (twitter_username or "")[:100],
                    (bio or "")[:400],
                    (email_local or "")[:120],
                    (commit_name or "")[:200],
                    (profile_json or "")[:16000],
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

    def roles_day_completed(self, repository: str, run_date: str) -> bool:
        return self.roles_day_info(repository, run_date) is not None

    def roles_day_info(self, repository: str, run_date: str) -> dict | None:
        """Return daily roles run metadata if ``run_date`` (YYYY-MM-DD) already completed."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT run_date, completed_at, via, prs_discovered, roles_granted
                FROM roles_daily_runs
                WHERE lower(repository) = lower(?) AND run_date = ?
                ORDER BY completed_at DESC
                LIMIT 1
                """,
                (repository, run_date),
            ).fetchone()
        return dict(row) if row else None

    def mark_roles_day(
        self,
        repository: str,
        run_date: str,
        completed_at: str,
        *,
        via: str = "scheduler",
        prs_discovered: int = 0,
        roles_granted: int = 0,
    ) -> bool:
        """Record a completed daily contributor-roles run. Returns False if already marked."""
        with self._lock:
            existing = self._conn.execute(
                "SELECT 1 FROM roles_daily_runs WHERE lower(repository) = lower(?) AND run_date = ?",
                (repository, run_date),
            ).fetchone()
            if existing is not None:
                return False
            cursor = self._conn.execute(
                """
                INSERT INTO roles_daily_runs (
                    repository, run_date, completed_at, via, prs_discovered, roles_granted
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repository, run_date) DO NOTHING
                """,
                (
                    repository,
                    run_date,
                    completed_at,
                    via,
                    int(prs_discovered),
                    int(roles_granted),
                ),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def get_author_merged_count(self, repository: str, github_login: str) -> dict | None:
        key = (github_login or "").strip().lower()
        if not key:
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT repository, github_login, github_user_id, total_count, fetched_at
                FROM author_merged_counts
                WHERE repository = ? AND github_login = ?
                """,
                (repository, key),
            ).fetchone()
        return dict(row) if row else None

    def set_author_merged_count(
        self,
        repository: str,
        github_login: str,
        total_count: int,
        fetched_at: str,
        *,
        github_user_id: int | None = None,
    ) -> None:
        key = (github_login or "").strip().lower()
        if not key or not _LOGIN.fullmatch(key):
            raise ValueError("Invalid GitHub login")
        if not isinstance(total_count, int) or total_count < 0:
            raise ValueError("Invalid merged PR count")
        uid = github_user_id if isinstance(github_user_id, int) and github_user_id > 0 else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO author_merged_counts (
                    repository, github_login, github_user_id, total_count, fetched_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(repository, github_login) DO UPDATE SET
                    github_user_id = COALESCE(excluded.github_user_id, author_merged_counts.github_user_id),
                    total_count = excluded.total_count,
                    fetched_at = excluded.fetched_at
                """,
                (repository, key, uid, total_count, fetched_at),
            )
            self._conn.commit()

    def increment_author_merged_count(
        self,
        repository: str,
        github_login: str,
        delta: int = 1,
    ) -> int | None:
        """Increment a cached lifetime merged PR count if present. Returns new total or None."""
        key = (github_login or "").strip().lower()
        if not key or delta <= 0:
            return None
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE author_merged_counts
                SET total_count = total_count + ?
                WHERE repository = ? AND github_login = ?
                """,
                (delta, repository, key),
            )
            if cursor.rowcount == 0:
                return None
            row = self._conn.execute(
                "SELECT total_count FROM author_merged_counts WHERE repository = ? AND github_login = ?",
                (repository, key),
            ).fetchone()
            self._conn.commit()
        return int(row["total_count"]) if row else None

    def record_pr_link_poster(
        self,
        repo: str,
        pr_number: int,
        discord_id: str,
        message_id: str,
        ts: str,
        *,
        message_text: str = "",
    ) -> bool:
        """Record a non-bot Discord user who posted a PR link. Returns True if newly inserted."""
        norm_repo = (repo or "").strip().lower()
        if (
            not _REPO.fullmatch(norm_repo)
            or not isinstance(pr_number, int)
            or pr_number <= 0
            or not is_discord_id(str(discord_id or ""))
            or not is_discord_id(str(message_id or ""))
        ):
            return False
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO pr_link_posters (
                    repo, pr_number, discord_id, message_id, ts, message_text
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    norm_repo,
                    pr_number,
                    str(discord_id),
                    str(message_id),
                    str(ts or ""),
                    (message_text or "")[:500],
                ),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def has_pr_link_poster_message(self, repo: str, pr_number: int, message_id: str) -> bool:
        norm_repo = (repo or "").strip().lower()
        with self._lock:
            row = self._conn.execute(
                """
                SELECT 1 FROM pr_link_posters
                WHERE repo = ? AND pr_number = ? AND message_id = ?
                """,
                (norm_repo, pr_number, str(message_id)),
            ).fetchone()
        return row is not None

    def pr_link_posters_for_pulls(self, repo: str, pr_numbers: list[int] | set[int]) -> list[dict]:
        """Return all pr_link_posters rows matching ``repo`` and any PR in ``pr_numbers``."""
        norm_repo = (repo or "").strip().lower()
        nums = sorted({int(n) for n in pr_numbers if isinstance(n, int) and n > 0})
        if not norm_repo or not nums:
            return []
        out: list[dict] = []
        with self._lock:
            for offset in range(0, len(nums), 500):
                chunk = nums[offset: offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = self._conn.execute(
                    f"""
                    SELECT repo, pr_number, discord_id, message_id, ts, message_text
                    FROM pr_link_posters
                    WHERE repo = ? AND pr_number IN ({placeholders})
                    ORDER BY pr_number ASC, ts ASC, message_id ASC
                    """,
                    [norm_repo, *chunk],
                ).fetchall()
                out.extend(dict(row) for row in rows)
        return out

    def pr_link_posters_for_discord(self, repo: str, discord_id: str) -> list[dict]:
        """Return all pr_link_posters rows posted by ``discord_id`` in ``repo``."""
        norm_repo = (repo or "").strip().lower()
        if not norm_repo or not is_discord_id(discord_id):
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT repo, pr_number, discord_id, message_id, ts, message_text
                FROM pr_link_posters
                WHERE repo = ? AND discord_id = ?
                ORDER BY pr_number ASC, ts ASC, message_id ASC
                """,
                (norm_repo, discord_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def all_pr_link_posters(self, repo: str) -> list[dict]:
        """Return all pr_link_posters rows for ``repo``."""
        norm_repo = (repo or "").strip().lower()
        if not norm_repo:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT repo, pr_number, discord_id, message_id, ts, message_text
                FROM pr_link_posters
                WHERE repo = ?
                ORDER BY pr_number ASC, ts ASC, message_id ASC
                """,
                (norm_repo,),
            ).fetchall()
        return [dict(row) for row in rows]

    def pr_authors_for_numbers(self, repository: str, pr_numbers: list[int] | set[int]) -> dict[int, str]:
        """Return ``{pr_number: lowercased_github_login}`` for merged PRs stored in SQLite."""
        nums = sorted({int(n) for n in pr_numbers if isinstance(n, int) and n > 0})
        if not repository or not nums:
            return {}
        out: dict[int, str] = {}
        with self._lock:
            for offset in range(0, len(nums), 500):
                chunk = nums[offset: offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = self._conn.execute(
                    f"""
                    SELECT number, github_login
                    FROM merged_pull_requests
                    WHERE lower(repository) = lower(?) AND number IN ({placeholders})
                    """,
                    [repository, *chunk],
                ).fetchall()
                for row in rows:
                    login = str(row["github_login"] or "").strip().lower()
                    if login:
                        out[int(row["number"])] = login
        return out

    def pr_link_posters_count(self, repo: str | None = None) -> int:
        with self._lock:
            if repo is None:
                row = self._conn.execute("SELECT COUNT(*) AS n FROM pr_link_posters").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM pr_link_posters WHERE repo = ?",
                    (repo.strip().lower(),),
                ).fetchone()
        return int(row["n"]) if row else 0

    def all_review_threads(self, repository: str | None = None) -> list[dict]:
        with self._lock:
            if repository is None:
                rows = self._conn.execute(
                    """
                    SELECT repository, number, channel_id, message_id, thread_id, created_at
                    FROM review_threads
                    ORDER BY number DESC
                    """
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT repository, number, channel_id, message_id, thread_id, created_at
                    FROM review_threads
                    WHERE lower(repository) = ?
                    ORDER BY number DESC
                    """,
                    (repository.strip().lower(),),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_match_attempt(self, login: str) -> dict | None:
        key = (login or "").strip().lower()
        if not key:
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT login, attempted_at, signature, outcome
                FROM match_attempts
                WHERE login = ?
                """,
                (key,),
            ).fetchone()
        return dict(row) if row else None

    def save_match_attempt(
        self,
        login: str,
        *,
        attempted_at: str,
        signature: str,
        outcome: str,
    ) -> None:
        key = (login or "").strip().lower()
        if not key:
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO match_attempts (login, attempted_at, signature, outcome)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(login) DO UPDATE SET
                    attempted_at = excluded.attempted_at,
                    signature = excluded.signature,
                    outcome = excluded.outcome
                """,
                (key, str(attempted_at), str(signature), str(outcome)),
            )
            self._conn.commit()

    def delete_match_attempt(self, login: str) -> None:
        key = (login or "").strip().lower()
        if not key:
            return
        with self._lock:
            self._conn.execute("DELETE FROM match_attempts WHERE login = ?", (key,))
            self._conn.commit()

    @staticmethod
    def _snowflake_int(raw: object) -> int:
        text = str(raw or "").strip()
        return int(text) if text.isdigit() else 0

    @classmethod
    def _decode_identity_note(cls, row) -> dict:
        data = dict(row)
        try:
            ev = json.loads(data.get("evidence_json") or "[]")
            evidence = [str(item) for item in ev if str(item).strip()] if isinstance(ev, list) else []
        except ValueError:
            evidence = []
        try:
            links = json.loads(data.get("message_links_json") or "[]")
            message_links = [str(item) for item in links if str(item).strip()] if isinstance(links, list) else []
        except ValueError:
            message_links = []
        last_scanned = str(data.get("last_message_id") or "").strip()
        last_seen = str(data.get("last_seen_message_id") or "").strip()
        needs_refresh = cls._snowflake_int(last_seen) > cls._snowflake_int(last_scanned)
        raw_login = data.get("inferred_github_login")
        inferred = str(raw_login).strip().lower() if isinstance(raw_login, str) and raw_login.strip() else None
        return {
            "discord_id": str(data["discord_id"]),
            "inferred_github_login": inferred,
            "evidence": evidence,
            "message_links": message_links,
            "confidence": float(data.get("confidence") or 0.0),
            "last_message_id": last_scanned,
            "last_seen_message_id": last_seen,
            "messages_scanned": int(data.get("messages_scanned") or 0),
            "updated_at": str(data.get("updated_at") or ""),
            "needs_refresh": needs_refresh,
        }

    def get_member_identity_note(self, discord_id: str) -> dict | None:
        uid = str(discord_id or "").strip()
        if not is_discord_id(uid):
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT discord_id, inferred_github_login, evidence_json, message_links_json,
                       confidence, last_message_id, last_seen_message_id, messages_scanned, updated_at
                FROM member_identity_notes
                WHERE discord_id = ?
                """,
                (uid,),
            ).fetchone()
        return self._decode_identity_note(row) if row else None

    def save_member_identity_note(
        self,
        discord_id: str,
        *,
        inferred_github_login: str | None,
        evidence: list[str] | tuple[str, ...] = (),
        message_links: list[str] | tuple[str, ...] = (),
        confidence: float = 0.0,
        last_message_id: str = "",
        messages_scanned: int = 0,
        updated_at: str,
    ) -> None:
        uid = str(discord_id or "").strip()
        if not is_discord_id(uid):
            return
        inferred = (inferred_github_login or "").strip().lower() or None
        clean_evidence = [str(item).strip()[:300] for item in evidence if str(item).strip()][:20]
        clean_links = [str(item).strip()[:200] for item in message_links if str(item).strip()][:20]
        scanned_id = str(last_message_id or "").strip()
        if scanned_id and not is_discord_id(scanned_id):
            scanned_id = ""
        with self._lock:
            existing = self._conn.execute(
                "SELECT last_seen_message_id FROM member_identity_notes WHERE discord_id = ?",
                (uid,),
            ).fetchone()
            prev_seen = str(existing["last_seen_message_id"] or "") if existing else ""
            seen_id = scanned_id if self._snowflake_int(scanned_id) >= self._snowflake_int(prev_seen) else prev_seen
            self._conn.execute(
                """
                INSERT INTO member_identity_notes (
                    discord_id, inferred_github_login, evidence_json, message_links_json,
                    confidence, last_message_id, last_seen_message_id, messages_scanned, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(discord_id) DO UPDATE SET
                    inferred_github_login = excluded.inferred_github_login,
                    evidence_json = excluded.evidence_json,
                    message_links_json = excluded.message_links_json,
                    confidence = excluded.confidence,
                    last_message_id = excluded.last_message_id,
                    last_seen_message_id = excluded.last_seen_message_id,
                    messages_scanned = excluded.messages_scanned,
                    updated_at = excluded.updated_at
                """,
                (
                    uid,
                    inferred,
                    json.dumps(clean_evidence),
                    json.dumps(clean_links),
                    float(confidence),
                    scanned_id,
                    seen_id,
                    max(0, int(messages_scanned)),
                    str(updated_at),
                ),
            )
            self._conn.commit()

    def note_member_activity(self, discord_id: str, message_id: str) -> bool:
        """Advance ``last_seen_message_id`` on an existing ``member_identity_notes`` row when a new message arrives."""
        uid = str(discord_id or "").strip()
        mid = str(message_id or "").strip()
        if not is_discord_id(uid) or not is_discord_id(mid):
            return False
        with self._lock:
            row = self._conn.execute(
                "SELECT last_seen_message_id FROM member_identity_notes WHERE discord_id = ?",
                (uid,),
            ).fetchone()
            if not row:
                return False
            if self._snowflake_int(mid) <= self._snowflake_int(row["last_seen_message_id"]):
                return False
            self._conn.execute(
                "UPDATE member_identity_notes SET last_seen_message_id = ? WHERE discord_id = ?",
                (mid, uid),
            )
            self._conn.commit()
            return True

    def member_identity_notes_for_login(self, github_login: str) -> list[dict]:
        key = (github_login or "").strip().lower()
        if not key:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT discord_id, inferred_github_login, evidence_json, message_links_json,
                       confidence, last_message_id, last_seen_message_id, messages_scanned, updated_at
                FROM member_identity_notes
                WHERE lower(inferred_github_login) = ?
                ORDER BY confidence DESC, updated_at DESC
                """,
                (key,),
            ).fetchall()
        return [self._decode_identity_note(row) for row in rows]

    def all_member_identity_notes(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT discord_id, inferred_github_login, evidence_json, message_links_json,
                       confidence, last_message_id, last_seen_message_id, messages_scanned, updated_at
                FROM member_identity_notes
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [self._decode_identity_note(row) for row in rows]

    def delete_member_identity_note(self, discord_id: str) -> None:
        uid = str(discord_id or "").strip()
        if not is_discord_id(uid):
            return
        with self._lock:
            self._conn.execute("DELETE FROM member_identity_notes WHERE discord_id = ?", (uid,))
            self._conn.commit()

    def acquire_lease(
        self,
        lock_name: str,
        owner_id: str,
        ttl_seconds: float,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Atomically acquire or re-acquire a DB-backed lease lock across processes."""
        name = (lock_name or "").strip()
        owner = (owner_id or "").strip()
        if not name or not owner:
            raise ValueError("Invalid lease lock_name or owner_id")
        moment = now or datetime.now(UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        acquired_iso = iso(moment)
        expires_iso = iso(moment + timedelta(seconds=max(1.0, float(ttl_seconds))))
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO job_locks (lock_name, owner_id, acquired_at, expires_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(lock_name) DO UPDATE SET
                        owner_id = excluded.owner_id,
                        acquired_at = excluded.acquired_at,
                        expires_at = excluded.expires_at
                    WHERE job_locks.owner_id = excluded.owner_id
                       OR job_locks.expires_at <= excluded.acquired_at
                    """,
                    (name, owner, acquired_iso, expires_iso),
                )
                row = self._conn.execute(
                    "SELECT owner_id FROM job_locks WHERE lock_name = ?",
                    (name,),
                ).fetchone()
                return bool(row is not None and row["owner_id"] == owner)

    def renew_lease(
        self,
        lock_name: str,
        owner_id: str,
        ttl_seconds: float,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Extend the lease expiration if still held by ``owner_id``."""
        name = (lock_name or "").strip()
        owner = (owner_id or "").strip()
        if not name or not owner:
            return False
        moment = now or datetime.now(UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        expires_iso = iso(moment + timedelta(seconds=max(1.0, float(ttl_seconds))))
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "UPDATE job_locks SET expires_at = ? WHERE lock_name = ? AND owner_id = ?",
                    (expires_iso, name, owner),
                )
                return cur.rowcount > 0

    def release_lease(self, lock_name: str, owner_id: str) -> None:
        """Release a DB-backed lease lock only if held by ``owner_id``."""
        name = (lock_name or "").strip()
        owner = (owner_id or "").strip()
        if not name or not owner:
            return
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM job_locks WHERE lock_name = ? AND owner_id = ?",
                    (name, owner),
                )

    def get_lease(self, lock_name: str) -> dict | None:
        name = (lock_name or "").strip()
        if not name:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT lock_name, owner_id, acquired_at, expires_at FROM job_locks WHERE lock_name = ?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return {
            "lock_name": row["lock_name"],
            "owner_id": row["owner_id"],
            "acquired_at": row["acquired_at"],
            "expires_at": row["expires_at"],
        }

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
