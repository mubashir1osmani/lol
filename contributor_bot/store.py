"""SQLite persistence for GitHub links, merged PRs, and granted roles.

The weekly thank-you ledger stays in the JSON state file. This database is
separate so role history cannot reset announcement dedupe.

The SQL is ordinary (explicit columns, no SQLite-only types) so a Postgres
backend can replace this module later: keep the method names, switch
placeholders from ``?`` to ``%s``, and translate ``INSERT OR IGNORE`` to
``INSERT ... ON CONFLICT DO NOTHING``. Callers only use these methods.
"""

import json
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
