"""Catch up on merged pull requests that landed before a webhook was delivered.

Search is paginated and split the same way as the weekly roundup, so GitHub's
1,000-result cap does not drop history. Each pull request is fetched, stored,
and skipped on the next run. Rate-limit responses pause and retry.
"""

import logging
import threading
import time
from datetime import datetime, timedelta

from .api import TransientAPIError, fetch_merged_numbers, fetch_pull_files, fetch_pull_request
from .config import Config
from .rewards import _is_bot, _label_names, apply_roles
from .schedule import UTC, iso, parse_time
from .store import Store

WINDOW = timedelta(days=28)


def ingest_number(config: Config, store: Store, number: int) -> bool:
    """Fetch one pull request and store it when a human merged it.

    Returns False when the pull request is not a merged human contribution.
    """
    pull = fetch_pull_request(config.github_repository, number, config.effective_github_token)
    user = pull.get("user") if isinstance(pull.get("user"), dict) else {}
    if not pull.get("merged") or _is_bot(user):
        return False
    login = user.get("login")
    user_id = user.get("id")
    merged_raw = pull.get("merged_at")
    if not isinstance(login, str) or not isinstance(user_id, int) or not isinstance(merged_raw, str):
        raise RuntimeError(f"Merged PR #{number} is missing its author or merge time")
    files = fetch_pull_files(config.github_repository, number, config.effective_github_token)
    title = pull.get("title") if isinstance(pull.get("title"), str) else ""
    store.upsert_pull(
        repository=config.github_repository,
        number=number,
        github_user_id=user_id,
        github_login=login,
        merged_at=iso(parse_time(merged_raw)),
        title=title,
        labels=_label_names(pull),
        files=files,
    )
    return True


def _retry(description: str, func):
    for attempt in range(5):
        try:
            return func()
        except TransientAPIError as error:
            delay = min(60, 5 * (attempt + 1))
            logging.warning("%s deferred (%s); retrying in %ss", description, error, delay)
            time.sleep(delay)
    raise RuntimeError(f"{description} kept hitting a GitHub rate limit")


def sync_author(config: Config, store: Store, login: str, github_user_id: int, discord_id: str) -> None:
    """Pull this author's merged history, then evaluate roles once."""
    if config.github_token:
        since = parse_time(config.backfill_since)
        now = datetime.now(UTC)
        try:
            numbers = fetch_merged_numbers(
                config.github_repository, since, now, config.github_token, author=login,
            )
        except TransientAPIError as error:
            logging.warning(
                "Could not list merged PRs for %s (%s); evaluating what is already recorded",
                login, error,
            )
            numbers = set()
        for number in sorted(numbers):
            if store.get_pull(config.github_repository, number):
                continue
            try:
                _retry(f"Catch-up for {login} #{number}", lambda number=number: ingest_number(config, store, number))
            except (TransientAPIError, RuntimeError) as error:
                logging.warning("Stopped catch-up for %s on #%s (%s)", login, number, error)
                break
    else:
        logging.warning(
            "GITHUB_TOKEN is unset; %s only gets credit for pull requests already recorded",
            login,
        )
    apply_roles(config, store, discord_id, github_user_id, shoutout=False)


def schedule_author_sync(config: Config, store: Store, login: str, github_user_id: int, discord_id: str) -> None:
    """Grant roles for PRs we already have, then catch up the rest in the background."""
    apply_roles(config, store, discord_id, github_user_id, shoutout=False)

    def _run() -> None:
        try:
            sync_author(config, store, login, github_user_id, discord_id)
        except Exception:
            logging.exception("GitHub catch-up failed for %s", login)

    threading.Thread(target=_run, name=f"link-sync-{discord_id}", daemon=True).start()


def backfill_merged_pulls(config: Config) -> None:
    """Scan historical merged PRs, then evaluate every linked Discord account."""
    store = Store(config.roles_db_path)
    try:
        cursor = parse_time(config.backfill_since)
        now = datetime.now(UTC)
        while cursor < now:
            end = min(now, cursor + WINDOW)
            start_iso, end_iso = iso(cursor), iso(end)
            if not store.backfill_done(config.github_repository, start_iso, end_iso):
                numbers = _retry(
                    f"Backfill search {start_iso}..{end_iso}",
                    lambda start=cursor, stop=end: fetch_merged_numbers(
                        config.github_repository, start, stop, config.github_token,
                    ),
                )
                stored = 0
                for number in sorted(numbers):
                    existing = store.get_pull(config.github_repository, number)
                    if existing:
                        continue
                    if _retry(f"Backfill PR #{number}", lambda number=number: ingest_number(config, store, number)):
                        stored += 1
                    if stored and stored % 25 == 0:
                        logging.info("Backfill stored %s new pull requests in %s..%s", stored, start_iso, end_iso)
                store.mark_backfill_done(config.github_repository, start_iso, end_iso, iso(datetime.now(UTC)))
                logging.info(
                    "Backfill window %s..%s complete (%s new pull requests)",
                    start_iso, end_iso, stored,
                )
            cursor = end
        total = store.pull_count(config.github_repository)
        if not config.contributor_roles_enabled:
            logging.info(
                "Stored %s merged pull requests. CONTRIBUTOR_ROLES_ENABLED is false, so no Discord roles were granted.",
                total,
            )
            return
        links = store.all_links()
        logging.info("Evaluating roles for %s linked accounts (%s pull requests stored)", len(links), total)
        for link in links:
            apply_roles(config, store, link["discord_id"], link["github_user_id"], shoutout=False)
            time.sleep(0.4)
    finally:
        store.close()
