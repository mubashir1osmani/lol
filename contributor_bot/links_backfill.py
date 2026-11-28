"""Capture and backfill Discord users who posted GitHub PR links.

Records ``pr_link_posters(repo, pr_number, discord_id, message_id, ts)`` from:
1. Live non-bot guild messages via ``gateway.on_message``.
2. Stored ``review_threads`` source message IDs.
3. Mod-only ``/links-backfill`` paging ``GET /guilds/{id}/messages/search``
   (handling HTTP 202 index-building and ``retry_after`` up to offset 9975).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime

from .api import TransientAPIError, fetch_message, search_guild_messages
from .config import Config, is_discord_id
from .schedule import UTC, iso, parse_time
from .store import Store

GUILD_SEARCH_PAGE_SIZE = 25
GUILD_SEARCH_MAX_OFFSET = 9975
LINKS_BACKFILL_DEADLINE_SECONDS = 240.0

_links_backfill_lock = threading.Lock()


@dataclass(frozen=True)
class LinksBackfillResult:
    review_threads_scanned: int
    review_threads_recorded: int
    search_pages: int
    messages_scanned: int
    links_recorded: int
    total_results: int
    rate_limited: bool = False
    timed_out: bool = False


def extract_pr_links_from_text(text: str, repository: str = "BerriAI/litellm") -> list[int]:
    """Return sorted unique PR numbers referenced as ``github.com/<repo>/pull/<N>``."""
    if not text or not repository:
        return []
    repo_escaped = re.escape(repository.strip())
    pattern = re.compile(
        rf"(?:https?://)?(?:www\.)?github\.com/{repo_escaped}/pull/([1-9][0-9]*)\b",
        re.IGNORECASE,
    )
    numbers = {int(match.group(1)) for match in pattern.finditer(text)}
    return sorted(numbers)


def _normalize_message_ts(raw: object) -> str:
    if isinstance(raw, datetime):
        return iso(raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC))
    if isinstance(raw, str) and raw.strip():
        try:
            return iso(parse_time(raw))
        except Exception:
            return raw.strip()
    return iso(datetime.now(UTC))


def record_pr_links_from_message(
    store: Store | None,
    repository: str,
    *,
    discord_id: str,
    message_id: str,
    content: str,
    ts: object = None,
    embeds: list[dict] | None = None,
) -> int:
    """Extract PR links from one non-bot guild message and persist to ``pr_link_posters``."""
    if store is None or not is_discord_id(str(discord_id or "")) or not is_discord_id(str(message_id or "")):
        return 0
    combined = content or ""
    if embeds:
        extra = " ".join(str(embed.get("url") or "") for embed in embeds if isinstance(embed, dict))
        if extra.strip():
            combined = f"{combined} {extra}".strip()
    pr_numbers = extract_pr_links_from_text(combined, repository)
    if not pr_numbers:
        return 0
    ts_iso = _normalize_message_ts(ts)
    recorded = 0
    for pr_number in pr_numbers:
        if store.record_pr_link_poster(
            repository,
            pr_number,
            str(discord_id),
            str(message_id),
            ts_iso,
            message_text=(content or "")[:500],
        ):
            recorded += 1
    return recorded


def backfill_pr_link_posters_from_review_threads(
    config: Config,
    store: Store,
    *,
    message_fetcher=None,
    deadline: float | None = None,
) -> tuple[int, int, bool]:
    """Derive PR-link posters from stored ``review_threads`` source message IDs."""
    fetcher = message_fetcher or fetch_message
    repo = config.github_repository
    threads = store.all_review_threads(repo)
    scanned = 0
    recorded = 0
    rate_limited = False
    for row in threads:
        if deadline is not None and time.monotonic() >= deadline:
            break
        pr_number = int(row["number"])
        message_id = str(row["message_id"])
        channel_id = str(row["channel_id"])
        if store.has_pr_link_poster_message(repo, pr_number, message_id):
            continue
        scanned += 1
        try:
            msg = fetcher(channel_id, message_id, config.discord_token)
        except TransientAPIError as error:
            logging.warning("Review thread message lookup rate-limited on PR #%d: %s", pr_number, error)
            rate_limited = True
            break
        except Exception as error:
            logging.info("Skipping review_threads message %s for PR #%d: %s", message_id, pr_number, error)
            continue
        if not isinstance(msg, dict):
            continue
        author = msg.get("author") if isinstance(msg.get("author"), dict) else {}
        if author.get("bot") or msg.get("webhook_id"):
            continue
        author_id = str(author.get("id") or "")
        if not is_discord_id(author_id):
            continue
        ts_iso = _normalize_message_ts(msg.get("timestamp") or row.get("created_at"))
        content = str(msg.get("content") or "")
        if store.record_pr_link_poster(
            repo,
            pr_number,
            author_id,
            message_id,
            ts_iso,
            message_text=content[:500],
        ):
            recorded += 1
    return scanned, recorded, rate_limited


def run_links_backfill(
    config: Config,
    store: Store | None,
    *,
    search_fn=None,
    message_fetcher=None,
    deadline_seconds: float | None = LINKS_BACKFILL_DEADLINE_SECONDS,
    sleep_fn=time.sleep,
) -> LinksBackfillResult | None:
    """Backfill ``pr_link_posters`` from ``review_threads`` and Discord guild message search."""
    if store is None or not config.discord_guild_id:
        return None
    if not _links_backfill_lock.acquire(blocking=False):
        logging.info("PR links backfill skipped: already running")
        return None

    t_start = time.monotonic()
    deadline = (t_start + deadline_seconds) if deadline_seconds is not None else None
    repo = config.github_repository
    search_query = f"github.com/{repo}/pull"
    do_search = search_fn or search_guild_messages
    try:
        rt_scanned, rt_recorded, rt_rate_limited = backfill_pr_link_posters_from_review_threads(
            config,
            store,
            message_fetcher=message_fetcher,
            deadline=(deadline - 15.0) if deadline is not None else None,
        )

        search_pages = 0
        messages_scanned = 0
        links_recorded = rt_recorded
        total_results = 0
        rate_limited = rt_rate_limited
        timed_out = False

        offset = 0
        while offset <= GUILD_SEARCH_MAX_OFFSET and not rate_limited:
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break
            try:
                payload = do_search(
                    config.discord_guild_id,
                    search_query,
                    config.discord_token,
                    offset=offset,
                    sleep_fn=sleep_fn,
                )
            except TransientAPIError as error:
                retry_after = getattr(error, "retry_after", None)
                if isinstance(retry_after, (int, float)) and 0 < retry_after <= 10.0 and (
                    deadline is None or (deadline - time.monotonic()) > retry_after + 2.0
                ):
                    sleep_fn(float(retry_after))
                    try:
                        payload = do_search(
                            config.discord_guild_id,
                            search_query,
                            config.discord_token,
                            offset=offset,
                            sleep_fn=sleep_fn,
                        )
                    except TransientAPIError:
                        rate_limited = True
                        break
                else:
                    logging.warning("PR links backfill search rate-limited at offset %d: %s", offset, error)
                    rate_limited = True
                    break
            except Exception as error:
                logging.warning("PR links backfill search failed at offset %d: %s", offset, error)
                break

            search_pages += 1
            reported_total = payload.get("total_results")
            if isinstance(reported_total, int) and reported_total >= 0:
                total_results = max(total_results, reported_total)
            groups = payload.get("messages")
            if not isinstance(groups, list) or not groups:
                break

            for group in groups:
                items = group if isinstance(group, list) else ([group] if isinstance(group, dict) else [])
                # Prefer explicit hit=True messages inside each context window; fallback to all dicts.
                hits = [m for m in items if isinstance(m, dict) and m.get("hit")]
                target_messages = hits if hits else [m for m in items if isinstance(m, dict)]
                for msg in target_messages:
                    messages_scanned += 1
                    author = msg.get("author") if isinstance(msg.get("author"), dict) else {}
                    if author.get("bot") or msg.get("webhook_id"):
                        continue
                    links_recorded += record_pr_links_from_message(
                        store,
                        repo,
                        discord_id=str(author.get("id") or ""),
                        message_id=str(msg.get("id") or ""),
                        content=str(msg.get("content") or ""),
                        ts=msg.get("timestamp"),
                        embeds=msg.get("embeds") if isinstance(msg.get("embeds"), list) else None,
                    )

            offset += GUILD_SEARCH_PAGE_SIZE
            if total_results and offset >= total_results:
                break
            if len(groups) < GUILD_SEARCH_PAGE_SIZE:
                break

        logging.info(
            "PR links backfill finished for %s: review_threads=%d (recorded=%d), "
            "search_pages=%d, messages=%d, new_links=%d, total_results=%d, "
            "rate_limited=%s, timed_out=%s in %.2fs",
            repo,
            rt_scanned,
            rt_recorded,
            search_pages,
            messages_scanned,
            links_recorded,
            total_results,
            rate_limited,
            timed_out,
            time.monotonic() - t_start,
        )
        return LinksBackfillResult(
            review_threads_scanned=rt_scanned,
            review_threads_recorded=rt_recorded,
            search_pages=search_pages,
            messages_scanned=messages_scanned,
            links_recorded=links_recorded,
            total_results=total_results,
            rate_limited=rate_limited,
            timed_out=timed_out,
        )
    finally:
        _links_backfill_lock.release()


def format_links_backfill_result(result: LinksBackfillResult | None) -> str:
    """Ephemeral moderator summary for /links-backfill."""
    if result is None:
        return "pr links backfill is already running or DISCORD_GUILD_ID is not configured."
    flags = []
    if result.rate_limited:
        flags.append("rate-limited")
    if result.timed_out:
        flags.append("time-bounded")
    suffix = f" ({', '.join(flags)})" if flags else ""
    return "\n".join([
        f"**PR links backfill finished{suffix}**",
        f"- Review threads checked: **{result.review_threads_scanned}** (**{result.review_threads_recorded}** new links)",
        f"- Guild search pages scanned: **{result.search_pages}** (**{result.messages_scanned}** messages, total_results={result.total_results})",
        f"- Total new PR-link poster rows recorded: **{result.links_recorded}**",
    ])
