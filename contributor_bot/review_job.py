"""Discord thread lifecycle for a LiteLLM pull-request pre-review.

Posts only in Discord. GitHub is read through review_github; the model call
goes through review_llm. A per-PR cooldown and an in-process lock limit
repeat work.
"""

import logging
import threading
from datetime import datetime

from .api import (
    TransientAPIError,
    edit_original_interaction,
    fetch_latest_message_id,
    fetch_channel_messages,
    post_message,
    start_public_thread,
)
from .config import Config, is_discord_id
from .review_github import Snapshot, gather_snapshot
from .review_llm import run_llm_review
from .reviews import (
    assess_gate,
    cooldown_remaining,
    render_status,
    render_written,
    review_chunks,
    recheck_number,
    should_review_message,
    tag_line,
)
from .schedule import UTC, iso
from .store import Store

_inflight: set[tuple[str, int]] = set()
_inflight_guard = threading.Lock()


class ReviewError(RuntimeError):
    """A failure safe to show in Discord. The message is written by this bot."""


def _spawn(target) -> None:
    threading.Thread(target=target, name="pr-review", daemon=True).start()


def _try_lock(repository: str, number: int) -> bool:
    key = (repository.lower(), number)
    with _inflight_guard:
        if key in _inflight:
            return False
        _inflight.add(key)
        return True


def _unlock(repository: str, number: int) -> None:
    with _inflight_guard:
        _inflight.discard((repository.lower(), number))


def _pull_url(repository: str, number: int) -> str:
    return f"https://github.com/{repository}/pull/{number}"


def _post(channel_id: str, token: str, content: str, *, user_id: str | None = None, role_id: str | None = None) -> None:
    users = [user_id] if user_id else []
    post_message(
        channel_id,
        token,
        {"content": content, "allowed_mentions": {"parse": [], "users": users}},
        mention_roles=[role_id] if role_id else None,
    )


def _ensure_thread(config: Config, store: Store, snapshot: Snapshot, channel_id: str, message_id: str) -> str:
    repository = config.github_repository
    record = store.review_thread(repository, snapshot.number)
    if record:
        return record["thread_id"]
    if not is_discord_id(channel_id) or not is_discord_id(message_id):
        raise ReviewError("couldn't tell which message to thread.")
    try:
        thread = start_public_thread(channel_id, message_id, config.discord_token, f"pr {snapshot.number} review")
    except RuntimeError as error:
        if "HTTP 403" in str(error):
            logging.error(
                "Could not open a review thread in channel %s: %s. "
                "The bot needs Create Public Threads and Send Messages in Threads.",
                channel_id, error,
            )
            raise ReviewError("couldn't open a thread. the bot needs create public threads and send messages in threads.") from error
        logging.error("Could not open a review thread for #%s: %s", snapshot.number, error)
        raise ReviewError("couldn't open a thread on that message.") from error
    now = iso(datetime.now(UTC))
    store.save_review_thread(
        repository=repository,
        number=snapshot.number,
        channel_id=channel_id,
        message_id=message_id,
        thread_id=thread["id"],
        created_at=now,
    )
    return thread["id"]


def _publish(config: Config, thread_id: str, chunks: list[str], user_id: str | None) -> None:
    role_id = config.review_maintainer_role_id
    ready = bool(chunks) and chunks[0].startswith("ready for maintainer")
    for chunk in chunks:
        is_tag = chunk.startswith("<@")
        _post(
            thread_id,
            config.discord_token,
            chunk,
            user_id=user_id if is_tag else None,
            role_id=role_id if is_tag and ready else None,
        )


def perform_review(
    config: Config,
    store: Store,
    number: int,
    *,
    channel_id: str | None = None,
    message_id: str | None = None,
    source: str = "message",
) -> dict:
    """Run one pre-review. Returns a small status dict for the caller."""
    repository = config.github_repository
    if not _try_lock(repository, number):
        return {"status": "busy", "thread_id": None, "detail": "already checking that pr."}
    try:
        return _perform_review(
            config, store, number, channel_id=channel_id, message_id=message_id, source=source,
        )
    finally:
        _unlock(repository, number)


def _perform_review(config, store, number, *, channel_id, message_id, source) -> dict:
    repository = config.github_repository
    record = store.review_thread(repository, number)
    remaining = cooldown_remaining(record, config.review_cooldown_seconds, datetime.now(UTC))
    if remaining:
        minutes = max(1, (remaining + 59) // 60)
        detail = f"checked recently. try again in {minutes}m."
        if record:
            detail = f"checked recently. see <#{record['thread_id']}>. try again in {minutes}m."
        logging.info("PR review for #%s skipped (%s); cooldown %ss", number, source, remaining)
        return {"status": "cooldown", "thread_id": record["thread_id"] if record else None, "detail": detail}
    logging.info("Reviewing #%s (%s)", number, source)
    try:
        snapshot = gather_snapshot(repository, number, config.github_token)
    except TransientAPIError as error:
        logging.warning("PR review for #%s deferred: %s", number, error)
        return {"status": "error", "thread_id": record["thread_id"] if record else None, "detail": "couldn't reach github. try again in a minute."}
    except RuntimeError as error:
        logging.warning("PR review for #%s failed: %s", number, error)
        if "HTTP 404" in str(error):
            detail = "that pr isn't on github, or it's private."
        else:
            detail = "couldn't read that pr from github."
        return {"status": "error", "thread_id": record["thread_id"] if record else None, "detail": detail}
    thread_channel = channel_id or (record["channel_id"] if record else None)
    thread_message = message_id or (record["message_id"] if record else None)
    try:
        thread_id = _ensure_thread(config, store, snapshot, thread_channel or "", thread_message or "")
    except ReviewError as error:
        return {"status": "error", "thread_id": None, "detail": str(error)}
    written = ""
    if snapshot.state == "open" and not snapshot.merged:
        try:
            written = render_written(run_llm_review(snapshot.diff, config.review_model))
        except RuntimeError as error:
            logging.warning("Written review for #%s failed: %s", number, error)
            if "not installed" in str(error):
                written = render_written(None, "litellm isn't installed on this bot, so this is the gate only")
            else:
                written = render_written(None, "the model call failed. the gate above is the part to trust")
        except Exception:
            logging.exception("Written review for #%s failed", number)
            written = render_written(None, "the model call failed. the gate above is the part to trust")
    gate = assess_gate(
        bots=config.review_bot_logins,
        head_sha=snapshot.head_sha,
        draft=snapshot.draft,
        merged=snapshot.merged,
        state=snapshot.state,
        mergeable=snapshot.mergeable,
        mergeable_state=snapshot.mergeable_state,
        reviews=snapshot.reviews,
        comments=snapshot.comments,
        threads=snapshot.threads,
        check_runs=snapshot.check_runs,
        combined_state=snapshot.combined_state,
    )
    status = render_status(
        gate,
        number=number,
        title=snapshot.title,
        url=_pull_url(repository, number),
        head_sha=snapshot.head_sha,
    )
    chunks = review_chunks(status, written)
    link = store.link_for_github_user(snapshot.author_id) if snapshot.author_id else None
    user_id = link["discord_id"] if link else None
    tag = tag_line(user_id, config.review_maintainer_role_id, ready=gate.ready)
    if tag:
        chunks.append(tag)
    try:
        _publish(config, thread_id, chunks, user_id)
    except RuntimeError as error:
        logging.error(
            "Could not post the review for #%s in thread %s: %s. "
            "The bot needs Send Messages in Threads.",
            number, thread_id, error,
        )
        return {"status": "error", "thread_id": thread_id, "detail": "couldn't post in the review thread."}
    store.mark_review_done(repository, number, iso(datetime.now(UTC)), snapshot.head_sha)
    logging.info("Posted a %s review for #%s in thread %s", "ready" if gate.ready else "needs-work", number, thread_id)
    return {"status": "posted", "thread_id": thread_id, "detail": ""}


def _ephemeral(content: str) -> dict:
    return {"type": 4, "data": {"content": content, "flags": 64, "allowed_mentions": {"parse": []}}}


def _defer(*, ephemeral: bool = False) -> dict:
    data: dict = {}
    if ephemeral:
        data["flags"] = 64
    return {"type": 5, "data": data}


def _option(payload: dict, name: str) -> str:
    for option in (payload.get("data") or {}).get("options") or []:
        if isinstance(option, dict) and option.get("name") == name and isinstance(option.get("value"), str):
            return option["value"]
    return ""


def handle_review_command(config: Config, store: Store, payload: dict) -> dict:
    name = (payload.get("data") or {}).get("name")
    if name == "recheck":
        return _recheck_command(config, store, payload)
    if name == "review-pr":
        return _review_command(config, store, payload)
    return _ephemeral("unknown command.")


def _review_command(config: Config, store: Store, payload: dict) -> dict:
    channel = payload.get("channel") or {}
    if isinstance(channel, dict) and channel.get("type") in {10, 11, 12}:
        return _ephemeral("run /review-pr in the channel. inside a review thread, use /recheck.")
    channel_id = payload.get("channel_id")
    if channel_id not in config.review_channel_ids:
        return _ephemeral("run /review-pr in a channel this bot watches for reviews.")
    numbers = should_review_message(
        {"content": _option(payload, "url"), "author": {"bot": False}},
        config.github_repository,
        require_ask=False,
    )
    if not numbers:
        return _ephemeral("that isn't a berriai/litellm pull request url.")
    number = numbers[0]
    record = store.review_thread(config.github_repository, number)
    if record:
        remaining = cooldown_remaining(record, config.review_cooldown_seconds, datetime.now(UTC))
        if remaining:
            minutes = max(1, (remaining + 59) // 60)
            return _ephemeral(f"checked recently. see <#{record['thread_id']}>. try again in {minutes}m.")
        _spawn(lambda: perform_review(config, store, number, source="recheck"))
        extra = ""
        if len(numbers) > 1:
            extra = " i only take one url per command."
        return _ephemeral(f"updating the review in <#{record['thread_id']}>.{extra}")
    token = payload.get("token") if isinstance(payload.get("token"), str) else ""
    application_id = str(payload.get("application_id") or "")
    _spawn(lambda: _finish_new_command(config, store, number, channel_id, application_id, token))
    return _defer(ephemeral=False)


def _finish_new_command(config, store, number, channel_id, application_id, token) -> None:
    def edit(content: str) -> dict | None:
        if not token:
            return None
        try:
            return edit_original_interaction(application_id, token, {
                "content": content,
                "allowed_mentions": {"parse": []},
            })
        except Exception:
            logging.exception("Could not update the /review-pr reply for #%s", number)
            return None

    try:
        ack = edit(f"checking #{number}. i'll put the review in a thread.")
        message_id = ack.get("id") if isinstance(ack, dict) else None
        result = perform_review(
            config, store, number,
            channel_id=channel_id,
            message_id=message_id if is_discord_id(message_id) else None,
            source="new",
        )
    except Exception:
        logging.exception("PR review command failed for #%s", number)
        edit("couldn't review that pr. try again in a minute.")
        return
    thread_id = result.get("thread_id")
    if result.get("status") == "posted" and thread_id:
        edit(f"review for #{number} is in <#{thread_id}>.")
        return
    edit(result.get("detail") or "couldn't review that pr. try again in a minute.")


def _recheck_command(config: Config, store: Store, payload: dict) -> dict:
    channel_id = payload.get("channel_id")
    record = store.review_thread_by_discord(channel_id) if is_discord_id(channel_id) else None
    if not record or record["repository"].lower() != config.github_repository.lower():
        return _ephemeral("run /recheck inside a pr review thread.")
    remaining = cooldown_remaining(record, config.review_cooldown_seconds, datetime.now(UTC))
    if remaining:
        minutes = max(1, (remaining + 59) // 60)
        return _ephemeral(f"checked recently. try again in {minutes}m.")
    number = int(record["number"])
    _spawn(lambda: perform_review(config, store, number, source="recheck"))
    return _ephemeral("rechecking. i'll post an update in this thread.")


def schedule_webhook_recheck(config: Config, store: Store, event: str, payload: dict) -> None:
    """Re-check a pull request we already have a thread for. No-op unless enabled."""
    if not config.pr_review_enabled or not config.review_webhook_recheck:
        return
    number = recheck_number(event, payload)
    if number is None:
        return
    repository = payload.get("repository") if isinstance(payload.get("repository"), dict) else {}
    full_name = repository.get("full_name") if isinstance(repository, dict) else None
    if not isinstance(full_name, str) or full_name.lower() != config.github_repository.lower():
        return
    record = store.review_thread(config.github_repository, number)
    if not record:
        return
    _spawn(lambda: perform_review(config, store, number, source="webhook"))


def handle_channel_message(config: Config, store: Store, channel_id: str, message: dict) -> None:
    numbers = should_review_message(message, config.github_repository, require_ask=config.review_require_ask)
    message_id = str(message.get("id") or "")
    for number in numbers:
        record = store.review_thread(config.github_repository, number)
        if record:
            remaining = cooldown_remaining(record, config.review_cooldown_seconds, datetime.now(UTC))
            if remaining:
                logging.info("Skipping #%s in channel %s; cooldown has %ss left", number, channel_id, remaining)
                continue
            _spawn(lambda number=number: perform_review(config, store, number, source="recheck"))
            continue
        _spawn(lambda number=number, message_id=message_id: perform_review(
            config, store, number, channel_id=channel_id, message_id=message_id, source="new",
        ))


def poll_review_channels(config: Config, store: Store) -> None:
    """Read new messages in the review channels. The first poll does not backfill."""
    if not config.pr_review_enabled:
        return
    for channel_id in config.review_channel_ids:
        cursor = store.review_cursor(channel_id)
        if cursor is None:
            latest = fetch_latest_message_id(channel_id, config.discord_token)
            if latest:
                store.set_review_cursor(channel_id, latest)
            logging.info(
                "PR review is watching channel %s from message %s and will not backfill older links",
                channel_id, latest or "none",
            )
            continue
        messages = list(fetch_channel_messages(channel_id, config.discord_token, cursor))
        messages.sort(key=lambda item: int(item["id"]))
        for message in messages:
            try:
                handle_channel_message(config, store, channel_id, message)
            except Exception:
                logging.exception("PR review failed for message %s", message.get("id"))
            store.set_review_cursor(channel_id, str(message["id"]))
