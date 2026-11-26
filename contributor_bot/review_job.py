"""Discord thread lifecycle for a LiteLLM pull-request pre-review.

Posts only in Discord. GitHub is read through review_github; the model call
goes through review_llm. A per-PR cooldown and an in-process lock limit
repeat work.
"""

import logging
import threading
import time
from datetime import datetime

from .ack import acknowledge_message
from .api import (
    edit_original_interaction,
    fetch_latest_message_id,
    fetch_channel_messages,
    fetch_recent_messages,
    post_message,
    start_public_thread,
)
from .config import Config, is_discord_id
from .review_agent import history_text, run_review_session
from .review_llm import complete_litellm
from .reviews import (
    classify_review_request,
    cooldown_remaining,
    recheck_number,
    should_review_message,
    wants_recheck,
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


def _post(channel_id: str, token: str, content: str, *, user_id: str | None = None, role_id: str | None = None) -> None:
    users = [user_id] if user_id else []
    post_message(
        channel_id,
        token,
        {"content": content, "allowed_mentions": {"parse": [], "users": users}},
        mention_roles=[role_id] if role_id else None,
    )


def _ensure_thread(config: Config, store: Store, number: int, channel_id: str, message_id: str) -> str:
    repository = config.github_repository
    record = store.review_thread(repository, number)
    if record:
        return record["thread_id"]
    if not is_discord_id(channel_id) or not is_discord_id(message_id):
        raise ReviewError("couldn't tell which message to thread.")
    try:
        thread = start_public_thread(channel_id, message_id, config.discord_token, f"pr {number} review")
    except RuntimeError as error:
        if "HTTP 403" in str(error):
            logging.error(
                "Could not open a review thread in channel %s: %s. "
                "The bot needs Create Public Threads and Send Messages in Threads.",
                channel_id, error,
            )
            raise ReviewError("couldn't open a thread. the bot needs create public threads and send messages in threads.") from error
        logging.error("Could not open a review thread for #%s: %s", number, error)
        raise ReviewError("couldn't open a thread on that message.") from error
    now = iso(datetime.now(UTC))
    store.save_review_thread(
        repository=repository,
        number=number,
        channel_id=channel_id,
        message_id=message_id,
        thread_id=thread["id"],
        created_at=now,
    )
    return thread["id"]


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
        if source == "recheck" and record:
            try:
                _post(record["thread_id"], config.discord_token, detail)
            except Exception:
                logging.warning("Could not post the cooldown note for #%s", number)
        return {"status": "cooldown", "thread_id": record["thread_id"] if record else None, "detail": detail}
    day = datetime.now(UTC).date().isoformat()
    if not store.claim_daily_run(day, config.review_daily_run_budget):
        detail = "ran out of budget for today. try again tomorrow."
        if record:
            try:
                _post(record["thread_id"], config.discord_token, detail)
            except Exception:
                logging.exception("Could not post the daily budget note for #%s", number)
        return {"status": "budget", "thread_id": record["thread_id"] if record else None, "detail": detail}
    logging.info("Reviewing #%s (%s)", number, source)
    thread_channel = channel_id or (record["channel_id"] if record else None)
    thread_message = message_id or (record["message_id"] if record else None)
    try:
        thread_id = _ensure_thread(config, store, number, thread_channel or "", thread_message or "")
    except ReviewError as error:
        return {"status": "error", "thread_id": None, "detail": str(error)}
    try:
        history = ""
        try:
            history = history_text(fetch_recent_messages(thread_id, config.discord_token, limit=15))
        except Exception:
            logging.warning("Could not read thread history for #%s", number)
        result = run_review_session(config, store, number, thread_id, mode="review", history=history)
    except Exception:
        logging.exception("Review agent failed for #%s", number)
        return {"status": "error", "thread_id": thread_id, "detail": "the reviewer hit an error. try /recheck in a bit."}
    store.mark_review_done(repository, number, iso(datetime.now(UTC)), result.head_sha or "")
    if result.verdict_posted:
        logging.info(
            "Posted a %s review for #%s in thread %s",
            "ready" if result.ready else "needs-work", number, thread_id,
        )
    elif result.exhausted:
        logging.info("Review for #%s ran out of budget (%s)", number, result.exhausted)
    return {"status": "posted", "thread_id": thread_id, "detail": result.text}


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
    acknowledge_message(config, channel_id, _source_message_id(payload))
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


def review_record_for_command(config: Config, store: Store, payload: dict) -> dict | None:
    """The review thread a slash command was used in.

    The channel id is the thread. A parent channel id is not a thread id, and
    the parent does not have to be listed in REVIEW_CHANNEL_IDS.
    """
    channel = payload.get("channel") if isinstance(payload.get("channel"), dict) else {}
    channel_id = payload.get("channel_id")
    nested_id = channel.get("id")
    if channel.get("type") in {10, 11, 12} and is_discord_id(nested_id):
        channel_id = nested_id
    if not is_discord_id(channel_id):
        logging.warning(
            "/recheck did not include a thread id (channel_id=%s parent_id=%s type=%s)",
            channel_id, channel.get("parent_id"), channel.get("type"),
        )
        return None
    record = store.review_thread_by_discord(str(channel_id))
    if not record or record["repository"].lower() != config.github_repository.lower():
        logging.warning(
            "/recheck channel %s is not a review thread (parent_id=%s type=%s)",
            channel_id, channel.get("parent_id"), channel.get("type"),
        )
        return None
    return record


def recheck_followup_text(result: dict) -> str:
    """Text for the slash-command follow-up. Cooldown and budget are shown, not only logged."""
    status = result.get("status")
    detail = result.get("detail") if isinstance(result.get("detail"), str) else ""
    thread_id = result.get("thread_id")
    if status == "posted" and thread_id:
        return f"updated the review in <#{thread_id}>."
    if status == "budget":
        return detail or "ran out of budget for today. try again tomorrow."
    if status == "cooldown":
        return detail or "checked recently. try again in a bit."
    if status == "busy":
        return detail or "already checking that pr."
    return detail or "couldn't recheck. try again in a bit."


def _source_message_id(payload: dict) -> str | None:
    """A slash command can name a message. Most do not, and then there is nothing to react to."""
    message = payload.get("message")
    if isinstance(message, dict) and is_discord_id(message.get("id")):
        return str(message["id"])
    return None


def _recheck_command(config: Config, store: Store, payload: dict) -> dict:
    record = review_record_for_command(config, store, payload)
    if not record:
        return _ephemeral("run /recheck inside a pr review thread.")
    acknowledge_message(config, payload.get("channel_id"), _source_message_id(payload))
    remaining = cooldown_remaining(record, config.review_cooldown_seconds, datetime.now(UTC))
    if remaining:
        minutes = max(1, (remaining + 59) // 60)
        detail = f"checked recently. try again in {minutes}m."
        _post_note(record["thread_id"], config.discord_token, detail, record.get("number"), "cooldown")
        return _ephemeral(detail)
    number = int(record["number"])
    token = payload.get("token") if isinstance(payload.get("token"), str) else ""
    application_id = str(payload.get("application_id") or "")
    # Defer first (type 5). The agent run follows up by editing this response,
    # so Discord gets an ack inside 3 seconds and the user sees the outcome.
    _spawn(lambda: _finish_recheck(config, store, number, application_id, token))
    return _defer(ephemeral=True)


def _post_note(thread_id: str, token: str, detail: str, number, kind: str) -> None:
    try:
        _post(thread_id, token, detail)
    except Exception:
        logging.exception("Could not post the /recheck %s note for #%s", kind, number)


def _finish_recheck(config, store, number: int, application_id: str, token: str) -> None:
    try:
        result = perform_review(config, store, number, source="recheck")
        detail = recheck_followup_text(result)
    except Exception:
        logging.exception("Recheck failed for #%s", number)
        detail = "couldn't recheck. try again in a bit."
    _edit_recheck_followup(application_id, token, detail, number)


def _edit_recheck_followup(application_id: str, token: str, detail: str, number: int) -> None:
    if not token:
        logging.error("No interaction token for the /recheck follow-up on #%s", number)
        return
    delay = 0.3
    for attempt in range(4):
        try:
            edit_original_interaction(application_id, token, {
                "content": detail,
                "allowed_mentions": {"parse": []},
            })
            return
        except RuntimeError as error:
            if "HTTP 404" not in str(error) or attempt == 3:
                logging.exception("Could not send the /recheck follow-up for #%s", number)
                return
            # The deferred ack may not be stored yet. Retry so the note is shown.
            time.sleep(delay)
            delay *= 2
        except Exception:
            logging.exception("Could not send the /recheck follow-up for #%s", number)
            return


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


def _classify_complete(config: Config, system: str, text: str) -> str:
    model = config.review_classifier_model or config.review_model
    return complete_litellm(model, [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ])


def message_decision(config: Config, message: dict) -> tuple[bool, list[int]]:
    """Whether this message wants a review, and which LiteLLM PR numbers it names."""
    if not config.review_classify:
        numbers = should_review_message(message, config.github_repository, require_ask=config.review_require_ask)
        return bool(numbers), numbers
    text = message.get("content") or ""
    asked = classify_review_request(
        text,
        config.github_repository,
        complete=lambda system, body: _classify_complete(config, system, body),
    )
    if not asked:
        return False, []
    return True, should_review_message(message, config.github_repository, require_ask=False)


def handle_thread_message(config: Config, store: Store, message: dict) -> None:
    """Answer a follow-up inside a review thread, or re-review when they say they fixed it."""
    if not config.pr_review_enabled:
        return
    author = message.get("author") or {}
    if author.get("bot") or message.get("webhook_id"):
        return
    channel_id = str(message.get("channel_id") or "")
    record = store.review_thread_by_discord(channel_id) if is_discord_id(channel_id) else None
    if not record or record["repository"].lower() != config.github_repository.lower():
        return
    number = int(record["number"])
    text = message.get("content") or ""
    message_id = message.get("id")
    if wants_recheck(text):
        acknowledge_message(config, channel_id, message_id)
        perform_review(config, store, number, source="recheck")
        return
    remaining = cooldown_remaining(
        {"last_reviewed_at": record.get("last_followup_at")},
        config.review_followup_cooldown_seconds,
        datetime.now(UTC),
    )
    if remaining:
        logging.info("Skipping follow-up on #%s; cooldown has %ss left", number, remaining)
        return
    day = datetime.now(UTC).date().isoformat()
    if not store.claim_daily_run(day, config.review_daily_run_budget):
        acknowledge_message(config, channel_id, message_id)
        _post(channel_id, config.discord_token, "ran out of budget for today. try again tomorrow.")
        return
    acknowledge_message(config, channel_id, message_id)
    try:
        history = history_text(fetch_recent_messages(channel_id, config.discord_token, limit=15))
    except Exception:
        logging.warning("Could not read follow-up history for #%s", number)
        history = text
    try:
        run_review_session(config, store, number, channel_id, mode="followup", history=history)
    except Exception:
        logging.exception("Follow-up agent failed for #%s", number)
    store.mark_followup(config.github_repository, number, iso(datetime.now(UTC)))


def handle_channel_message(config: Config, store: Store, channel_id: str, message: dict) -> None:
    asked, numbers = message_decision(config, message)
    message_id = str(message.get("id") or "")
    if asked and not numbers:
        acknowledge_message(config, channel_id, message_id)
        _post(channel_id, config.discord_token, "drop the berriai/litellm pull url and i'll open a review thread.")
        return
    planned = []
    for number in numbers:
        record = store.review_thread(config.github_repository, number)
        if record:
            remaining = cooldown_remaining(record, config.review_cooldown_seconds, datetime.now(UTC))
            if remaining:
                logging.info("Skipping #%s in channel %s; cooldown has %ss left", number, channel_id, remaining)
                continue
        planned.append((number, record))
    if not planned:
        return
    acknowledge_message(config, channel_id, message_id)
    for number, record in planned:
        if record:
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
