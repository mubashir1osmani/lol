"""Discord thread lifecycle for a LiteLLM pull-request pre-review.

Posts only in Discord. GitHub is read through review_github; the model call
goes through review_llm. A per-PR cooldown and an in-process lock limit
repeat work.
"""

import logging
import re
import threading
import time
from datetime import datetime, timedelta

from .ack import acknowledge_message
from .routing import explicit_review
from .triggers import apply_triggers
from .api import (
    TransientAPIError,
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
from .schedule import UTC, iso, parse_time
from .store import Store

_DISCORD_EPOCH_MS = 1420070400000
_RECOVERY_LOOKBACK_HOURS = 24
_RECOVERY_MAX_ATTEMPTS = 3
_HASH_PR_RE = re.compile(r"(?<![A-Za-z0-9])#([1-9][0-9]{0,7})\b")

_inflight: set[tuple[str, int]] = set()
_inflight_guard = threading.Lock()
_message_locks: dict[tuple[str, str], threading.Lock] = {}
_message_locks_guard = threading.Lock()


class ReviewError(RuntimeError):
    """A failure safe to show in Discord. The message is written by this bot."""


def _message_lock(channel_id: str, message_id: str) -> threading.Lock:
    key = (str(channel_id), str(message_id))
    with _message_locks_guard:
        if len(_message_locks) > 1000:
            _message_locks.clear()
        lock = _message_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _message_locks[key] = lock
        return lock


def _spawn(target) -> None:
    import contextvars
    from .tracing import flush_tracing

    ctx = contextvars.copy_context()

    def _runner() -> None:
        try:
            ctx.run(target)
        except Exception:
            logging.exception("Review worker failed")
        finally:
            flush_tracing()

    threading.Thread(target=_runner, name="pr-review", daemon=True).start()


def _spawn_review(number: int, fn) -> None:
    def _task() -> None:
        try:
            fn()
        except Exception:
            logging.exception("PR review worker crashed for #%s", number)

    _spawn(_task)


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
    with _message_lock(channel_id, message_id):
        record = store.review_thread(repository, number)
        if record:
            return record["thread_id"]
        existing_on_message = store.review_thread_by_message(channel_id, message_id)
        if existing_on_message:
            thread_id = existing_on_message["thread_id"]
        else:
            try:
                thread = start_public_thread(channel_id, message_id, config.discord_token, f"pr {number} review")
            except RuntimeError as error:
                if "HTTP 403" in str(error):
                    logging.error(
                        "Could not open a review thread in channel %s: %s. "
                        "The bot needs Create Public Threads and Send Messages in Threads.",
                        channel_id, error,
                    )
                    raise ReviewError(
                        "couldn't open a thread. the bot needs create public threads and send messages in threads."
                    ) from error
                logging.error("Could not open a review thread for #%s: %s", number, error)
                raise ReviewError("couldn't open a thread on that message.") from error
            thread_id = thread["id"]
        now = iso(datetime.now(UTC))
        store.save_review_thread(
            repository=repository,
            number=number,
            channel_id=channel_id,
            message_id=message_id,
            thread_id=thread_id,
            created_at=now,
        )
        return thread_id


def _record_request_outcome(
    store: Store,
    repository: str,
    number: int,
    message_id: str | None,
    *,
    status: str,
    error: str = "",
    increment_attempts: bool = False,
) -> None:
    if not message_id or not is_discord_id(message_id):
        return
    try:
        store.mark_review_request(
            repository,
            number,
            message_id,
            status,
            error=error,
            updated_at=iso(datetime.now(UTC)),
            increment_attempts=increment_attempts,
        )
    except Exception:
        logging.warning("Could not update review_requests status for #%s (msg %s)", number, message_id)


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
    except Exception as error:
        logging.exception("Unexpected error during PR review for #%s (%s)", number, source)
        _record_request_outcome(
            store, repository, number, message_id, status="failed", error=str(error), increment_attempts=True,
        )
        return {"status": "error", "thread_id": None, "detail": "the reviewer hit an error. try /recheck in a bit."}
    finally:
        _unlock(repository, number)


def _perform_review(config, store, number, *, channel_id, message_id, source) -> dict:
    repository = config.github_repository
    record = store.review_thread(repository, number)
    request_message_id = message_id if is_discord_id(message_id) else (record["message_id"] if record else None)
    remaining = cooldown_remaining(record, config.review_cooldown_seconds, datetime.now(UTC))
    if remaining:
        minutes = max(1, (remaining + 59) // 60)
        detail = f"checked recently. try again in {minutes}m."
        if record:
            detail = f"checked recently. see <#{record['thread_id']}>. try again in {minutes}m."
        logging.info("PR review for #%s skipped (%s); cooldown %ss", number, source, remaining)
        if record and record.get("last_reviewed_at"):
            _record_request_outcome(store, repository, number, request_message_id, status="done")
        if source == "recheck" and record:
            try:
                _post(record["thread_id"], config.discord_token, detail)
            except Exception:
                logging.warning("Could not post the cooldown note for #%s", number)
        return {"status": "cooldown", "thread_id": record["thread_id"] if record else None, "detail": detail}
    day = datetime.now(UTC).date().isoformat()
    if not store.claim_daily_run(day, config.review_daily_run_budget):
        detail = "ran out of budget for today. try again tomorrow."
        _record_request_outcome(
            store, repository, number, request_message_id, status="failed", error=detail, increment_attempts=True,
        )
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
        _record_request_outcome(
            store, repository, number, request_message_id, status="failed", error=str(error), increment_attempts=True,
        )
        return {"status": "error", "thread_id": None, "detail": str(error)}
    try:
        history = ""
        try:
            history = history_text(fetch_recent_messages(thread_id, config.discord_token, limit=15))
        except Exception:
            logging.warning("Could not read thread history for #%s", number)
        if source != "watch" and (config.github_actor_token or config.review_dry_run):
            apply_triggers(config, store, number, lambda text: _post(thread_id, config.discord_token, text))
        result = run_review_session(config, store, number, thread_id, mode="review", history=history)
    except Exception as error:
        logging.exception("Review agent failed for #%s", number)
        _record_request_outcome(
            store, repository, number, request_message_id, status="failed", error=str(error), increment_attempts=True,
        )
        return {"status": "error", "thread_id": thread_id, "detail": "the reviewer hit an error. try /recheck in a bit."}
    store.mark_review_done(repository, number, iso(datetime.now(UTC)), result.head_sha or "")
    _record_request_outcome(store, repository, number, request_message_id, status="done", increment_attempts=True)
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
    _ack(config, channel_id, _source_message_id(payload), payload)
    number = numbers[0]
    record = store.review_thread(config.github_repository, number)
    if record:
        remaining = cooldown_remaining(record, config.review_cooldown_seconds, datetime.now(UTC))
        if remaining:
            minutes = max(1, (remaining + 59) // 60)
            return _ephemeral(f"checked recently. see <#{record['thread_id']}>. try again in {minutes}m.")
        _spawn_review(number, lambda: perform_review(config, store, number, source="recheck"))
        extra = ""
        if len(numbers) > 1:
            extra = " i only take one url per command."
        return _ephemeral(f"updating the review in <#{record['thread_id']}>.{extra}")
    token = payload.get("token") if isinstance(payload.get("token"), str) else ""
    application_id = str(payload.get("application_id") or "")
    _spawn_review(number, lambda: _finish_new_command(config, store, number, channel_id, application_id, token))
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


def _ack(config: Config, channel_id: object, message_id: object, source: dict | None = None) -> None:
    if not isinstance(source, dict):
        acknowledge_message(config, channel_id, message_id)
        return
    guild = source.get("_guild")
    guild_id = source.get("guild_id")
    loop = source.get("_loop")
    if guild is None and not guild_id and loop is None:
        acknowledge_message(config, channel_id, message_id)
        return
    acknowledge_message(config, channel_id, message_id, guild=guild, guild_id=guild_id, loop=loop)


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
    _ack(config, payload.get("channel_id"), _source_message_id(payload), payload)
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
    _spawn_review(number, lambda: _finish_recheck(config, store, number, application_id, token))
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
    _spawn_review(number, lambda: perform_review(config, store, number, source="webhook"))


def _classify_complete(config: Config, system: str, text: str, *, session_id: str | None = None) -> str:
    from .tracing import AGENT_PR_REVIEW, build_trace_metadata
    model = config.review_classifier_model or config.review_model
    return complete_litellm(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        metadata=build_trace_metadata(AGENT_PR_REVIEW, session_id=session_id),
    )


def message_decision(config: Config, message: dict) -> tuple[bool, list[int]]:
    """Whether this message wants a review, and which LiteLLM PR numbers it names."""
    if not config.review_classify:
        numbers = should_review_message(message, config.github_repository, require_ask=config.review_require_ask)
        return bool(numbers), numbers
    from contextlib import nullcontext
    from .tracing import (
        AGENT_PR_REVIEW,
        OPENINFERENCE_SPAN_KIND_CHAIN,
        agent_span,
        child_span,
        current_agent_name,
    )

    text = message.get("content") or ""
    channel_id = message.get("channel_id") or message.get("id") or ""
    session_id = f"{AGENT_PR_REVIEW}:classifier:{channel_id}" if channel_id else None
    outer_cm = (
        agent_span(
            AGENT_PR_REVIEW,
            session_id=session_id,
            input_value=text[:800],
            attributes={"discord.channel_id": str(channel_id)},
        )
        if current_agent_name() is None
        else nullcontext()
    )
    with outer_cm as outer_span:
        with child_span(
            "pr_review_classifier",
            kind=OPENINFERENCE_SPAN_KIND_CHAIN,
            input_value=text[:800],
            attributes={"discord.channel_id": str(channel_id)},
        ) as span:
            asked = classify_review_request(
                text,
                config.github_repository,
                complete=lambda system, body: _classify_complete(config, system, body, session_id=session_id),
            )
            numbers = should_review_message(message, config.github_repository, require_ask=False) if asked else []
            out = {"asked": asked, "numbers": numbers}
            span.set_output(out)
            if outer_span is not None:
                outer_span.set_output(out)
            return asked, numbers


def _mentioned_numbers_in_thread(text: str, repository: str, records: list[dict]) -> list[dict]:
    known = {int(rec["number"]): rec for rec in records}
    mentioned: list[int] = []
    for num in should_review_message({"content": text, "author": {"bot": False}}, repository, require_ask=False):
        if num in known and num not in mentioned:
            mentioned.append(num)
    for match in _HASH_PR_RE.finditer(text or ""):
        num = int(match.group(1))
        if num in known and num not in mentioned:
            mentioned.append(num)
    return [known[num] for num in mentioned]


def handle_thread_message(config: Config, store: Store, message: dict) -> None:
    """Answer a follow-up inside a review thread, or re-review when they say they fixed it."""
    if not config.pr_review_enabled:
        return
    author = message.get("author") or {}
    if author.get("bot") or message.get("webhook_id"):
        return
    channel_id = str(message.get("channel_id") or "")
    records = [
        rec for rec in (store.review_threads_by_discord(channel_id) if is_discord_id(channel_id) else [])
        if rec["repository"].lower() == config.github_repository.lower()
    ]
    if not records:
        return
    text = message.get("content") or ""
    message_id = message.get("id")
    matched = _mentioned_numbers_in_thread(text, config.github_repository, records) if len(records) > 1 else []
    if wants_recheck(text):
        _ack(config, channel_id, message_id, message)
        targets = matched or records
        if len(targets) == 1:
            perform_review(config, store, int(targets[0]["number"]), source="recheck")
        else:
            for rec in targets:
                num = int(rec["number"])
                _spawn_review(num, lambda num=num: perform_review(config, store, num, source="recheck"))
        return
    record = matched[0] if matched else records[0]
    number = int(record["number"])
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
        _ack(config, channel_id, message_id, message)
        _post(channel_id, config.discord_token, "ran out of budget for today. try again tomorrow.")
        return
    _ack(config, channel_id, message_id, message)
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


def handle_review_channel_message(config: Config, store: Store, channel_id: str, message: dict) -> dict:
    """Process a Discord channel message already routed to ``pr_review_agent``."""
    message_id = str(message.get("id") or "")
    asked, numbers = message_decision(config, message)
    if asked and not numbers:
        _ack(config, channel_id, message_id, message)
        _post(channel_id, config.discord_token, "drop the berriai/litellm pull url and i'll open a review thread.")
        return {"asked": True, "numbers": [], "prompted_for_url": True}
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
        return {"asked": asked, "numbers": numbers, "planned": []}
    now_iso = iso(datetime.now(UTC))
    if is_discord_id(channel_id) and is_discord_id(message_id):
        for number, record in planned:
            try:
                store.upsert_review_request(
                    repository=config.github_repository,
                    number=number,
                    channel_id=channel_id,
                    message_id=message_id,
                    status="pending",
                    source="recheck" if record else "new",
                    now_iso=now_iso,
                )
            except Exception:
                logging.warning("Could not persist review request for #%s (msg %s)", number, message_id)
    _ack(config, channel_id, message_id, message)
    for number, record in planned:
        if record:
            _spawn_review(
                number,
                lambda number=number, message_id=message_id: perform_review(
                    config, store, number, message_id=message_id or None, source="recheck",
                ),
            )
            continue
        _spawn_review(
            number,
            lambda number=number, message_id=message_id: perform_review(
                config, store, number, channel_id=channel_id, message_id=message_id, source="new",
            ),
        )
    return {"asked": asked, "numbers": numbers, "planned": [n for n, _ in planned]}


def handle_channel_message(config: Config, store: Store, channel_id: str, message: dict) -> None:
    """Route a channel message through the single message router (PR review or support)."""
    from .message_router import route_and_dispatch_message

    route_and_dispatch_message(config, store, channel_id, message)


def _answer_if_decided(config: Config, channel_id: str, message: dict) -> None:
    """Delegate support Q&A to ``support_agent`` when enabled and allowlisted."""
    from .support_agent import handle_routed_message as handle_support

    handle_support(config, None, channel_id, message)


def _routed_complete(config, tier, *, session_id):
    def complete(system, body):
        from .routing import complete_auto
        from .tracing import AGENT_SUPPORT

        return complete_auto(config, [
            {"role": "system", "content": system},
            {"role": "user", "content": body},
        ], session_id=session_id, tier=tier, agent=AGENT_SUPPORT)
    return complete


def _message_created_at(message: dict) -> datetime | None:
    raw_ts = message.get("timestamp")
    if isinstance(raw_ts, str) and raw_ts.strip():
        try:
            return parse_time(raw_ts)
        except ValueError:
            pass
    mid = str(message.get("id") or "")
    if is_discord_id(mid):
        ts_ms = (int(mid) >> 22) + _DISCORD_EPOCH_MS
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
    return None


_NON_REVIEW_BOT_PREFIXES = (
    "checked recently.",
    "ran out of budget",
    "already checking",
    "couldn't ",
    "the reviewer hit an error",
)


def _thread_already_has_bot_review(
    config: Config,
    thread_id: str,
    number: int,
    *,
    multi_pr: bool,
    cache: dict[str, list[dict]],
) -> bool:
    """Check Discord thread history so boot recovery never duplicates a review after a DB reset."""
    if not is_discord_id(thread_id):
        return False
    messages = cache.get(thread_id)
    if messages is None:
        try:
            messages = fetch_recent_messages(thread_id, config.discord_token, limit=20)
        except Exception:
            messages = []
        cache[thread_id] = messages
    pr_tag = f"#{number}"
    pr_path = f"/pull/{number}"
    for msg in messages:
        author = msg.get("author") or {}
        if not author.get("bot"):
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        lowered = content.lower()
        if any(lowered.startswith(prefix) for prefix in _NON_REVIEW_BOT_PREFIXES):
            continue
        if not multi_pr:
            return True
        if pr_tag in content or pr_path in lowered:
            return True
    return False


def recover_pending_reviews(
    config: Config,
    store: Store,
    *,
    now: datetime | None = None,
    lookback_hours: int = _RECOVERY_LOOKBACK_HOURS,
    max_attempts: int = _RECOVERY_MAX_ATTEMPTS,
    scan_channels: bool = True,
) -> list[int]:
    """Retry pending/failed review requests from roughly the last 24h without duplicating completed reviews.

    Also recovers interrupted pre-migration review threads (``last_reviewed_at IS NULL``)
    and unreviewed PR review requests posted in watched channels within the lookback window.
    """
    if not config.pr_review_enabled:
        return []
    moment = now or datetime.now(UTC)
    cutoff = moment - timedelta(hours=max(1, int(lookback_hours)))
    cutoff_iso = iso(cutoff)
    now_iso = iso(moment)
    repo = config.github_repository

    # 1. Any review_threads created in the last 24h that never finished (e.g. mid-review restart).
    for row in store.incomplete_review_threads(repo, since_iso=cutoff_iso):
        if is_discord_id(row.get("channel_id")) and is_discord_id(row.get("message_id")):
            try:
                store.upsert_review_request(
                    repository=repo,
                    number=int(row["number"]),
                    channel_id=str(row["channel_id"]),
                    message_id=str(row["message_id"]),
                    status="pending",
                    source="recovery",
                    now_iso=now_iso,
                )
            except Exception:
                logging.warning("Could not enqueue incomplete review thread #%s", row.get("number"))

    # 2. Recent channel messages in the last 24h that were never reviewed (e.g. crashed before DB save).
    if scan_channels and config.discord_token:
        thread_msg_cache: dict[str, list[dict]] = {}
        for channel_id in config.review_channel_ids:
            try:
                recent = fetch_recent_messages(channel_id, config.discord_token, limit=50)
            except (TransientAPIError, RuntimeError) as error:
                logging.warning(
                    "Could not scan recent review messages in channel %s on boot: %s",
                    channel_id, error,
                )
                continue
            recent.sort(key=lambda item: int(item["id"]) if is_discord_id(item.get("id")) else 0)
            for message in recent:
                author = message.get("author") or {}
                if author.get("bot") or message.get("webhook_id"):
                    continue
                msg_id = str(message.get("id") or "")
                if not is_discord_id(msg_id):
                    continue
                created_at = _message_created_at(message)
                if created_at is not None and created_at < cutoff:
                    continue
                text = message.get("content") or ""
                if not explicit_review(text, repo):
                    continue
                candidate_numbers = should_review_message(message, repo, require_ask=False)
                if not candidate_numbers:
                    continue
                msg_thread = message.get("thread") or {}
                msg_thread_id = str(msg_thread.get("id") or "")
                multi_pr = len(candidate_numbers) > 1
                unreviewed: list[int] = []
                for number in candidate_numbers:
                    existing_thread = store.review_thread(repo, number)
                    if existing_thread and existing_thread.get("last_reviewed_at"):
                        continue
                    existing_req = store.get_review_request(repo, number, msg_id)
                    if existing_req and existing_req.get("status") == "done":
                        continue
                    if (
                        not existing_thread
                        and is_discord_id(msg_thread_id)
                        and _thread_already_has_bot_review(
                            config, msg_thread_id, number, multi_pr=multi_pr, cache=thread_msg_cache,
                        )
                    ):
                        continue
                    unreviewed.append(number)
                if not unreviewed:
                    continue
                try:
                    asked, _ = message_decision(config, message)
                except Exception:
                    logging.warning("Boot recovery classification failed for msg %s", msg_id)
                    asked = False
                if not asked:
                    continue
                for number in unreviewed:
                    try:
                        store.upsert_review_request(
                            repository=repo,
                            number=number,
                            channel_id=channel_id,
                            message_id=msg_id,
                            status="pending",
                            source="recovery",
                            now_iso=now_iso,
                        )
                    except Exception:
                        logging.warning("Could not enqueue recovered review #%s (msg %s)", number, msg_id)

    # 3. Replay all pending/failed requests from the last 24h without duplicating finished reviews.
    pending = store.pending_review_requests(repo, since_iso=cutoff_iso, max_attempts=max_attempts)
    queued: list[int] = []
    seen_numbers: set[int] = set()
    for req in pending:
        number = int(req["number"])
        msg_id = str(req["message_id"])
        ch_id = str(req["channel_id"])
        existing_thread = store.review_thread(repo, number)
        if existing_thread and existing_thread.get("last_reviewed_at"):
            if req.get("source") in {"new", "recovery"} or existing_thread["last_reviewed_at"] >= req["created_at"]:
                _record_request_outcome(store, repo, number, msg_id, status="done")
                continue
        if number in seen_numbers:
            continue
        seen_numbers.add(number)
        queued.append(number)
        _spawn_review(
            number,
            lambda number=number, ch_id=ch_id, msg_id=msg_id: perform_review(
                config, store, number, channel_id=ch_id, message_id=msg_id, source="recovery",
            ),
        )
    if queued:
        logging.info(
            "Recovered %d pending/failed PR review request(s) from the last %dh: %s",
            len(queued),
            lookback_hours,
            ", ".join(f"#{n}" for n in queued),
        )
    return queued


def poll_review_channels(config: Config, store: Store) -> None:
    """Read new messages in the review channels. The first poll does not backfill."""
    if not config.pr_review_enabled:
        return
    for channel_id in config.review_channel_ids:
        cursor = store.review_cursor(channel_id)
        if cursor is None:
            try:
                latest = fetch_latest_message_id(channel_id, config.discord_token)
            except TransientAPIError as error:
                logging.warning(
                    "Skipping initial PR review cursor check for channel %s due to transient Discord error: %s",
                    channel_id, error,
                )
                continue
            if latest:
                store.set_review_cursor(channel_id, latest)
            logging.info(
                "PR review is watching channel %s from message %s and will not backfill older links",
                channel_id, latest or "none",
            )
            continue
        try:
            messages = list(fetch_channel_messages(channel_id, config.discord_token, cursor))
        except TransientAPIError as error:
            logging.warning(
                "Skipping PR review poll for channel %s due to transient Discord error: %s",
                channel_id, error,
            )
            continue
        messages.sort(key=lambda item: int(item["id"]))
        for message in messages:
            try:
                handle_channel_message(config, store, channel_id, message)
            except Exception:
                logging.exception("PR review failed for message %s", message.get("id"))
            store.advance_review_cursor(channel_id, str(message["id"]))
