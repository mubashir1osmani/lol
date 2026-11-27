"""Record merged pull requests and grant the Discord roles they earn."""

import logging
import re
from datetime import datetime

from .api import (
    TransientAPIError,
    add_member_role,
    fetch_pull_files,
    post_issue_comment,
    post_message,
)
from .config import Config
from .present import announcement_text, invite_comment
from .roles import ROLE_KEYS, DEFAULT_ROLE_NAMES, MergedPull, earned_role_keys
from .schedule import UTC, iso, parse_time
from .store import Store

_LOGIN = re.compile(r"[A-Za-z0-9-]+(?:\[bot\])?")
_ROLE_ORDER = {key: index for index, key in enumerate(ROLE_KEYS)}


def _is_bot(user: dict) -> bool:
    login = user.get("login") if isinstance(user, dict) else ""
    return bool(user.get("type") == "Bot" or (isinstance(login, str) and login.endswith("[bot]")))


def _label_names(pull: dict) -> list[str]:
    names = []
    for label in pull.get("labels") or []:
        if isinstance(label, dict) and isinstance(label.get("name"), str):
            names.append(label["name"])
        elif isinstance(label, str):
            names.append(label)
    return names


def _models(rows: list[dict]) -> list[MergedPull]:
    models = []
    for row in rows:
        models.append(MergedPull(
            number=int(row["number"]),
            merged_at=parse_time(row["merged_at"]),
            labels=tuple(row["labels"]),
            files=tuple(row["files"]),
        ))
    return models


def assign_role(config: Config, discord_id: str, role_key: str) -> bool:
    """Give one role. Missing permissions and role hierarchy are logged, not raised."""
    role_id = config.role_ids.get(role_key, "")
    role_name = config.role_names.get(role_key) or DEFAULT_ROLE_NAMES.get(role_key, role_key)
    if not role_id:
        logging.error("No Discord role id configured for %s", role_key)
        return False
    try:
        add_member_role(config.discord_guild_id or "", discord_id, role_id, config.discord_token)
    except RuntimeError as error:
        message = str(error)
        if "HTTP 403" in message or "HTTP 404" in message:
            logging.error(
                "Could not assign Discord role %s (%s) to user %s: %s. "
                "The bot needs Manage Roles, and its role must be above %s in the server role list. "
                "A 404 means the member or the role is not in this server.",
                role_name, role_id, discord_id, message, role_name,
            )
            return False
        raise
    logging.info("Assigned Discord role %s (%s) to user %s", role_name, role_id, discord_id)
    return True


def apply_roles(
    config: Config,
    store: Store,
    discord_id: str,
    github_user_id: int,
    *,
    pr_number: int | None = None,
    shoutout: bool = False,
    announce: bool = True,
) -> list[str]:
    """Grant any newly earned roles and optionally post one announcement when something is new.

    Returns the list of role keys newly granted during this call.
    """
    if not config.contributor_roles_enabled:
        return []
    earned = earned_role_keys(_models(store.pulls_for_author(config.github_repository, github_user_id)))
    already = store.grant_keys(discord_id)
    now = iso(datetime.now(UTC))
    # A role Discord refused to add is left unannounced. Skipping the merge
    # shoutout too lets the next attempt post one combined message.
    assignment_blocked = False
    newly_granted: list[str] = []
    for key in earned:
        if key in already:
            continue
        if not assign_role(config, discord_id, key):
            assignment_blocked = True
            continue
        name = config.role_names.get(key) or DEFAULT_ROLE_NAMES[key]
        if store.record_grant(discord_id, key, name, now):
            newly_granted.append(key)
    if not announce:
        if newly_granted:
            store.mark_grants_announced(discord_id, newly_granted, now)
        return newly_granted
    pending = store.unannounced_grants(discord_id)
    pending.sort(key=lambda grant: _ROLE_ORDER.get(grant["role_key"], len(ROLE_KEYS)))
    if assignment_blocked and not pending:
        return newly_granted
    recorded = store.get_pull(config.github_repository, pr_number) if pr_number else None
    want_shoutout = bool(
        shoutout and config.merge_shoutouts_enabled and pr_number
        and recorded and not recorded["shoutout_at"]
    )
    if not pending and not want_shoutout:
        return newly_granted
    channel_id = config.role_announcements_channel_id
    names = [grant["role_name"] for grant in pending]
    if not channel_id:
        logging.warning(
            "ROLE_ANNOUNCEMENTS_CHANNEL_ID is unset; not announcing roles for discord user %s",
            discord_id,
        )
        return newly_granted
    content = announcement_text(discord_id, names, pr_number if want_shoutout else None)
    try:
        post_message(channel_id, config.discord_token, {
            "content": content,
            "allowed_mentions": {"parse": [], "users": [discord_id]},
        })
    except TransientAPIError:
        raise
    except RuntimeError as error:
        logging.error(
            "Could not post a role announcement in channel %s: %s. "
            "The bot needs View Channel and Send Messages there.",
            channel_id, error,
        )
        return newly_granted
    if pending:
        store.mark_grants_announced(discord_id, [grant["role_key"] for grant in pending], now)
    if want_shoutout and pr_number:
        store.mark_shoutout(config.github_repository, pr_number, now)
    return newly_granted


def _maybe_invite(config: Config, store: Store, login: str, github_user_id: int, number: int) -> str:
    """Return ``retry`` when the caller should answer the webhook with HTTP 500."""
    if not config.pr_comment_invites_enabled or not config.discord_invite_url:
        return "skip"
    if store.link_for_github_user(github_user_id):
        return "skip"
    if not config.github_token:
        logging.error(
            "PR comment invites are enabled but GITHUB_TOKEN is unset; not commenting on #%s",
            number,
        )
        return "skip"
    if not store.claim_invite(config.github_repository, number):
        return "skip"
    try:
        post_issue_comment(
            config.github_repository,
            number,
            config.github_token,
            invite_comment(login, config.discord_invite_url),
        )
    except TransientAPIError as error:
        store.release_invite(config.github_repository, number)
        logging.warning("Invite comment on #%s deferred: %s", number, error)
        return "retry"
    except RuntimeError as error:
        store.release_invite(config.github_repository, number)
        logging.error(
            "Could not comment on #%s: %s. The token needs permission to write issue comments.",
            number, error,
        )
        return "skip"
    store.finish_invite(config.github_repository, number, iso(datetime.now(UTC)))
    return "ok"


def handle_github_event(config: Config, store: Store, event: str, payload: dict) -> int:
    """Handle one signed GitHub webhook. Returns the HTTP status to send back."""
    if event == "ping" or event != "pull_request":
        return 200
    pull = payload.get("pull_request")
    repository = payload.get("repository") if isinstance(payload.get("repository"), dict) else {}
    full_name = repository.get("full_name") if isinstance(repository, dict) else None
    if not isinstance(pull, dict) or not isinstance(full_name, str):
        return 400
    if full_name.lower() != config.github_repository.lower():
        logging.info("Ignoring pull_request webhook for %s", full_name)
        return 200
    number = pull.get("number")
    if not isinstance(number, int) or number <= 0:
        return 400
    if payload.get("action") != "closed" or pull.get("merged") is not True:
        return 200
    user = pull.get("user") if isinstance(pull.get("user"), dict) else {}
    login = user.get("login")
    user_id = user.get("id")
    if not isinstance(login, str) or not _LOGIN.fullmatch(login) or not isinstance(user_id, int) or user_id <= 0:
        logging.error("Merged PR #%s has no usable author; ignoring", number)
        return 200
    if _is_bot(user):
        logging.info("Ignoring merged PR #%s from bot %s", number, login)
        return 200
    existing = store.get_pull(config.github_repository, number)
    if existing:
        return _finish_known_pull(
            config, store, existing["github_login"], int(existing["github_user_id"]), number,
        )
    raw_merged_at = pull.get("merged_at")
    try:
        merged_at = iso(parse_time(raw_merged_at if isinstance(raw_merged_at, str) else ""))
    except (TypeError, ValueError):
        logging.error("Merged PR #%s has no usable merged_at; ignoring", number)
        return 200
    try:
        files = fetch_pull_files(config.github_repository, number, config.effective_github_token)
    except TransientAPIError as error:
        logging.warning("File list for #%s deferred: %s", number, error)
        return 500
    except RuntimeError as error:
        logging.error("File list for #%s failed: %s", number, error)
        return 500
    title = pull.get("title") if isinstance(pull.get("title"), str) else ""
    store.upsert_pull(
        repository=config.github_repository,
        number=number,
        github_user_id=user_id,
        github_login=login,
        merged_at=merged_at,
        title=title,
        labels=_label_names(pull),
        files=files,
    )
    return _finish_known_pull(config, store, login, user_id, number)


def _finish_known_pull(config: Config, store: Store, login: str, user_id: int, number: int) -> int:
    link = store.link_for_github_user(user_id)
    if link:
        apply_roles(
            config, store, link["discord_id"], user_id,
            pr_number=number, shoutout=True,
        )
    if _maybe_invite(config, store, login, user_id, number) == "retry":
        return 500
    return 200
