"""Discord slash commands for linking GitHub and showing pull requests."""

import logging
import secrets
import threading
from datetime import datetime, timedelta

from .api import (
    TransientAPIError,
    edit_original_interaction,
    fetch_open_pulls_for_author,
    fetch_pull_request,
    fetch_pull_reviews,
)
from .config import Config, is_discord_id
from .oauth import authorize_url
from .present import my_prs_text, review_status
from .roles import DEFAULT_ROLE_NAMES, MergedPull, earned_role_keys
from .schedule import UTC, iso, parse_time
from .store import Store

SLASH_COMMANDS = [
    {
        "name": "link-github",
        "description": "Link your GitHub account to earn LiteLLM contributor roles",
        "type": 1,
        "dm_permission": False,
    },
    {
        "name": "unlink-github",
        "description": "Unlink your GitHub account from Discord",
        "type": 1,
        "dm_permission": False,
    },
    {
        "name": "my-prs",
        "description": "Show your open LiteLLM PRs, merged count, and roles",
        "type": 1,
        "dm_permission": False,
    },
]

_EPHEMERAL = 64


def _ephemeral(content: str) -> dict:
    return {
        "type": 4,
        "data": {"content": content, "flags": _EPHEMERAL, "allowed_mentions": {"parse": []}},
    }


def handle_interaction(config: Config, store: Store, payload: dict) -> dict:
    if payload.get("type") == 1:
        return {"type": 1}
    if payload.get("type") != 2:
        return _ephemeral("that interaction isn't supported.")
    if payload.get("guild_id") != config.discord_guild_id:
        return _ephemeral("run this in the litellm discord server.")
    if str(payload.get("application_id") or "") != (config.discord_application_id or ""):
        return _ephemeral("this command isn't for this bot.")
    user = (payload.get("member") or {}).get("user") or {}
    discord_id = user.get("id")
    if not is_discord_id(discord_id):
        return _ephemeral("couldn't tell who ran that command.")
    name = (payload.get("data") or {}).get("name")
    if name == "link-github":
        return _link(config, store, discord_id)
    if name == "unlink-github":
        return _unlink(store, discord_id)
    if name == "my-prs":
        return _my_prs(config, store, discord_id, payload.get("token"))
    return _ephemeral("unknown command.")


def _link(config: Config, store: Store, discord_id: str) -> dict:
    state = secrets.token_urlsafe(32)
    now = datetime.now(UTC)
    store.put_oauth_state(state, discord_id, iso(now + timedelta(minutes=10)), iso(now))
    url = authorize_url(
        config.github_oauth_client_id or "",
        config.github_oauth_redirect_uri or "",
        state,
    )
    return _ephemeral(
        "connect github so merged litellm prs can earn you roles:\n"
        f"{url}\n"
        "this link expires in 10 minutes and only works for you."
    )


def _unlink(store: Store, discord_id: str) -> dict:
    if store.unlink(discord_id):
        return _ephemeral("unlinked. roles you already earned stay on your discord account.")
    return _ephemeral("your github isn't linked. run /link-github first.")


def _my_prs(config: Config, store: Store, discord_id: str, interaction_token: object) -> dict:
    link = store.link_for_discord(discord_id)
    if not link:
        return _ephemeral("your github isn't linked yet. run /link-github to connect it.")
    if not isinstance(interaction_token, str):
        return _ephemeral("couldn't open that reply. try /my-prs again.")
    threading.Thread(
        target=_finish_my_prs,
        args=(config, store, link, interaction_token),
        name="my-prs",
        daemon=True,
    ).start()
    return {"type": 5, "data": {"flags": _EPHEMERAL}}


def _finish_my_prs(config: Config, store: Store, link: dict, interaction_token: str) -> None:
    try:
        content = my_prs_content(config, store, link)
    except Exception:
        logging.exception("my-prs failed for discord user %s", link.get("discord_id"))
        content = "couldn't load your prs just now. try again in a minute."
    try:
        edit_original_interaction(config.discord_application_id or "", interaction_token, {
            "content": content,
            "allowed_mentions": {"parse": []},
        })
    except Exception:
        logging.exception("Could not send the /my-prs follow-up")


def my_prs_content(config: Config, store: Store, link: dict) -> str:
    rows = store.pulls_for_author(config.github_repository, link["github_user_id"])
    models = [
        MergedPull(int(row["number"]), parse_time(row["merged_at"]), tuple(row["labels"]), tuple(row["files"]))
        for row in rows
    ]
    names = [
        config.role_names.get(key) or DEFAULT_ROLE_NAMES[key]
        for key in earned_role_keys(models)
    ]
    open_prs: list[dict] = []
    open_total = None
    open_error = False
    try:
        items, open_total = fetch_open_pulls_for_author(
            config.github_repository, link["github_login"], config.github_token,
        )
        for item in items[:8]:
            open_prs.append(_with_review(config, item))
    except (TransientAPIError, RuntimeError, ValueError):
        logging.warning("Open PR lookup failed for %s", link["github_login"])
        open_error = True
    return my_prs_text(
        link["github_login"], open_prs, open_total, len(rows), names, open_error=open_error,
    )


def _with_review(config: Config, item: dict) -> dict:
    try:
        pull = fetch_pull_request(config.github_repository, item["number"], config.github_token)
        reviews = fetch_pull_reviews(config.github_repository, item["number"], config.github_token)
        status = review_status(pull, reviews)
        title = pull.get("title") if isinstance(pull.get("title"), str) else item["title"]
    except (TransientAPIError, RuntimeError, ValueError):
        logging.warning("Review status for #%s is unavailable", item["number"])
        status = "review status unknown"
        title = item["title"]
    return {"number": item["number"], "title": title, "review_status": status}
