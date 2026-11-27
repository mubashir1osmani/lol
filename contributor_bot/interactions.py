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
    post_interaction_followup,
)
from .config import Config, is_discord_id
from .oauth import authorize_url
from .pr_board import format_pr_board_summary
from .present import my_prs_text, review_status
from .review_job import handle_review_command
from .roles import DEFAULT_ROLE_NAMES, MergedPull, earned_role_keys
from .schedule import UTC, iso, parse_time
from .store import Store

OAUTH_SLASH_COMMANDS = [
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
]

ROLE_BASE_SLASH_COMMANDS = [
    {
        "name": "my-prs",
        "description": "Show your open LiteLLM PRs, merged count, and roles",
        "type": 1,
        "dm_permission": False,
    },
]

SLASH_COMMANDS = [*OAUTH_SLASH_COMMANDS, *ROLE_BASE_SLASH_COMMANDS]

_PERM_ADMINISTRATOR = 1 << 3
_PERM_MANAGE_GUILD = 1 << 5
_PERM_MANAGE_MESSAGES = 1 << 13
_PERM_MODERATE_MEMBERS = 1 << 40
_MOD_PERMISSION_MASK = (
    _PERM_ADMINISTRATOR
    | _PERM_MANAGE_GUILD
    | _PERM_MANAGE_MESSAGES
    | _PERM_MODERATE_MEMBERS
)

ROLES_BACKFILL_SLASH_COMMANDS = [
    {
        "name": "roles-backfill",
        "description": "Run the time-bounded historical contributor roles backfill",
        "type": 1,
        "dm_permission": False,
        "default_member_permissions": str(_PERM_MANAGE_MESSAGES),
    },
]


def role_slash_commands(config: Config) -> list[dict]:
    """Slash commands to register when contributor roles are enabled.

    /link-github and /unlink-github stay hidden unless GitHub OAuth is configured.
    """
    if not config.contributor_roles_enabled:
        return []
    commands = list(ROLE_BASE_SLASH_COMMANDS) + list(ROLES_BACKFILL_SLASH_COMMANDS)
    if config.github_oauth_enabled:
        return list(OAUTH_SLASH_COMMANDS) + commands
    return commands

# Registered whenever MERGED_WEEKLY_ENABLED=true and DISCORD_GUILD_ID is set.
MERGED_SLASH_COMMANDS = [
    {
        "name": "merged-this-week",
        "description": "Preview community pull requests merged in the last 7 days",
        "type": 1,
        "dm_permission": False,
        "default_member_permissions": str(_PERM_MANAGE_MESSAGES),
    },
]

# Registered whenever PR_BOARD_PROJECT_ID (or PR_BOARD_OWNER + PR_BOARD_NUMBER) and DISCORD_GUILD_ID are set.
PR_BOARD_SLASH_COMMANDS = [
    {
        "name": "pr-board",
        "description": "Show the contributor PR backlog board link and counts per status",
        "type": 1,
        "dm_permission": False,
        "default_member_permissions": str(_PERM_MANAGE_MESSAGES),
    },
]

REVIEW_SLASH_COMMANDS = [
    {
        "name": "review-pr",
        "description": "Pre-review a LiteLLM pull request in a thread",
        "type": 1,
        "dm_permission": False,
        "options": [
            {
                "type": 3,
                "name": "url",
                "description": "https://github.com/BerriAI/litellm/pull/123",
                "required": True,
            },
        ],
    },
    {
        "name": "recheck",
        "description": "Re-check the pull request this thread is tracking",
        "type": 1,
        "dm_permission": False,
    },
]

_EPHEMERAL = 64
_SUPPRESS_EMBEDS = 4


def _ephemeral(content: str) -> dict:
    return {
        "type": 4,
        "data": {"content": content, "flags": _EPHEMERAL, "allowed_mentions": {"parse": []}},
    }


def is_server_moderator(payload: dict) -> bool:
    """True when the invoker has administrator, manage_guild, manage_messages, or moderate_members."""
    member = payload.get("member") or {}
    raw = member.get("permissions")
    if raw is None:
        return False
    if hasattr(raw, "value"):
        raw = raw.value
    try:
        bits = int(raw)
    except (TypeError, ValueError):
        return False
    return bool(bits & _MOD_PERMISSION_MASK)


def handle_interaction(config: Config, store: Store, payload: dict) -> dict:
    if payload.get("type") == 1:
        return {"type": 1}
    if payload.get("type") != 2:
        return _ephemeral("that interaction isn't supported.")
    if payload.get("guild_id") != config.discord_guild_id:
        return _ephemeral("run this in the litellm discord server.")
    if config.discord_application_id and str(payload.get("application_id") or "") != config.discord_application_id:
        return _ephemeral("this command isn't for this bot.")
    user = (payload.get("member") or {}).get("user") or {}
    discord_id = user.get("id")
    if not is_discord_id(discord_id):
        return _ephemeral("couldn't tell who ran that command.")
    name = (payload.get("data") or {}).get("name")
    app_id = str(payload.get("application_id") or config.discord_application_id or "")
    if name == "link-github":
        if not config.github_oauth_enabled:
            return _ephemeral(
                "github account linking is disabled on this bot; contributor roles are assigned automatically."
            )
        return _link(config, store, discord_id)
    if name == "unlink-github":
        if not config.github_oauth_enabled:
            return _ephemeral(
                "github account linking is disabled on this bot; contributor roles are assigned automatically."
            )
        return _unlink(store, discord_id)
    if name == "my-prs":
        return _my_prs(config, store, discord_id, payload.get("token"), app_id)
    if name == "roles-backfill":
        if not is_server_moderator(payload):
            return _ephemeral("only mods can run this")
        if not config.contributor_roles_enabled:
            return _ephemeral("contributor roles are off on this bot.")
        return _roles_backfill(
            config,
            store,
            payload.get("token"),
            app_id,
            guild=payload.get("_guild"),
            loop=payload.get("_loop"),
        )
    if name == "merged-this-week":
        if not is_server_moderator(payload):
            return _ephemeral("only mods can run this")
        return _merged_preview(
            config,
            store,
            payload.get("token"),
            application_id=app_id,
            channel_id=str(payload.get("channel_id") or ""),
            guild=payload.get("_guild"),
            loop=payload.get("_loop"),
        )
    if name == "pr-board":
        if not is_server_moderator(payload):
            return _ephemeral("only mods can run this")
        return _ephemeral(format_pr_board_summary(config, store))
    if name in {"review-pr", "recheck"}:
        if not config.pr_review_enabled:
            return _ephemeral("pr pre-review is off on this bot.")
        return handle_review_command(config, store, payload)
    return _ephemeral("unknown command.")


def _merged_preview(
    config: Config,
    store: Store,
    interaction_token: object,
    *,
    application_id: str = "",
    channel_id: str = "",
    guild=None,
    loop=None,
) -> dict:
    """Defer immediately, then send an ephemeral preview without marking the week posted."""
    if not isinstance(interaction_token, str) or not interaction_token:
        return _ephemeral("couldn't open that reply. try /merged-this-week again.")
    app_id = application_id or (config.discord_application_id or "")
    threading.Thread(
        target=_finish_merged_preview,
        args=(config, store, interaction_token, app_id),
        kwargs={"channel_id": channel_id, "guild": guild, "loop": loop},
        name="merged-this-week",
        daemon=True,
    ).start()
    return {"type": 5, "data": {"flags": _EPHEMERAL}}


def _finish_merged_preview(
    config: Config,
    store: Store,
    interaction_token: str,
    application_id: str,
    *,
    channel_id: str = "",
    guild=None,
    loop=None,
    now: datetime | None = None,
) -> None:
    from .weekly import WeeklyFetchError, build_weekly_report, chunk_mention_user_ids

    moment = now or datetime.now(UTC)
    try:
        report = build_weekly_report(config, store, moment, guild=guild, loop=loop, pace=False)
        chunks = list(report.chunks)
        if report.matched_count == 0 and report.diagnostics is not None and chunks:
            diag_line = report.diagnostics.summary_line()
            candidate = f"{chunks[0]}\n\n{diag_line}"
            if len(candidate) <= 1900:
                chunks[0] = candidate
    except WeeklyFetchError as error:
        logging.warning("Weekly merge preview GitHub fetch failed: %s", error)
        if error.rate_limited:
            chunks = [
                "github search rate limit hit while loading this week's merges. "
                "set GITHUB_TOKEN or try again in a minute."
            ]
        else:
            chunks = ["couldn't load this week's merges from github. try again in a minute."]
    except Exception:
        logging.exception("Could not build the weekly merge preview")
        chunks = ["couldn't load this week's merges. try again in a minute."]

    if not chunks:
        chunks = ["no discord-matched community pull requests merged in the last 7 days."]
    try:
        edit_original_interaction(application_id, interaction_token, {
            "content": chunks[0],
            "flags": _EPHEMERAL | _SUPPRESS_EMBEDS,
            "allowed_mentions": {"parse": [], "users": chunk_mention_user_ids(chunks[0])},
        })
        for chunk in chunks[1:]:
            post_interaction_followup(application_id, interaction_token, {
                "content": chunk,
                "flags": _EPHEMERAL | _SUPPRESS_EMBEDS,
                "allowed_mentions": {"parse": [], "users": chunk_mention_user_ids(chunk)},
            })
    except Exception:
        logging.exception("Could not send the /merged-this-week follow-up")


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


def _roles_backfill(
    config: Config,
    store: Store,
    interaction_token: object,
    application_id: str,
    *,
    guild=None,
    loop=None,
) -> dict:
    if not isinstance(interaction_token, str) or not interaction_token:
        return _ephemeral("couldn't open that reply. try /roles-backfill again.")
    threading.Thread(
        target=_finish_roles_backfill,
        args=(config, store, interaction_token, application_id, guild, loop),
        name="roles-backfill",
        daemon=True,
    ).start()
    return {"type": 5, "data": {"flags": _EPHEMERAL}}


def _finish_roles_backfill(
    config: Config,
    store: Store,
    interaction_token: str,
    application_id: str,
    guild=None,
    loop=None,
) -> None:
    from .roles_sync import format_backfill_result, run_roles_backfill

    try:
        result = run_roles_backfill(config, store, guild=guild, loop=loop)
        content = format_backfill_result(result)
    except Exception:
        logging.exception("/roles-backfill failed")
        content = "contributor roles backfill hit an unexpected error. check logs and try again."
    try:
        edit_original_interaction(application_id, interaction_token, {
            "content": content,
            "flags": _EPHEMERAL,
            "allowed_mentions": {"parse": []},
        })
    except Exception:
        logging.exception("Could not send the /roles-backfill follow-up")


def _resolved_identity_for_discord(config: Config, store: Store, discord_id: str) -> dict | None:
    link = store.link_for_discord(discord_id)
    if link:
        return link
    matched = store.confirmed_match_for_discord(discord_id)
    if matched:
        return {
            "discord_id": discord_id,
            "github_login": matched["github_login"],
            "github_user_id": matched.get("github_user_id"),
        }
    for login, mapped_id in (config.github_discord_map or {}).items():
        if mapped_id == discord_id:
            return {"discord_id": discord_id, "github_login": login, "github_user_id": None}
    return None


def _my_prs(
    config: Config,
    store: Store,
    discord_id: str,
    interaction_token: object,
    application_id: str = "",
) -> dict:
    link = _resolved_identity_for_discord(config, store, discord_id)
    if not link:
        if config.github_oauth_enabled:
            return _ephemeral("your github isn't linked yet. run /link-github to connect it.")
        return _ephemeral("couldn't match your discord account to a merged litellm github contributor yet.")
    if not isinstance(interaction_token, str):
        return _ephemeral("couldn't open that reply. try /my-prs again.")
    resolved_app_id = application_id or (config.discord_application_id or "")
    threading.Thread(
        target=_finish_my_prs,
        args=(config, store, link, interaction_token, resolved_app_id),
        name="my-prs",
        daemon=True,
    ).start()
    return {"type": 5, "data": {"flags": _EPHEMERAL}}


def _finish_my_prs(
    config: Config,
    store: Store,
    link: dict,
    interaction_token: str,
    application_id: str = "",
) -> None:
    try:
        content = my_prs_content(config, store, link)
    except Exception:
        logging.exception("my-prs failed for discord user %s", link.get("discord_id"))
        content = "couldn't load your prs just now. try again in a minute."
    try:
        edit_original_interaction(application_id or (config.discord_application_id or ""), interaction_token, {
            "content": content,
            "allowed_mentions": {"parse": []},
        })
    except Exception:
        logging.exception("Could not send the /my-prs follow-up")


def my_prs_content(config: Config, store: Store, link: dict) -> str:
    github_user_id = link.get("github_user_id")
    if isinstance(github_user_id, int) and github_user_id > 0:
        rows = store.pulls_for_author(config.github_repository, github_user_id)
    else:
        rows = store.pulls_for_author_login(config.github_repository, link["github_login"])
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
            config.github_repository, link["github_login"], config.effective_github_token,
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
        pull = fetch_pull_request(config.github_repository, item["number"], config.effective_github_token)
        reviews = fetch_pull_reviews(config.github_repository, item["number"], config.effective_github_token)
        status = review_status(pull, reviews)
        title = pull.get("title") if isinstance(pull.get("title"), str) else item["title"]
    except (TransientAPIError, RuntimeError, ValueError):
        logging.warning("Review status for #%s is unavailable", item["number"])
        status = "review status unknown"
        title = item["title"]
    return {"number": item["number"], "title": title, "review_status": status}
