"""GitHub OAuth: the browser proves which account belongs to a Discord user."""

import logging
from datetime import datetime
from urllib.parse import parse_qs, urlencode

from .api import TransientAPIError, exchange_github_oauth, fetch_github_user
from .backfill import schedule_author_sync
from .config import Config
from .schedule import UTC, iso
from .store import Store


def authorize_url(client_id: str, redirect_uri: str, state: str) -> str:
    query = urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": "read:user",
        "state": state,
        "allow_signup": "false",
    })
    return f"https://github.com/login/oauth/authorize?{query}"


def _one(params: dict, name: str) -> str | None:
    values = params.get(name) or []
    if len(values) != 1 or not isinstance(values[0], str):
        return None
    return values[0]


def callback_page(config: Config, store: Store, query: str) -> tuple[int, str]:
    """Finish OAuth. The GitHub access token is used once and never stored."""
    params = parse_qs(query, keep_blank_values=False, max_num_fields=10)
    if params.get("error"):
        return 400, "github authorization was cancelled. run /link-github in discord to try again."
    code = _one(params, "code")
    state = _one(params, "state")
    if not code or not state:
        return 400, "that link is missing its github confirmation. run /link-github again."
    discord_id = store.pop_oauth_state(state, iso(datetime.now(UTC)))
    if not discord_id:
        return 400, "that link expired. run /link-github again."
    try:
        access_token = exchange_github_oauth(
            config.github_oauth_client_id or "",
            config.github_oauth_client_secret or "",
            code,
            config.github_oauth_redirect_uri or "",
        )
        user = fetch_github_user(access_token)
    except (TransientAPIError, RuntimeError, ValueError) as error:
        logging.warning("GitHub OAuth callback failed: %s", error)
        return 400, "github didn't confirm that login. run /link-github again."
    failure = store.link_account(discord_id, user["login"], user["id"], iso(datetime.now(UTC)))
    if failure:
        return 400, failure
    schedule_author_sync(config, store, user["login"], user["id"], discord_id)
    return 200, f"linked github account {user['login']}. you can close this tab and go back to discord."
