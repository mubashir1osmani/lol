import json
import logging
import math
import re
import time
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .config import is_discord_id
from .schedule import iso, parse_time
from .releases import RELEASE_INDEX, parse_release_index, parse_new_contributors, release_url


GITHUB_API = "https://api.github.com"
DISCORD_API = "https://discord.com/api/v10"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_last_search_request = 0.0


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        # Never forward bot/API credentials to a redirect destination.
        raise RuntimeError("API redirects are disabled")


# Do not let HTTP(S)_PROXY environment variables reroute credentialed requests.
_open_request = build_opener(ProxyHandler({}), _NoRedirects()).open


_OWNER_REPO = r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
_SNOWFLAKE = r"[1-9][0-9]{0,19}"
_PR_NUMBER = r"[1-9][0-9]*"
# Interaction tokens are untrusted path segments. Keep them to one segment.
_INTERACTION_TOKEN = r"[A-Za-z0-9._~=-]{10,256}"


def _github_request_allowed(method: str, path: str) -> bool:
    pull = rf"/repos/{_OWNER_REPO}/pulls/{_PR_NUMBER}"
    if method == "GET" and (
        re.fullmatch(pull, path)
        or re.fullmatch(pull + r"/files", path)
        or re.fullmatch(pull + r"/reviews", path)
        or re.fullmatch(rf"/repos/{_OWNER_REPO}/pulls", path)
        or path in {"/search/issues", "/user"}
    ):
        return True
    # The only GitHub write: an optional invite comment on a merged pull request.
    return method == "POST" and re.fullmatch(
        rf"/repos/{_OWNER_REPO}/issues/{_PR_NUMBER}/comments", path
    ) is not None


def _discord_request_allowed(method: str, path: str) -> bool:
    if method == "GET" and re.fullmatch(rf"/api/v10/channels/{_SNOWFLAKE}(?:/messages)?", path):
        return True
    if method == "POST" and re.fullmatch(rf"/api/v10/channels/{_SNOWFLAKE}/messages", path):
        return True
    # Add an existing role to a member. Creating roles, editing members, and
    # removing roles stay blocked.
    if method == "PUT" and re.fullmatch(
        rf"/api/v10/guilds/{_SNOWFLAKE}/members/{_SNOWFLAKE}/roles/{_SNOWFLAKE}", path
    ):
        return True
    if method == "PUT" and re.fullmatch(
        rf"/api/v10/applications/{_SNOWFLAKE}/guilds/{_SNOWFLAKE}/commands", path
    ):
        return True
    # Deferred slash-command replies. Only the original interaction response.
    return method == "PATCH" and re.fullmatch(
        rf"/api/v10/webhooks/{_SNOWFLAKE}/{_INTERACTION_TOKEN}/messages/@original", path
    ) is not None


def _validate_request(request: Request) -> None:
    """Defense in depth, not a sandbox against malicious replacement code."""
    url = urlsplit(request.full_url)
    method = request.get_method()
    if url.scheme != "https" or url.fragment or any(
        segment in {".", ".."} for segment in url.path.split("/")
    ):
        raise ValueError("Only approved HTTPS API requests are allowed")
    if url.netloc == "api.github.com":
        allowed = _github_request_allowed(method, url.path)
    elif url.netloc == "github.com":
        # OAuth code exchange only. The browser, not this bot, opens the authorize page.
        allowed = method == "POST" and url.path == "/login/oauth/access_token" and not url.query
    elif url.netloc == "docs.litellm.ai":
        allowed = (method == "GET" and not url.query and not request.has_header("Authorization")
                   and (request.full_url == RELEASE_INDEX or release_url(request.full_url)))
    elif url.netloc == "discord.com":
        allowed = _discord_request_allowed(method, url.path)
    else:
        allowed = False
    if not allowed:
        raise ValueError("Request destination or operation is not allowed")


class TransientAPIError(RuntimeError):
    """A temporary upstream failure; retry on the next poll."""


def _http(request: Request, attempts: int = 2) -> tuple[int, Any, bytes]:
    _validate_request(request)
    for attempt in range(attempts):
        try:
            with _open_request(request, timeout=15) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise RuntimeError("API response exceeds the size limit")
                return response.status, response.headers, body
        except HTTPError as error:
            if error.code == 304:
                error.close()
                return 304, error.headers, b""
            details = error.read(500).decode("utf-8", errors="replace")
            error.close()
            retryable = error.code in _RETRYABLE_STATUS or (
                error.code == 403 and error.headers.get("X-RateLimit-Remaining") == "0"
            )
            delay = 1.0 if error.code in {500, 502, 503, 504} else None
            if error.code == 429:
                # Discord's short per-channel cooldown is common during
                # history pagination. Honor it so a large backfill can finish.
                try:
                    delay = float(error.headers.get("Retry-After") or json.loads(details)["retry_after"])
                except (ValueError, TypeError, KeyError):
                    delay = None
            if (
                delay is not None and math.isfinite(delay) and 0 <= delay <= 60
                and attempt + 1 < attempts
            ):
                time.sleep(delay)
                continue
            raise (TransientAPIError if retryable else RuntimeError)(
                # Do not log upstream bodies, which can echo credentials or
                # other untrusted content. Status and host identify failures.
                f"{request.host} returned HTTP {error.code}"
            ) from error
        except OSError as error:
            if attempt + 1 < attempts:
                time.sleep(1)
                continue
            raise TransientAPIError(f"Could not reach {request.host}: {error}") from error
    raise AssertionError("unreachable")


def _request_json(request: Request) -> Any:
    _status, _headers, body = _http(request)
    return json.loads(body) if body else None


_conditional_cache: dict[str, tuple[str, bytes]] = {}


def conditional_get(url: str, headers: dict[str, str]) -> bytes:
    cached = _conditional_cache.get(url)
    if cached:
        headers = {**headers, "If-None-Match": cached[0]}
    status, response_headers, body = _http(Request(url, headers=headers))
    if status == 304 and cached:
        return cached[1]
    etag = response_headers.get("ETag")
    if etag:
        _conditional_cache[url] = (etag, body)
    else:
        _conditional_cache.pop(url, None)
    return body


def _discord_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bot {token}",
        "User-Agent": "DiscordBot (https://github.com/BerriAI/litellm, 2.0.0)",
    }


def fetch_channel_messages(
    channel_id: str, token: str, after: str | None = None
) -> Iterator[dict[str, Any]]:
    """Walk history newest-first, stopping at the saved message checkpoint.

    Use `before` pagination, not an `after` query whose ordering could skip
    messages when more than one page arrives between polls.
    """
    before = None
    while True:
        params = {"limit": "100"}
        if before:
            params["before"] = before
        messages = _request_json(Request(
            f"{DISCORD_API}/channels/{channel_id}/messages?{urlencode(params)}",
            headers=_discord_headers(token),
        ))
        if not isinstance(messages, list):
            raise RuntimeError("Discord returned an unexpected messages response")
        if not messages:
            return
        oldest = min(int(message["id"]) for message in messages)
        if before is not None and oldest >= int(before):
            raise RuntimeError("Discord history pagination did not advance")
        for message in messages:
            if after is None or int(message["id"]) > int(after):
                yield message
        if (after is not None and oldest <= int(after)) or len(messages) < 100:
            return
        before = str(oldest)


def fetch_pull_request(repository: str, number: int, token: str | None) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "litellm-contributor-discord-bot",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    pull = json.loads(conditional_get(f"{GITHUB_API}/repos/{repository}/pulls/{number}", headers))
    if (
        not isinstance(pull, dict)
        or pull.get("number") != number
        or not isinstance(pull.get("merged"), bool)
        or not isinstance(pull.get("user"), dict)
        or not isinstance(pull["user"].get("login"), str)
    ):
        raise RuntimeError("GitHub returned an unexpected pull-request response")
    return pull


def fetch_release_entries() -> dict:
    return parse_release_index(conditional_get(RELEASE_INDEX, {}).decode("utf-8"))


def fetch_release_prs(url: str, repository: str) -> list[int]:
    if release_url(url) != url:
        raise ValueError("Only official LiteLLM release-note URLs are allowed")
    return parse_new_contributors(conditional_get(url, {}).decode("utf-8"), repository)


def github_headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "litellm-contributor-discord-bot",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def wait_for_github_budget(headers, *, now: float | None = None, sleep=time.sleep) -> None:
    """Sleep when a GitHub response says the remaining budget is almost gone."""
    raw = headers.get("X-RateLimit-Remaining") if headers is not None else None
    if raw is None:
        return
    try:
        remaining = int(raw)
    except (TypeError, ValueError):
        return
    if remaining >= 5:
        return
    reset_raw = headers.get("X-RateLimit-Reset")
    current = time.time() if now is None else now
    try:
        reset = float(reset_raw)
    except (TypeError, ValueError):
        reset = current + 2
    delay = reset - current
    if not math.isfinite(delay):
        return
    delay = min(max(delay, 0.0), 90.0) + 0.25
    logging.info("GitHub rate limit has %s requests left; waiting %.1fs", remaining, delay)
    sleep(delay)


def github_get_json(url: str, token: str | None, *, pace: bool = False):
    status, headers, body = _http(Request(url, headers=github_headers(token)))
    if status != 200:
        raise RuntimeError(f"GitHub returned HTTP {status}")
    if pace:
        wait_for_github_budget(headers)
    return json.loads(body) if body else None


def _clean_path(path: object) -> str:
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
        raise RuntimeError("GitHub returned an unexpected file path")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise RuntimeError("GitHub returned an unexpected file path")
    return path


def fetch_pull_files(repository: str, number: int, token: str | None) -> list[str]:
    """Changed paths for a pull request, paginated and rate-limit aware."""
    files: list[str] = []
    page = 1
    while page <= 30:
        payload = github_get_json(
            f"{GITHUB_API}/repos/{repository}/pulls/{number}/files?per_page=100&page={page}",
            token,
            pace=True,
        )
        if not isinstance(payload, list):
            raise RuntimeError("GitHub returned an unexpected files response")
        for item in payload:
            if not isinstance(item, dict):
                raise RuntimeError("GitHub returned an unexpected files response")
            files.append(_clean_path(item.get("filename")))
        if len(payload) < 100:
            return files
        page += 1
    raise RuntimeError("Pull request file list is too large")


def fetch_pull_reviews(repository: str, number: int, token: str | None) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    page = 1
    while page <= 10:
        payload = github_get_json(
            f"{GITHUB_API}/repos/{repository}/pulls/{number}/reviews?per_page=100&page={page}",
            token,
            pace=True,
        )
        if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
            raise RuntimeError("GitHub returned an unexpected reviews response")
        reviews.extend(payload)
        if len(payload) < 100:
            return reviews
        page += 1
    return reviews


def fetch_open_pulls_for_author(repository: str, author: str, token: str | None) -> tuple[list[dict[str, Any]], int]:
    """Open pull requests authored by ``author``. Returns ``(items, total_count)``."""
    if not re.fullmatch(r"[A-Za-z0-9-]+", author):
        raise ValueError("Unexpected GitHub login")
    _pace_github_search(token)
    params = urlencode({
        "q": f"repo:{repository} is:pr is:open author:{author}",
        "per_page": 20,
        "page": 1,
    })
    payload = _request_json(Request(
        f"{GITHUB_API}/search/issues?{params}",
        headers=github_headers(token),
    ))
    if (not isinstance(payload, dict) or payload.get("incomplete_results") is not False
            or not isinstance(payload.get("total_count"), int)
            or not isinstance(payload.get("items"), list)):
        raise TransientAPIError("GitHub search is incomplete")
    items = []
    for item in payload["items"]:
        number = item.get("number") if isinstance(item, dict) else None
        title = item.get("title") if isinstance(item, dict) else None
        if not isinstance(number, int) or number <= 0 or not isinstance(title, str):
            raise RuntimeError("Unexpected GitHub search result")
        items.append({"number": number, "title": title})
    return items, payload["total_count"]


def exchange_github_oauth(client_id: str, client_secret: str, code: str, redirect_uri: str) -> str:
    """Trade an OAuth code for a user access token. The token is not logged or stored."""
    if not re.fullmatch(r"[A-Za-z0-9._-]{8,200}", code):
        raise ValueError("Unexpected GitHub OAuth code")
    payload = _request_json(Request(
        "https://github.com/login/oauth/access_token",
        data=json.dumps({
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        }).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "litellm-contributor-discord-bot",
        },
    ))
    token = payload.get("access_token") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        raise RuntimeError("GitHub OAuth token exchange was declined")
    return token


def fetch_github_user(access_token: str) -> dict[str, Any]:
    """Identity for an OAuth access token. Callers must discard the token afterwards."""
    user = github_get_json(f"{GITHUB_API}/user", access_token)
    login = user.get("login") if isinstance(user, dict) else None
    user_id = user.get("id") if isinstance(user, dict) else None
    if (not isinstance(user, dict) or user.get("type") != "User"
            or not isinstance(login, str) or not re.fullmatch(r"[A-Za-z0-9-]+", login)
            or not isinstance(user_id, int) or user_id <= 0):
        raise RuntimeError("GitHub OAuth did not return a user account")
    return {"login": login, "id": user_id}


def add_member_role(guild_id: str, user_id: str, role_id: str, token: str) -> None:
    status, _headers, _body = _http(Request(
        f"{DISCORD_API}/guilds/{guild_id}/members/{user_id}/roles/{role_id}",
        data=b"",
        method="PUT",
        headers=_discord_headers(token),
    ))
    if status != 204:
        raise RuntimeError(f"discord.com returned HTTP {status}")


def register_guild_commands(application_id: str, guild_id: str, token: str, commands: list[dict[str, Any]]) -> None:
    response = _request_json(Request(
        f"{DISCORD_API}/applications/{application_id}/guilds/{guild_id}/commands",
        data=json.dumps(commands).encode("utf-8"),
        method="PUT",
        headers={**_discord_headers(token), "Content-Type": "application/json"},
    ))
    if not isinstance(response, list):
        raise RuntimeError("Discord did not accept the slash command list")


def edit_original_interaction(application_id: str, interaction_token: str, message: dict[str, Any]) -> None:
    """Finish a deferred slash command. Uses the interaction token, not the bot token."""
    if not is_discord_id(application_id) or not re.fullmatch(_INTERACTION_TOKEN, interaction_token):
        raise ValueError("Unexpected interaction token")
    mentions = message.get("allowed_mentions") or {}
    users = mentions.get("users", [])
    if not isinstance(users, list) or any(not is_discord_id(user_id) for user_id in users):
        raise ValueError("Only explicit Discord user IDs may be mentioned")
    message = {**message, "allowed_mentions": {"parse": [], "users": users}}
    try:
        _edit_original(application_id, interaction_token, message)
    except RuntimeError as error:
        if "HTTP 404" not in str(error):
            raise
        # Discord may not have recorded the deferred response yet.
        time.sleep(0.4)
        _edit_original(application_id, interaction_token, message)


def _edit_original(application_id: str, interaction_token: str, message: dict[str, Any]) -> None:
    response = _request_json(Request(
        f"{DISCORD_API}/webhooks/{application_id}/{interaction_token}/messages/@original",
        data=json.dumps(message).encode("utf-8"),
        method="PATCH",
        headers={"Content-Type": "application/json", "User-Agent": "litellm-contributor-discord-bot"},
    ))
    if not isinstance(response, dict) or not is_discord_id(response.get("id")):
        raise RuntimeError("Discord did not return an interaction follow-up")


def post_issue_comment(repository: str, number: int, token: str, body: str) -> dict[str, Any]:
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required to comment on pull requests")
    response = _request_json(Request(
        f"{GITHUB_API}/repos/{repository}/issues/{number}/comments",
        data=json.dumps({"body": body}).encode("utf-8"),
        method="POST",
        headers={**github_headers(token), "Content-Type": "application/json"},
    ))
    if not isinstance(response, dict) or not isinstance(response.get("id"), int):
        raise RuntimeError("GitHub did not return a comment receipt")
    return response


def _pace_github_search(token: str | None) -> None:
    global _last_search_request
    delay = (2.1 if token else 6.1) - (time.monotonic() - _last_search_request)
    if delay > 0:
        time.sleep(delay)
    _last_search_request = time.monotonic()


def fetch_merged_numbers(repository: str, start, end, token: str | None, author: str | None = None) -> set[int]:
    """Search is discovery only; each selected PR is verified through /pulls/N.

    Split large windows to avoid GitHub search's 1,000-result truncation.
    Search ranges are inclusive; exact [start,end) checks happen on PR data.
    ``author`` limits discovery to one GitHub login (used when somebody links).
    """
    if author is not None and not re.fullmatch(r"[A-Za-z0-9-]+", author):
        raise ValueError("Unexpected GitHub login")
    author_query = f" author:{author}" if author else ""
    found = set()
    page, total = 1, None
    while True:
        params = urlencode({
            "q": f"repo:{repository} is:pr is:merged{author_query} merged:{iso(start)}..{iso(end)}",
            "per_page": 100,
            "page": page,
        })
        # Search has a separate 10/min anonymous (30/min authenticated) budget.
        # Pace across recursive windows too, or >1,000 results can hit the same
        # rate limit on every restart before ever completing discovery.
        _pace_github_search(token)
        payload = _request_json(Request(
            f"{GITHUB_API}/search/issues?{params}",
            headers=github_headers(token),
        ))
        if (not isinstance(payload, dict) or payload.get("incomplete_results") is not False
                or not isinstance(payload.get("total_count"), int)
                or not isinstance(payload.get("items"), list)):
            raise TransientAPIError("GitHub search is incomplete; roundup deferred")
        if total is None:
            total = payload["total_count"]
        if payload["total_count"] > 1000:
            if (end - start).total_seconds() < 4:
                raise RuntimeError("Too many merged PRs in a one-second search window")
            midpoint = parse_time(iso(start + (end - start) / 2))
            return (fetch_merged_numbers(repository, start, midpoint, token, author)
                    | fetch_merged_numbers(repository, midpoint, end, token, author))
        items = payload["items"]
        for item in items:
            number = item.get("number")
            if (not isinstance(number, int) or number <= 0
                    or (item.get("pull_request") or {}).get("url")
                    != f"{GITHUB_API}/repos/{repository}/pulls/{number}"):
                raise RuntimeError("Unexpected GitHub search result")
            found.add(number)
        if page * 100 >= max(total, payload["total_count"]):
            if len(found) < max(total, payload["total_count"]):
                raise TransientAPIError("GitHub search changed during pagination; retrying later")
            return found
        if not items:
            raise TransientAPIError("GitHub search ended before all results were read")
        page += 1


def fetch_channel(channel_id: str, token: str) -> dict[str, Any]:
    channel = _request_json(Request(
        f"{DISCORD_API}/channels/{channel_id}",
        headers=_discord_headers(token),
    ))
    if not isinstance(channel, dict) or channel.get("id") != channel_id:
        raise RuntimeError("Discord returned an unexpected channel response")
    return channel


def post_message(channel_id: str, token: str, message: dict[str, Any]) -> dict[str, Any]:
    mentions = message.get("allowed_mentions") or {}
    users = mentions.get("users", [])
    if (
        not isinstance(users, list)
        or len(users) > 100
        or any(not is_discord_id(user_id) for user_id in users)
    ):
        raise ValueError("Only explicit Discord user IDs may be mentioned")
    message = {**message, "allowed_mentions": {"parse": [], "users": users}}
    response = _request_json(Request(
        f"{DISCORD_API}/channels/{channel_id}/messages",
        data=json.dumps(message).encode("utf-8"),
        method="POST",
        headers={**_discord_headers(token), "Content-Type": "application/json"},
    ))
    if (
        not isinstance(response, dict)
        or not is_discord_id(response.get("id"))
        or response.get("channel_id") != channel_id
    ):
        raise RuntimeError("Discord did not return a valid message receipt")
    return response
