import json
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


def _validate_request(request: Request) -> None:
    """Defense in depth, not a sandbox against malicious replacement code."""
    url = urlsplit(request.full_url)
    method = request.get_method()
    if url.scheme != "https" or url.fragment or any(
        segment in {".", ".."} for segment in url.path.split("/")
    ):
        raise ValueError("Only approved HTTPS API requests are allowed")
    if url.netloc == "api.github.com":
        allowed = method == "GET" and (re.fullmatch(
            r"/repos/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pulls/[1-9][0-9]*", url.path
        ) or url.path == "/search/issues")
    elif url.netloc == "docs.litellm.ai":
        allowed = (method == "GET" and not url.query and not request.has_header("Authorization")
                   and (request.full_url == RELEASE_INDEX or release_url(request.full_url)))
    elif url.netloc == "discord.com":
        allowed = (
            method == "GET" and re.fullmatch(r"/api/v10/channels/[1-9][0-9]*(?:/messages)?", url.path)
        ) or (
            method == "POST" and re.fullmatch(r"/api/v10/channels/[1-9][0-9]*/messages", url.path)
        )
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


def fetch_merged_numbers(repository: str, start, end, token: str | None) -> set[int]:
    """Search is discovery only; each selected PR is verified through /pulls/N.

    Split large windows to avoid GitHub search's 1,000-result truncation.
    Search ranges are inclusive; exact [start,end) checks happen on PR data.
    """
    global _last_search_request
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "litellm-contributor-discord-bot"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    found = set()
    page, total = 1, None
    while True:
        params = urlencode({"q": f"repo:{repository} is:pr is:merged merged:{iso(start)}..{iso(end)}",
                            "per_page": 100, "page": page})
        # Search has a separate 10/min anonymous (30/min authenticated) budget.
        # Pace across recursive windows too, or >1,000 results can hit the same
        # rate limit on every restart before ever completing discovery.
        delay = (2.1 if token else 6.1) - (time.monotonic() - _last_search_request)
        if delay > 0:
            time.sleep(delay)
        _last_search_request = time.monotonic()
        payload = _request_json(Request(f"{GITHUB_API}/search/issues?{params}", headers=headers))
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
            return (fetch_merged_numbers(repository, start, midpoint, token)
                    | fetch_merged_numbers(repository, midpoint, end, token))
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
