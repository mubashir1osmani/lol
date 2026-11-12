import json
import math
import time
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import is_discord_id


GITHUB_API = "https://api.github.com"
DISCORD_API = "https://discord.com/api/v10"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class TransientAPIError(RuntimeError):
    """A temporary upstream failure; retry on the next poll."""


def _http(request: Request, attempts: int = 2) -> tuple[int, Any, bytes]:
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=15) as response:
                return response.status, response.headers, response.read()
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
                f"{request.host} returned {error.code}: {details}"
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


def fetch_channel(channel_id: str, token: str) -> dict[str, Any]:
    channel = _request_json(Request(
        f"{DISCORD_API}/channels/{channel_id}",
        headers=_discord_headers(token),
    ))
    if not isinstance(channel, dict) or channel.get("id") != channel_id:
        raise RuntimeError("Discord returned an unexpected channel response")
    return channel


def post_message(channel_id: str, token: str, message: dict[str, Any]) -> dict[str, Any]:
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
