import json
import logging
import math
import re
import threading
import time
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
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


_ALLOWED_PROJECT_V2_MUTATIONS = frozenset({
    "addProjectV2ItemById",
    "updateProjectV2ItemFieldValue",
})


def _allowed_project_v2_mutation(folded: str) -> bool:
    """Allow only single-operation GitHub Projects v2 board add/update mutations."""
    if not re.match(r"^mutation\b", folded, re.I):
        return False
    # Reject multiple operations, subscriptions, or comments/strings hiding other fields.
    if re.search(r"\b(subscription|fragment)\b", folded, re.I):
        return False
    if len(re.findall(r"\bmutation\b", folded, re.I)) != 1:
        return False
    open_brace = folded.find("{")
    if open_brace < 0 or not folded.endswith("}"):
        return False
    # Walk the top-level selection set (brace depth 1, paren depth 0) and inspect every field name.
    brace_depth = 0
    paren_depth = 0
    in_string = False
    escaped = False
    top_tokens: list[str] = []
    token_chars: list[str] = []
    for ch in folded[open_brace:]:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if brace_depth == 1 and paren_depth == 0 and token_chars:
                top_tokens.append("".join(token_chars))
                token_chars = []
            brace_depth += 1
            continue
        if ch == "}":
            if brace_depth == 1 and paren_depth == 0 and token_chars:
                top_tokens.append("".join(token_chars))
                token_chars = []
            brace_depth -= 1
            if brace_depth < 0:
                return False
            continue
        if brace_depth == 1:
            if ch == "(":
                if paren_depth == 0 and token_chars:
                    top_tokens.append("".join(token_chars))
                    token_chars = []
                paren_depth += 1
                continue
            if ch == ")":
                paren_depth -= 1
                if paren_depth < 0:
                    return False
                continue
            if paren_depth == 0:
                if ch.isalnum() or ch in "_:":
                    token_chars.append(ch)
                elif token_chars:
                    top_tokens.append("".join(token_chars))
                    token_chars = []
    if brace_depth != 0 or paren_depth != 0 or in_string:
        return False
    # Resolve optional GraphQL aliases (e.g. "item: addProjectV2ItemById" -> "addProjectV2ItemById").
    fields: list[str] = []
    skip_alias = False
    for tok in top_tokens:
        if tok == ":":
            skip_alias = True
            continue
        if tok.endswith(":"):
            skip_alias = True
            continue
        if ":" in tok:
            fields.append(tok.split(":", 1)[1])
            skip_alias = False
            continue
        if skip_alias:
            fields.append(tok)
            skip_alias = False
            continue
        fields.append(tok)
    return bool(fields) and all(field in _ALLOWED_PROJECT_V2_MUTATIONS for field in fields)


def _graphql_query_allowed(request: Request) -> bool:
    """Allow read-only GraphQL queries plus the two Projects v2 board mutations."""
    if request.get_method() != "POST" or urlsplit(request.full_url).query:
        return False
    data = request.data
    if not isinstance(data, bytes) or not data or len(data) > 20_000:
        return False
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or set(payload) - {"query", "variables"}:
        return False
    query = payload.get("query")
    variables = payload.get("variables", {})
    if not isinstance(query, str) or not query.strip() or len(query) > 10_000:
        return False
    if variables is not None and not isinstance(variables, dict):
        return False
    folded = " ".join(query.split())
    if re.search(r"\bsubscription\b", folded, re.I):
        return False
    if re.search(r"\bmutation\b", folded, re.I):
        return _allowed_project_v2_mutation(folded)
    lowered = folded.lower()
    return lowered.startswith("query ") or lowered.startswith("{")


def _github_request_allowed(method: str, path: str) -> bool:
    pull = rf"/repos/{_OWNER_REPO}/pulls/{_PR_NUMBER}"
    commit = rf"/repos/{_OWNER_REPO}/commits/[0-9a-fA-F]{{40}}"
    check_run = rf"/repos/{_OWNER_REPO}/check-runs/{_PR_NUMBER}"
    if method == "GET" and (
        re.fullmatch(pull, path)
        or re.fullmatch(pull + r"/files", path)
        or re.fullmatch(pull + r"/commits", path)
        or re.fullmatch(pull + r"/reviews", path)
        or re.fullmatch(pull + r"/comments", path)
        or re.fullmatch(commit + r"/check-runs", path)
        or re.fullmatch(commit + r"/status", path)
        or re.fullmatch(check_run, path)
        or re.fullmatch(check_run + r"/annotations", path)
        or re.fullmatch(rf"/repos/{_OWNER_REPO}/contents(?:/[^/]+)+", path)
        or re.fullmatch(rf"/repos/{_OWNER_REPO}/pulls", path)
        or re.fullmatch(r"/orgs/[A-Za-z0-9._-]+/public_members", path)
        or re.fullmatch(r"/users/[A-Za-z0-9-]+", path)
        or path in {"/search/issues", "/search/code", "/user"}
    ):
        return True
    # The only GitHub write: an optional invite comment on a merged pull request.
    # PR pre-review must not approve, merge, or comment. GraphQL is query-only.
    return method == "POST" and re.fullmatch(
        rf"/repos/{_OWNER_REPO}/issues/{_PR_NUMBER}/comments", path
    ) is not None


def _discord_request_allowed(method: str, path: str) -> bool:
    if method == "GET" and (
        re.fullmatch(rf"/api/v10/channels/{_SNOWFLAKE}(?:/messages)?", path)
        or re.fullmatch(rf"/api/v10/channels/{_SNOWFLAKE}/messages/{_SNOWFLAKE}", path)
    ):
        return True
    if method == "POST" and re.fullmatch(rf"/api/v10/channels/{_SNOWFLAKE}/messages", path):
        return True
    # A public thread hung off an existing message. One thread per message.
    if method == "POST" and re.fullmatch(
        rf"/api/v10/channels/{_SNOWFLAKE}/messages/{_SNOWFLAKE}/threads", path
    ):
        return True
    # Edit a message this bot already posted. Not roles, members, or channels.
    if method == "PATCH" and re.fullmatch(
        rf"/api/v10/channels/{_SNOWFLAKE}/messages/{_SNOWFLAKE}", path
    ):
        return True
    if method == "GET" and (
        re.fullmatch(rf"/api/v10/guilds/{_SNOWFLAKE}/channels", path)
        or re.fullmatch(rf"/api/v10/guilds/{_SNOWFLAKE}/members", path)
        or re.fullmatch(rf"/api/v10/guilds/{_SNOWFLAKE}/members/search", path)
        or re.fullmatch(rf"/api/v10/guilds/{_SNOWFLAKE}/messages/search", path)
    ):
        return True
    # Add or remove an existing contributor role on a member. Creating/deleting
    # roles, kicking/banning/editing members, and deleting channels stay blocked.
    if method in {"PUT", "DELETE"} and re.fullmatch(
        rf"/api/v10/guilds/{_SNOWFLAKE}/members/{_SNOWFLAKE}/roles/{_SNOWFLAKE}", path
    ):
        return True
    if method == "PUT" and re.fullmatch(
        rf"/api/v10/applications/{_SNOWFLAKE}/guilds/{_SNOWFLAKE}/commands", path
    ):
        return True
    # React to a message the bot is about to review. Not a reaction delete.
    if method == "PUT" and re.fullmatch(
        rf"/api/v10/channels/{_SNOWFLAKE}/messages/{_SNOWFLAKE}/reactions/[^/]+/@me", path
    ):
        return True
    # Deferred slash-command replies: edit @original or post follow-up chunks.
    if method == "PATCH" and re.fullmatch(
        rf"/api/v10/webhooks/{_SNOWFLAKE}/{_INTERACTION_TOKEN}/messages/@original", path
    ):
        return True
    return method == "POST" and re.fullmatch(
        rf"/api/v10/webhooks/{_SNOWFLAKE}/{_INTERACTION_TOKEN}", path
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
        if url.path == "/graphql":
            allowed = _graphql_query_allowed(request)
        else:
            allowed = _github_request_allowed(method, url.path)
    elif url.netloc == "github.com":
        # OAuth code exchange only. The browser, not this bot, opens the authorize page.
        allowed = method == "POST" and url.path == "/login/oauth/access_token" and not url.query
    elif url.netloc == "docs.litellm.ai":
        allowed = (method == "GET" and not url.query and not request.has_header("Authorization")
                   and (request.full_url == RELEASE_INDEX or release_url(request.full_url)))
    elif url.netloc == "discord.com":
        allowed = _discord_request_allowed(method, url.path)
    elif url.netloc == "api.typesafe.ai":
        # Jev System One decisions only. No other TypeSafe route.
        allowed = method == "POST" and url.path == "/v1/systemone" and not url.query
    else:
        allowed = False
    if not allowed:
        raise ValueError("Request destination or operation is not allowed")


class TransientAPIError(RuntimeError):
    """A temporary upstream failure; retry on the next poll."""

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


_github_rate_lock = threading.Lock()
_github_rate_state: dict[str, float | int | None] = {"remaining": None, "reset": None}


def reset_github_rate_limit_state() -> None:
    """Test helper."""
    with _github_rate_lock:
        _github_rate_state["remaining"] = None
        _github_rate_state["reset"] = None


def github_rate_limit_status() -> tuple[int | None, float | None]:
    """Return latest observed (X-RateLimit-Remaining, X-RateLimit-Reset epoch)."""
    with _github_rate_lock:
        rem = _github_rate_state["remaining"]
        rst = _github_rate_state["reset"]
    return (int(rem) if isinstance(rem, int) else None, float(rst) if isinstance(rst, (int, float)) else None)


def _record_github_rate_headers(host: str | None, headers) -> None:
    if host != "api.github.com" or headers is None:
        return
    raw_rem = headers.get("X-RateLimit-Remaining")
    raw_reset = headers.get("X-RateLimit-Reset")
    rem = None
    rst = None
    try:
        if raw_rem is not None:
            rem = int(raw_rem)
    except (TypeError, ValueError):
        rem = None
    try:
        if raw_reset is not None:
            rst = float(raw_reset)
    except (TypeError, ValueError):
        rst = None
    if rem is None and rst is None:
        return
    with _github_rate_lock:
        if rem is not None:
            _github_rate_state["remaining"] = rem
        if rst is not None:
            _github_rate_state["reset"] = rst


def _rate_limit_delay(error: HTTPError, details: str, *, now: float | None = None) -> float | None:
    retry_after = error.headers.get("Retry-After") if error.headers is not None else None
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except (ValueError, TypeError):
            pass
    if error.code == 429:
        try:
            return max(0.0, float(json.loads(details)["retry_after"]))
        except (ValueError, TypeError, KeyError):
            pass
    reset_raw = error.headers.get("X-RateLimit-Reset") if error.headers is not None else None
    if reset_raw is not None:
        try:
            current = time.time() if now is None else now
            return max(0.0, float(reset_raw) - current) + 0.5
        except (ValueError, TypeError):
            pass
    return None


def _github_json_error_summary(host: str | None, details: str) -> str:
    """Extract safe JSON error fields from a GitHub API error body for logs."""
    if host != "api.github.com" or not details:
        return ""
    try:
        parsed = json.loads(details)
    except (ValueError, TypeError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    parts: list[str] = []
    msg = parsed.get("message")
    if isinstance(msg, str) and msg.strip():
        parts.append(msg.strip()[:200])
    errs = parsed.get("errors")
    if isinstance(errs, list):
        for item in errs[:3]:
            if isinstance(item, dict):
                e_type = str(item.get("type") or "").strip()
                e_msg = str(item.get("message") or "").strip()
                e_path = item.get("path")
                chunk = ": ".join(p for p in (e_type, e_msg) if p)
                if e_path:
                    chunk = f"{chunk} (path={e_path})" if chunk else f"path={e_path}"
                if chunk:
                    parts.append(chunk[:160])
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip()[:160])
    return "; ".join(parts)[:300]


def _http(
    request: Request,
    attempts: int = 2,
    *,
    timeout: float = 15.0,
    max_retry_delay: float = 60.0,
) -> tuple[int, Any, bytes]:
    _validate_request(request)
    for attempt in range(attempts):
        try:
            with _open_request(request, timeout=timeout) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise RuntimeError("API response exceeds the size limit")
                _record_github_rate_headers(request.host, response.headers)
                return response.status, response.headers, body
        except HTTPError as error:
            if error.code == 304:
                _record_github_rate_headers(request.host, error.headers)
                error.close()
                return 304, error.headers, b""
            details = error.read(1024).decode("utf-8", errors="replace")
            _record_github_rate_headers(request.host, error.headers)
            error.close()
            has_retry_after = bool(error.headers and error.headers.get("Retry-After"))
            github_rate_limited = error.code == 403 and (
                (error.headers and error.headers.get("X-RateLimit-Remaining") == "0")
                or has_retry_after
            )
            retryable = error.code in _RETRYABLE_STATUS or github_rate_limited
            delay = min(max_retry_delay, float(1 << attempt)) if error.code in {500, 502, 503, 504} else None
            if error.code == 429 or github_rate_limited:
                delay = _rate_limit_delay(error, details)
            if (
                delay is not None and math.isfinite(delay) and 0 <= delay <= max_retry_delay
                and attempt + 1 < attempts
            ):
                time.sleep(delay)
                continue
            message = f"{request.host} returned HTTP {error.code}"
            gh_detail = _github_json_error_summary(request.host, details)
            if retryable:
                exc: Exception = TransientAPIError(message, retry_after=delay)
            else:
                exc = RuntimeError(message)
            if gh_detail:
                setattr(exc, "github_error_detail", gh_detail)
            raise exc from error
        except OSError as error:
            if attempt + 1 < attempts:
                time.sleep(1)
                continue
            raise TransientAPIError(f"Could not reach {request.host}: {error}") from error
    raise AssertionError("unreachable")


def _request_json(
    request: Request,
    *,
    attempts: int = 2,
    timeout: float = 15.0,
    max_retry_delay: float = 60.0,
) -> Any:
    _status, _headers, body = _http(
        request,
        attempts=attempts,
        timeout=timeout,
        max_retry_delay=max_retry_delay,
    )
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
        messages = _request_json(
            Request(
                f"{DISCORD_API}/channels/{channel_id}/messages?{urlencode(params)}",
                headers=_discord_headers(token),
            ),
            attempts=3,
        )
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


def github_get_json(
    url: str,
    token: str | None,
    *,
    pace: bool = False,
    attempts: int = 2,
    timeout: float = 15.0,
    max_retry_delay: float = 60.0,
):
    status, headers, body = _http(
        Request(url, headers=github_headers(token)),
        attempts=attempts,
        timeout=timeout,
        max_retry_delay=max_retry_delay,
    )
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


def fetch_pull_diff(repository: str, number: int, token: str | None) -> str:
    """Unified diff for a pull request. Same read URL as the pull, with the diff media type."""
    headers = github_headers(token)
    headers["Accept"] = "application/vnd.github.diff"
    status, _headers, body = _http(Request(f"{GITHUB_API}/repos/{repository}/pulls/{number}", headers=headers))
    if status != 200:
        raise RuntimeError(f"GitHub returned HTTP {status}")
    text = body.decode("utf-8", errors="replace")
    if text.lstrip().startswith("{"):
        raise RuntimeError("GitHub did not return a diff")
    return text


def github_graphql(
    token: str | None,
    query: str,
    variables: dict[str, Any],
    *,
    timeout: float = 15.0,
    attempts: int = 2,
    max_retry_delay: float = 5.0,
) -> Any:
    """One GraphQL call (read-only queries or Projects v2 board mutations)."""
    payload = _request_json(
        Request(
            f"{GITHUB_API}/graphql",
            data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
            method="POST",
            headers={**github_headers(token), "Content-Type": "application/json"},
        ),
        attempts=attempts,
        timeout=timeout,
        max_retry_delay=max_retry_delay,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("GitHub returned an unexpected GraphQL response")
    return payload


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


def remove_member_role(guild_id: str, user_id: str, role_id: str, token: str) -> None:
    status, _headers, _body = _http(Request(
        f"{DISCORD_API}/guilds/{guild_id}/members/{user_id}/roles/{role_id}",
        method="DELETE",
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


def edit_original_interaction(application_id: str, interaction_token: str, message: dict[str, Any]) -> dict[str, Any]:
    """Finish a deferred slash command. Uses the interaction token, not the bot token."""
    payload = _interaction_message(application_id, interaction_token, message)
    return _retry_interaction_404(lambda: _edit_original(application_id, interaction_token, payload))


def post_interaction_followup(application_id: str, interaction_token: str, message: dict[str, Any]) -> dict[str, Any]:
    """Send an additional chunk for a deferred slash command."""
    payload = _interaction_message(application_id, interaction_token, message)
    return _retry_interaction_404(lambda: _post_followup(application_id, interaction_token, payload))


def _interaction_message(application_id: str, interaction_token: str, message: dict[str, Any]) -> dict[str, Any]:
    if not is_discord_id(application_id) or not re.fullmatch(_INTERACTION_TOKEN, interaction_token):
        raise ValueError("Unexpected interaction token")
    mentions = message.get("allowed_mentions") or {}
    users = mentions.get("users", [])
    if not isinstance(users, list) or any(not is_discord_id(user_id) for user_id in users):
        raise ValueError("Only explicit Discord user IDs may be mentioned")
    return {**message, "allowed_mentions": {"parse": [], "users": users}}


def _retry_interaction_404(action):
    for delay in (0.4, 0.8):
        try:
            return action()
        except RuntimeError as error:
            if "HTTP 404" not in str(error):
                raise
            # Discord may not have recorded the deferred response yet.
            time.sleep(delay)
    return action()


def _edit_original(application_id: str, interaction_token: str, message: dict[str, Any]) -> dict[str, Any]:
    response = _request_json(Request(
        f"{DISCORD_API}/webhooks/{application_id}/{interaction_token}/messages/@original",
        data=json.dumps(message).encode("utf-8"),
        method="PATCH",
        headers={"Content-Type": "application/json", "User-Agent": "litellm-contributor-discord-bot"},
    ))
    if not isinstance(response, dict) or not is_discord_id(response.get("id")):
        raise RuntimeError("Discord did not return an interaction follow-up")
    return response


def _post_followup(application_id: str, interaction_token: str, message: dict[str, Any]) -> dict[str, Any]:
    response = _request_json(Request(
        f"{DISCORD_API}/webhooks/{application_id}/{interaction_token}",
        data=json.dumps(message).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "litellm-contributor-discord-bot"},
    ))
    if not isinstance(response, dict) or not is_discord_id(response.get("id")):
        raise RuntimeError("Discord did not return an interaction follow-up")
    return response


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
        if payload["total_count"] >= 1000:
            logging.warning(
                "GitHub merged PR search window %s..%s for %s hit %d results (>=1000 cap); splitting",
                iso(start),
                iso(end),
                repository,
                payload["total_count"],
            )
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


def fetch_author_merged_count(
    repository: str,
    author: str,
    token: str | None,
    *,
    timeout: float = 15.0,
) -> int:
    """Return an author's lifetime merged PR total_count via a single GitHub search call."""
    if not author or not re.fullmatch(r"[A-Za-z0-9-]+", author):
        raise ValueError("Unexpected GitHub login")
    params = urlencode({
        "q": f"repo:{repository} is:pr is:merged author:{author}",
        "per_page": 1,
    })
    _pace_github_search(token)
    payload = _request_json(
        Request(
            f"{GITHUB_API}/search/issues?{params}",
            headers=github_headers(token),
        ),
        attempts=2,
        timeout=timeout,
        max_retry_delay=5.0,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("incomplete_results") is True
        or not isinstance(payload.get("total_count"), int)
        or payload["total_count"] < 0
    ):
        raise TransientAPIError("GitHub search returned an incomplete author merged count")
    return int(payload["total_count"])


def fetch_recent_messages(
    channel_id: str,
    token: str,
    limit: int = 20,
    *,
    timeout: float = 15.0,
    attempts: int = 2,
    max_retry_delay: float = 10.0,
) -> list[dict[str, Any]]:
    """Newest messages in a channel or thread. One page, no history backfill."""
    if not is_discord_id(channel_id) or not 1 <= limit <= 100:
        raise ValueError("Unexpected Discord channel id")
    messages = _request_json(
        Request(
            f"{DISCORD_API}/channels/{channel_id}/messages?limit={limit}",
            headers=_discord_headers(token),
        ),
        attempts=attempts,
        timeout=timeout,
        max_retry_delay=max_retry_delay,
    )
    if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
        raise RuntimeError("Discord returned an unexpected messages response")
    return messages


def edit_message(
    channel_id: str,
    message_id: str,
    token: str,
    content: str,
    *,
    mention_roles: list[str] | None = None,
    mention_users: list[str] | None = None,
) -> dict[str, Any]:
    """Edit one message the bot sent. Role pings still require mention_roles."""
    if not is_discord_id(channel_id) or not is_discord_id(message_id):
        raise ValueError("Unexpected Discord message id")
    users = list(mention_users or [])
    roles = list(mention_roles or [])
    if any(not is_discord_id(user_id) for user_id in users) or len(users) > 10:
        raise ValueError("Only explicit Discord user IDs may be mentioned")
    if len(roles) > 1 or any(not is_discord_id(role_id) for role_id in roles):
        raise ValueError("Only one explicit Discord role ID may be mentioned")
    allowed: dict[str, Any] = {"parse": [], "users": users}
    if roles:
        allowed["roles"] = roles
    response = _request_json(Request(
        f"{DISCORD_API}/channels/{channel_id}/messages/{message_id}",
        data=json.dumps({"content": content, "allowed_mentions": allowed}).encode("utf-8"),
        method="PATCH",
        headers={**_discord_headers(token), "Content-Type": "application/json"},
    ))
    if not isinstance(response, dict) or response.get("id") != message_id:
        raise RuntimeError("Discord did not return an edited message")
    return response


def fetch_latest_message_id(channel_id: str, token: str) -> str | None:
    """The newest message id, or None when the channel is empty. Does not page history."""
    messages = _request_json(
        Request(
            f"{DISCORD_API}/channels/{channel_id}/messages?limit=1",
            headers=_discord_headers(token),
        ),
        attempts=3,
    )
    if not isinstance(messages, list):
        raise RuntimeError("Discord returned an unexpected messages response")
    if not messages:
        return None
    message_id = messages[0].get("id") if isinstance(messages[0], dict) else None
    if not is_discord_id(message_id):
        raise RuntimeError("Discord returned an unexpected messages response")
    return message_id


def fetch_message(channel_id: str, message_id: str, token: str) -> dict[str, Any]:
    if not is_discord_id(channel_id) or not is_discord_id(message_id):
        raise ValueError("Unexpected Discord message id")
    message = _request_json(Request(
        f"{DISCORD_API}/channels/{channel_id}/messages/{message_id}",
        headers=_discord_headers(token),
    ))
    if not isinstance(message, dict) or message.get("id") != message_id:
        raise RuntimeError("Discord returned an unexpected message response")
    return message


def start_public_thread(channel_id: str, message_id: str, token: str, name: str) -> dict[str, Any]:
    """Start one public thread on a message. Reuse the thread if that message already has one."""
    if not is_discord_id(channel_id) or not is_discord_id(message_id):
        raise ValueError("Thread requires Discord channel and message IDs")
    cleaned = re.sub(r"\s+", " ", name).strip()[:100] or "pr review"
    last_error: RuntimeError | None = None
    for duration in (1440, 60):
        try:
            response = _request_json(Request(
                f"{DISCORD_API}/channels/{channel_id}/messages/{message_id}/threads",
                data=json.dumps({"name": cleaned, "auto_archive_duration": duration}).encode("utf-8"),
                method="POST",
                headers={**_discord_headers(token), "Content-Type": "application/json"},
            ))
        except RuntimeError as error:
            last_error = error
            if "HTTP 400" not in str(error):
                raise
            existing = _thread_on_message(channel_id, message_id, token)
            if existing:
                return existing
            continue
        if not isinstance(response, dict) or not is_discord_id(response.get("id")):
            raise RuntimeError("Discord did not return a thread")
        return response
    if last_error:
        raise last_error
    raise RuntimeError("Discord did not return a thread")


def _thread_on_message(channel_id: str, message_id: str, token: str) -> dict[str, Any] | None:
    try:
        message = fetch_message(channel_id, message_id, token)
    except (RuntimeError, TransientAPIError, ValueError):
        return None
    thread = message.get("thread")
    if isinstance(thread, dict) and is_discord_id(thread.get("id")):
        return thread
    return None


def fetch_channel(channel_id: str, token: str) -> dict[str, Any]:
    channel = _request_json(Request(
        f"{DISCORD_API}/channels/{channel_id}",
        headers=_discord_headers(token),
    ))
    if not isinstance(channel, dict) or channel.get("id") != channel_id:
        raise RuntimeError("Discord returned an unexpected channel response")
    return channel


def fetch_guild_channels(guild_id: str, token: str) -> list[dict[str, Any]]:
    """List channels in a Discord guild over REST."""
    if not is_discord_id(guild_id):
        raise ValueError("Unexpected Discord guild ID")
    channels = _request_json(Request(
        f"{DISCORD_API}/guilds/{guild_id}/channels",
        headers=_discord_headers(token),
    ))
    if not isinstance(channels, list):
        return []
    return [item for item in channels if isinstance(item, dict)]


def fetch_guild_members(
    guild_id: str,
    token: str,
    *,
    max_pages: int = 50,
) -> list[dict[str, Any]]:
    """List guild members over Discord REST (1,000 per page).

    Works whenever Server Members Intent is enabled in the Discord Developer
    Portal, even if the gateway client connected without intents.members or the
    interaction arrived over HTTP with no gateway guild object.
    """
    if not is_discord_id(guild_id):
        raise ValueError("Unexpected Discord guild ID")
    collected: list[dict[str, Any]] = []
    after = "0"
    for _page in range(max_pages):
        params = urlencode({"limit": 1000, "after": after})
        batch = _request_json(Request(
            f"{DISCORD_API}/guilds/{guild_id}/members?{params}",
            headers=_discord_headers(token),
        ))
        if not isinstance(batch, list) or not batch:
            break
        last_id = None
        for item in batch:
            if not isinstance(item, dict):
                continue
            collected.append(item)
            user = item.get("user") if isinstance(item.get("user"), dict) else {}
            uid = user.get("id")
            if isinstance(uid, str) and is_discord_id(uid):
                last_id = uid
        if len(batch) < 1000 or not last_id or last_id == after:
            break
        after = last_id
    return collected


def search_guild_members(
    guild_id: str,
    query: str,
    token: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search guild members by username/nickname prefix over REST (no members intent needed)."""
    if not is_discord_id(guild_id):
        raise ValueError("Unexpected Discord guild ID")
    cleaned = (query or "").strip()
    if not cleaned:
        return []
    bounded = max(1, min(int(limit), 100))
    params = urlencode({"query": cleaned[:32], "limit": bounded})
    payload = _request_json(Request(
        f"{DISCORD_API}/guilds/{guild_id}/members/search?{params}",
        headers=_discord_headers(token),
    ))
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def search_guild_messages(
    guild_id: str,
    content: str = "",
    token: str = "",
    *,
    author_id: str | None = None,
    min_id: str | None = None,
    offset: int = 0,
    max_202_retries: int = 6,
    timeout: float = 15.0,
    sleep_fn=time.sleep,
) -> dict[str, Any]:
    """Search guild messages via GET /guilds/{id}/messages/search.

    Supports filtering by ``content``, ``author_id``, and/or ``min_id`` (at least
    one of ``content`` or ``author_id`` is required). Handles HTTP 202 (Discord
    search index still building) with ``retry_after`` and bounds offset to 9975.
    """
    if not is_discord_id(guild_id):
        raise ValueError("Unexpected Discord guild ID")
    cleaned = (content or "").strip()
    cleaned_author = (author_id or "").strip() if author_id is not None else ""
    if cleaned_author and not is_discord_id(cleaned_author):
        raise ValueError("Unexpected Discord author ID")
    cleaned_min = (min_id or "").strip() if min_id is not None else ""
    if cleaned_min and not is_discord_id(cleaned_min):
        raise ValueError("Unexpected Discord min_id")
    if not cleaned and not cleaned_author:
        raise ValueError("Search content or author_id must not be empty")
    if not isinstance(offset, int) or offset < 0 or offset > 9975:
        raise ValueError("Search offset must be between 0 and 9975")
    query_params: dict[str, Any] = {"offset": offset}
    if cleaned:
        query_params["content"] = cleaned
    if cleaned_author:
        query_params["author_id"] = cleaned_author
    if cleaned_min:
        query_params["min_id"] = cleaned_min
    params = urlencode(query_params)
    url = f"{DISCORD_API}/guilds/{guild_id}/messages/search?{params}"
    for attempt in range(max(1, max_202_retries) + 1):
        status, headers, body = _http(
            Request(url, headers=_discord_headers(token)),
            attempts=2,
            timeout=timeout,
            max_retry_delay=15.0,
        )
        payload = json.loads(body) if body else {}
        if status == 202:
            if attempt >= max_202_retries:
                raise TransientAPIError("Discord guild search index is still building (HTTP 202)")
            retry_after = 2.0
            if isinstance(payload, dict) and isinstance(payload.get("retry_after"), (int, float)):
                retry_after = max(0.0, min(float(payload["retry_after"]), 15.0))
            elif headers is not None and headers.get("Retry-After") is not None:
                try:
                    retry_after = max(0.0, min(float(headers.get("Retry-After")), 15.0))
                except (TypeError, ValueError):
                    retry_after = 2.0
            sleep_fn(retry_after)
            continue
        if status != 200:
            raise RuntimeError(f"discord.com returned HTTP {status}")
        if not isinstance(payload, dict):
            raise RuntimeError("Discord returned an unexpected guild message search response")
        return payload
    raise TransientAPIError("Discord guild search index did not become ready")


def add_reaction(channel_id: str, message_id: str, emoji: str, token: str) -> None:
    """PUT one reaction on a message. Callers log and swallow Discord errors."""
    if not is_discord_id(channel_id) or not is_discord_id(message_id):
        raise ValueError("Unexpected Discord message id")
    if not isinstance(emoji, str) or not emoji or any(ch in emoji for ch in "/\\"):
        raise ValueError("Unexpected reaction")
    status, _headers, _body = _http(Request(
        f"{DISCORD_API}/channels/{channel_id}/messages/{message_id}/reactions/{quote(emoji, safe='')}/@me",
        data=b"",
        method="PUT",
        headers=_discord_headers(token),
    ))
    if status not in {200, 204}:
        raise RuntimeError(f"discord.com returned HTTP {status}")


def post_message(
    channel_id: str,
    token: str,
    message: dict[str, Any],
    *,
    mention_roles: list[str] | None = None,
) -> dict[str, Any]:
    mentions = message.get("allowed_mentions") or {}
    users = mentions.get("users", [])
    roles = list(mention_roles or [])
    if (
        not isinstance(users, list)
        or len(users) > 100
        or any(not is_discord_id(user_id) for user_id in users)
    ):
        raise ValueError("Only explicit Discord user IDs may be mentioned")
    # Roles stuffed into the message are ignored. Callers pass mention_roles.
    if len(roles) > 1 or any(not is_discord_id(role_id) for role_id in roles):
        raise ValueError("Only one explicit Discord role ID may be mentioned")
    allowed: dict[str, Any] = {"parse": [], "users": users}
    if roles:
        allowed["roles"] = roles
    message = {**message, "allowed_mentions": allowed}
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
