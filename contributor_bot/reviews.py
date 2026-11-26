"""Pure PR-review detection, gate, and Discord formatting.

Network and the LiteLLM SDK stay in review_github, review_llm, and review_job.
Nothing here approves, merges, or comments on GitHub.
"""

import re
from dataclasses import dataclass

from .poller import extract_pr_numbers
from .schedule import parse_time

# Logins observed on recent BerriAI/litellm pull requests (for example #54722):
# cursor[bot] is Cursor Bugbot (reviews include <!-- BUGBOT_REVIEW -->),
# greptile-apps[bot] is Greptile, and veria-ai[bot] is Veria.
DEFAULT_REVIEW_BOTS = ("cursor[bot]", "greptile-apps[bot]", "veria-ai[bot]")

DISCORD_MESSAGE_LIMIT = 2000
MAX_REVIEW_POSTS = 8
MAX_PRS_PER_MESSAGE = 3
MAX_FINDINGS_PER_BOT = 5
_STALE_MARKER = "BUGBOT_REVIEW_STALE"
_ACTIVE_REVIEW = {"COMMENTED", "APPROVED", "CHANGES_REQUESTED"}
_PASSING_CONCLUSIONS = {"success", "neutral", "skipped"}
_ASK = re.compile(
    r"(?i)\b(reviews?|reviewer|ptal|lgtm|take a look|have a look|needs? a look|ready for review)\b"
)


def asks_for_review(text: str) -> bool:
    """True when the message is asking someone to look at a pull request."""
    return _ASK.search(text or "") is not None


def should_review_message(message: dict, repository: str, *, require_ask: bool) -> list[int]:
    """Pull request numbers worth a thread, or an empty list.

    Bot and webhook messages are ignored so the bot cannot review its own posts.
    ``require_ask`` still needs a BerriAI/litellm pull URL; the ask is extra.
    """
    author = message.get("author") or {}
    if author.get("bot") or message.get("webhook_id"):
        return []
    if message.get("type") not in (None, 0, 19):
        return []
    numbers = extract_pr_numbers(message, repository)
    if not numbers:
        return []
    if require_ask and not asks_for_review(message.get("content") or ""):
        return []
    return numbers[:MAX_PRS_PER_MESSAGE]


def cooldown_remaining(record: dict | None, cooldown_seconds: int, now) -> int:
    """Seconds left before the same PR can be reviewed again. Zero means go ahead."""
    if not record or not record.get("last_reviewed_at"):
        return 0
    try:
        reviewed = parse_time(record["last_reviewed_at"])
    except (TypeError, ValueError):
        return 0
    remaining = cooldown_seconds - (now - reviewed).total_seconds()
    if remaining <= 0:
        return 0
    return int(remaining)


def recheck_number(event: str, payload: dict) -> int | None:
    """Pull request number for an optional webhook re-check, else None."""
    if not isinstance(payload, dict):
        return None
    if event == "pull_request" and payload.get("action") in {"synchronize", "ready_for_review"}:
        pass
    elif event == "pull_request_review" and payload.get("action") == "submitted":
        pass
    else:
        return None
    pull = payload.get("pull_request")
    number = pull.get("number") if isinstance(pull, dict) else None
    if isinstance(number, int) and number > 0:
        return number
    return None


def discord_length(text: str) -> int:
    """Discord counts UTF-16 code units, not Python code points."""
    return len(text.encode("utf-16-le")) // 2


def _prefix_within(text: str, limit: int) -> int:
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if discord_length(text[:mid]) <= limit:
            low = mid
        else:
            high = mid - 1
    return low


def split_discord_messages(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split ``text`` so every part fits in one Discord message."""
    if limit < 1:
        raise ValueError("Discord message limit must be positive")
    remaining = (text or "").strip()
    if not remaining:
        return []
    if discord_length(remaining) <= limit:
        return [remaining]
    parts: list[str] = []
    while remaining:
        if discord_length(remaining) <= limit:
            parts.append(remaining)
            break
        cut = _prefix_within(remaining, limit)
        if cut <= 0:
            cut = 1
        newline = remaining.rfind("\n", 0, cut)
        if newline >= max(cut // 2, 1):
            cut = newline
        chunk = remaining[:cut].rstrip("\n")
        if not chunk:
            cut = max(_prefix_within(remaining, limit), 1)
            chunk = remaining[:cut]
        parts.append(chunk)
        remaining = remaining[cut:].lstrip("\n")
    return [part for part in parts if part.strip()]


def discord_safe(text: str) -> str:
    """Stop model text and comment bodies from pinging the room."""
    cleaned = (text or "").replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
    return re.sub(r"<@[&!]?", "‹@", cleaned)


def plain(text: str, limit: int = 140) -> str:
    cleaned = discord_safe(text).replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > limit:
        return cleaned[: limit - 3] + "..."
    return cleaned


def summarize_comment(body: str) -> str:
    """First readable line of a bot review comment."""
    text = re.sub(r"<!--.*?-->", " ", body or "", flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("**", " ").replace("`", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(?:#+\s*)+", "", text)
    text = plain(text, 140)
    return text or "unresolved comment"


def normalize_thread(raw: dict) -> dict | None:
    """GraphQL or REST-shaped review thread → a stable dict."""
    if not isinstance(raw, dict):
        return None
    comments = raw.get("comments") or []
    if isinstance(comments, dict):
        comments = comments.get("nodes") or []
    normalized = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        login = comment.get("login") if isinstance(comment.get("login"), str) else ""
        if not login:
            author = comment.get("author") or comment.get("user") or {}
            login = author.get("login") if isinstance(author, dict) else ""
        commit = comment.get("commit") if isinstance(comment.get("commit"), dict) else {}
        commit_id = comment.get("commit_id") or commit.get("oid") or ""
        normalized.append({
            "login": login.lower() if isinstance(login, str) else "",
            "body": comment.get("body") if isinstance(comment.get("body"), str) else "",
            "commit_id": commit_id if isinstance(commit_id, str) else "",
        })
    if not normalized:
        return None
    path = raw.get("path") if isinstance(raw.get("path"), str) else ""
    return {
        "resolved": bool(raw.get("isResolved") if "isResolved" in raw else raw.get("resolved")),
        "outdated": bool(raw.get("isOutdated") if "isOutdated" in raw else raw.get("outdated")),
        "path": path,
        "comments": normalized,
    }


def _login(obj: dict) -> str:
    user = obj.get("user") or obj.get("author") or {}
    login = user.get("login") if isinstance(user, dict) else ""
    return login.lower() if isinstance(login, str) else ""


def _bot_set(bots: tuple[str, ...]) -> set[str]:
    return {bot.lower() for bot in bots}


@dataclass(frozen=True)
class Finding:
    bot: str
    path: str
    summary: str


@dataclass(frozen=True)
class BotGate:
    login: str
    reviewed_head: bool
    ever_reviewed: bool
    findings: tuple[Finding, ...] = ()


@dataclass(frozen=True)
class Gate:
    ready: bool
    reasons: tuple[str, ...]
    bots: tuple[BotGate, ...]
    ci: str


def _covers_head_review(review: dict, head_sha: str) -> bool:
    if review.get("commit_id") != head_sha:
        return False
    if review.get("state") not in _ACTIVE_REVIEW:
        return False
    return _STALE_MARKER not in (review.get("body") or "")


def _activity(login: str, reviews: list[dict], comments: list[dict], head_sha: str) -> tuple[bool, bool]:
    ever = False
    on_head = False
    for review in reviews:
        if _login(review) != login:
            continue
        ever = True
        if _covers_head_review(review, head_sha):
            on_head = True
    for comment in comments:
        if _login(comment) != login:
            continue
        ever = True
        body = comment.get("body") or ""
        if comment.get("commit_id") == head_sha and _STALE_MARKER not in body:
            on_head = True
    return ever, on_head


def findings_from_threads(threads: list[dict], bots: tuple[str, ...]) -> list[Finding]:
    """Unresolved, current review threads opened by a configured bot."""
    wanted = _bot_set(bots)
    found: list[Finding] = []
    for raw in threads:
        thread = normalize_thread(raw)
        if not thread:
            continue
        opener = thread["comments"][0]["login"]
        if opener not in wanted:
            continue
        if thread["resolved"] or thread["outdated"]:
            continue
        body = thread["comments"][0]["body"]
        if _STALE_MARKER in body:
            continue
        found.append(Finding(opener, thread.get("path") or "", summarize_comment(body)))
    return found


def findings_from_comments(comments: list[dict], bots: tuple[str, ...], head_sha: str) -> list[Finding]:
    """Fallback when GraphQL resolution is unavailable: top-level notes on the head commit."""
    wanted = _bot_set(bots)
    found: list[Finding] = []
    for comment in comments:
        login = _login(comment)
        if login not in wanted or comment.get("in_reply_to_id"):
            continue
        if comment.get("commit_id") != head_sha:
            continue
        body = comment.get("body") or ""
        if _STALE_MARKER in body:
            continue
        path = comment.get("path") if isinstance(comment.get("path"), str) else ""
        found.append(Finding(login, path, summarize_comment(body)))
    return found


def _worse(left: str, right: str) -> str:
    rank = {"passing": 0, "stale": 1, "pending": 2, "failing": 3}
    return left if rank[left] >= rank[right] else right


def _check_bucket(run: dict) -> str:
    status = str(run.get("status") or "").lower()
    conclusion = str(run.get("conclusion") or "").lower()
    if status != "completed":
        return "pending"
    if conclusion in _PASSING_CONCLUSIONS:
        return "passing"
    if conclusion == "stale":
        return "stale"
    return "failing"


def summarize_ci(check_runs: list[dict] | None, combined_state: str | None) -> tuple[str, tuple[str, ...]]:
    """Return ``(kind, names)``. ``kind`` is green, failing, pending, stale, missing, or unknown."""
    if check_runs is None and combined_state is None:
        return "unknown", ()
    buckets: dict[str, str] = {}
    order: list[str] = []
    for run in check_runs or []:
        if not isinstance(run, dict):
            continue
        name = str(run.get("name") or "check").strip() or "check"
        bucket = _check_bucket(run)
        if name not in buckets:
            order.append(name)
            buckets[name] = bucket
        else:
            buckets[name] = _worse(buckets[name], bucket)
    failing = [name for name in order if buckets[name] == "failing"]
    pending = [name for name in order if buckets[name] == "pending"]
    stale = [name for name in order if buckets[name] == "stale"]
    if combined_state in {"failure", "error"}:
        failing.append("status")
    elif combined_state == "pending":
        pending.append("status")
    if failing:
        return "failing", tuple(failing)
    if pending:
        return "pending", tuple(pending)
    if stale:
        return "stale", tuple(stale)
    if check_runs is None:
        return ("green", ()) if combined_state == "success" else ("unknown", ())
    if not check_runs and combined_state is None:
        return "missing", ()
    return "green", ()


def _ci_reason(kind: str, names: tuple[str, ...]) -> str | None:
    shown = ", ".join(names[:5])
    extra = len(names) - 5
    if extra > 0:
        shown += f" (+{extra} more)"
    if kind == "failing":
        return f"ci failing: {shown}"
    if kind == "pending":
        return f"ci still running: {shown}"
    if kind == "stale":
        return f"ci stale: {shown}"
    if kind == "missing":
        return "ci has not reported yet"
    if kind == "unknown":
        return "couldn't read ci checks"
    return None


def _bot_reasons(bot: BotGate) -> list[str]:
    reasons = []
    if not bot.ever_reviewed:
        reasons.append(f"{bot.login} has not reviewed this pr")
    elif not bot.reviewed_head:
        reasons.append(f"{bot.login} has not re-reviewed the latest commit")
    shown = bot.findings[:MAX_FINDINGS_PER_BOT]
    for finding in shown:
        where = f" on {finding.path}" if finding.path else ""
        reasons.append(f"{bot.login}: unresolved{where} - {finding.summary}")
    extra = len(bot.findings) - len(shown)
    if extra:
        reasons.append(f"{bot.login}: {extra} more unresolved comments")
    return reasons


def assess_gate(
    *,
    bots: tuple[str, ...],
    head_sha: str,
    draft: bool,
    merged: bool,
    state: str,
    mergeable: bool | None,
    mergeable_state: str | None,
    reviews: list[dict],
    comments: list[dict],
    threads: list[dict] | None,
    check_runs: list[dict] | None,
    combined_state: str | None,
) -> Gate:
    """Decide ready-for-maintainer vs needs-work from already-fetched GitHub data.

    ``threads is None`` means GraphQL resolution could not be read. The gate
    then stays needs-work and treats latest-commit comments as still open.
    """
    if merged:
        return Gate(False, ("already merged",), (), "skipped")
    if state != "open":
        return Gate(False, ("pr is closed",), (), "skipped")

    wanted = tuple(bot.lower() for bot in bots)
    if threads is None:
        findings = findings_from_comments(comments, wanted, head_sha)
        resolution_known = False
    else:
        findings = findings_from_threads(threads, wanted)
        resolution_known = True
    by_bot: dict[str, list[Finding]] = {bot: [] for bot in wanted}
    for finding in findings:
        if finding.bot in by_bot:
            by_bot[finding.bot].append(finding)
    bot_gates = []
    for login in wanted:
        ever, on_head = _activity(login, reviews, comments, head_sha)
        bot_gates.append(BotGate(login, on_head, ever, tuple(by_bot[login])))

    reasons: list[str] = []
    if draft:
        reasons.append("still a draft")
    if mergeable_state == "dirty" or mergeable is False:
        reasons.append("merge conflicts")
    elif mergeable is not True:
        reasons.append("mergeability isn't known yet")
    ci_kind, ci_names = summarize_ci(check_runs, combined_state)
    ci_reason = _ci_reason(ci_kind, ci_names)
    if ci_reason:
        reasons.append(ci_reason)
    if not resolution_known:
        reasons.append("couldn't read review-thread resolution from github")
    for bot in bot_gates:
        reasons.extend(_bot_reasons(bot))
    return Gate(not reasons, tuple(reasons), tuple(bot_gates), ci_kind)


def short_sha(sha: str) -> str:
    if re.fullmatch(r"[0-9a-f]{7,40}", sha or ""):
        return sha[:7]
    return "unknown"


def render_status(gate: Gate, *, number: int, title: str, url: str, head_sha: str) -> str:
    lines = [
        "ready for maintainer ✅" if gate.ready else "needs work",
        f"#{number} {plain(title, 180)}",
        url,
        f"commit {short_sha(head_sha)}",
        "",
    ]
    if gate.ready:
        for bot in gate.bots:
            lines.append(f"- {bot.login}: re-reviewed, no outstanding issues")
        lines.append("ci: green")
        lines.append("mergeable, not a draft")
    else:
        lines.append("fix these before a maintainer looks:")
        for reason in gate.reasons:
            lines.append(f"- {discord_safe(reason)}")
    return "\n".join(lines)


def render_written(review: dict | None, error: str | None = None) -> str:
    if error:
        return f"written review: {discord_safe(error)}"
    if not review:
        return ""
    lines = ["written review:", discord_safe(str(review.get("summary") or "no written summary"))]
    for label, key in (("risks", "risks"), ("missing tests", "missing_tests"), ("checklist", "checklist")):
        items = [discord_safe(str(item)).strip() for item in (review.get(key) or []) if str(item).strip()]
        if not items:
            continue
        lines.append("")
        lines.append(f"{label}:")
        for item in items[:5]:
            lines.append(f"- {item}")
    omitted = [plain(str(name), 80) for name in (review.get("omitted") or []) if str(name).strip()]
    if omitted:
        lines.append("")
        suffix = "" if len(omitted) <= 8 else f" (+{len(omitted) - 8} more)"
        lines.append("didn't read every file: " + ", ".join(omitted[:8]) + suffix)
    return "\n".join(lines)


def review_chunks(status: str, written: str, *, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    chunks = split_discord_messages(status, limit)
    if written.strip():
        chunks.extend(split_discord_messages(written, limit))
    if len(chunks) > MAX_REVIEW_POSTS:
        chunks = chunks[: MAX_REVIEW_POSTS - 1] + ["review truncated to fit discord."]
    return chunks


def tag_line(user_id: str | None, role_id: str | None, *, ready: bool) -> str | None:
    """Ping the linked contributor, and the maintainer role only when the gate is ready."""
    parts = []
    if user_id:
        parts.append(f"<@{user_id}>")
    if ready and role_id:
        parts.append(f"<@&{role_id}>")
    if not parts:
        return None
    return " ".join(parts)
