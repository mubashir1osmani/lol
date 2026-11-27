"""Ask Bugbot and Greptile to review the latest commit, as the GitHub user.

Only an allowlist of comments is posted. Veria has no documented re-run
comment, so it is not triggered. A missing actor token is explained in Discord
instead of posting. ``REVIEW_DRY_RUN`` logs the comment and does not post it.
"""

import logging
from datetime import datetime, timedelta

from .api import post_issue_comment
from .review_github import gather_snapshot
from .reviews import assess_gate
from .schedule import UTC, iso

# https://cursor.com/docs/bugbot — top-level comment, from a user account.
# https://www.greptile.com/docs/code-review/developer-essentials — `@greptileai`.
DEFAULT_TRIGGERS = (
    ("cursor[bot]", "bugbot run"),
    ("greptile-apps[bot]", "@greptileai"),
)
_KNOWN = {
    "bugbot run": "cursor[bot]",
    "cursor review": "cursor[bot]",
    "@greptileai": "greptile-apps[bot]",
    "@greptile": "greptile-apps[bot]",
}


def parse_trigger_comments(raw: str | None) -> tuple[tuple[str, str], ...]:
    """``login=comment`` pairs, or a known comment string. Unset uses the defaults."""
    if raw is None or not str(raw).strip():
        return DEFAULT_TRIGGERS
    chosen = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            raise ValueError("REVIEW_TRIGGER_COMMENTS has an empty entry")
        if "=" in text:
            login, body = text.split("=", 1)
            login, body = login.strip().lower(), " ".join(body.strip().split())
            if not login or not body or len(body) > 80:
                raise ValueError("REVIEW_TRIGGER_COMMENTS entries look like cursor[bot]=bugbot run")
            chosen.append((login, body))
            continue
        body = " ".join(text.split())
        login = _KNOWN.get(body.lower()) or _KNOWN.get(body)
        if not login:
            raise ValueError(
                f"REVIEW_TRIGGER_COMMENTS entry {text!r} is not a known trigger. "
                "Use bugbot run, @greptileai, or login=comment."
            )
        chosen.append((login, body if body.lower() not in _KNOWN else _canonical(body)))
    if not chosen:
        raise ValueError("REVIEW_TRIGGER_COMMENTS is empty")
    return tuple(chosen)


def _canonical(body: str) -> str:
    lowered = body.lower()
    for comment, _login in _KNOWN.items():
        if comment == lowered or comment == body:
            return comment
    return " ".join(body.split())


def allowlist(triggers: tuple[tuple[str, str], ...]) -> frozenset[str]:
    return frozenset(body for _login, body in triggers)


def plan_triggers(gate, triggers: tuple[tuple[str, str], ...], posted: set[str]) -> list[tuple[str, str]]:
    """Triggers whose bot has not reviewed this head sha, and that were not posted yet."""
    by_login = {bot.login.lower(): bot for bot in gate.bots}
    planned = []
    for login, body in triggers:
        bot = by_login.get(login.lower())
        if bot is None or bot.reviewed_head:
            continue
        if body in posted:
            continue
        planned.append((login, body))
    return planned


def trigger_note(planned: list[tuple[str, str]], *, mode: str) -> str:
    """mode is posted, dry-run, or manual."""
    if not planned:
        return ""
    lines = ["i'll get the missing reviews onto the latest commit."]
    for login, body in planned:
        if mode == "dry-run":
            lines.append(f"- dry run: would comment `{body}` for {login}")
        elif mode == "manual":
            lines.append(f"- comment `{body}` on the pull request for {login}")
        else:
            lines.append(f"- commented `{body}` for {login}")
    if mode == "manual":
        lines.append("i don't have GITHUB_ACTOR_TOKEN, so i can't post those myself.")
    return "\n".join(lines)


def posted_bodies(store, repository: str, number: int, head_sha: str, triggers) -> set[str]:
    return {
        body for _login, body in triggers
        if store.trigger_posted(repository, number, head_sha, body)
    }


def apply_triggers(config, store, number: int, thread_post) -> list[tuple[str, str]]:
    """Post missing triggers or describe them. ``thread_post`` receives the Discord note.

    Returns the triggers that were posted, logged, or handed to the contributor.
    """
    token = getattr(config, "github_actor_token", None)
    dry = bool(getattr(config, "review_dry_run", False))
    if not token and not dry:
        return []
    triggers = config.review_trigger_comments or DEFAULT_TRIGGERS
    try:
        snapshot = gather_snapshot(config.github_repository, number, config.effective_github_token)
    except Exception as error:
        logging.warning("Could not decide review triggers for #%s: %s", number, error)
        return []
    if not snapshot.head_sha:
        return []
    gate = assess_gate(
        bots=tuple(config.review_bot_logins),
        head_sha=snapshot.head_sha,
        draft=snapshot.draft,
        merged=snapshot.merged,
        state=snapshot.state,
        mergeable=snapshot.mergeable,
        mergeable_state=snapshot.mergeable_state,
        reviews=snapshot.reviews,
        comments=snapshot.comments,
        threads=snapshot.threads,
        check_runs=snapshot.check_runs,
        combined_state=snapshot.combined_state,
    )
    already = posted_bodies(store, config.github_repository, number, snapshot.head_sha, triggers)
    planned = plan_triggers(gate, triggers, already)
    if not planned:
        return []
    since = iso(datetime.now(UTC) - timedelta(hours=1))
    if store.triggers_since(config.github_repository, number, since) >= config.review_trigger_hourly_limit:
        thread_post("not commenting again yet. this pull request hit the trigger rate limit.")
        return []
    if dry or not token:
        mode = "dry-run" if dry else "manual"
        note = trigger_note(planned, mode=mode)
        logging.info("Review triggers for #%s (%s): %s", number, mode, note.replace("\n", " | "))
        thread_post(note)
        return planned
    posted = []
    for login, body in planned:
        try:
            post_issue_comment(config.github_repository, number, token, body)
        except Exception as error:
            logging.warning("Could not comment `%s` on #%s: %s", body, number, error)
            continue
        store.save_trigger(config.github_repository, number, snapshot.head_sha, body, iso(datetime.now(UTC)))
        posted.append((login, body))
    if posted:
        thread_post(trigger_note(posted, mode="posted"))
    return posted


def comment_is_allowed(body: str, triggers: tuple[tuple[str, str], ...], *, summary: bool) -> str | None:
    """Return the normalized comment, or None when it is not allowed."""
    text = " ".join((body or "").split())
    if text in allowlist(triggers):
        return text
    if summary and text and len(text) <= 400 and not text.lower().startswith(("@", "bugbot", "cursor review")):
        return text
    return None
