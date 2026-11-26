"""Casual lowercase copy for role announcements, invites, and /my-prs."""

from .config import is_discord_id


def _article(phrase: str) -> str:
    return "an" if phrase[:1].lower() in "aeiou" else "a"


def _safe(text: str) -> str:
    """Keep operator-configured names from becoming mentions or newlines."""
    return text.replace("\n", " ").replace("\r", " ").replace("@", "@\u200b").replace("<", "‹")


def role_phrase(role_names: list[str]) -> str:
    parts = []
    for name in role_names:
        lowered = _safe(name.strip().lower())
        if not lowered:
            continue
        parts.append(f"{_article(lowered)} {lowered}")
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def announcement_text(discord_id: str, role_names: list[str], pr_number: int | None = None) -> str:
    """One lowercase line for #shipped. Mentions only the linked Discord user."""
    if not is_discord_id(discord_id):
        raise ValueError("Announcement requires a Discord user id")
    mention = f"<@{discord_id}>"
    roles = role_phrase(role_names)
    if pr_number and roles:
        return f"{mention} just shipped #{pr_number} and became {roles} 🎉"
    if roles:
        return f"{mention} just became {roles} 🎉"
    if pr_number:
        return f"{mention} just shipped #{pr_number} 🎉"
    raise ValueError("Nothing to announce")


def invite_comment(login: str, invite_url: str) -> str:
    return (
        f"hey @{login} - this pr just merged. link your github in the litellm discord "
        f"with `/link-github` to pick up a contributor role: {invite_url}"
    )


def _title(text: str) -> str:
    cleaned = _safe(text).strip()
    if len(cleaned) > 80:
        return cleaned[:77] + "..."
    return cleaned or "untitled"


def my_prs_text(
    login: str,
    open_prs: list[dict],
    open_total: int | None,
    merged_count: int,
    role_names: list[str],
    *,
    open_error: bool = False,
) -> str:
    """Ephemeral /my-prs reply. ``open_prs`` items need number, title, and review_status."""
    lines = [f"hey {_safe(login)}, here's your litellm pr snapshot", ""]
    if open_error:
        lines.append("open: couldn't reach github just now")
    elif not open_prs:
        lines.append("open: none right now")
    else:
        lines.append("open:")
        shown = open_prs[:8]
        for pull in shown:
            lines.append(
                f"- #{pull['number']} {_title(str(pull.get('title') or ''))} - {pull.get('review_status') or 'review status unknown'}"
            )
        extra = max((open_total if open_total is not None else len(open_prs)) - len(shown), 0)
        if extra:
            lines.append(f"- and {extra} more open")
    lines.append("")
    lines.append(f"merged: {merged_count} recorded")
    if role_names:
        lines.append("roles: " + ", ".join(_safe(name.lower()) for name in role_names if name.strip()))
    else:
        lines.append("roles: none yet")
    text = "\n".join(lines)
    if len(text) > 1900:
        text = text[:1897] + "..."
    return text


def review_status(pull: dict, reviews: list[dict]) -> str:
    """Summarize the latest review from each reviewer.

    Dismissed reviews clear that reviewer. Comments alone do not decide status.
    """
    if pull.get("draft"):
        return "draft"
    latest: dict[object, str] = {}
    for review in reviews:
        user = review.get("user") or {}
        user_id = user.get("id")
        state = review.get("state")
        if user_id is None or not isinstance(state, str):
            continue
        if state == "DISMISSED":
            latest.pop(user_id, None)
        elif state in {"APPROVED", "CHANGES_REQUESTED"}:
            latest[user_id] = state
    states = set(latest.values())
    if "CHANGES_REQUESTED" in states:
        return "changes requested"
    if "APPROVED" in states:
        return "approved"
    return "waiting on review"
