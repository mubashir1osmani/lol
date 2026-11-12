import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import urlsplit

from .api import TransientAPIError
from .config import Config, is_discord_id
from .state import load_state, save_state


def extract_pr_numbers(message: dict[str, Any], repository: str) -> list[int]:
    text = message.get("content") or ""
    # Link previews can carry the URL even when the surrounding text does not.
    for embed in message.get("embeds") or []:
        text += " " + (embed.get("url") or "")
    numbers = set()
    for candidate in re.findall(r"https?://[^\s<>]+", text, re.IGNORECASE):
        try:
            url = urlsplit(candidate.rstrip(".,;!?:)]}'\""))
        except ValueError:
            continue
        if url.hostname not in {"github.com", "www.github.com"} or url.username or url.password:
            continue
        match = re.fullmatch(r"/([^/]+/[^/]+)/pull/([1-9][0-9]*)(?:/.*)?", url.path)
        if match and match[1].lower() == repository.lower():
            numbers.add(int(match[2]))
    return sorted(numbers)


def _thank_you(user_ids: list[str], pr_count: int, channel_id: str) -> str:
    noun = "PR" if pr_count == 1 else "PRs"
    return (
        "🎉 **Thank you for contributing to LiteLLM!**\n\n"
        f"These contributors had {pr_count} {noun} shared in <#{channel_id}> "
        "merged on GitHub:\n\n"
        + "\n".join(f"- <@{user_id}>" for user_id in user_ids)
        + "\n\nYour contributions make LiteLLM better for everyone. 💚"
    )


def build_announcements(config: Config, state: dict[str, Any]) -> list[dict[str, Any]]:
    contributors: dict[str, list[str]] = {}
    for number, pull in state["pull_requests"].items():
        if not pull["merged"] or pull["announced"]:
            continue
        user_id = config.github_discord_map.get(pull["github_author"].lower())
        if not user_id and config.contributor_attribution == "message_author":
            user_id = pull["discord_id"]
        if not user_id:
            logging.warning("No Discord mapping for GitHub contributor %s (PR #%s)", pull["github_author"], number)
            continue
        contributors.setdefault(user_id, []).append(number)

    batches: list[dict[str, Any]] = []

    def make_batch(user_ids: list[str]) -> dict[str, Any]:
        numbers = sorted((number for user_id in user_ids for number in contributors[user_id]), key=int)
        # A saved outbox plus Discord's nonce deduplication makes retries of
        # the same announcement safe during Discord's recent-message window.
        identity = json.dumps([config.scope, user_ids, numbers], sort_keys=True)
        nonce = hashlib.sha256(identity.encode()).hexdigest()[:24]
        return {
            "discord_ids": list(user_ids),
            "pr_numbers": numbers,
            "payload": {
                "content": _thank_you(user_ids, len(numbers), config.contributions_channel_id),
                "allowed_mentions": {"parse": [], "users": list(user_ids)},
                "nonce": nonce,
                "enforce_nonce": True,
            },
        }

    batch: list[str] = []
    for user_id in sorted(contributors, key=int):
        candidate = make_batch(batch + [user_id])
        length = len(candidate["payload"]["content"].encode("utf-16-le")) // 2
        if batch and (length > 2000 or len(batch) >= 100):
            batches.append(make_batch(batch))
            batch = []
        batch.append(user_id)
    if batch:
        batches.append(make_batch(batch))
    return batches


def _flush_outbox(config: Config, state: dict[str, Any], post: Callable) -> None:
    while state["outbox"]:
        batch = state["outbox"][0]
        post(config.announcements_channel_id, config.discord_token, batch["payload"])
        for number in batch["pr_numbers"]:
            state["pull_requests"][number]["announced"] = True
        state["contributor_ids"] = sorted(
            set(state["contributor_ids"]) | set(batch["discord_ids"]), key=int
        )
        state["outbox"].pop(0)
        save_state(config.state_file, state)
        logging.info("Thanked Discord contributors: %s", ", ".join(batch["discord_ids"]))


def poll_once(
    config: Config,
    fetch_messages: Callable[[str, str, str | None], Iterable[dict[str, Any]]],
    fetch_pull: Callable[[str, int, str | None], dict[str, Any]],
    post: Callable[[str, str, dict[str, Any]], None],
) -> None:
    state = load_state(config)
    # Finish interrupted announcements before advancing channel history.
    _flush_outbox(config, state, post)
    messages = sorted(
        fetch_messages(config.contributions_channel_id, config.discord_token, state["last_message_id"]),
        key=lambda message: int(message["id"]),
    )
    for message in messages:
        author = message.get("author") or {}
        user_id = str(author.get("id", ""))
        if not author.get("bot") and not message.get("webhook_id") and is_discord_id(user_id):
            for number in extract_pr_numbers(message, config.github_repository):
                # The earliest human sharer owns this PR unless the GitHub
                # author's explicit mapping overrides them. Reposts do not
                # produce extra credit or announcements.
                state["pull_requests"].setdefault(str(number), {
                    "discord_id": user_id,
                    "message_id": str(message["id"]),
                    "github_author": None,
                    "merged": False,
                    "announced": False,
                })
        state["last_message_id"] = str(message["id"])
    # Persist discovery before calling GitHub so outages cannot lose PRs.
    save_state(config.state_file, state)

    checked_numbers = []
    for number, pull in state["pull_requests"].items():
        if pull["announced"] or pull["merged"]:
            continue
        try:
            result = fetch_pull(config.github_repository, int(number), config.github_token)
        except TransientAPIError as error:
            logging.warning("GitHub check deferred: %s", error)
            break
        except Exception:
            logging.exception("Could not check PR #%s; will retry next poll", number)
            checked_numbers.append(number)
            continue
        checked_numbers.append(number)
        # Closed != merged. Never announce based on a Discord claim or a
        # GitHub issue/PR's closed state alone.
        if result.get("merged") is True:
            pull["github_author"] = result["user"]["login"]
            pull["merged"] = True

    # Persist fair round-robin ordering: if a rate limit interrupts the scan,
    # the next poll starts with unchecked PRs instead of the same old open PRs.
    for number in checked_numbers:
        state["pull_requests"][number] = state["pull_requests"].pop(number)

    state["outbox"] = build_announcements(config, state)
    save_state(config.state_file, state)
    _flush_outbox(config, state, post)
    logging.info("Tracking %d PRs; %d unique contributors thanked", len(state["pull_requests"]), len(state["contributor_ids"]))
