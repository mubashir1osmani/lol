import copy
import hashlib
import json
import logging
import re
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

from .api import (fetch_channel_messages, fetch_pull_request, fetch_release_entries,
                  fetch_release_prs, fetch_merged_numbers, post_message)
from .config import Config, is_discord_id
from .schedule import UTC, first_friday, iso, new_window, parse_time, shift_week
from .state import load_state, save_state

MAX_CHECKS_PER_POLL = 40
OPENING = "Thank you to this week's contributors!"
CLOSING = "We appreciate everything you do to help build LiteLLM!"


def extract_pr_numbers(message: dict[str, Any], repository: str) -> list[int]:
    text = message.get("content") or ""
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


def _pull(state, number):
    return state["pull_requests"].setdefault(str(number), {
        "merged": False, "announced": False, "github_author": None, "sharers": [],
    })


def discover(config: Config, state: dict, now: datetime) -> set[str]:
    """Discover both sources. Any failure stops publication, not just that source."""
    messages = sorted(fetch_channel_messages(config.contributions_channel_id,
                      config.discord_token, state["last_message_id"]), key=lambda m: int(m["id"]))
    for message in messages:
        author = message.get("author") or {}
        uid = author.get("id")
        if not author.get("bot") and not message.get("webhook_id") and is_discord_id(uid):
            for number in extract_pr_numbers(message, config.github_repository):
                pull = _pull(state, number)
                evidence = {"discord_id": uid, "message_id": str(message["id"])}
                if evidence not in pull.setdefault("sharers", []):
                    pull["sharers"].append(evidence)
                pull.setdefault("discord_id", uid)
                pull.setdefault("message_id", str(message["id"]))
        state["last_message_id"] = str(message["id"])
    save_state(config.state_file, state)
    start, end = (parse_time(state["weekly"][key]) for key in ("start", "end"))
    entries = fetch_release_entries()
    if not entries or min(entries.values()) > start:
        raise RuntimeError("Release index does not cover the weekly start; manual catch-up required")
    for url, published in entries.items():
        if start <= published < end and published <= now:
            state["releases"].setdefault(url, {"published_at": iso(published), "pr_numbers": []})
    release_numbers = set()
    for url, release in state["releases"].items():
        published = parse_time(release["published_at"])
        if start <= published < end and published <= now:
            numbers = fetch_release_prs(url, config.github_repository)
            release["pr_numbers"] = [str(n) for n in numbers]
            for number in numbers:
                _pull(state, number)
                release_numbers.add(str(number))
    save_state(config.state_file, state)
    return release_numbers


def verify(config: Config, state: dict, numbers: set[str], now: datetime, due: bool) -> bool:
    """Checkpoint each lookup, so anonymous rate limits don't restart the work."""
    checks = 0
    end = parse_time(state["weekly"]["end"])
    for number in sorted(numbers, key=int):
        pull = state["pull_requests"][number]
        if pull["announced"] or (pull.get("merged") and pull.get("merged_at")):
            continue
        checked = parse_time(pull["checked_at"]) if pull.get("checked_at") else None
        if checked and ((due and checked >= end) or (not due and now - checked < timedelta(hours=1))):
            continue
        if checks >= MAX_CHECKS_PER_POLL:
            return False
        result = fetch_pull_request(config.github_repository, int(number), config.github_token)
        author = result.get("user") or {}
        login = author.get("login", "")
        if (result.get("number") != int(number) or not isinstance(result.get("merged"), bool)
                or not re.fullmatch(r"[A-Za-z0-9-]+(?:\[bot\])?", login)):
            raise RuntimeError(f"Invalid GitHub verification for PR #{number}")
        merged_at = result.get("merged_at")
        if result["merged"]:
            parse_time(merged_at or "")
        pull.update(merged=result["merged"], merged_at=merged_at, github_author=login,
                    github_author_id=author.get("id"), is_bot=author.get("type") == "Bot" or login.endswith("[bot]"),
                    checked_at=iso(now))
        checks += 1
        save_state(config.state_file, state)
    return True


def contributor_list(config: Config, state: dict, release_numbers: set[str]) -> list[dict]:
    start, end = (parse_time(state["weekly"][key]) for key in ("start", "end"))
    groups = {}
    for number, pull in state["pull_requests"].items():
        if pull["announced"] or not pull.get("merged_at") or pull.get("is_bot"):
            continue
        merged_at = parse_time(pull["merged_at"])
        discord_source = bool(pull.get("discord_id") or pull.get("sharers"))
        if not (pull["merged"] and merged_at < end and
                (number in release_numbers or (discord_source and start <= merged_at))):
            continue
        login = pull["github_author"]
        identity = str(pull.get("github_author_id") or login.lower())
        group = groups.setdefault(identity, {"github_login": login, "github_id": pull.get("github_author_id"),
                                            "pr_numbers": [], "discord_id": None, "candidates": []})
        group["pr_numbers"].append(number)
        mapped = config.github_discord_map.get(login.lower())
        if mapped:
            if group["discord_id"] and group["discord_id"] != mapped:
                raise RuntimeError(f"Conflicting Discord mappings for GitHub account {identity}")
            group["discord_id"] = mapped
        evidence = pull.get("sharers") or ([{"discord_id": pull["discord_id"], "message_id": pull.get("message_id")}] if pull.get("discord_id") else [])
        for sharer in evidence:
            group["candidates"].append({**sharer, "pr_number": number})
    for group in groups.values():
        group["pr_numbers"].sort(key=int)
    return sorted(groups.values(), key=lambda g: g["github_login"].lower())


def build_announcements(config: Config, state: dict, contributors: list[dict], preview=False) -> list[dict]:
    """Use the requested copy; only confirmed mappings are taggable."""
    def row(person):
        login, uid = person["github_login"], person["discord_id"]
        name = f"<@{uid}> ({login})" if uid else f"[{login}](https://github.com/{login})"
        numbers = person["pr_numbers"]
        links = [f"[#{n}](https://github.com/{config.github_repository}/pull/{n})" for n in numbers[:15]]
        suffix = f" (+{len(numbers) - 15} more)" if len(numbers) > 15 else ""
        return f"- {name} - merged " + ", ".join(links) + suffix

    def batch(people):
        content = OPENING + "\n\n" + "\n".join(row(p) for p in people) + "\n\n" + CLOSING
        ids = sorted({p["discord_id"] for p in people if p["discord_id"]}, key=int)
        numbers = sorted({n for p in people for n in p["pr_numbers"]}, key=int)
        identity = json.dumps([config.scope, state["weekly"]["start"], state["weekly"]["end"], content, preview], sort_keys=True)
        return {"pr_numbers": numbers, "discord_ids": ids,
                "github_logins": [p["github_login"] for p in people],
                "payload": {"content": content, "allowed_mentions": {"parse": [], "users": [] if preview else ids},
                            "nonce": hashlib.sha256(identity.encode()).hexdigest()[:24], "enforce_nonce": True,
                            "flags": 4}}

    batches, people = [], []
    for person in contributors:
        candidate = batch(people + [person])
        if (len(candidate["payload"]["content"].encode("utf-16-le")) // 2 > 2000
                or len(candidate["discord_ids"]) > 100):
            if not people:
                raise RuntimeError("A contributor row exceeds Discord's message limit")
            batches.append(batch(people))
            people = []
        people.append(person)
    if people:
        batches.append(batch(people))
    return batches


def _flush_outbox(config, state):
    while state["outbox"]:
        batch = state["outbox"][0]
        receipt = post_message(config.announcements_channel_id, config.discord_token, batch["payload"])
        for number in batch["pr_numbers"]:
            state["pull_requests"][number]["announced"] = True
        state["contributor_ids"] = sorted(set(state["contributor_ids"]) | set(batch["discord_ids"]), key=int)
        state["roundups"][-1]["message_ids"].append(receipt["id"])
        state["outbox"].pop(0)
        save_state(config.state_file, state)


def _finish(config, state):
    end = parse_time(state["weekly"]["end"])
    state["weekly"] = new_window(config, shift_week(config, end))
    save_state(config.state_file, state)


def poll_once(config: Config, now: datetime | None = None) -> None:
    if not config.announcements_enabled:
        return
    now = now or datetime.now(UTC)
    state = load_state(config)
    if state["weekly"] is None:
        state["weekly"] = new_window(config, first_friday(config, now))
        save_state(config.state_file, state)
    if state["weekly"]["prepared"]:
        _flush_outbox(config, state)
        _finish(config, state)
        return
    release_numbers = discover(config, state, now)
    due = now >= parse_time(state["weekly"]["end"])
    numbers = set(release_numbers)
    if due:
        if state["weekly"]["merge_numbers"] is None:
            state["weekly"]["merge_numbers"] = sorted(fetch_merged_numbers(config.github_repository,
                parse_time(state["weekly"]["start"]), parse_time(state["weekly"]["end"]), config.github_token))
            save_state(config.state_file, state)
        numbers.update(str(n) for n in state["weekly"]["merge_numbers"]
                       if str(n) in state["pull_requests"] and state["pull_requests"][str(n)].get("discord_id"))
    if not verify(config, state, numbers, now, due) or not due:
        return
    people = contributor_list(config, state, release_numbers)
    state["outbox"] = build_announcements(config, state, people)
    state["weekly"]["prepared"] = True
    state["roundups"].append({"start": state["weekly"]["start"], "end": state["weekly"]["end"],
        "contributors": people, "message_ids": []})
    save_state(config.state_file, state)
    _flush_outbox(config, state)
    _finish(config, state)
    logging.info("Weekly roundup complete: %d contributors; %d PRs", len(people), sum(len(p["pr_numbers"]) for p in people))


def preview_roundup(config: Config, now: datetime | None = None) -> dict:
    """Independent cache, no production checkpoints/outbox/announced flags changed."""
    now = now or datetime.now(UTC)
    production = load_state(config)
    preview_config = replace(config, state_file=config.state_file + ".preview")
    state = load_state(preview_config)
    for number, pull in production["pull_requests"].items():
        cached = state["pull_requests"].get(number, {})
        state["pull_requests"][number] = copy.deepcopy(pull)
        if not pull.get("merged_at") and cached.get("merged_at"):
            state["pull_requests"][number].update({key: cached[key] for key in
                ("merged", "merged_at", "github_author", "github_author_id", "is_bot", "checked_at")})
    state["weekly"] = copy.deepcopy(production["weekly"] or new_window(config, first_friday(config, now)))
    state["weekly"].update(end=iso(min(now, parse_time(state["weekly"]["end"]))), merge_numbers=None, prepared=False)
    state["outbox"] = []
    release_numbers = discover(preview_config, state, now)
    merged = fetch_merged_numbers(config.github_repository, parse_time(state["weekly"]["start"]),
                                 parse_time(state["weekly"]["end"]), config.github_token)
    numbers = release_numbers | {str(n) for n in merged if str(n) in state["pull_requests"] and state["pull_requests"][str(n)].get("discord_id")}
    if not verify(preview_config, state, numbers, now, True):
        raise RuntimeError("Preview verification is still in progress; run preview again to resume its separate cache")
    people = contributor_list(config, state, release_numbers)
    return {"start": state["weekly"]["start"], "end": state["weekly"]["end"], "contributors": people,
            "messages": [b["payload"] for b in build_announcements(config, state, people, preview=True)]}
