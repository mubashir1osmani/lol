"""Fetch RSS/Atom feeds (blog posts, social updates) with the standard library."""

import html
import logging
import re
from collections.abc import Callable
from typing import Any
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from .config import Config
from .state import load_ids, save_ids


FeedEntry = dict[str, str]

ATOM_NAMESPACE = "{http://www.w3.org/2005/Atom}"


def _text(element: ElementTree.Element | None) -> str:
    return (element.text or "").strip() if element is not None else ""


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", text)).strip()


def _rss_entries(channel: ElementTree.Element) -> list[FeedEntry]:
    entries = []
    for item in channel.findall("item"):
        link = _text(item.find("link"))
        entries.append(
            {
                "id": _text(item.find("guid")) or link,
                "title": _strip_html(_text(item.find("title"))),
                "link": link,
                "summary": _strip_html(_text(item.find("description"))),
                "published": _text(item.find("pubDate")),
            }
        )
    return entries


def _atom_entries(feed: ElementTree.Element) -> list[FeedEntry]:
    entries = []
    for item in feed.findall(f"{ATOM_NAMESPACE}entry"):
        link = ""
        for candidate in item.findall(f"{ATOM_NAMESPACE}link"):
            if candidate.get("rel") in (None, "alternate"):
                link = candidate.get("href", "")
                break
        entries.append(
            {
                "id": _text(item.find(f"{ATOM_NAMESPACE}id")) or link,
                "title": _strip_html(_text(item.find(f"{ATOM_NAMESPACE}title"))),
                "link": link,
                "summary": _strip_html(
                    _text(item.find(f"{ATOM_NAMESPACE}summary"))
                    or _text(item.find(f"{ATOM_NAMESPACE}content"))
                ),
                "published": _text(item.find(f"{ATOM_NAMESPACE}published"))
                or _text(item.find(f"{ATOM_NAMESPACE}updated")),
            }
        )
    return entries


def parse_feed(document: str) -> list[FeedEntry]:
    """Parse an RSS 2.0 or Atom document into newest-first entries."""
    root = ElementTree.fromstring(document)
    if root.tag == "rss":
        channel = root.find("channel")
        entries = _rss_entries(channel) if channel is not None else []
    elif root.tag == f"{ATOM_NAMESPACE}feed":
        entries = _atom_entries(root)
    else:
        raise RuntimeError(f"Unsupported feed format: {root.tag}")
    return [entry for entry in entries if entry["id"]]


def fetch_feed(url: str) -> list[FeedEntry]:
    request = Request(url, headers={"User-Agent": "litellm-release-discord-bot"})
    with urlopen(request, timeout=15) as response:
        return parse_feed(response.read().decode("utf-8", errors="replace"))


def feed_message(
    entry: FeedEntry, label: str, blurb: str | None = None
) -> dict[str, Any]:
    if label == "blog":
        content = "📝 **New post on the LiteLLM blog!**"
        color = 0x5865F2
        footer = "docs.litellm.ai/blog"
    else:
        content = "📣 **LiteLLM update**"
        color = 0x1DA1F2
        footer = "Social media"

    description = blurb or entry.get("summary", "")
    if len(description) > 2_000:
        description = f"{description[:1_980].rstrip()}…"

    return {
        "content": content,
        "embeds": [
            {
                "title": entry["title"][:256] or entry["link"],
                "url": entry["link"],
                "description": description,
                "color": color,
                "footer": {"text": footer},
            }
        ],
        "allowed_mentions": {"parse": []},
    }


def poll_feeds_once(
    config: Config,
    fetch: Callable[[str], list[FeedEntry]],
    post: Callable[[str, str, dict[str, Any]], None],
    state_file: str = ".data/feeds.json",
    write_blurb: Callable[[FeedEntry, str], str | None] = lambda _entry, _label: None,
) -> None:
    """Post feed entries not seen before. The first fetch of a feed only seeds
    state, so a fresh bot does not flood the channel with the entire archive."""
    for label, url in config.feeds:
        try:
            entries = fetch(url)
        except Exception:
            logging.exception("Could not fetch %s feed %s", label, url)
            continue

        initialized, seen = load_ids(state_file, url)
        seen_ids = set(seen)
        entry_ids = [entry["id"] for entry in entries]

        if not initialized:
            save_ids(state_file, url, entry_ids)
            logging.info("Seeded %s feed with %d entries", label, len(entry_ids))
            continue

        unseen = [entry for entry in entries if entry["id"] not in seen_ids]
        unseen.reverse()
        for entry in unseen:
            post(
                config.updates_channel_id,
                config.discord_token,
                feed_message(entry, label, write_blurb(entry, label)),
            )
            seen_ids.add(entry["id"])
            save_ids(state_file, url, list(seen_ids))
            logging.info("Posted %s entry: %s", label, entry["title"])
