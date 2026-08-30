import logging
import signal
import threading
import time
from typing import Any

from . import agent as claude
from .api import (
    add_thread_member,
    apply_mention,
    fetch_active_threads,
    fetch_first_message,
    fetch_releases,
    post_message,
    release_message,
)
from .config import Config, get_config, load_dotenv
from .feeds import fetch_feed, poll_feeds_once
from .poller import poll_once
from .state import load_state, save_state
from .threads import poll_threads_once


def build_agent(config: Config) -> "claude.ClaudeAgent | None":
    if not config.anthropic_api_key:
        return None
    try:
        return claude.ClaudeAgent(config.anthropic_api_key, config.claude_model)
    except ImportError:
        logging.warning(
            "ANTHROPIC_API_KEY is set but the anthropic package is not "
            "installed (pip install anthropic); running without Claude"
        )
        return None


def main() -> None:
    load_dotenv()
    config = get_config()
    stop_event = threading.Event()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    def stop(_signal_number: int, _frame: object) -> None:
        logging.info("Shutting down")
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ai = build_agent(config)

    def announce(channel_id: str, token: str, message: dict[str, Any]) -> None:
        post_message(
            channel_id,
            token,
            apply_mention(message, config.mention_role_id, config.mention_everyone),
        )

    def post_release_with_digest(
        channel_id: str, token: str, release: dict[str, Any], repository: str
    ) -> None:
        digest = claude.try_complete(ai, claude.release_prompt(release, repository))
        announce(channel_id, token, release_message(release, repository, digest))

    def write_blurb(entry: dict[str, str], label: str) -> str | None:
        return claude.try_complete(ai, claude.feed_prompt(entry, label))

    def greet_thread(thread: dict[str, Any]) -> None:
        if ai is None or not config.agent_replies_in_threads:
            return
        thread_id = str(thread["id"])
        first_message = fetch_first_message(thread_id, config.discord_token)
        reply = claude.try_complete(
            ai, claude.thread_prompt(thread.get("name", ""), first_message)
        )
        if reply:
            post_message(
                thread_id,
                config.discord_token,
                {"content": reply[:2_000], "allowed_mentions": {"parse": []}},
            )

    def check_releases() -> None:
        poll_once(
            config, fetch_releases, post_release_with_digest, load_state, save_state
        )

    def check_feeds() -> None:
        poll_feeds_once(config, fetch_feed, announce, write_blurb=write_blurb)

    def check_threads() -> None:
        poll_threads_once(
            config, fetch_active_threads, add_thread_member, greet=greet_thread
        )

    tasks = [("releases", check_releases, config.poll_interval_seconds)]
    if config.feeds:
        tasks.append(("feeds", check_feeds, config.feed_poll_interval_seconds))
    if config.support_user_id:
        tasks.append(("threads", check_threads, config.thread_poll_interval_seconds))

    logging.info("Watching GitHub releases for %s", config.github_repository)
    if config.feeds:
        logging.info(
            "Watching %d feed(s): %s",
            len(config.feeds),
            ", ".join(url for _label, url in config.feeds),
        )
    if config.support_user_id:
        logging.info(
            "Adding user %s to new threads in guild %s",
            config.support_user_id,
            config.discord_guild_id,
        )
    logging.info(
        "Claude agent: %s",
        f"enabled ({config.claude_model})" if ai else "disabled (no ANTHROPIC_API_KEY)",
    )

    next_run = {name: 0.0 for name, _task, _interval in tasks}
    while not stop_event.is_set():
        now = time.monotonic()
        for name, task, interval in tasks:
            if now < next_run[name]:
                continue
            try:
                task()
            except Exception:
                logging.exception("%s check failed", name)
            next_run[name] = time.monotonic() + interval
        pending = min(next_run.values()) - time.monotonic()
        stop_event.wait(max(pending, 1))
