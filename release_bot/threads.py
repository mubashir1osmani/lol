"""Add the support user to new threads so they never miss a support request."""

import logging
from collections.abc import Callable
from typing import Any

from .config import Config
from .state import load_ids, save_ids


Thread = dict[str, Any]


def poll_threads_once(
    config: Config,
    fetch_threads: Callable[[str, str], list[Thread]],
    add_member: Callable[[str, str, str], None],
    state_file: str = ".data/threads.json",
    greet: Callable[[Thread], None] | None = None,
) -> None:
    if not config.support_user_id or not config.discord_guild_id:
        return

    threads = fetch_threads(config.discord_guild_id, config.discord_token)
    if config.support_channel_ids:
        threads = [
            thread
            for thread in threads
            if str(thread.get("parent_id")) in config.support_channel_ids
        ]

    initialized, joined = load_ids(state_file, "joined")
    joined_ids = set(joined)

    if not initialized:
        # Seed with existing threads so a fresh bot only reacts to new ones.
        save_ids(state_file, "joined", [str(thread["id"]) for thread in threads])
        logging.info("Seeded thread state with %d active threads", len(threads))
        return

    for thread in threads:
        thread_id = str(thread["id"])
        if thread_id in joined_ids:
            continue
        try:
            add_member(thread_id, config.support_user_id, config.discord_token)
            logging.info(
                "Added support user to thread %s (%s)",
                thread_id,
                thread.get("name", "unnamed"),
            )
        except Exception:
            logging.exception("Could not add support user to thread %s", thread_id)
            continue
        joined_ids.add(thread_id)
        save_ids(state_file, "joined", list(joined_ids))
        if greet is not None:
            try:
                greet(thread)
            except Exception:
                logging.exception("Could not greet thread %s", thread_id)
