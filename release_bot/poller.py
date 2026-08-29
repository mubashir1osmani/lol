import logging
from collections.abc import Callable
from typing import Any

from .config import Config


Release = dict[str, Any]


def poll_once(
    config: Config,
    fetch_releases: Callable[[str, str | None], list[Release]],
    post_release: Callable[[str, str, Release, str], None],
    load_state: Callable[[str], tuple[bool, list[str]]],
    save_state: Callable[[str, list[str]], None],
) -> None:
    releases = [
        release
        for release in fetch_releases(config.github_repository, config.github_token)
        if not release.get("draft")
        and (config.include_prereleases or not release.get("prerelease"))
    ]
    initialized, posted_release_ids = load_state(config.state_file)

    if not initialized:
        fetched_ids = [str(release["id"]) for release in releases]
        if config.post_latest_on_startup and releases:
            post_release(
                config.discord_channel_id,
                config.discord_token,
                releases[0],
                config.github_repository,
            )
            logging.info("Posted initial release %s", releases[0]["tag_name"])
        save_state(config.state_file, fetched_ids)
        return

    posted_ids = set(posted_release_ids)
    unseen = [
        release for release in releases if str(release["id"]) not in posted_ids
    ]
    unseen.reverse()

    for release in unseen:
        post_release(
            config.discord_channel_id,
            config.discord_token,
            release,
            config.github_repository,
        )
        posted_ids.add(str(release["id"]))
        save_state(config.state_file, list(posted_ids))
        logging.info("Posted release %s", release["tag_name"])

    if not unseen:
        logging.info("No new releases")
