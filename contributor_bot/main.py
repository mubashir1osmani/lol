import argparse
import logging
import os
import signal
import threading
import uuid

from .api import (
    TransientAPIError,
    fetch_channel,
    fetch_channel_messages,
    fetch_pull_request,
    post_message,
)
from .config import get_config, is_discord_id, load_dotenv
from .poller import poll_once


def send_test_message(channel_id: str, token: str) -> str:
    """Send one delivery test, without touching production config or state."""
    if not is_discord_id(channel_id):
        raise ValueError("The test channel must be a Discord channel ID")
    if not token:
        raise ValueError("Missing required environment variable: DISCORD_BOT_TOKEN")
    channel = fetch_channel(channel_id, token)
    if (
        channel.get("name") not in {"moderator-only", "moderators-only"}
        or channel.get("type") != 0
        or not is_discord_id(channel.get("guild_id"))
    ):
        raise ValueError("The test destination must be the #moderator-only or #moderators-only text channel")
    receipt = post_message(channel_id, token, {
        "content": (
            "🧪 **LiteLLM contributor bot test**\n\n"
            "This is a delivery test for the contributor thank-you bot. "
            "No PRs are scanned or marked as thanked by this test."
        ),
        "allowed_mentions": {"parse": []},
        "nonce": uuid.uuid4().hex[:24],
        "enforce_nonce": True,
    })
    return f"https://discord.com/channels/{channel['guild_id']}/{channel_id}/{receipt['id']}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LiteLLM contributor thank-you bot")
    parser.add_argument(
        "--test-channel-id",
        metavar="ID",
        help="Send one unpinged test to #moderators-only and exit (only needs DISCORD_BOT_TOKEN)",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    if args.test_channel_id is not None:
        url = send_test_message(args.test_channel_id, os.environ.get("DISCORD_BOT_TOKEN", "").strip())
        print(f"Test message sent: {url}")
        return

    config = get_config()
    stop_event = threading.Event()

    def stop(_signal_number: int, _frame: object) -> None:
        logging.info("Shutting down")
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if not config.announcements_enabled:
        logging.info(
            "Automatic announcements are paused (ANNOUNCEMENTS_ENABLED=false). "
            "Use --test-channel-id for a one-time delivery test. "
            "Set ANNOUNCEMENTS_ENABLED=true and redeploy to start contributor checks."
        )
        stop_event.wait()
        return
    logging.info(
        "Watching channel %s for merged %s PRs; thanking contributors in %s",
        config.contributions_channel_id,
        config.github_repository,
        config.announcements_channel_id,
    )
    while not stop_event.is_set():
        try:
            poll_once(config, fetch_channel_messages, fetch_pull_request, post_message)
        except TransientAPIError as error:
            logging.warning("Contributor check deferred: %s", error)
        except Exception:
            logging.exception("Contributor check failed")
        stop_event.wait(config.poll_interval_seconds)
