import argparse
import json
import logging
import os
import signal
import threading
import uuid
from datetime import datetime

from .api import (
    TransientAPIError,
    fetch_channel,
    post_message,
)
from .config import get_config, is_discord_id, load_dotenv
from .poller import poll_once, preview_roundup
from .schedule import UTC, parse_time
from .state import load_state, save_state


def moderator_channel(channel_id: str, token: str) -> dict:
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
    return channel


def send_test_message(channel_id: str, token: str) -> str:
    """Send one delivery test, without touching production config or state."""
    channel = moderator_channel(channel_id, token)
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


def run_preview(config, channel_id=None):
    channel = moderator_channel(channel_id, config.discord_token) if channel_id else None
    report = preview_roundup(config)
    report_path = config.state_file + ".preview-report.json"
    save_state(report_path, report)
    if channel:
        payloads = [{"content": f"**Moderator preview (no pings)**\n{report['start']} to {report['end']}\n"
                     "Combines Discord PRs and new release-note contributors. Production history is unchanged.",
                     "nonce": uuid.uuid4().hex[:24], "enforce_nonce": True}] + report["messages"]
        if not report["contributors"]:
            payloads[0]["content"] += "\nNo eligible contributors in this window."
        candidate_rows = []
        for person in report["contributors"]:
            if person["discord_id"]:
                continue
            ids = sorted({evidence["discord_id"] for evidence in person["candidates"]}, key=int)
            if ids:
                suffix = f" (+{len(ids) - 15} more in the report)" if len(ids) > 15 else ""
                candidate_rows.append(f"- {person['github_login']}: " + ", ".join(f"<@{uid}> (`{uid}`)" for uid in ids[:15]) + suffix)
        if candidate_rows:
            heading = ("**Possible Discord matches - NOT confirmed (no pings)**\n"
                       "These members shared the PR links; they may not be the authors. "
                       "Confirm identity before adding a GITHUB_DISCORD_MAP entry.\n")
            content = heading
            for row in candidate_rows:
                if len((content + row).encode('utf-16-le')) // 2 > 1900:
                    payloads.append({"content": content})
                    content = heading
                content += row + "\n"
            payloads.append({"content": content})
        for payload in payloads:
            receipt = post_message(channel_id, config.discord_token,
                                   {**payload, "allowed_mentions": {"parse": [], "users": []}})
            print(f"Preview sent: https://discord.com/channels/{channel['guild_id']}/{channel_id}/{receipt['id']}")
        print(f"Contributor list and candidate Discord matches: {report_path}")
    else:
        print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LiteLLM contributor thank-you bot")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--test-channel-id",
        metavar="ID",
        help="Send one unpinged test to #moderators-only and exit (only needs DISCORD_BOT_TOKEN)",
    )
    mode.add_argument("--preview", action="store_true", help="Print this week's preview and candidate Discord IDs; never post")
    mode.add_argument("--preview-channel-id", metavar="ID", help="Send a no-ping weekly preview to #moderator-only")
    mode.add_argument("--once", action="store_true", help="Collect once; publish only if the Friday roundup is due and enabled")
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
    if args.preview or args.preview_channel_id:
        run_preview(config, args.preview_channel_id)
        return
    if args.once:
        poll_once(config)
        return
    stop_event = threading.Event()

    def stop(_signal_number: int, _frame: object) -> None:
        logging.info("Shutting down")
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if not config.announcements_enabled:
        logging.info(
            "Automatic announcements are paused (ANNOUNCEMENTS_ENABLED=false). "
            "Use --preview-channel-id for a no-ping weekly preview. "
            "Set ANNOUNCEMENTS_ENABLED=true and restart to enable the Friday schedule."
        )
        stop_event.wait()
        return
    logging.info(
        "Weekly contributor roundup: Friday %s %s, destination %s",
        config.roundup_time, config.roundup_timezone, config.announcements_channel_id,
    )
    while not stop_event.is_set():
        delay = config.poll_interval_seconds
        try:
            poll_once(config)
            window = load_state(config)["weekly"]
            if window:
                remaining = (parse_time(window["end"]) - datetime.now(UTC)).total_seconds()
                if remaining > 0:
                    delay = min(delay, remaining)
        except TransientAPIError as error:
            logging.warning("Contributor check deferred: %s", error)
        except Exception:
            logging.exception("Contributor check failed")
        stop_event.wait(delay)
