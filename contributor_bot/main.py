import argparse
import json
import logging
import os
import signal
import threading
import time
import uuid
from datetime import datetime

from .api import (
    TransientAPIError,
    fetch_channel,
    post_message,
    register_guild_commands,
)
from .backfill import backfill_merged_pulls
from .config import get_config, is_discord_id, load_dotenv
from .gateway import start_gateway, stop_gateway
from .interactions import (
    LINKS_BACKFILL_SLASH_COMMANDS,
    MERGED_SLASH_COMMANDS,
    PR_BOARD_SLASH_COMMANDS,
    REVIEW_SLASH_COMMANDS,
    role_slash_commands,
)
from .poller import poll_once, preview_roundup
from .pr_board import sync_pr_board
from .review_job import poll_review_channels
from .roles_sync import (
    log_roles_daily_boot_status,
    revalidate_all_grants_on_boot,
    run_daily_roles_if_due,
    run_roles_backfill,
    should_run_boot_backfill,
    should_run_grants_revalidation,
)
from .store import Store
from .schedule import UTC, parse_time
from .server import start_http_server, stop_http_server
from .signatures import ed25519_public_key_ok
from .state import load_state, save_state
from .tracing import configure_litellm_tracing
from .weekly import log_weekly_boot_status, post_weekly_if_due, weekly_schedule_status


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
    mode.add_argument(
        "--backfill-merged-prs",
        action="store_true",
        help="Scan historical merged PRs and grant contributor roles to linked Discord users",
    )
    mode.add_argument(
        "--post-weekly-now",
        action="store_true",
        help="Post this week's contributor summary to ANNOUNCEMENTS_CHANNEL_ID right now and exit",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    configure_litellm_tracing()
    if args.test_channel_id is not None:
        url = send_test_message(args.test_channel_id, os.environ.get("DISCORD_BOT_TOKEN", "").strip())
        print(f"Test message sent: {url}")
        return

    config = get_config()
    if args.preview or args.preview_channel_id:
        run_preview(config, args.preview_channel_id)
        return
    if args.backfill_merged_prs:
        run_backfill(config)
        return
    if args.post_weekly_now:
        store = Store(config.roles_db_path)
        try:
            posted = post_weekly_if_due(config, store, force=True, via="cli")
        finally:
            store.close()
        if posted is None:
            raise SystemExit("Weekly summary was not posted (0 matched contributors or missing destination channel)")
        print("Weekly summary posted.")
        return
    if args.once:
        poll_once(config)
        if config.pr_review_enabled or config.pr_board_enabled or config.contributor_roles_enabled:
            store = Store(config.roles_db_path)
            try:
                if config.pr_review_enabled:
                    poll_review_channels(config, store)
                if config.pr_board_enabled:
                    sync_pr_board(config, store)
                if config.contributor_roles_enabled:
                    run_daily_roles_if_due(config, store)
            finally:
                store.close()
        return
    server = start_contributor_roles(config)
    store = getattr(server, "store", None) if server is not None else None
    gateway = start_gateway(config, store)
    stop_event = threading.Event()

    def stop(_signal_number: int, _frame: object) -> None:
        logging.info("Shutting down")
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        _run_loop(config, stop_event, store, gateway)
    finally:
        stop_gateway(gateway)
        if server is not None:
            stop_http_server(server)


def run_backfill(config) -> None:
    if not config.effective_github_token:
        raise SystemExit("GITHUB_TOKEN or GITHUB_ACTOR_TOKEN is required for --backfill-merged-prs")
    backfill_merged_pulls(config)


def _gateway_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("discord") is not None
    except Exception:
        return False


def start_contributor_roles(config):
    """Listen for GitHub webhooks and, when configured, Discord HTTP interactions.

    A Render web service sets PORT, so /health is served even while features are off.
    A failure here does not stop the weekly summary loop.
    """
    features = (
        config.contributor_roles_enabled
        or config.pr_review_enabled
        or config.pr_board_enabled
        or (config.merged_weekly_enabled and bool(config.discord_guild_id))
    )
    if not features and not os.environ.get("PORT", "").strip():
        return None
    server = None
    try:
        server = start_http_server(config)
    except OSError as error:
        logging.error("Contributor HTTP server failed to start: %s", error)
    commands = []
    if config.contributor_roles_enabled:
        commands.extend(role_slash_commands(config))
    if config.pr_review_enabled:
        commands.extend(REVIEW_SLASH_COMMANDS)
    if config.merged_weekly_enabled and config.discord_guild_id:
        commands.extend(MERGED_SLASH_COMMANDS)
    if (config.contributor_roles_enabled or config.merged_weekly_enabled) and config.discord_guild_id:
        commands.extend(LINKS_BACKFILL_SLASH_COMMANDS)
    if config.pr_board_enabled and config.discord_guild_id:
        commands.extend(PR_BOARD_SLASH_COMMANDS)
    # When discord.py is installed, gateway.py syncs guild commands via CommandTree on
    # connect. Skip the duplicate REST PUT here so startup does not trigger a Discord 429.
    if features and commands and config.discord_application_id and config.discord_guild_id and not _gateway_available():
        try:
            register_guild_commands(
                config.discord_application_id,
                config.discord_guild_id,
                config.discord_token,
                commands,
            )
            logging.info(
                "Registered slash commands in guild %s: %s",
                config.discord_guild_id,
                ", ".join(command["name"] for command in commands),
            )
        except Exception:
            logging.exception(
                "Could not register slash commands. Check DISCORD_APPLICATION_ID, "
                "that the bot invite includes applications.commands, and that the bot is in the server."
            )
    elif features and commands and not (config.discord_application_id and config.discord_guild_id) and not _gateway_available():
        logging.warning(
            "Skipping REST slash-command registration: DISCORD_APPLICATION_ID or DISCORD_GUILD_ID is unset"
        )
    if features and config.discord_public_key and not ed25519_public_key_ok(config.discord_public_key):
        logging.error(
            "DISCORD_PUBLIC_KEY is not a valid Ed25519 point. "
            "Discord interactions will be rejected until it is corrected."
        )
    if features and not config.effective_github_token:
        logging.warning(
            "GITHUB_TOKEN and GITHUB_ACTOR_TOKEN are unset. GitHub reads may hit anonymous 60/hr rate limits."
        )
    if config.pr_review_enabled:
        logging.info(
            "PR review agent is on for %s using %s. It only posts in Discord.",
            ", ".join(config.review_channel_ids), config.review_model,
        )
    if config.contributor_roles_enabled:
        if not config.roles_announce_enabled:
            logging.info("Role announcements disabled (ROLES_ANNOUNCE_ENABLED=false); roles will be granted silently")
        elif not config.role_announcements_channel_id:
            logging.warning("ROLE_ANNOUNCEMENTS_CHANNEL_ID is unset; new roles will be granted without a #shipped post")
    return server


def _run_loop(config, stop_event: threading.Event, store=None, gateway=None) -> None:
    reviews = bool(config.pr_review_enabled and store is not None)
    weekly_watchdog = bool(config.merged_weekly_enabled and store is not None)
    roles_daily = bool(config.contributor_roles_enabled and store is not None)
    pr_board_sync = False
    if not config.pr_board_enabled:
        logging.info(
            "Contributor PR board is disabled "
            "(set PR_BOARD_PROJECT_ID or PR_BOARD_OWNER + PR_BOARD_NUMBER to enable)."
        )
    elif not config.github_actor_token:
        logging.warning(
            "Contributor PR board is enabled but GITHUB_ACTOR_TOKEN is unset "
            "(requires 'project' scope); board sync is paused."
        )
    elif store is None:
        logging.error(
            "Contributor PR board is enabled but the HTTP server did not start, "
            "so board sync will not run"
        )
    else:
        pr_board_sync = True
        logging.info(
            "Contributor PR board sync active for %s (every %ds).",
            config.github_repository,
            config.pr_board_sync_seconds,
        )
    if config.contributor_roles_enabled:
        if store is None:
            logging.error(
                "Contributor roles are enabled but the HTTP server/store did not start, "
                "so daily contributor role runs will not execute"
            )
        else:
            log_roles_daily_boot_status(config, store)
    if config.merged_weekly_enabled:
        if store is not None:
            log_weekly_boot_status(config, store)
        else:
            logging.info("Weekly contributor summary: %s", weekly_schedule_status(config, store))
    if config.announcements_enabled:
        logging.info(
            "Legacy roundup poller (ANNOUNCEMENTS_ENABLED) is retired; "
            "weekly posts run only via the Friday %s %s summary scheduler.",
            config.merged_time, config.merged_timezone,
        )
    if not reviews and not weekly_watchdog and not pr_board_sync and not roles_daily:
        if not config.announcements_enabled:
            logging.info("Automatic announcements are paused (ANNOUNCEMENTS_ENABLED=false).")
        stop_event.wait()
        return
    if config.pr_review_enabled and not reviews:
        logging.error("PR pre-review is enabled but the HTTP server did not start, so channel reviews will not run")
    first_tick = True
    next_board_sync = 0.0
    boot_backfill_started = False
    while not stop_event.is_set():
        gateway_live = gateway is not None and gateway.connected.is_set()
        guild, loop = (
            gateway.current_guild_and_loop(config)
            if gateway is not None and hasattr(gateway, "current_guild_and_loop")
            else (None, None)
        )
        if reviews and not gateway_live:
            try:
                poll_review_channels(config, store)
            except Exception:
                logging.exception("PR review poll failed")
        # Give a freshly started gateway one brief cycle to connect and use its
        # cached guild member list; thereafter run a 60s watchdog regardless so
        # a dead gateway or stopped tasks.loop can never silence scheduled jobs.
        if not (first_tick and gateway is not None and not gateway_live):
            if weekly_watchdog:
                try:
                    post_weekly_if_due(config, store, guild=guild, loop=loop)
                except Exception:
                    logging.exception("Weekly merge summary watchdog check failed")
            if roles_daily:
                if should_run_grants_revalidation(config, store):
                    try:
                        revalidate_all_grants_on_boot(config, store, guild=guild, loop=loop)
                    except Exception:
                        logging.exception("Boot all-grants re-validation failed")
                if not boot_backfill_started and should_run_boot_backfill(config, store):
                    boot_backfill_started = True
                    logging.info("Starting one-time contributor roles backfill on boot (ROLES_BACKFILL_ON_BOOT=true)")
                    threading.Thread(
                        target=_run_boot_backfill,
                        args=(config, store, guild, loop),
                        name="roles-boot-backfill",
                        daemon=True,
                    ).start()
                try:
                    run_daily_roles_if_due(config, store, guild=guild, loop=loop)
                except Exception:
                    logging.exception("Daily contributor roles watchdog check failed")
        if pr_board_sync and time.monotonic() >= next_board_sync:
            next_board_sync = time.monotonic() + float(config.pr_board_sync_seconds)
            try:
                sync_pr_board(config, store)
            except Exception:
                logging.exception("Contributor PR board sync failed")
        delay = float(config.poll_interval_seconds)
        if weekly_watchdog or roles_daily:
            delay = min(delay, 60.0)
        if pr_board_sync:
            delay = min(delay, max(1.0, next_board_sync - time.monotonic()))
        first_tick = False
        stop_event.wait(delay)


def _run_boot_backfill(config, store, guild=None, loop=None) -> None:
    try:
        run_roles_backfill(config, store, guild=guild, loop=loop)
    except Exception:
        logging.exception("Boot contributor roles backfill failed")
