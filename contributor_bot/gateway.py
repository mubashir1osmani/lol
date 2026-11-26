"""Discord gateway so the bot shows online and receives messages and slash commands.

discord.py is imported lazily. Without it the process keeps using REST and stays
offline. GitHub webhooks and OAuth stay on the HTTP server. Leave the
Interactions Endpoint URL empty to deliver slash commands here. If that URL is
set, Discord sends interactions to HTTP instead, and that handler still works.
"""

import asyncio
import logging
import threading

from .config import Config
from .interactions import handle_interaction
from .review_job import handle_channel_message, handle_thread_message
from .store import Store


PRESENCE_REVIEW = "for PR reviews"
PRESENCE_ROUNDUP = "LiteLLM contributors"


class Gateway:
    def __init__(self):
        self.connected = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.client = None

    def start(self, config: Config, store: Store | None) -> None:
        self._thread = threading.Thread(
            target=self._run,
            args=(config, store),
            name="discord-gateway",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        client, loop = self.client, self.loop
        if client is not None and loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(client.close(), loop)
            try:
                future.result(timeout=3)
            except Exception:
                logging.warning("Discord gateway close timed out")
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _run(self, config: Config, store: Store | None) -> None:
        try:
            discord = _load_discord()
        except ImportError:
            logging.warning(
                "discord.py is not installed, so the bot stays offline and uses REST. "
                "pip install -r requirements.txt to show it online."
            )
            return
        try:
            asyncio.run(self._serve(discord, config, store))
        except Exception:
            logging.exception("Discord gateway stopped")
        finally:
            self.connected.clear()

    async def _serve(self, discord, config: Config, store: Store | None) -> None:
        intents = build_intents(discord, members=config.discord_members_intent)
        client = discord.Client(intents=intents)
        self.client = client

        @client.event
        async def on_ready():
            watching = PRESENCE_REVIEW if config.pr_review_enabled else PRESENCE_ROUNDUP
            await client.change_presence(
                status=discord.Status.online,
                activity=discord.Activity(type=discord.ActivityType.watching, name=watching),
            )
            self.connected.set()
            logging.info("Discord gateway connected as %s, watching %s", client.user, watching)

        @client.event
        async def on_disconnect():
            self.connected.clear()

        @client.event
        async def on_message(message):
            if self._stop.is_set() or message.author.bot or message.guild is None:
                return
            if not config.pr_review_enabled or store is None:
                return
            payload = _message_payload(message)
            channel_type = getattr(message.channel.type, "value", message.channel.type)
            if channel_type in {10, 11, 12}:
                threading.Thread(
                    target=handle_thread_message,
                    args=(config, store, payload),
                    name="review-followup",
                    daemon=True,
                ).start()
                return
            channel_id = str(message.channel.id)
            if not config.review_watch_all_channels and channel_id not in config.review_channel_ids:
                return
            if channel_type not in {0, 5}:
                return
            threading.Thread(
                target=handle_channel_message,
                args=(config, store, channel_id, payload),
                name="review-message",
                daemon=True,
            ).start()

        @client.event
        async def on_interaction(interaction):
            if interaction.type != discord.InteractionType.application_command:
                return
            await _dispatch_command(discord, interaction, config, store)

        self.loop = asyncio.get_running_loop()
        if self._stop.is_set():
            return
        await client.start(config.discord_token)


def _load_discord():
    import discord
    return discord


def build_intents(discord, *, members: bool):
    """Guilds, guild messages, and message content. Members only when requested."""
    intents = discord.Intents.none()
    intents.guilds = True
    intents.guild_messages = True
    intents.message_content = True
    if members:
        intents.members = True
    return intents


def _message_payload(message) -> dict:
    author = message.author
    return {
        "id": str(message.id),
        "type": 19 if message.reference else 0,
        "content": message.content or "",
        "webhook_id": str(message.webhook_id) if message.webhook_id else None,
        "author": {"id": str(author.id), "bot": bool(author.bot)},
        "channel_id": str(message.channel.id),
    }


def _interaction_payload(interaction) -> dict:
    data = interaction.data or {}
    options = []
    for option in data.get("options") or []:
        if isinstance(option, dict):
            options.append({"name": option.get("name"), "value": option.get("value")})
    channel = interaction.channel
    channel_type = 0
    parent_id = None
    if channel is not None:
        channel_type = int(getattr(channel.type, "value", channel.type))
        parent = getattr(channel, "parent_id", None)
        if parent:
            parent_id = str(parent)
    payload = {
        "type": 2,
        "guild_id": str(interaction.guild_id or ""),
        "application_id": str(interaction.application_id or ""),
        "channel_id": str(interaction.channel_id or ""),
        "token": interaction.token,
        "member": {"user": {"id": str(interaction.user.id)}},
        "data": {"name": data.get("name"), "options": options},
        "channel": {"id": str(interaction.channel_id or ""), "type": channel_type},
    }
    if parent_id:
        payload["channel"]["parent_id"] = parent_id
    return payload


async def _dispatch_command(discord, interaction, config: Config, store: Store | None) -> None:
    if store is None:
        await interaction.response.send_message("this bot isn't ready for commands yet.", ephemeral=True)
        return
    response = handle_interaction(config, store, _interaction_payload(interaction))
    data = response.get("data") or {}
    content = data.get("content") or "ok"
    ephemeral = bool(data.get("flags", 0) & 64)
    if response.get("type") == 5:
        await interaction.response.defer(ephemeral=ephemeral)
        return
    await interaction.response.send_message(content, ephemeral=ephemeral)


def start_gateway(config: Config, store: Store | None) -> Gateway:
    gateway = Gateway()
    gateway.start(config, store)
    return gateway


def stop_gateway(gateway: Gateway | None) -> None:
    if gateway is not None:
        gateway.stop()
