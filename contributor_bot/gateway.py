"""Discord gateway so the bot shows online and receives messages and slash commands.

discord.py is imported lazily. Without it the process keeps using REST and stays
offline. GitHub webhooks and OAuth stay on the HTTP server. Leave the
Interactions Endpoint URL empty to deliver slash commands here. If that URL is
set, Discord sends interactions to HTTP instead, and that handler still works.

Guild slash commands are registered on a discord.py CommandTree and synced to
the guild on connect, so they show up in channels and in threads. An empty tree
is never synced, because that would wipe commands already registered over REST.
"""

import asyncio
import logging
import threading

from .ack import bind_ack_reactions, emoji_index
from .config import Config
from .interactions import REVIEW_SLASH_COMMANDS, SLASH_COMMANDS, handle_interaction
from .review_job import handle_channel_message, handle_thread_message
from .store import Store


PRESENCE_REVIEW = "for PR reviews"
PRESENCE_ROUNDUP = "LiteLLM contributors"
THREAD_CHANNEL_TYPES = {10, 11, 12}

# Interaction channel objects for threads omit fields discord.py's Thread
# constructor requires. Filling them is what lets a slash command in a thread
# finish parsing instead of being dropped before our handler runs.
_THREAD_STUB_METADATA = {
    "archived": False,
    "auto_archive_duration": 1440,
    "archive_timestamp": "2020-01-01T00:00:00.000000+00:00",
}
_patched = False


class Gateway:
    def __init__(self):
        self.connected = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.client = None
        self._commands_synced = False

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
            if config.pr_review_enabled:
                bind_ack_reactions(config, [])
            return
        try:
            asyncio.run(self._serve(discord, config, store))
        except Exception:
            logging.exception("Discord gateway stopped")
        finally:
            self.connected.clear()

    async def _serve(self, discord, config: Config, store: Store | None) -> None:
        install_gateway_patches(discord)
        intents = build_intents(discord, members=config.discord_members_intent)
        client = discord.Client(intents=intents)
        self.client = client
        tree = _bind_command_tree(discord, client, config, store)

        @client.event
        async def on_ready():
            watching = PRESENCE_REVIEW if config.pr_review_enabled else PRESENCE_ROUNDUP
            await client.change_presence(
                status=discord.Status.online,
                activity=discord.Activity(type=discord.ActivityType.watching, name=watching),
            )
            self.connected.set()
            logging.info("Discord gateway connected as %s, watching %s", client.user, watching)
            try:
                bind_ack_reactions(config, _guild_emojis(client, config.discord_guild_id))
            except Exception:
                logging.exception("Could not resolve ack reactions")
            if self._commands_synced or self._stop.is_set():
                return
            try:
                synced = await sync_guild_commands(tree, discord, config)
            except Exception:
                logging.exception(
                    "Could not sync slash commands to guild %s. "
                    "Commands registered over REST stay in place.",
                    config.discord_guild_id,
                )
                return
            if synced:
                self._commands_synced = True

        @client.event
        async def on_disconnect():
            self.connected.clear()

        @client.event
        async def on_message(message):
            if self._stop.is_set() or message.author.bot or message.guild is None:
                return
            if not config.pr_review_enabled or store is None:
                return
            try:
                payload = _message_payload(message)
            except Exception:
                logging.exception("Could not read a Discord message")
                return
            channel_id = payload["channel_id"]
            if message_is_thread(message, store):
                _start_worker(handle_thread_message, (config, store, payload), "review-followup")
                return
            if not config.review_watch_all_channels and channel_id not in config.review_channel_ids:
                return
            channel_type = channel_type_value(getattr(message.channel, "type", None))
            if channel_type not in {0, 5}:
                return
            _start_worker(handle_channel_message, (config, store, channel_id, payload), "review-message")

        @client.event
        async def on_interaction(interaction):
            # The command tree answers commands it knows. This covers a command
            # that arrived before the tree had it, or a REST-only command.
            if interaction.type != discord.InteractionType.application_command:
                return
            data = interaction.data or {}
            name = data.get("name") if isinstance(data, dict) else None
            if tree_handles(tree, name, interaction.guild_id):
                return
            await _dispatch_command(discord, interaction, config, store)

        self.loop = asyncio.get_running_loop()
        if self._stop.is_set():
            return
        await client.start(config.discord_token)


def _load_discord():
    import discord
    return discord


def _start_worker(target, args, name: str) -> None:
    def run():
        try:
            target(*args)
        except Exception:
            logging.exception("%s failed", name)

    threading.Thread(target=run, name=name, daemon=True).start()


def build_intents(discord, *, members: bool):
    """Guilds, guild messages, and message content. Members only when requested."""
    intents = discord.Intents.none()
    intents.guilds = True
    intents.guild_messages = True
    intents.message_content = True
    if members:
        intents.members = True
    return intents


def channel_type_value(channel_type) -> int | None:
    if channel_type is None:
        return None
    value = getattr(channel_type, "value", channel_type)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def thread_payload(data: dict) -> dict:
    """Partial thread channel from an interaction, plus the fields Thread() indexes.

    Discord's interaction channel object is partial. discord.py 2.7 still reads
    owner_id, message_count, member_count, and thread_metadata, and raises
    KeyError when they are missing. That aborts interaction parsing, so a slash
    command used inside a thread never reaches the bot.
    """
    filled = dict(data)
    filled.setdefault("owner_id", 0)
    filled.setdefault("message_count", 0)
    filled.setdefault("member_count", 0)
    if not filled.get("parent_id"):
        filled["parent_id"] = filled.get("id", 0)
    filled.setdefault("name", "thread")
    meta = dict(filled["thread_metadata"]) if isinstance(filled.get("thread_metadata"), dict) else {}
    meta.setdefault("archived", False)
    meta.setdefault("auto_archive_duration", _THREAD_STUB_METADATA["auto_archive_duration"])
    meta.setdefault("archive_timestamp", _THREAD_STUB_METADATA["archive_timestamp"])
    filled["thread_metadata"] = meta
    return filled


def install_gateway_patches(discord) -> None:
    """Remember raw channel ids, and let partial threads build."""
    global _patched
    if _patched:
        return
    _patched = True
    original_thread = discord.Thread._from_data

    def _thread_from_data(self, data):
        return original_thread(self, thread_payload(data))

    discord.Thread._from_data = _thread_from_data
    original_interaction = discord.Interaction._from_data

    def _interaction_from_data(self, data):
        raw_channel = data.get("channel") or {}
        channel_id = data.get("channel_id") or raw_channel.get("id")
        parent = raw_channel.get("parent_id")
        # Interaction is a slotted class, so this cannot grow new attributes.
        # extras is created before _from_data. Interaction.channel_id is None
        # when the thread is uncached and the partial channel cannot be built.
        self.extras["litellm_channel_id"] = str(channel_id) if channel_id else ""
        self.extras["litellm_parent_id"] = str(parent) if parent else None
        self.extras["litellm_channel_type"] = raw_channel.get("type")
        try:
            return original_interaction(self, data)
        except Exception:
            logging.exception(
                "Discord interaction in channel %s (type %s) failed to parse",
                self.extras.get("litellm_channel_id"),
                self.extras.get("litellm_channel_type"),
            )
            raise

    discord.Interaction._from_data = _interaction_from_data


def commands_for_guild(config: Config) -> list[dict]:
    """Slash commands the gateway syncs to the guild. Empty means do not sync."""
    commands = []
    if config.contributor_roles_enabled:
        commands.extend(SLASH_COMMANDS)
    if config.pr_review_enabled:
        commands.extend(REVIEW_SLASH_COMMANDS)
    return commands


def should_sync_guild_commands(config: Config) -> bool:
    """Guild sync replaces the guild command list. Never sync an empty one."""
    return bool(config.discord_guild_id) and bool(commands_for_guild(config))


def tree_handles(tree, name: str | None, guild_id) -> bool:
    if tree is None or not name:
        return False
    try:
        guild_key = int(guild_id) if guild_id else None
    except (TypeError, ValueError):
        guild_key = None
    guild_commands = getattr(tree, "_guild_commands", {})
    if guild_key and name in guild_commands.get(guild_key, {}):
        return True
    return name in getattr(tree, "_global_commands", {})


def interaction_place(interaction) -> dict:
    """Thread id, parent id, and channel type for a slash command.

    Prefer the raw gateway channel id. discord.py's ``channel_id`` property
    returns None when the channel object was not cached and could not be built,
    which is the usual case for a thread the bot created over REST.
    """
    extras = getattr(interaction, "extras", None)
    extras = extras if isinstance(extras, dict) else {}
    raw_id = str(extras.get("litellm_channel_id") or getattr(interaction, "_litellm_channel_id", "") or "")
    raw_parent = extras.get("litellm_parent_id")
    if raw_parent is None:
        raw_parent = getattr(interaction, "_litellm_parent_id", None)
    parent_id = str(raw_parent) if raw_parent else None
    raw_type = extras.get("litellm_channel_type")
    if raw_type is None:
        raw_type = getattr(interaction, "_litellm_channel_type", None)
    channel_type = channel_type_value(raw_type)
    channel = getattr(interaction, "channel", None)
    resolved_id = ""
    if channel is not None:
        resolved_id = str(getattr(channel, "id", "") or "")
        parent = getattr(channel, "parent_id", None)
        if parent and not parent_id:
            parent_id = str(parent)
        if channel_type is None:
            channel_type = channel_type_value(getattr(channel, "type", None))
    property_id = ""
    try:
        prop = interaction.channel_id
    except Exception:
        prop = None
    if prop:
        property_id = str(prop)
    channel_id = raw_id or resolved_id or property_id
    if channel_type in THREAD_CHANNEL_TYPES and raw_id:
        # The command was used in the thread. Do not substitute the parent channel.
        channel_id = raw_id
    return {
        "channel_id": channel_id,
        "parent_id": parent_id,
        "channel_type": 0 if channel_type is None else channel_type,
    }


def message_channel_id(message) -> str:
    raw = getattr(message, "_litellm_channel_id", None)
    if raw:
        return str(raw)
    channel = getattr(message, "channel", None)
    if channel is not None and getattr(channel, "id", None):
        return str(channel.id)
    if getattr(message, "channel_id", None):
        return str(message.channel_id)
    return ""


def message_is_thread(message, store: Store | None) -> bool:
    """True for a real thread, or for any channel id that is a stored review thread.

    An uncached thread arrives as a partial channel with no type. Its id is still
    the thread id, and that id does not need to be in REVIEW_CHANNEL_IDS.
    """
    if channel_type_value(getattr(getattr(message, "channel", None), "type", None)) in THREAD_CHANNEL_TYPES:
        return True
    channel_id = message_channel_id(message)
    if store is None or not channel_id:
        return False
    return store.review_thread_by_discord(channel_id) is not None


def _message_payload(message) -> dict:
    author = message.author
    return {
        "id": str(message.id),
        "type": 19 if message.reference else 0,
        "content": message.content or "",
        "webhook_id": str(message.webhook_id) if message.webhook_id else None,
        "author": {"id": str(author.id), "bot": bool(author.bot)},
        "channel_id": message_channel_id(message),
    }


def _interaction_payload(interaction) -> dict:
    data = interaction.data or {}
    options = []
    for option in data.get("options") or []:
        if isinstance(option, dict):
            options.append({"name": option.get("name"), "value": option.get("value")})
        else:
            # discord.py app commands pass resolved options on the namespace.
            name = getattr(option, "name", None)
            value = getattr(option, "value", None)
            if name:
                options.append({"name": name, "value": value})
    # A tree callback's interaction.data options are the raw dicts. Also copy
    # a resolved namespace so /review-pr still sees the url parameter.
    namespace = getattr(interaction, "namespace", None)
    if namespace is not None and not options:
        for name, value in getattr(namespace, "__dict__", {}).items():
            if name.startswith("_") or value is None:
                continue
            options.append({"name": name, "value": value})
    place = interaction_place(interaction)
    payload = {
        "type": 2,
        "guild_id": str(interaction.guild_id or ""),
        "application_id": str(interaction.application_id or ""),
        "channel_id": place["channel_id"],
        "token": interaction.token,
        "member": {"user": {"id": str(interaction.user.id)}},
        "data": {"name": data.get("name"), "options": options},
        "channel": {"id": place["channel_id"], "type": place["channel_type"]},
    }
    if place["parent_id"]:
        payload["channel"]["parent_id"] = place["parent_id"]
    message = getattr(interaction, "message", None)
    message_id = getattr(message, "id", None) if message is not None else None
    if message_id:
        payload["message"] = {"id": str(message_id)}
    return payload


def _guild_emojis(client, guild_id: str | None):
    """Custom emoji on the configured guild. Empty when the guild is not cached yet."""
    guild = None
    if guild_id and hasattr(client, "get_guild"):
        try:
            guild = client.get_guild(int(guild_id))
        except (TypeError, ValueError):
            guild = None
    return emoji_index(getattr(guild, "emojis", ()) if guild is not None else ())


async def _dispatch_command(discord, interaction, config: Config, store: Store | None) -> None:
    try:
        if store is None:
            await interaction.response.send_message("this bot isn't ready for commands yet.", ephemeral=True)
            return
        response = handle_interaction(config, store, _interaction_payload(interaction))
        data = response.get("data") or {}
        content = data.get("content") or "ok"
        ephemeral = bool(data.get("flags", 0) & 64)
        if response.get("type") == 5:
            # Ack within Discord's 3 second window. The long review edits this
            # response when it finishes, including cooldown and budget notes.
            if not interaction.response.is_done():
                await interaction.response.defer(ephemeral=ephemeral)
            return
        if interaction.response.is_done():
            await interaction.followup.send(content, ephemeral=ephemeral)
            return
        await interaction.response.send_message(content, ephemeral=ephemeral)
    except Exception:
        logging.exception("Slash command failed")
        try:
            if interaction.response.is_done():
                await interaction.followup.send("couldn't run that command. try again in a bit.", ephemeral=True)
            else:
                await interaction.response.send_message(
                    "couldn't run that command. try again in a bit.", ephemeral=True,
                )
        except Exception:
            logging.exception("Could not send the slash-command error")


def _bind_command_tree(discord, client, config: Config, store: Store | None):
    """Register guild commands on the client's tree. Sync happens on ready."""
    if not should_sync_guild_commands(config):
        return None
    tree = discord.app_commands.CommandTree(client)
    guild = discord.Object(id=int(config.discord_guild_id))
    app = discord.app_commands

    def add(spec: dict, callback):
        wrapped = app.allowed_contexts(guilds=True, dms=False, private_channels=False)(callback)
        tree.command(name=spec["name"], description=spec["description"], guild=guild)(wrapped)

    if config.contributor_roles_enabled:
        for spec in SLASH_COMMANDS:
            async def callback(interaction):
                await _dispatch_command(discord, interaction, config, store)

            callback.__name__ = spec["name"].replace("-", "_")
            add(spec, callback)
    if config.pr_review_enabled:
        recheck_spec = next(spec for spec in REVIEW_SLASH_COMMANDS if spec["name"] == "recheck")
        review_spec = next(spec for spec in REVIEW_SLASH_COMMANDS if spec["name"] == "review-pr")

        async def recheck(interaction):
            await _dispatch_command(discord, interaction, config, store)

        add(recheck_spec, recheck)

        async def review_pr(interaction, url: str):
            await _dispatch_command(discord, interaction, config, store)

        described = app.describe(url=review_spec["options"][0]["description"])(review_pr)
        add(review_spec, described)
    return tree


async def sync_guild_commands(tree, discord, config: Config):
    """Upload the tree to one guild so commands appear immediately, including in threads."""
    if tree is None or not should_sync_guild_commands(config):
        return []
    guild = discord.Object(id=int(config.discord_guild_id))
    synced = await tree.sync(guild=guild)
    logging.info(
        "Synced %s slash commands to guild %s. They are available in channels and threads.",
        len(synced), config.discord_guild_id,
    )
    return synced


def start_gateway(config: Config, store: Store | None) -> Gateway:
    gateway = Gateway()
    gateway.start(config, store)
    return gateway


def stop_gateway(gateway: Gateway | None) -> None:
    if gateway is not None:
        gateway.stop()
