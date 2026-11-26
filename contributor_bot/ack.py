"""Immediate Discord reactions so a slow review still shows the bot saw the message.

Reactions are resolved lazily per guild at reaction time from ``guild.emojis``,
falling back to ``await guild.fetch_emojis()`` (cached per guild with a TTL)
when the guild emoji cache is empty. Reactions are added before the agent run
and never block it. A missing Add Reactions permission, an unknown emoji, or a
Discord error is logged and swallowed. Slash commands with no source message
are not reacted to; the deferred reply is the acknowledgement.
"""

import asyncio
import inspect
import logging
import re
import threading
import time

from .api import add_reaction
from .config import Config, is_discord_id

EYES = "👀"
FALLBACK_TY = "🙏"
# First matching guild emoji is the single custom "ty" reaction.
TY_EMOJI_NAMES = ("ty", "litellm_ty", "thankyou", "thank_you")
EMOJI_CACHE_TTL_SECONDS = 300.0

_CUSTOM_MENTION = re.compile(r"^<(?:a)?:([\w]{1,32}):([1-9][0-9]{0,19})>$")
_CUSTOM_PAIR = re.compile(r"^([\w]{1,32}):([1-9][0-9]{0,19})$")
_PLAIN_NAME = re.compile(r"^[\w]{1,32}$")

_lock = threading.Lock()
_guild_emoji_cache: dict[str, tuple[list[tuple[str, str]], float]] = {}
_logged_guilds: set[str] = set()


def reset_ack_cache() -> None:
    """Test helper. Clears per-guild emoji cache and first-resolution log flags."""
    with _lock:
        _guild_emoji_cache.clear()
        _logged_guilds.clear()


def parse_ack_reactions(raw: str | None) -> tuple[str, ...] | None:
    """Validate ACK_REACTIONS. None means the default eyes + ty sequence."""
    if raw is None or not str(raw).strip():
        return None
    parts = [part.strip() for part in str(raw).split(",")]
    if any(not part for part in parts):
        raise ValueError("ACK_REACTIONS has an empty entry")
    for part in parts:
        _classify(part)
    return tuple(parts)


def _classify(token: str) -> tuple[str, str]:
    if len(token) > 80 or any(ch.isspace() or ch in "/\\" for ch in token):
        raise ValueError(
            "ACK_REACTIONS entries must be an emoji, <:name:id>, name:id, or an emoji name"
        )
    mention = _CUSTOM_MENTION.fullmatch(token)
    if mention:
        return ("custom", f"{mention.group(1)}:{mention.group(2)}")
    pair = _CUSTOM_PAIR.fullmatch(token)
    if pair:
        return ("custom", f"{pair.group(1)}:{pair.group(2)}")
    if token.startswith("<") or token.endswith(">") or ":" in token:
        raise ValueError(
            f"ACK_REACTIONS entry {token!r} is not an emoji, <:name:id>, name:id, or an emoji name"
        )
    if _PLAIN_NAME.fullmatch(token):
        return ("name", token)
    return ("unicode", token)


def _needs_guild_emojis(spec: tuple[str, ...] | None) -> bool:
    if spec is None:
        return True
    return any(_classify(token)[0] == "name" for token in spec)


def emoji_index(emojis) -> list[tuple[str, str]]:
    """(name, id) pairs from discord.py emoji objects, dicts, or (name, id) tuples."""
    found = []
    for emoji in emojis or []:
        if isinstance(emoji, tuple) and len(emoji) >= 2:
            name, emoji_id = emoji[0], emoji[1]
        elif isinstance(emoji, dict):
            name, emoji_id = emoji.get("name"), emoji.get("id")
        else:
            name = getattr(emoji, "name", None)
            emoji_id = getattr(emoji, "id", None)
        if isinstance(name, str) and name and emoji_id is not None and is_discord_id(str(emoji_id)):
            found.append((name, str(emoji_id)))
    return found


def _lookup(name: str, emojis: list[tuple[str, str]]) -> str | None:
    wanted = name.casefold()
    for emoji_name, emoji_id in emojis:
        if emoji_name.casefold() == wanted:
            return f"{emoji_name}:{emoji_id}"
    return None


def _named_or_fallback(names: tuple[str, ...], fallback: str, emojis: list[tuple[str, str]]) -> tuple[str, str]:
    for name in names:
        found = _lookup(name, emojis)
        if found:
            return found, f"{found} ({name})"
    tried = ", ".join(names)
    return fallback, f"{fallback} (fallback; no guild emoji named {tried})"


def resolve_ack_reactions(spec: tuple[str, ...] | None, emojis) -> tuple[tuple[str, ...], str]:
    """API reaction strings, and one log line describing how they were chosen."""
    index = emoji_index(emojis)
    if spec is None:
        eyes = (EYES, EYES)
        ty = _named_or_fallback(TY_EMOJI_NAMES, FALLBACK_TY, index)
        chosen = (eyes, ty)
    else:
        chosen = tuple(_resolve_token(token, index) for token in spec)
    api = tuple(item[0] for item in chosen)
    detail = ", ".join(item[1] for item in chosen) or "(none)"
    return api, f"Ack reactions: {detail}"


def _resolve_token(token: str, emojis: list[tuple[str, str]]) -> tuple[str, str]:
    kind, value = _classify(token)
    if kind == "custom":
        return value, value
    if kind == "unicode":
        return value, value
    found = _lookup(value, emojis)
    if found:
        return found, f"{found} ({value})"
    return "", f"(skipped {value}; no guild emoji with that name)"


def _guild_id(guild, guild_id: str | None, config: Config) -> str:
    raw = getattr(guild, "id", None) if guild is not None else None
    if raw is not None and str(raw):
        return str(raw)
    if guild_id:
        return str(guild_id)
    return str(config.discord_guild_id or "")


def _cached_or_guild_emojis(guild, gid: str, now: float) -> tuple[list[tuple[str, str]], bool]:
    """Return (emojis, should_fetch) from guild.emojis or the per-guild TTL cache."""
    if guild is not None:
        direct = emoji_index(getattr(guild, "emojis", ()))
        if direct:
            return direct, False
    if gid:
        with _lock:
            entry = _guild_emoji_cache.get(gid)
            if entry is not None and now < entry[1]:
                return list(entry[0]), False
    return [], True


def _store_fetched_emojis(gid: str, fetched, ttl_seconds: float, now: float) -> list[tuple[str, str]]:
    emojis = emoji_index(fetched)
    if gid and ttl_seconds > 0:
        with _lock:
            _guild_emoji_cache[gid] = (list(emojis), now + ttl_seconds)
    return emojis


def _finalize_resolution(config: Config, gid: str, emojis: list[tuple[str, str]], *, record_log: bool = True) -> tuple[str, ...]:
    api, line = resolve_ack_reactions(config.ack_reactions, emojis)
    usable = tuple(item for item in api if item)
    if record_log:
        log_key = gid or "(no-guild)"
        should_log = False
        with _lock:
            if log_key not in _logged_guilds:
                _logged_guilds.add(log_key)
                should_log = True
        if should_log:
            names = ", ".join(name for name, _emoji_id in emojis) or "(none)"
            logging.info(
                "Guild %s emojis (%d): %s | %s",
                gid or "unknown",
                len(emojis),
                names,
                line,
            )
    return usable


async def _await_fetch_emojis(guild):
    result = guild.fetch_emojis()
    if inspect.isawaitable(result):
        return await result
    return result


def _fetch_emojis_sync(guild, loop):
    if loop is not None and loop.is_running():
        return asyncio.run_coroutine_threadsafe(_await_fetch_emojis(guild), loop).result(timeout=10)
    result = guild.fetch_emojis()
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


async def resolve_reactions_for_guild_async(
    config: Config,
    guild=None,
    *,
    guild_id: str | None = None,
    ttl_seconds: float = EMOJI_CACHE_TTL_SECONDS,
    clock=time.monotonic,
) -> tuple[str, ...]:
    """Resolve ack reactions lazily from guild.emojis, falling back to await guild.fetch_emojis()."""
    gid = _guild_id(guild, guild_id, config)
    now = clock()
    emojis, should_fetch = _cached_or_guild_emojis(guild, gid, now)
    fetch_failed = False
    if should_fetch and _needs_guild_emojis(config.ack_reactions):
        if guild is not None and callable(getattr(guild, "fetch_emojis", None)):
            try:
                fetched = await _await_fetch_emojis(guild)
                emojis = _store_fetched_emojis(gid, fetched, ttl_seconds, now)
            except Exception as error:
                fetch_failed = True
                logging.warning("Could not fetch emojis for guild %s: %s", gid or "unknown", error)
                emojis = []
    return _finalize_resolution(config, gid, emojis, record_log=not fetch_failed)


def resolve_reactions_for_guild(
    config: Config,
    guild=None,
    *,
    guild_id: str | None = None,
    loop=None,
    ttl_seconds: float = EMOJI_CACHE_TTL_SECONDS,
    clock=time.monotonic,
) -> tuple[str, ...]:
    """Thread-safe lazy resolver. Bridges await guild.fetch_emojis() onto the gateway loop."""
    gid = _guild_id(guild, guild_id, config)
    now = clock()
    emojis, should_fetch = _cached_or_guild_emojis(guild, gid, now)
    fetch_failed = False
    if should_fetch and _needs_guild_emojis(config.ack_reactions):
        if guild is not None and callable(getattr(guild, "fetch_emojis", None)):
            try:
                fetched = _fetch_emojis_sync(guild, loop)
                emojis = _store_fetched_emojis(gid, fetched, ttl_seconds, now)
            except Exception as error:
                fetch_failed = True
                logging.warning("Could not fetch emojis for guild %s: %s", gid or "unknown", error)
                emojis = []
    return _finalize_resolution(config, gid, emojis, record_log=not fetch_failed)


def apply_reactions(
    config: Config,
    channel_id: str,
    message_id: str,
    *,
    guild=None,
    guild_id: str | None = None,
    loop=None,
    ttl_seconds: float = EMOJI_CACHE_TTL_SECONDS,
) -> None:
    """Resolve reactions lazily for this guild and add them in order."""
    reactions = resolve_reactions_for_guild(
        config,
        guild,
        guild_id=guild_id,
        loop=loop,
        ttl_seconds=ttl_seconds,
    )
    for emoji in reactions:
        try:
            add_reaction(channel_id, message_id, emoji, config.discord_token)
        except Exception as error:
            logging.warning(
                "Could not react to message %s in %s with %s: %s",
                message_id, channel_id, emoji, error,
            )


def acknowledge_message(
    config: Config,
    channel_id: object,
    message_id: object,
    *,
    guild=None,
    guild_id: str | None = None,
    loop=None,
) -> None:
    """Fire-and-forget. Returns before Discord answers so the agent can start."""
    channel = str(channel_id or "")
    message = str(message_id or "")
    if not is_discord_id(channel) or not is_discord_id(message):
        return
    threading.Thread(
        target=apply_reactions,
        args=(config, channel, message),
        kwargs={"guild": guild, "guild_id": guild_id, "loop": loop},
        name="ack-react",
        daemon=True,
    ).start()
