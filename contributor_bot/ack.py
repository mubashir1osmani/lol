"""Immediate Discord reactions so a slow review still shows the bot saw the message.

Reactions are added before the agent run and never block it. A missing Add
Reactions permission, an unknown emoji, or a Discord error is logged and
swallowed. Slash commands with no source message are not reacted to; the
deferred reply is the acknowledgement.
"""

import logging
import re
import threading

from .api import add_reaction
from .config import Config, is_discord_id

EYES = "👀"
FALLBACK_TY = "🙏"
# First matching guild emoji is the single custom "ty" reaction.
TY_EMOJI_NAMES = ("ty", "litellm_ty", "thankyou", "thank_you")

_CUSTOM_MENTION = re.compile(r"^<(?:a)?:([\w]{1,32}):([1-9][0-9]{0,19})>$")
_CUSTOM_PAIR = re.compile(r"^([\w]{1,32}):([1-9][0-9]{0,19})$")
_PLAIN_NAME = re.compile(r"^[\w]{1,32}$")

_lock = threading.Lock()
_current: tuple[str, ...] | None = None
_logged = False


def reset_ack_cache() -> None:
    """Test helper. Production resolves once per process."""
    global _current, _logged
    with _lock:
        _current = None
        _logged = False


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


def emoji_index(emojis) -> list[tuple[str, str]]:
    """(name, id) pairs from discord.py emoji objects or (name, id) tuples."""
    found = []
    for emoji in emojis or []:
        if isinstance(emoji, tuple) and len(emoji) >= 2:
            name, emoji_id = emoji[0], emoji[1]
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


def bind_ack_reactions(config: Config, emojis) -> tuple[str, ...]:
    """Remember the resolved list and log it once."""
    global _current, _logged
    api, line = resolve_ack_reactions(config.ack_reactions, emojis)
    usable = tuple(item for item in api if item)
    with _lock:
        _current = usable
        if not _logged:
            logging.info("%s", line)
            _logged = True
    return usable


def current_ack_reactions(config: Config) -> tuple[str, ...]:
    """The list to apply. Before the gateway resolves guild emojis, names fall back."""
    with _lock:
        if _current is not None:
            return _current
    api, _line = resolve_ack_reactions(config.ack_reactions, ())
    return tuple(item for item in api if item)


def apply_reactions(config: Config, channel_id: str, message_id: str) -> None:
    """Add each ack reaction in order. One failure does not stop the rest or the review."""
    for emoji in current_ack_reactions(config):
        try:
            add_reaction(channel_id, message_id, emoji, config.discord_token)
        except Exception as error:
            logging.warning(
                "Could not react to message %s in %s with %s: %s",
                message_id, channel_id, emoji, error,
            )


def acknowledge_message(config: Config, channel_id: object, message_id: object) -> None:
    """Fire-and-forget. Returns before Discord answers so the agent can start."""
    channel = str(channel_id or "")
    message = str(message_id or "")
    if not is_discord_id(channel) or not is_discord_id(message):
        return
    threading.Thread(
        target=apply_reactions,
        args=(config, channel, message),
        name="ack-react",
        daemon=True,
    ).start()
