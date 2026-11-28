"""Backward-compatible facade for support Q&A, now owned by ``support_agent``."""

from __future__ import annotations

from .api import post_message
from .support_agent import (
    PROMPT as _SYSTEM,
    _recent_channel_reply as _recent,
    answer_support_question,
    is_support_question as is_server_question,
)


def answer_question(config, channel_id: str, text: str, *, complete=None) -> None:
    """Delegate to ``support_agent`` (opens top-level ``support_agent`` AGENT span)."""
    answer_support_question(
        config,
        channel_id,
        text,
        complete=complete,
        poster=lambda cid, body: post_message(
            cid,
            config.discord_token,
            {"content": body, "allowed_mentions": {"parse": []}},
        ),
        fetch_pages=False,
    )


__all__ = [
    "_SYSTEM",
    "_recent",
    "answer_question",
    "is_server_question",
]
