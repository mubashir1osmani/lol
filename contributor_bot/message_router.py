"""Single Discord message router for the three-agent Aria architecture.

Decides which top-level agent handles an incoming Discord channel message:
- PR review request -> ``pr_review_agent``
- LiteLLM support question (in an allowlisted channel when enabled) -> ``support_agent``
- Anything else (chit-chat, disabled feature, non-allowlisted channel, bot message) -> ``none``

Tracing invariant:
- Routing is NEVER emitted as its own top-level ``AGENT`` span.
- When routing to ``pr_review_agent`` or ``support_agent``, the target agent opens
  its top-level ``AGENT`` span and records ``message_router`` as a ``CHAIN`` child span.
- When routing to ``none``, ZERO spans are emitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Config, is_discord_id
from .poller import extract_pr_numbers
from .reviews import asks_for_review, wants_recheck
from .routing import explicit_review, obvious_skip
from .store import Store
from .support_agent import (
    LiteLLMKnowledgeBase,
    is_support_channel_allowed,
    is_support_question,
)
from .tracing import AGENT_PR_REVIEW, AGENT_SUPPORT

ROUTE_PR_REVIEW = AGENT_PR_REVIEW
ROUTE_SUPPORT = AGENT_SUPPORT
ROUTE_NONE = "none"


@dataclass(frozen=True)
class RouteDecision:
    target: str  # "pr_review_agent" | "support_agent" | "none"
    reason: str
    pr_numbers: tuple[int, ...] = ()
    tier: str = "SIMPLE"
    source: str = "heuristic"

    def to_trace_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "reason": self.reason,
            "pr_numbers": list(self.pr_numbers),
            "tier": self.tier,
            "source": self.source,
        }


def is_review_channel_allowed(config: Config, channel_id: str) -> bool:
    """True when ``pr_review_agent`` is enabled and watches ``channel_id``."""
    if not getattr(config, "pr_review_enabled", False):
        return False
    return bool(config.review_watch_all_channels) or str(channel_id) in config.review_channel_ids


def classify_route(config: Config, channel_id: str, message: dict) -> RouteDecision:
    """Classify a Discord channel message into ``pr_review_agent``, ``support_agent``, or ``none``.

    Pure heuristic — emits zero spans and makes zero LLM calls so ``none`` messages
    never create stray traces.
    """
    author = message.get("author") or {}
    if author.get("bot") or message.get("webhook_id"):
        return RouteDecision(ROUTE_NONE, "bot_or_webhook")

    text = str(message.get("content") or "")
    if not text.strip():
        return RouteDecision(ROUTE_NONE, "empty_message")

    repo = config.github_repository
    pr_numbers = tuple(extract_pr_numbers(message, repo))

    # 1. PR review requests take priority over support Q&A.
    if is_review_channel_allowed(config, channel_id):
        if pr_numbers or explicit_review(text, repo):
            return RouteDecision(
                ROUTE_PR_REVIEW,
                "explicit_pr_review",
                pr_numbers=pr_numbers,
                tier="COMPLEX",
                source="heuristic",
            )
        if config.review_classify and (asks_for_review(text) or wants_recheck(text)):
            return RouteDecision(
                ROUTE_PR_REVIEW,
                "review_ask_without_link",
                pr_numbers=(),
                tier="COMPLEX",
                source="heuristic",
            )

    # Even in a support-only channel, never misroute a PR URL or explicit review ask as support Q&A.
    if pr_numbers or asks_for_review(text) or wants_recheck(text):
        return RouteDecision(ROUTE_NONE, "pr_review_not_enabled_for_channel", pr_numbers=pr_numbers)

    # 2. Support questions (gated behind SUPPORT_AGENT_ENABLED + SUPPORT_AGENT_CHANNEL_IDS).
    if is_support_channel_allowed(config, channel_id):
        if obvious_skip(text):
            return RouteDecision(ROUTE_NONE, "obvious_chatter")
        if is_support_question(text, repo):
            return RouteDecision(
                ROUTE_SUPPORT,
                "support_question",
                pr_numbers=(),
                tier="SIMPLE",
                source="heuristic",
            )

    return RouteDecision(ROUTE_NONE, "unmatched")


def route_and_dispatch_message(
    config: Config,
    store: Store | None,
    channel_id: str,
    message: dict,
    *,
    kb: LiteLLMKnowledgeBase | None = None,
    support_complete=None,
    support_poster=None,
) -> RouteDecision:
    """Route one Discord channel message and dispatch to the chosen top-level agent.

    Advances the review cursor on watched review channels before routing.
    When routed to ``none``, emits zero spans and performs no agent work.
    """
    message_id = str(message.get("id") or "")
    if (
        store is not None
        and getattr(config, "pr_review_enabled", False)
        and str(channel_id) in config.review_channel_ids
        and is_discord_id(message_id)
    ):
        try:
            store.advance_review_cursor(str(channel_id), message_id)
        except Exception:
            logging.warning("Could not advance review cursor for channel %s", channel_id)

    decision = classify_route(config, str(channel_id), message)
    if decision.target == ROUTE_PR_REVIEW:
        if store is None:
            logging.warning("PR review message routed without a Store; skipping")
            return RouteDecision(ROUTE_NONE, "missing_store")
        from .pr_review_agent import handle_routed_message as handle_pr_review

        handle_pr_review(
            config,
            store,
            str(channel_id),
            message,
            route_info=decision.to_trace_dict(),
        )
        return decision

    if decision.target == ROUTE_SUPPORT:
        from .support_agent import handle_routed_message as handle_support

        handle_support(
            config,
            store,
            str(channel_id),
            message,
            kb=kb,
            complete=support_complete,
            poster=support_poster,
            route_info=decision.to_trace_dict(),
        )
        return decision

    return decision


__all__ = [
    "ROUTE_NONE",
    "ROUTE_PR_REVIEW",
    "ROUTE_SUPPORT",
    "RouteDecision",
    "classify_route",
    "is_review_channel_allowed",
    "route_and_dispatch_message",
]
