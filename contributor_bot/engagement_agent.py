"""Engagement Agent (``engagement_agent``).

Dedicated top-level OpenInference ``AGENT`` module owning contributor recognition:
- Role matching (``role_matcher`` child span): GitHub enrichment (GraphQL + REST),
  message-evidence identity notes, list-wise LLM matching, negative cache
- Contributor role grants & daily/boot revalidation with Discord guild presence
  + LiteLLM GitHub as source of truth (announcements remain gated by
  ``ROLES_ANNOUNCE_ENABLED``, default ``false``)
- Weekly community merged-PR summary (``weekly_summary`` child span)
"""

from __future__ import annotations

from datetime import datetime

from .config import Config
from .schedule import UTC
from .store import Store
from .tracing import AGENT_ENGAGEMENT, agent_span

AGENT_NAME = AGENT_ENGAGEMENT

PROMPT = (
    "Match external LiteLLM GitHub contributors to Discord guild members using "
    "GitHub profile/commit enrichment, PR-link message evidence, and guild member "
    "metadata, then summarize merged PRs into a warm, concise weekly community thank-you."
)

TOOLS = (
    {
        "name": "enrich_github_profiles",
        "description": "Batch-enrich GitHub contributor profiles, commits, socials, and READMEs.",
    },
    {
        "name": "scan_message_evidence",
        "description": "Infer GitHub <-> Discord links from message history and PR-link posts.",
    },
    {
        "name": "match_contributors",
        "description": "Resolve GitHub PR authors to Discord guild members (deterministic + LLM).",
    },
    {
        "name": "sync_discord_roles",
        "description": "Grant or revoke tiered contributor roles in Discord (announcements gated by ROLES_ANNOUNCE_ENABLED).",
    },
    {
        "name": "summarize_weekly_merges",
        "description": "Turn merged PR titles into plain-English weekly shout-out descriptions.",
    },
    {
        "name": "post_weekly_summary",
        "description": "Post the weekly community merged-PR thank-you to Discord.",
    },
)


def run_role_matching(
    groups: list[dict],
    config: Config,
    store: Store,
    **kwargs,
):
    """Run contributor role matching inside the ``engagement_agent`` AGENT span."""
    from .matcher import match_community_contributors_with_diagnostics

    moment = kwargs.get("now") or datetime.now(UTC)
    repo = getattr(config, "github_repository", "") or "repo"
    with agent_span(
        AGENT_NAME,
        session_id=f"{AGENT_NAME}:role_matcher:{repo}:{moment.date().isoformat()}",
        input_value={"task": "role_matching", "repository": repo, "contributor_count": len(groups)},
        attributes={"engagement.task": "role_matching", "engagement.repository": repo},
    ) as span:
        result = match_community_contributors_with_diagnostics(groups, config, store, **kwargs)
        diagnostics = result[3]
        span.set_output(
            {
                "contributors": diagnostics.contributor_count,
                "members": diagnostics.member_count,
                "candidates": diagnostics.candidate_pair_count,
                "matched": diagnostics.matched_count,
                "skipped": diagnostics.skipped_count,
                "methods": dict(diagnostics.method_counts),
                "skip_reasons": dict(diagnostics.skip_reasons),
            }
        )
        return result


def summarize_weekly_prs(config: Config, groups: list[dict], **kwargs) -> dict[int, str]:
    """Summarize merged PR titles inside the ``engagement_agent`` AGENT span."""
    from .weekly import summarize_pr_titles

    pr_count = sum(len(g.get("pulls") or ()) for g in groups)
    with agent_span(
        AGENT_NAME,
        session_id=f"{AGENT_NAME}:weekly_summary:{pr_count}prs",
        input_value={"task": "weekly_summary", "pr_count": pr_count},
        attributes={"engagement.task": "weekly_summary", "weekly.pr_count": pr_count},
    ) as span:
        summaries = summarize_pr_titles(config, groups, **kwargs)
        span.set_output({"summarized_count": len(summaries)})
        return summaries


def run_weekly_summary(config: Config, store: Store, **kwargs):
    """Check and post the weekly merged-PR community summary inside ``engagement_agent``."""
    from .weekly import post_weekly_if_due

    return post_weekly_if_due(config, store, **kwargs)


def run_daily_roles(config: Config, store: Store, **kwargs):
    """Run daily contributor role sync inside ``engagement_agent``."""
    from .roles_sync import run_daily_roles_if_due

    return run_daily_roles_if_due(config, store, **kwargs)


def run_grants_revalidation(config: Config, store: Store, **kwargs):
    """Revalidate existing contributor role grants inside ``engagement_agent``."""
    from .roles_sync import revalidate_all_grants_on_boot

    return revalidate_all_grants_on_boot(config, store, **kwargs)


def run_roles_backfill(config: Config, store: Store, **kwargs):
    """Run contributor roles backfill inside ``engagement_agent``."""
    from .roles_sync import run_roles_backfill as _run_roles_backfill

    moment = kwargs.get("now") or datetime.now(UTC)
    repo = getattr(config, "github_repository", "") or "repo"
    with agent_span(
        AGENT_NAME,
        session_id=f"{AGENT_NAME}:roles_backfill:{repo}:{moment.date().isoformat()}",
        input_value={"task": "roles_backfill", "repository": repo},
        attributes={"engagement.task": "roles_backfill", "engagement.repository": repo},
    ) as span:
        result = _run_roles_backfill(config, store, **kwargs)
        span.set_output({"ran": result is not None})
        return result


__all__ = [
    "AGENT_NAME",
    "PROMPT",
    "TOOLS",
    "run_daily_roles",
    "run_grants_revalidation",
    "run_role_matching",
    "run_roles_backfill",
    "run_weekly_summary",
    "summarize_weekly_prs",
]
