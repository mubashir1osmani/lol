"""Decide whether to answer, then pick a cheap or capable model with LiteLLM Autorouter.

Jev (TypeSafe System One) is one HTTP call. It is not a chat model. The custom
classifier only reads the tier that call already chose, so Jev is not called again.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass

from .reviews import asks_for_review, wants_recheck
from .poller import extract_pr_numbers

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_INPUT_USD_PER_TOKEN = 4.2e-8
FAST_MODEL = "anthropic/claude-haiku-4-5"
_DECISION_CACHE: dict[str, "Decision"] = {}
_DAILY_COST = 0.0

ROUTING_CONFIG = {
    "classifier_type": "custom",
    "classifier_fallback": "heuristic",
    "tiers": {
        "SIMPLE": "fast",
        "MEDIUM": "fast",
        "COMPLEX": "capable",
        "REASONING": "capable",
    },
    "keyword_tier_rules": [
        {"keywords": ["stack trace", "traceback", "proxy config", "pull request", " pr "], "tier": "COMPLEX"},
    ],
    "session_affinity": True,
    "enable_context_window_escalation": True,
    "return_raw_model_name": True,
}


class TierClassifier:
    """Return the tier Jev already chose. A miss falls back to the heuristic."""

    async def classify(self, context):
        metadata = context.metadata or {}
        tier = metadata.get("decision_tier")
        return tier if isinstance(tier, str) and tier else None


@dataclass(frozen=True)
class Decision:
    reply: bool
    route: str
    tier: str
    reason: str
    model_id: str
    source: str
    input_tokens: int = 0
    cost_usd: float = 0.0


def obvious_skip(text: str) -> bool:
    body = (text or "").strip()
    if not body or len(body) < 8:
        return True
    letters = [ch for ch in body if ch.isalpha()]
    if len(letters) < 3:
        return True
    if body.lower() in {"thanks", "thank you", "ty", "lol", "lgtm", "ok", "okay", "nice", "gm"}:
        return True
    return False


def explicit_review(text: str, repository: str) -> bool:
    if wants_recheck(text or ""):
        return True
    numbers = extract_pr_numbers({"content": text or ""}, repository)
    return bool(numbers) and (asks_for_review(text or "") or True)


def _cache_key(channel: str, text: str) -> str:
    return hashlib.sha256(f"{channel}\n{text}".encode()).hexdigest()


def _questions() -> dict:
    return {
        "should_reply": {
            "type": "noul",
            "instructions": "Is this a question or request the bot can usefully answer, rather than chatter or something only a maintainer can decide?",
            "criteria": {
                "true": "A question, bug report, or request a bot can answer",
                "false": "Chatter, thanks, or a call only a maintainer should make",
            },
        },
        "route": {
            "type": "choice",
            "instructions": "Which route should handle this message?",
            "criteria": {
                "simple_qa": "A short factual question",
                "pr_review": "A request to review a pull request",
                "bug_report": "A bug report or stack trace",
                "maintainer_only": "Merging, permissions, or another maintainer decision",
            },
        },
        "complexity": {
            "type": "score",
            "instructions": "How much reasoning does a good answer need?",
            "criteria": [
                "A short factual answer",
                "Some code or configuration detail",
                "Design, debugging, or a pull-request review",
            ],
        },
    }


def _tier(route: str, score: float) -> str:
    if score >= 1.5 or route in {"pr_review", "bug_report"}:
        return "REASONING" if score >= 1.5 and route == "bug_report" else "COMPLEX"
    if route == "simple_qa":
        return "SIMPLE"
    return "MEDIUM"


def _from_jev(payload: dict, threshold: float) -> Decision:
    answers = payload.get("answers") or {}
    noul = ((answers.get("should_reply") or {}).get("noul"))
    choice = (answers.get("route") or {}).get("choice") or "maintainer_only"
    score = (answers.get("complexity") or {}).get("score")
    score = float(score) if isinstance(score, (int, float)) else 0.0
    probability = float(noul) if isinstance(noul, (int, float)) else 0.0
    route = choice if choice in {"simple_qa", "pr_review", "bug_report", "maintainer_only"} else "maintainer_only"
    reply = probability >= threshold and route != "maintainer_only"
    usage = payload.get("usage") or {}
    tokens = usage.get("input_tokens") if isinstance(usage.get("input_tokens"), int) else 0
    return Decision(
        reply=reply,
        route=route,
        tier=_tier(route, score),
        reason=f"should_reply={probability:.2f} route={route}",
        model_id=str(payload.get("model") or ""),
        source="jev",
        input_tokens=tokens,
        cost_usd=tokens * JEV_INPUT_USD_PER_TOKEN,
    )


def call_jev(model: str, state: str, api_key: str, transport=None) -> dict:
    body = json.dumps({"model": model, "state": state, "questions": _questions()}).encode()
    if transport is not None:
        return transport(body)
    from urllib.request import Request
    from .api import _request_json
    return _request_json(Request(
        JEV_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "litellm-contributor-discord-bot",
        },
    ))


def decide(config, text: str, *, channel: str = "", context: str = "", transport=None) -> Decision:
    """One decision. Heuristics skip the network. Explicit reviews are not decided here."""
    if obvious_skip(text):
        return Decision(False, "skip", "SIMPLE", "obvious chatter", "", "heuristic")
    key = _cache_key(channel, text)
    cached = _DECISION_CACHE.get(key)
    if cached is not None:
        return cached
    state = json.dumps({
        "channel": channel,
        "context": (context or "")[:500],
        "message": (text or "")[:1500],
    })
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip() or getattr(config, "typesafe_api_key", None)
    model = getattr(config, "review_decision_model", None) or "jev-latest"
    threshold = float(getattr(config, "decision_reply_threshold", 0.7) or 0.7)
    if api_key or transport is not None:
        try:
            payload = call_jev(model, state, api_key or "", transport)
            decision = _from_jev(payload, threshold)
        except Exception as error:
            logging.warning("Jev decision failed, using the Haiku fallback: %s", error)
            decision = _haiku_decision(config, text)
    else:
        decision = _haiku_decision(config, text)
    logging.info(
        "Decision should_reply=%s route=%s model=%s source=%s cost_usd=%.8f",
        decision.reply, decision.route, decision.model_id or model, decision.source, decision.cost_usd,
    )
    _note_cost(decision.cost_usd)
    if len(_DECISION_CACHE) > 500:
        _DECISION_CACHE.clear()
    _DECISION_CACHE[key] = decision
    return decision


def _haiku_decision(config, text: str, complete=None) -> Decision:
    model = config.review_classifier_model or FAST_MODEL
    if complete is None:
        try:
            from .review_llm import complete_litellm
            raw = complete_litellm(model, [
                {"role": "system", "content": "Reply with JSON only: {\"reply\": true|false, \"route\": \"simple_qa|pr_review|bug_report|maintainer_only\", \"reason\": \"short\"}"},
                {"role": "user", "content": (text or "")[:800]},
            ])
        except Exception as error:
            logging.warning("Haiku decision fallback failed: %s", error)
            return Decision(False, "maintainer_only", "COMPLEX", "classifier unavailable", "", "haiku")
    else:
        raw = complete(text)
    try:
        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except (ValueError, AttributeError):
        parsed = {}
    route = parsed.get("route") if parsed.get("route") in {"simple_qa", "pr_review", "bug_report", "maintainer_only"} else "maintainer_only"
    reply = bool(parsed.get("reply")) and route != "maintainer_only"
    return Decision(reply, route, _tier(route, 0.0), str(parsed.get("reason") or "haiku"), model, "haiku")


def _note_cost(amount: float) -> None:
    global _DAILY_COST
    if amount <= 0:
        return
    _DAILY_COST += amount
    logging.info("Decision cost today: $%.6f", _DAILY_COST)


def reset_decision_cache() -> None:
    global _DAILY_COST
    _DECISION_CACHE.clear()
    _DAILY_COST = 0.0


def routing_config(plugin=None) -> dict:
    config = dict(ROUTING_CONFIG)
    config["classifier_plugin"] = plugin or TierClassifier()
    return config


def build_router(config):
    """The documented Autorouter. ``fast`` is Haiku. ``capable`` is REVIEW_MODEL."""
    import litellm
    capable = config.review_model or "anthropic/claude-sonnet-4-5"
    fast = getattr(config, "fast_model", None) or FAST_MODEL
    return litellm.Router(
        model_list=[
            {"model_name": "fast", "litellm_params": {"model": fast}},
            {"model_name": "capable", "litellm_params": {"model": capable}},
            {"model_name": "auto", "litellm_params": {
                "model": "auto_router/complexity_router",
                "complexity_router_config": routing_config(),
                "complexity_router_default_model": "capable",
            }},
        ],
        num_retries=0,
    )


def capable_model(config) -> str:
    """The model the COMPLEX and REASONING tiers name. Reviews use this directly."""
    return config.review_model or "anthropic/claude-sonnet-4-5"


def complete_auto(config, messages: list, *, session_id: str, tier: str) -> str:
    """One Autorouter completion. The classifier reads ``tier`` and does not call Jev."""
    import asyncio
    router = build_router(config)
    metadata = {"session_id": str(session_id), "decision_tier": tier}

    async def call():
        return await router.acompletion(model="auto", messages=messages, metadata=metadata)

    response = asyncio.run(call())
    selected = getattr(response, "model", "") or tier
    log_completion_cost(response, str(selected))
    choice = response.choices[0].message
    return getattr(choice, "content", "") or ""


def log_completion_cost(response, selected: str) -> float:
    try:
        import litellm
        cost = float(litellm.completion_cost(completion_response=response))
    except Exception:
        cost = 0.0
    logging.info("Reply model=%s cost_usd=%.6f", selected, cost)
    _note_cost(cost)
    return cost
