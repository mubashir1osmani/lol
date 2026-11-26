#!/usr/bin/env python3
"""Print the Jev routing table for sample Discord messages. No live model calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contributor_bot.config import Config  # noqa: E402
from contributor_bot.routing import decide, reset_decision_cache  # noqa: E402

# Mock Jev answers. Heuristic rows never reach the API.
SAMPLES = [
    {"text": "lol", "skip": True},
    {"text": "thanks!", "skip": True},
    {"text": "ok", "skip": True},
    {"text": "🔥🔥", "skip": True},
    {"text": "nice", "skip": True},
    {"text": "gm", "skip": True},
    {"text": "how do I add an openai provider?", "noul": 0.94, "route": "simple_qa", "score": 0.2},
    {"text": "what does num_retries do?", "noul": 0.9, "route": "simple_qa", "score": 0.4},
    {"text": "where is the proxy config documented?", "noul": 0.88, "route": "simple_qa", "score": 0.8},
    {"text": "can you explain the router strategy in one paragraph?", "noul": 0.86, "route": "simple_qa", "score": 1.1},
    {"text": "please review https://github.com/BerriAI/litellm/pull/12", "explicit": True},
    {"text": "check again", "explicit": True},
    {"text": "the proxy crashes, traceback attached, TypeError in router.py", "noul": 0.92, "route": "bug_report", "score": 1.7},
    {"text": "users get 401 after rotating the virtual key", "noul": 0.84, "route": "bug_report", "score": 1.2},
    {"text": "should we merge this for the design partner?", "noul": 0.8, "route": "maintainer_only", "score": 0.4},
    {"text": "can you grant me the maintainer role?", "noul": 0.77, "route": "maintainer_only", "score": 0.2},
    {"text": "hmm not sure", "noul": 0.42, "route": "simple_qa", "score": 0.3},
    {"text": "ship it", "noul": 0.2, "route": "simple_qa", "score": 0.1},
    {"text": "why is spend tracking double counting streaming calls?", "noul": 0.9, "route": "bug_report", "score": 1.6},
    {"text": "is redis required for the proxy?", "noul": 0.95, "route": "simple_qa", "score": 0.3},
]


def _payload(sample: dict) -> dict:
    route = sample.get("route", "simple_qa")
    return {
        "model": "jev-1.14.0",
        "answers": {
            "should_reply": {"type": "noul", "noul": sample.get("noul", 0.0)},
            "route": {"type": "choice", "choice": route, "confidence": 0.8, "probabilities": {route: 0.8}},
            "complexity": {"type": "score", "score": sample.get("score", 0.2), "confidence": 0.8, "probabilities": {"0": 0.5, "1": 0.5, "2": 0.0}},
        },
        "usage": {"input_tokens": 160, "output_tokens": 24},
    }


def table(config: Config) -> list[dict]:
    reset_decision_cache()
    by_text = {sample["text"]: sample for sample in SAMPLES}
    rows = []
    for sample in SAMPLES:
        if sample.get("explicit"):
            rows.append({
                "message": sample["text"],
                "action": "review",
                "route": "pr_review",
                "tier": "COMPLEX",
                "model": config.review_model or "anthropic/claude-sonnet-4-5",
                "reason": "explicit review skips the gate",
                "cost_usd": 0.0,
            })
            continue

        def transport(body, _sample=sample):
            state = json.loads(body.decode())["state"]
            message = json.loads(state)["message"]
            return _payload(by_text[message])

        decision = decide(config, sample["text"], channel="help", transport=transport)
        keywords = ("stack trace", "traceback", "proxy config", "pull request")
        tier = decision.tier
        if decision.reply and any(word in sample["text"].lower() for word in keywords):
            tier = "COMPLEX"
        if not decision.reply:
            action = "skip"
            model = ""
        elif tier in {"SIMPLE", "MEDIUM"}:
            action = "answer"
            model = config.fast_model
        else:
            action = "answer"
            model = config.review_model or "anthropic/claude-sonnet-4-5"
        rows.append({
            "message": sample["text"],
            "action": action,
            "route": decision.route,
            "tier": tier,
            "model": model,
            "reason": decision.reason,
            "cost_usd": decision.cost_usd,
            "source": decision.source,
        })
    return rows


def main() -> None:
    config = Config(
        "token", "123", "456",
        typesafe_api_key="dry-run",
        review_model="anthropic/claude-sonnet-4-5",
        fast_model="anthropic/claude-haiku-4-5",
    )
    print("mocked Jev: no live TYPESAFE_API_KEY call")
    print(f"{'action':8} {'route':16} {'tier':10} {'model':32} cost_usd  message")
    for row in table(config):
        print(f"{row['action']:8} {row['route']:16} {row['tier']:10} {row['model'][:32]:32} {row['cost_usd']:.8f}  {row['message'][:48]}")


if __name__ == "__main__":
    main()
