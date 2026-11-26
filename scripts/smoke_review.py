#!/usr/bin/env python3
"""Run the real review path against open BerriAI/litellm pull requests.

Uses the Agents SDK and LitellmModel. GitHub reads are real. When no provider
key is set, LiteLLM's completion is scripted so the tool loop, usage hooks,
and Discord post path still run. That case is printed as mocked.

    python3 scripts/smoke_review.py
    REVIEW_AGENT_MAX_TOKENS=1000000 python3 scripts/smoke_review.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contributor_bot.ack import EYES, bind_ack_reactions, reset_ack_cache  # noqa: E402
from contributor_bot.config import Config  # noqa: E402
from contributor_bot.review_agent import run_review_session  # noqa: E402
from contributor_bot.review_job import handle_channel_message, handle_thread_message  # noqa: E402
from contributor_bot.reviews import DEFAULT_REVIEW_BOTS  # noqa: E402
from contributor_bot.store import Store  # noqa: E402

LIVE_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY")
PRS = (54677, None)  # None is replaced with a small open pull request


def _live() -> str | None:
    for name in LIVE_KEYS:
        if os.environ.get(name, "").strip():
            return name
    return None


def _small_pr(token: str | None) -> int:
    from contributor_bot.api import github_get_json
    pulls = github_get_json(
        "https://api.github.com/repos/BerriAI/litellm/pulls?state=open&per_page=5&sort=updated",
        token,
        pace=True,
    )
    ranked = []
    for pull in pulls or []:
        number = pull.get("number") if isinstance(pull, dict) else None
        if not isinstance(number, int) or number == 54677:
            continue
        detail = github_get_json(
            f"https://api.github.com/repos/BerriAI/litellm/pulls/{number}",
            token,
            pace=True,
        )
        changed = detail.get("changed_files") if isinstance(detail, dict) else None
        title = detail.get("title") if isinstance(detail, dict) else ""
        ranked.append((changed if isinstance(changed, int) else 10_000, number, title or ""))
    ranked.sort()
    if not ranked:
        raise SystemExit("No open pull request to use as the small sample")
    print(f"small pull request: #{ranked[0][1]} ({ranked[0][0]} files) {ranked[0][2]}")
    return ranked[0][1]


def _config(cap: int) -> Config:
    token = os.environ.get("GITHUB_TOKEN", "").strip() or None
    return replace(
        Config("token", "123", "456", github_token=token),
        pr_review_enabled=True,
        review_model=os.environ.get("REVIEW_MODEL", "anthropic/claude-sonnet-4-5"),
        review_bot_logins=DEFAULT_REVIEW_BOTS,
        review_usage_cap=cap,
        review_max_steps=0,
        review_timeout_seconds=0,
        review_daily_run_budget=0,
        review_classify=False,
        github_repository="BerriAI/litellm",
        discord_guild_id="333",
        discord_application_id="444",
    )


def _install_fake_litellm():
    import litellm
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    state = {"n": 0}

    def _message(name: str | None, arguments: dict | None, content: str | None):
        if name:
            return Message(role="assistant", content=None, tool_calls=[{
                "id": f"call_{state['n']}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments or {})},
            }])
        return Message(role="assistant", content=content or "done")

    def _finished(messages) -> set[str]:
        done = set()
        for message in messages or []:
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
            if role != "tool":
                continue
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
            text = content if isinstance(content, str) else json.dumps(content)
            if "ready_ignoring_agent_bugs" in text:
                done.add("review_bundle")
            if "\"ready\"" in text:
                done.add("submit_verdict")
        return done

    async def acompletion(**kwargs):
        state["n"] += 1
        done = _finished(kwargs.get("messages") or [])
        if "review_bundle" not in done:
            name, arguments, content = "review_bundle", {}, None
        elif "submit_verdict" not in done:
            name, arguments, content = "submit_verdict", {
                "summary": "smoke: gate comes from review_bundle, not from reading every file",
                "risks": "",
                "missing_tests": "",
                "checklist": "",
                "blocking_json": "[]",
            }, None
        else:
            name, arguments, content = None, None, "smoke review finished"
        # Large prompt totals on purpose. Summing them exceeds 1_000_000.
        usage = Usage(prompt_tokens=600_000, completion_tokens=40, total_tokens=600_040)
        message = _message(name, arguments, content)
        finish = "tool_calls" if name else "stop"
        return ModelResponse(
            model=kwargs.get("model") or "fake",
            choices=[Choices(finish_reason=finish, index=0, message=message)],
            usage=usage,
        )

    litellm.acompletion = acompletion
    return state


def _time_tools():
    import contributor_bot.review_agent as review_agent
    real = review_agent.execute_tool

    def timed(ctx, name, arguments):
        started = time.perf_counter()
        try:
            output = real(ctx, name, arguments)
        except Exception:
            print(f"  tool {name} failed after {time.perf_counter() - started:.2f}s")
            raise
        print(f"  tool {name:16} {time.perf_counter() - started:6.2f}s  {len(output):6} chars")
        return output

    review_agent.execute_tool = timed
    return real


def _run_one(number: int, cap: int, posts: list) -> None:
    import contributor_bot.review_agent as review_agent
    config = _config(cap)
    directory = tempfile.TemporaryDirectory()
    store = Store(str(Path(directory.name) / "roles.db"))

    def post(channel_id, token, message, mention_roles=None):
        posts.append(message.get("content") or "")
        return {"id": str(len(posts)), "channel_id": channel_id}

    review_agent.post_message = post
    started = time.perf_counter()
    result = run_review_session(config, store, number, "900", mode="review")
    elapsed = time.perf_counter() - started
    store.close()
    directory.cleanup()
    print(f"\n=== PR #{number}  REVIEW_AGENT_MAX_TOKENS={cap or 'unset'}  {elapsed:.1f}s ===")
    print(f"steps={result.steps} tool_calls={result.tool_calls} "
          f"input_tokens={result.input_tokens} output_tokens={result.output_tokens} "
          f"exhausted={result.exhausted}")
    if result.step_seconds:
        print("step_seconds=" + ", ".join(f"{item:.2f}" for item in result.step_seconds))
    print("--- verdict ---")
    text = "\n".join(posts) if posts else result.text
    print(text[:2000] or "(empty)")
    posts.clear()
    if result.exhausted == "tokens":
        raise SystemExit(f"#{number} still stopped on the token budget with cap {cap}")


def _discord_path(number: int) -> None:
    print("\n=== discord fake client ===")
    reset_ack_cache()
    config = _config(0)
    bind_ack_reactions(config, [("ty", "111111111111111111")])
    directory = tempfile.TemporaryDirectory()
    store = Store(str(Path(directory.name) / "roles.db"))
    discord = {"reactions": [], "posts": [], "threads": []}

    def add_reaction(channel_id, message_id, emoji, token):
        discord["reactions"].append(emoji)
        print(f"  react {channel_id}/{message_id} {emoji}")

    def post(channel_id, token, message, mention_roles=None):
        discord["posts"].append(message.get("content") or "")
        return {"id": "1", "channel_id": channel_id}

    def start_thread(channel_id, message_id, token, name):
        discord["threads"].append(name)
        print(f"  thread {name} on {message_id}")
        return {"id": "900"}

    import contributor_bot.ack as ack
    import contributor_bot.review_agent as review_agent
    import contributor_bot.review_job as review_job
    ack.add_reaction = add_reaction
    review_agent.post_message = post
    review_job.post_message = post
    review_job.start_public_thread = start_thread
    review_job.fetch_recent_messages = lambda *args, **kwargs: []

    handle_channel_message(config, store, "321", {
        "id": "654",
        "author": {"id": "111", "bot": False},
        "content": f"please review https://github.com/BerriAI/litellm/pull/{number}",
    })
    for thread in threading.enumerate():
        if thread.name in {"ack-react", "pr-review"}:
            thread.join(timeout=120)
    print("reactions:", " ".join(discord["reactions"]))
    print("threads:", discord["threads"])
    print("verdict:", (discord["posts"][0][:300] if discord["posts"] else "(none)"))
    if discord["reactions"][:1] != [EYES]:
        raise SystemExit("discord path did not react with eyes first")
    handle_thread_message(config, store, {
        "id": "8888",
        "channel_id": "900",
        "content": "check again",
        "author": {"id": "111", "bot": False},
    })
    for thread in threading.enumerate():
        if thread.name in {"ack-react", "pr-review"}:
            thread.join(timeout=120)
    print("after check again, reactions:", " ".join(discord["reactions"]))
    store.close()
    directory.cleanup()


def main() -> None:
    os.environ.setdefault("REVIEW_AGENT_TRACING", "false")
    key = _live()
    if key:
        print(f"live model: {key} is set, model={os.environ.get('REVIEW_MODEL', 'anthropic/claude-sonnet-4-5')}")
    else:
        print("mocked model: no ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / GROQ_API_KEY")
        _install_fake_litellm()
    _time_tools()
    token = os.environ.get("GITHUB_TOKEN", "").strip() or None
    numbers = [54677, _small_pr(token)]
    posts: list[str] = []
    for number in numbers:
        for cap in (0, 1_000_000):
            _run_one(number, cap, posts)
    if not key:
        _discord_path(numbers[-1])
    print("\nsmoke finished")


if __name__ == "__main__":
    main()
