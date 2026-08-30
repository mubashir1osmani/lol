"""Claude-powered writing for the bot: release digests, feed blurbs, greetings."""

import logging
from typing import Any


SYSTEM_PROMPT = (
    "You are the LiteLLM community assistant in the official LiteLLM Discord "
    "server. You write short, friendly, technically accurate posts for "
    "developers who use LiteLLM. Use Discord markdown. Never invent features, "
    "links, or version numbers that are not in the source material. Reply with "
    "the post content only - no preamble and no closing remarks."
)

# Models where the server-side refusal fallback is worth enabling.
FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5")


def release_prompt(release: dict[str, Any], repository: str) -> str:
    notes = (release.get("body") or "").strip()[:20_000]
    return (
        f"A new release of {repository} just shipped: "
        f"{release.get('name') or release['tag_name']} ({release['tag_name']}).\n\n"
        "Write a Discord digest of these release notes for the community: a "
        "1-2 sentence TL;DR, then up to 6 bullet points covering the changes "
        "that matter most to users (new providers, breaking changes, fixes). "
        "Keep it under 250 words.\n\n"
        f"Release notes:\n{notes or '(no notes were provided)'}"
    )


def feed_prompt(entry: dict[str, str], label: str) -> str:
    source = "the LiteLLM blog" if label == "blog" else "LiteLLM's social media"
    return (
        f"A new post appeared on {source}: \"{entry['title']}\" "
        f"({entry['link']}).\n\n"
        "Write a 2-3 sentence Discord blurb telling the community what the "
        "post covers and why they might care.\n\n"
        f"Post summary:\n{entry.get('summary') or '(no summary available)'}"
    )


def thread_prompt(thread_name: str, first_message: str) -> str:
    return (
        f"A member just opened a support thread titled \"{thread_name}\".\n\n"
        f"Their message:\n{first_message[:4_000] or '(no message yet)'}\n\n"
        "Write a brief first reply: acknowledge their question, share a "
        "pointer to relevant LiteLLM docs ONLY if you are certain one exists "
        "(https://docs.litellm.ai), and let them know the team has been "
        "notified and will follow up. 2-4 sentences."
    )


class ClaudeAgent:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        # Imported here so the bot still runs without the SDK when no
        # ANTHROPIC_API_KEY is configured.
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 1_500) -> str:
        if self.model in FALLBACK_MODELS:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude declined to write this post")
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not text:
            raise RuntimeError("Claude returned an empty response")
        return text


def try_complete(agent: "ClaudeAgent | None", prompt: str) -> str | None:
    """Ask Claude, returning None on any failure so the caller can fall back
    to posting the raw content instead of dropping the update."""
    if agent is None:
        return None
    try:
        return agent.complete(prompt)
    except Exception:
        logging.exception("Claude request failed; falling back to raw content")
        return None
