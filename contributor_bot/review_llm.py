"""LiteLLM written review. Imported only when a review actually runs.

The SDK is imported inside the call so the Friday roundup and contributor
roles keep working when the package is not installed.
"""

import json
import re

from .reviews import (
    VERDICT_BELOW_STANDARDS,
    VERDICT_LOOKS_GOOD,
    VERDICT_NEEDS_WORK,
    format_finding_bullet,
    load_review_standards,
    normalize_verdict,
)
from .tracing import build_trace_metadata, configure_litellm_tracing

DIFF_CHUNK_CHARS = 20_000
DIFF_MAX_CHUNKS = 3

_SYSTEM_BASE = (
    "You review pull requests for BerriAI/litellm. "
    "Your notes are posted in Discord only. Do not approve, request changes, or sound like a GitHub review. "
    "Write like a busy senior maintainer on Discord: plain, direct, constructive, lowercase is fine. "
    "No praise padding, no 'Great PR!', no headings, no emojis, no em-dashes, no hedging, and no summary of what the PR does. "
    "Return one JSON object and nothing else, with keys "
    f"verdict (one of: '{VERDICT_LOOKS_GOOD}', '{VERDICT_NEEDS_WORK}', '{VERDICT_BELOW_STANDARDS}') "
    "and findings (0 to 5 short objects {path, line, reason, suggestion}; empty array when looks good). "
    "For non-clean verdicts, every finding must include a concrete non-empty suggestion saying what the PR should do instead "
    "(exact file, class/function, or change, keeping problem + fix to one or two short sentences). "
    "No markdown fences."
)


def _system_prompt() -> str:
    return f"{_SYSTEM_BASE}\n\n=== LITELLM CODE QUALITY STANDARDS ===\n{load_review_standards()}"


_SYSTEM = _system_prompt()


def chunk_diff(diff: str, *, chunk_chars: int = DIFF_CHUNK_CHARS, max_chunks: int = DIFF_MAX_CHUNKS) -> tuple[list[str], list[str]]:
    """Split a unified diff on file boundaries. Return ``(chunks, omitted file headers)``."""
    text = diff or ""
    if not text.strip():
        return [], []
    files = [part for part in re.split(r"(?=^diff --git )", text, flags=re.M) if part.strip()]
    if not files:
        files = [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    omitted: list[str] = []

    def header(part: str) -> str:
        line = part.splitlines()[0] if part.splitlines() else "diff"
        return line[:120]

    def flush() -> None:
        nonlocal current, size
        if current:
            chunks.append("".join(current))
            current = []
            size = 0

    for part in files:
        if len(chunks) >= max_chunks:
            omitted.append(header(part))
            continue
        if len(part) > chunk_chars:
            flush()
            if len(chunks) >= max_chunks:
                omitted.append(header(part))
                continue
            chunks.append(part[:chunk_chars] + "\n(file diff truncated)\n")
            continue
        if current and size + len(part) > chunk_chars:
            flush()
            if len(chunks) >= max_chunks:
                omitted.append(header(part))
                continue
        current.append(part)
        size += len(part)
    if current:
        if len(chunks) < max_chunks:
            flush()
        else:
            omitted.extend(header(part) for part in current)
    return chunks, omitted


def parse_review_json(text: str) -> dict:
    """Accept a JSON object, fenced or not. Prose becomes the summary."""
    raw = text or ""
    match = re.search(r"\{.*\}", raw, flags=re.S)
    data = None
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
    if not isinstance(data, dict):
        summary = re.sub(r"\s+", " ", raw).strip()
        return {
            "verdict": VERDICT_NEEDS_WORK,
            "findings": [],
            "summary": summary[:800] or "no written summary",
            "risks": [],
            "missing_tests": [],
            "checklist": [],
        }

    def strings(key: str) -> list[str]:
        value = data.get(key)
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = [str(item) for item in value]
        else:
            items = []
        cleaned = []
        for item in items:
            item = re.sub(r"\s+", " ", item).strip()
            if item:
                cleaned.append(item[:240])
        return cleaned[:5]

    raw_findings = data.get("findings")
    findings: list[str] = []
    if isinstance(raw_findings, list):
        for item in raw_findings[:5]:
            bullet = format_finding_bullet(item)
            if bullet:
                findings.append(bullet)
    verdict = normalize_verdict(data.get("verdict"))
    if verdict is None:
        verdict = VERDICT_NEEDS_WORK if findings else VERDICT_LOOKS_GOOD

    summary = data.get("summary") if isinstance(data.get("summary"), str) else ""
    summary = re.sub(r"\s+", " ", summary).strip()[:800]
    return {
        "verdict": verdict,
        "findings": findings,
        "summary": summary or "no written summary",
        "risks": strings("risks"),
        "missing_tests": strings("missing_tests"),
        "checklist": strings("checklist"),
    }


def complete_litellm(
    model: str,
    messages: list[dict],
    *,
    timeout: float = 60.0,
    metadata: dict | None = None,
) -> str:
    """One ``litellm.completion`` call. Provider keys come from the process environment."""
    try:
        import litellm
    except ImportError as error:
        raise RuntimeError("litellm is not installed") from error
    configure_litellm_tracing()
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 700,
        "timeout": timeout,
    }
    if metadata:
        kwargs["metadata"] = metadata
    response = litellm.completion(**kwargs)
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        raise RuntimeError("litellm returned an empty review")
    message = getattr(choices[0], "message", None)
    if message is None and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                texts.append(str(part.get("text") or ""))
            else:
                texts.append(str(getattr(part, "text", part)))
        content = "".join(texts)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("litellm returned an empty review")
    return content


def _chunk_messages(chunk: str, index: int, total: int) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"diff part {index} of {total}. reply with the json object only.\n\n{chunk}"
            ),
        },
    ]


def _merge_messages(notes: list[str], omitted: list[str]) -> list[dict]:
    omitted_line = ", ".join(omitted[:20]) if omitted else "none"
    partials = "\n\n".join(f"partial {index}:\n{note}" for index, note in enumerate(notes, start=1))
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                "combine these partial reviews of one pull request into one json object. "
                "drop duplicates. mention a truncated diff in the summary if needed.\n"
                f"files not included: {omitted_line}\n\n{partials}"
            ),
        },
    ]


def run_llm_review(diff: str, model: str, complete=None, *, session_id: str | None = None) -> dict:
    """Review a diff in at most a few chunks, then one combined JSON object."""
    if complete is None:
        metadata = build_trace_metadata("pr_review", session_id=session_id)
        complete = lambda m, msgs: complete_litellm(m, msgs, metadata=metadata)
    chunks, omitted = chunk_diff(diff)
    if not chunks:
        return {
            "summary": "no diff text to review",
            "risks": [],
            "missing_tests": [],
            "checklist": ["check the files tab, the diff was empty"],
            "omitted": omitted,
        }
    notes = [
        complete(model, _chunk_messages(chunk, index, len(chunks)))
        for index, chunk in enumerate(chunks, start=1)
    ]
    if len(notes) == 1 and not omitted:
        parsed = parse_review_json(notes[0])
    else:
        parsed = parse_review_json(complete(model, _merge_messages(notes, omitted)))
    parsed["omitted"] = omitted
    return parsed
