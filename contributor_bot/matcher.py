"""Automatic GitHub-to-Discord contributor matching for the weekly summary.

Matches community PR authors to LiteLLM Discord guild members without requiring
contributors to run /link-github:
1. Optional /link-github store or persisted SQLite auto-matches (computed once).
2. GitHub profile (login, name, blog, twitter_username, bio) + PR commit author
   names and email local parts.
3. Discord guild members from cache or guild.fetch_members (username, global_name, nick).
4. Deterministic strong signals (normalized login/name, Discord handle in bio/blog,
   matching twitter_username) when unique in the guild.
5. Single batched Jev (TypeSafe System One, REVIEW_DECISION_MODEL) noul request
   for ambiguous or fuzzy candidates, accepting >= 0.85, with Haiku fallback when
   TYPESAFE_API_KEY is absent.
"""

from __future__ import annotations

import asyncio
import difflib
import inspect
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

from .api import GITHUB_API, TransientAPIError, github_get_json
from .config import is_discord_id
from .routing import FAST_MODEL, JEV_INPUT_USD_PER_TOKEN, _note_cost, call_jev_questions
from .schedule import UTC, iso

JEV_MATCH_THRESHOLD = 0.85
JEV_RUNNER_UP_MARGIN = 0.05
MAX_CANDIDATES_PER_AUTHOR = 3
MAX_PAIRS_PER_BATCH = 30
FUZZY_CANDIDATE_THRESHOLD = 0.72

_DISCORD_USER_URL = re.compile(
    r"https?://(?:www\.)?(?:discord\.com|discordapp\.com)/users/([1-9][0-9]{15,19})\b",
    re.IGNORECASE,
)
_DISCORD_INVITE_URL = re.compile(
    r"https?://(?:www\.)?(?:discord\.gg|discord\.com/invite|discordapp\.com/invite)/\S+",
    re.IGNORECASE,
)
_DISCORD_HANDLE_IN_TEXT = re.compile(
    r"\b(?:discord|dc)\b(?:\s*(?:username|handle|tag|id|is))?\s*(?:[:=@|/\-]\s*|\s+)@?([A-Za-z0-9_.-]{2,32})(?:#[0-9]{4})?",
    re.IGNORECASE,
)
_LEGACY_DISCORD_TAG = re.compile(r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]{2,32})#[0-9]{4}\b")
_NOREPLY_ID_PREFIX = re.compile(r"^[0-9]+\+(.+)$")

_IGNORED_EMAIL_LOCALS = frozenset({
    "noreply", "no-reply", "donotreply", "action", "actions", "github", "github-actions",
    "bot", "dependabot", "renovate", "claude", "cursor", "admin", "root", "info",
    "contact", "hello", "hi", "mail", "email", "dev", "developer", "me", "support",
})
_IGNORED_COMMIT_NAMES = frozenset({
    "github", "web-flow", "github-actions", "github-actions[bot]",
    "dependabot[bot]", "renovate[bot]", "claude", "cursor",
})
_IGNORED_DISCORD_HANDLE_WORDS = frozenset({
    "server", "invite", "channel", "community", "guild", "bot", "link", "here", "join",
    "https", "http", "com", "gg", "org", "net", "io",
})


@dataclass(frozen=True)
class GitHubAuthorProfile:
    login: str
    user_id: int | None = None
    name: str = ""
    blog: str = ""
    twitter_username: str = ""
    bio: str = ""
    commit_names: tuple[str, ...] = ()
    email_locals: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscordMemberInfo:
    discord_id: str
    username: str = ""
    global_name: str = ""
    nick: str = ""


@dataclass(frozen=True)
class CandidatePair:
    profile: GitHubAuthorProfile
    member: DiscordMemberInfo
    heuristic_score: float
    signals: tuple[str, ...]


@dataclass(frozen=True)
class MatchDetail:
    github_login: str
    discord_id: str
    method: str
    confidence: float
    discord_username: str = ""
    pr_count: int = 0


def _fold_ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold()


def normalize_strict(text: str) -> str:
    """Lowercase and strip punctuation/spaces while preserving digits."""
    return re.sub(r"[^a-z0-9]+", "", _fold_ascii(text))


def normalize_base(text: str) -> str:
    """Lowercase, strip punctuation/spaces, and drop trailing digit suffixes."""
    strict = normalize_strict(text)
    if not strict:
        return ""
    stem = re.sub(r"\d+$", "", strict)
    return stem if len(stem) >= 3 else strict


def _name_variants(text: str) -> set[str]:
    out: set[str] = set()
    strict = normalize_strict(text)
    if strict:
        out.add(strict)
    base = normalize_base(text)
    if base:
        out.add(base)
    return out


def extract_discord_ids_from_text(text: str) -> list[str]:
    if not text:
        return []
    found: list[str] = []
    for match in _DISCORD_USER_URL.finditer(text):
        uid = match.group(1)
        if is_discord_id(uid) and uid not in found:
            found.append(uid)
    return found


def extract_discord_handles_from_text(text: str) -> list[str]:
    """Extract explicit Discord usernames from a GitHub bio or blog string."""
    if not text:
        return []
    cleaned = _DISCORD_INVITE_URL.sub(" ", text)
    cleaned = _DISCORD_USER_URL.sub(" ", cleaned)
    found: list[str] = []
    for pattern in (_DISCORD_HANDLE_IN_TEXT, _LEGACY_DISCORD_TAG):
        for match in pattern.finditer(cleaned):
            raw = (match.group(1) or "").strip().strip(".,;:!?()[]{}\"'")
            lowered = raw.casefold()
            if (
                len(raw) >= 2
                and lowered not in _IGNORED_DISCORD_HANDLE_WORDS
                and not lowered.startswith("discord.")
                and lowered not in {h.casefold() for h in found}
            ):
                found.append(raw)
    return found


def _extract_email_local(email: str) -> str | None:
    if not isinstance(email, str) or "@" not in email:
        return None
    local, domain = email.strip().split("@", 1)
    domain_lower = domain.strip().casefold()
    if domain_lower.endswith("users.noreply.github.com"):
        m = _NOREPLY_ID_PREFIX.match(local)
        local = m.group(1) if m else local
    local = local.strip()
    if not local or "[bot]" in local.casefold():
        return None
    if local.casefold() in _IGNORED_EMAIL_LOCALS:
        return None
    if len(normalize_strict(local)) < 3:
        return None
    return local


def _is_rate_limit_error(error: Exception) -> bool:
    if isinstance(error, TransientAPIError):
        return True
    text = str(error)
    return "HTTP 403" in text or "HTTP 429" in text or "rate limit" in text.lower()


def fetch_github_author_profile(
    login: str,
    repository: str,
    pull_numbers: list[int],
    token: str | None,
    *,
    user_id: int | None = None,
    getter=None,
) -> tuple[GitHubAuthorProfile, bool]:
    """Fetch GitHub user profile + PR commit author names and email local parts."""
    fetch = getter or github_get_json
    name = ""
    blog = ""
    twitter = ""
    bio = ""
    email_locals: list[str] = []
    commit_names: list[str] = []
    rate_limited = False

    def _add_email(raw_email: object) -> None:
        if isinstance(raw_email, str):
            local = _extract_email_local(raw_email)
            if local and local.casefold() not in {e.casefold() for e in email_locals}:
                email_locals.append(local)

    def _add_commit_name(raw_name: object) -> None:
        if not isinstance(raw_name, str):
            return
        cleaned = raw_name.strip()
        if not cleaned or cleaned.casefold() in _IGNORED_COMMIT_NAMES:
            return
        if cleaned.casefold() not in {n.casefold() for n in commit_names}:
            commit_names.append(cleaned)

    if login and re.fullmatch(r"[A-Za-z0-9-]+", login):
        try:
            profile = fetch(f"{GITHUB_API}/users/{login}", token, pace=False)
            if isinstance(profile, dict):
                name = (profile.get("name") or "").strip() if isinstance(profile.get("name"), str) else ""
                blog = (profile.get("blog") or "").strip() if isinstance(profile.get("blog"), str) else ""
                twitter = (
                    (profile.get("twitter_username") or "").strip().lstrip("@")
                    if isinstance(profile.get("twitter_username"), str) else ""
                )
                bio = (profile.get("bio") or "").strip() if isinstance(profile.get("bio"), str) else ""
                _add_email(profile.get("email"))
        except Exception as error:
            if _is_rate_limit_error(error):
                rate_limited = True
            logging.warning("Could not fetch GitHub profile for %s: %s", login, error)

    if not rate_limited and repository and pull_numbers:
        for pr_number in pull_numbers[:2]:
            if not isinstance(pr_number, int) or pr_number <= 0:
                continue
            try:
                commits = fetch(
                    f"{GITHUB_API}/repos/{repository}/pulls/{pr_number}/commits?per_page=30",
                    token,
                    pace=False,
                )
            except Exception as error:
                if _is_rate_limit_error(error):
                    rate_limited = True
                logging.warning("Could not fetch commits for %s#%s: %s", repository, pr_number, error)
                break
            if not isinstance(commits, list):
                continue
            for entry in commits:
                if not isinstance(entry, dict):
                    continue
                author_obj = entry.get("author")
                if isinstance(author_obj, dict):
                    commit_login = author_obj.get("login")
                    if isinstance(commit_login, str) and commit_login.casefold() != login.casefold():
                        continue
                commit = entry.get("commit") if isinstance(entry.get("commit"), dict) else {}
                c_author = commit.get("author") if isinstance(commit.get("author"), dict) else {}
                _add_commit_name(c_author.get("name"))
                _add_email(c_author.get("email"))
            if commit_names and email_locals:
                break

    return (
        GitHubAuthorProfile(
            login=login,
            user_id=user_id,
            name=name,
            blog=blog,
            twitter_username=twitter,
            bio=bio,
            commit_names=tuple(commit_names),
            email_locals=tuple(email_locals),
        ),
        rate_limited,
    )


def member_from_raw(member) -> DiscordMemberInfo | None:
    if isinstance(member, DiscordMemberInfo):
        return member
    if isinstance(member, dict):
        user = member.get("user") if isinstance(member.get("user"), dict) else {}
        if user.get("bot") or member.get("bot"):
            return None
        member_id = user.get("id") or member.get("id")
        if member_id is None or not is_discord_id(str(member_id)):
            return None
        username = str(user.get("username") or member.get("username") or member.get("name") or "").strip()
        global_name = str(user.get("global_name") or member.get("global_name") or "").strip()
        nick = str(member.get("nick") or "").strip()
        return DiscordMemberInfo(
            discord_id=str(member_id),
            username=username,
            global_name=global_name,
            nick=nick,
        )
    if getattr(member, "bot", False):
        return None
    member_id = getattr(member, "id", None)
    if member_id is None or not is_discord_id(str(member_id)):
        return None
    username = str(getattr(member, "name", None) or "").strip()
    global_name = str(getattr(member, "global_name", None) or "").strip()
    nick = str(getattr(member, "nick", None) or "").strip()
    return DiscordMemberInfo(
        discord_id=str(member_id),
        username=username,
        global_name=global_name,
        nick=nick,
    )


async def _collect_async_members(result) -> list:
    if hasattr(result, "__aiter__"):
        found = []
        async for member in result:
            found.append(member)
        return found
    if inspect.isawaitable(result):
        resolved = await result
        return list(resolved) if resolved is not None else []
    return list(result) if result is not None else []


def _run_guild_coro(coro, loop):
    if loop is not None and loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=30)
    return asyncio.run(coro)


def load_guild_members(guild, loop=None) -> list[DiscordMemberInfo]:
    """Load non-bot Discord members from guild cache, falling back to guild.fetch_members."""
    if guild is None:
        return []
    raw_cached = list(getattr(guild, "members", ()) or ())
    cached = [info for raw in raw_cached if (info := member_from_raw(raw)) is not None]
    member_count = getattr(guild, "member_count", None)
    fetcher = getattr(guild, "fetch_members", None)
    cache_looks_incomplete = (
        not cached
        or (isinstance(member_count, int) and member_count > 5 and len(cached) <= 1)
    )
    if cached and (not cache_looks_incomplete or not callable(fetcher)):
        return cached
    if callable(fetcher):
        try:
            async def run():
                return await _collect_async_members(fetcher(limit=None))

            fetched = _run_guild_coro(run(), loop)
            parsed = [info for raw in fetched if (info := member_from_raw(raw)) is not None]
            if parsed:
                return parsed
        except Exception as error:
            logging.warning("guild.fetch_members failed; using cached members: %s", error)
    return cached


def _member_fields(member: DiscordMemberInfo) -> tuple[str, ...]:
    return tuple(val for val in (member.username, member.global_name, member.nick) if val)


def _member_tokens(member: DiscordMemberInfo) -> tuple[set[str], set[str], set[str]]:
    raw_lower: set[str] = set()
    strict_set: set[str] = set()
    base_set: set[str] = set()
    for field in _member_fields(member):
        lowered = field.strip().casefold()
        if lowered:
            raw_lower.add(lowered)
        s = normalize_strict(field)
        if s:
            strict_set.add(s)
        b = normalize_base(field)
        if b:
            base_set.add(b)
    return raw_lower, strict_set, base_set


def _exact_strict_match(value: str, raw_lower: set[str], strict_set: set[str]) -> bool:
    cleaned = (value or "").strip()
    if not cleaned:
        return False
    lowered = cleaned.casefold()
    if len(lowered) >= 3 and lowered in raw_lower:
        return True
    strict = normalize_strict(cleaned)
    return len(strict) >= 4 and strict in strict_set


def _exact_base_match(value: str, base_set: set[str]) -> bool:
    cleaned = (value or "").strip()
    if not cleaned:
        return False
    base = normalize_base(cleaned)
    return len(base) >= 4 and base in base_set


def _exact_field_match(value: str, raw_lower: set[str], strict_set: set[str], base_set: set[str]) -> bool:
    return _exact_strict_match(value, raw_lower, strict_set) or _exact_base_match(value, base_set)


def _match_tier_hits(value: str, member_index: list) -> list[DiscordMemberInfo]:
    """Match strict normalized form first; only strip trailing digit suffixes if no strict match."""
    strict_hits = [
        member
        for member, raw_l, strict_s, _base_s in member_index
        if _exact_strict_match(value, raw_l, strict_s)
    ]
    if strict_hits:
        return strict_hits
    return [
        member
        for member, _raw_l, _strict_s, base_s in member_index
        if _exact_base_match(value, base_s)
    ]


def _name_initial_match(full_name: str, member: DiscordMemberInfo) -> float:
    """Check patterns like 'Jane Doe' <-> 'janed', 'jdoe', or 'Jane D'."""
    parts = [re.sub(r"[^a-z]+", "", _fold_ascii(p)) for p in re.split(r"[\s._-]+", full_name or "")]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return 0.0
    first, last = parts[0], parts[-1]
    if len(first) < 2 or len(last) < 2:
        return 0.0
    patterns = {
        f"{first[0]}{last}",
        f"{first}{last[0]}",
        f"{last}{first}",
    }
    _raw, strict_set, base_set = _member_tokens(member)
    all_member_norms = strict_set | base_set
    for pattern in patterns:
        if len(pattern) >= 4 and pattern in all_member_norms:
            return 0.84
    return 0.0


def _fuzzy_score_pair(profile: GitHubAuthorProfile, member: DiscordMemberInfo) -> tuple[float, list[str]]:
    raw_lower, strict_set, base_set = _member_tokens(member)
    member_norms = [t for t in (strict_set | base_set) if len(t) >= 3]
    best = 0.0
    signals: list[str] = []

    # Secondary exact signals (commit author names and email local parts) -> high-priority Jev candidates.
    for commit_name in profile.commit_names:
        if _exact_field_match(commit_name, raw_lower, strict_set, base_set):
            if 0.88 > best:
                best = 0.88
            signals.append(f"commit_name_match:{commit_name}")
        else:
            init_score = _name_initial_match(commit_name, member)
            if init_score > best:
                best = init_score
                signals.append(f"commit_initial_match:{commit_name}")

    for email_local in profile.email_locals:
        if _exact_field_match(email_local, raw_lower, strict_set, base_set):
            if 0.86 > best:
                best = 0.86
            signals.append(f"email_local_match:{email_local}")

    if profile.name:
        init_score = _name_initial_match(profile.name, member)
        if init_score > best:
            best = init_score
            signals.append(f"name_initial_match:{profile.name}")

    source_values: list[tuple[str, str]] = [("login", profile.login)]
    if profile.name:
        source_values.append(("name", profile.name))
    if profile.twitter_username:
        source_values.append(("twitter", profile.twitter_username))
    for cn in profile.commit_names:
        source_values.append(("commit_name", cn))
    for el in profile.email_locals:
        source_values.append(("email_local", el))

    for label, source in source_values:
        for src_norm in _name_variants(source):
            if len(src_norm) < 4:
                continue
            for mem_norm in member_norms:
                if len(mem_norm) < 4:
                    continue
                ratio = difflib.SequenceMatcher(None, src_norm, mem_norm).ratio()
                if ratio >= FUZZY_CANDIDATE_THRESHOLD:
                    if ratio > best:
                        best = ratio
                    tag = f"fuzzy_{label}:{ratio:.2f}"
                    if tag not in signals:
                        signals.append(tag)
                elif (
                    len(src_norm) >= 5
                    and len(mem_norm) >= 5
                    and (src_norm in mem_norm or mem_norm in src_norm)
                    and (min(len(src_norm), len(mem_norm)) / max(len(src_norm), len(mem_norm))) >= 0.65
                ):
                    contain_score = 0.76
                    if contain_score > best:
                        best = contain_score
                    tag = f"substring_{label}"
                    if tag not in signals:
                        signals.append(tag)

    return best, signals[:5]


def score_author_against_members(
    profile: GitHubAuthorProfile,
    members: list[DiscordMemberInfo],
    *,
    max_candidates: int = MAX_CANDIDATES_PER_AUTHOR,
) -> tuple[tuple[DiscordMemberInfo, str, float] | None, list[CandidatePair]]:
    """Return either a unique deterministic strong match OR ranked fuzzy/ambiguous candidates."""
    if not members:
        return None, []

    by_id = {m.discord_id: m for m in members}
    member_index = [(m, *_member_tokens(m)) for m in members]

    # 1. Direct Discord user URL/ID in bio or blog.
    for source_name, text in (("bio_discord_id", profile.bio), ("blog_discord_id", profile.blog)):
        hits = [by_id[uid] for uid in extract_discord_ids_from_text(text) if uid in by_id]
        if len(hits) == 1:
            return (hits[0], source_name, 0.99), []

    # 2. Explicit Discord handle in bio or blog.
    ambiguous_strong: dict[str, tuple[DiscordMemberInfo, list[str]]] = {}
    for source_name, text in (("bio_discord_handle", profile.bio), ("blog_discord_handle", profile.blog)):
        handles = extract_discord_handles_from_text(text)
        if not handles:
            continue
        hits_by_id: dict[str, DiscordMemberInfo] = {}
        for handle in handles:
            for member in _match_tier_hits(handle, member_index):
                hits_by_id[member.discord_id] = member
        hits = list(hits_by_id.values())
        if len(hits) == 1:
            return (hits[0], source_name, 0.98), []
        for member in hits:
            entry = ambiguous_strong.setdefault(member.discord_id, (member, []))
            entry[1].append(f"ambiguous_{source_name}")

    # 3. Exact normalized match of login, twitter_username, or GitHub profile name.
    strong_tiers = [
        ("exact_login", profile.login, 0.96),
        ("twitter_match", profile.twitter_username, 0.95),
        ("exact_name", profile.name, 0.93),
    ]
    for method, value, confidence in strong_tiers:
        if not value:
            continue
        hits = _match_tier_hits(value, member_index)
        if len(hits) == 1:
            return (hits[0], method, confidence), []
        for member in hits:
            entry = ambiguous_strong.setdefault(member.discord_id, (member, []))
            entry[1].append(f"ambiguous_{method}")

    # 4. Collect ambiguous strong collisions + fuzzy/commit/email candidates for Jev/Haiku.
    candidates: list[CandidatePair] = []
    seen_ids: set[str] = set()
    for uid, (member, sigs) in ambiguous_strong.items():
        seen_ids.add(uid)
        fuzzy_score, fuzzy_sigs = _fuzzy_score_pair(profile, member)
        combined_sigs = tuple(dict.fromkeys(sigs + fuzzy_sigs))
        candidates.append(CandidatePair(profile, member, max(0.90, fuzzy_score), combined_sigs))

    for member in members:
        if member.discord_id in seen_ids:
            continue
        score, sigs = _fuzzy_score_pair(profile, member)
        if score >= FUZZY_CANDIDATE_THRESHOLD and sigs:
            candidates.append(CandidatePair(profile, member, score, tuple(sigs)))

    candidates.sort(key=lambda item: (-item.heuristic_score, item.member.discord_id))
    return None, candidates[:max_candidates]


def _serialize_pair_for_model(pair_key: str, pair: CandidatePair) -> dict:
    return {
        "pair": pair_key,
        "github": {
            "login": pair.profile.login,
            "name": pair.profile.name,
            "twitter_username": pair.profile.twitter_username,
            "bio": pair.profile.bio[:200],
            "blog": pair.profile.blog[:160],
            "commit_names": list(pair.profile.commit_names[:4]),
            "email_locals": list(pair.profile.email_locals[:4]),
        },
        "discord": {
            "id": pair.member.discord_id,
            "username": pair.member.username,
            "global_name": pair.member.global_name,
            "nick": pair.member.nick,
        },
        "signals": list(pair.signals),
    }


def _score_pairs_with_jev(
    config,
    pairs: list[CandidatePair],
    api_key: str,
    *,
    jev_transport=None,
) -> dict[str, float]:
    """Send all candidate pairs to TypeSafe Jev in a single batched noul request."""
    model = getattr(config, "review_decision_model", None) or "jev-latest"
    state_items = []
    questions = {}
    for index, pair in enumerate(pairs):
        key = f"pair_{index}"
        state_items.append(_serialize_pair_for_model(key, pair))
        questions[key] = {
            "type": "noul",
            "instructions": (
                f"In {key}, are GitHub contributor '{pair.profile.login}' and "
                f"Discord member '{pair.member.username}' the same person?"
            ),
            "criteria": {
                "true": "Strong evidence from login, real name, commit name, email local part, bio/blog/twitter, and Discord username/global_name/nick that they are the same person",
                "false": "Different people, coincidence, or insufficient evidence to be sure",
            },
        }
    state = json.dumps({"candidate_pairs": state_items})
    payload = call_jev_questions(model, state, questions, api_key, transport=jev_transport)
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int):
        _note_cost(usage["input_tokens"] * JEV_INPUT_USD_PER_TOKEN)
    answers = payload.get("answers") if isinstance(payload, dict) else None
    scores: dict[str, float] = {}
    if isinstance(answers, dict):
        for key in questions:
            entry = answers.get(key)
            noul = entry.get("noul") if isinstance(entry, dict) else None
            if isinstance(noul, (int, float)):
                scores[key] = float(noul)
    return scores


def _score_pairs_with_haiku(
    config,
    pairs: list[CandidatePair],
    *,
    haiku_complete=None,
) -> dict[str, float]:
    """Single batched Haiku fallback when TYPESAFE_API_KEY is not configured."""
    model = (
        getattr(config, "review_classifier_model", None)
        or getattr(config, "fast_model", None)
        or FAST_MODEL
    )
    state_items = [
        _serialize_pair_for_model(f"pair_{index}", pair)
        for index, pair in enumerate(pairs)
    ]
    prompt = json.dumps({"candidate_pairs": state_items})
    messages = [
        {
            "role": "system",
            "content": (
                "For each candidate pair in candidate_pairs, estimate the probability (0.0 to 1.0) "
                "that the GitHub contributor and Discord member are the same person. "
                "Only give >= 0.85 when there is clear, specific evidence; give low scores when uncertain. "
                "Reply with a single JSON object mapping each pair key (e.g. \"pair_0\") to a float."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    if haiku_complete is not None:
        raw = haiku_complete(model, messages)
    else:
        from .review_llm import complete_litellm
        raw = complete_litellm(model, messages)
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        parsed = json.loads(raw[start: end + 1]) if start >= 0 and end >= start else {}
    except (ValueError, AttributeError):
        parsed = {}
    scores: dict[str, float] = {}
    if isinstance(parsed, dict):
        for index in range(len(pairs)):
            key = f"pair_{index}"
            val = parsed.get(key)
            if isinstance(val, (int, float)):
                scores[key] = max(0.0, min(1.0, float(val)))
    return scores


def disambiguate_candidates(
    config,
    candidates: list[CandidatePair],
    *,
    threshold: float = JEV_MATCH_THRESHOLD,
    jev_transport=None,
    haiku_complete=None,
) -> dict[str, tuple[DiscordMemberInfo, str, float]]:
    """Score ambiguous/fuzzy pairs in one batched Jev call (or Haiku fallback)."""
    if not candidates:
        return {}

    api_key = (
        os.environ.get("TYPESAFE_API_KEY", "").strip()
        or (getattr(config, "typesafe_api_key", None) or "").strip()
    )
    all_scores: dict[int, float] = {}
    source = "jev"

    for offset in range(0, len(candidates), MAX_PAIRS_PER_BATCH):
        batch = candidates[offset: offset + MAX_PAIRS_PER_BATCH]
        batch_scores: dict[str, float] = {}
        if api_key or jev_transport is not None:
            try:
                batch_scores = _score_pairs_with_jev(config, batch, api_key, jev_transport=jev_transport)
                source = "jev"
            except Exception as error:
                logging.warning("Jev contributor matching failed; trying Haiku fallback: %s", error)
                try:
                    batch_scores = _score_pairs_with_haiku(config, batch, haiku_complete=haiku_complete)
                    source = "haiku"
                except Exception as haiku_error:
                    logging.warning("Haiku contributor matching fallback failed: %s", haiku_error)
                    batch_scores = {}
        else:
            try:
                batch_scores = _score_pairs_with_haiku(config, batch, haiku_complete=haiku_complete)
                source = "haiku"
            except Exception as haiku_error:
                logging.warning("Haiku contributor matching unavailable: %s", haiku_error)
                batch_scores = {}

        for idx, _pair in enumerate(batch):
            key = f"pair_{idx}"
            if key in batch_scores:
                all_scores[offset + idx] = batch_scores[key]

    by_login: dict[str, list[tuple[float, DiscordMemberInfo]]] = {}
    for idx, pair in enumerate(candidates):
        score = all_scores.get(idx)
        if score is None:
            continue
        by_login.setdefault(pair.profile.login.casefold(), []).append((score, pair.member))

    winners: dict[str, tuple[DiscordMemberInfo, str, float]] = {}
    for login_key, scored in by_login.items():
        scored.sort(key=lambda item: -item[0])
        top_score, top_member = scored[0]
        if top_score < threshold:
            continue
        if len(scored) > 1:
            second_score = scored[1][0]
            if second_score >= threshold and (top_score - second_score) < JEV_RUNNER_UP_MARGIN:
                continue
        winners[login_key] = (top_member, source, top_score)
    return winners


def match_community_contributors(
    groups: list[dict],
    config,
    store,
    *,
    guild=None,
    loop=None,
    now: datetime | None = None,
    members: list[DiscordMemberInfo] | None = None,
    github_getter=None,
    jev_transport=None,
    haiku_complete=None,
    skip_github_enrichment: bool = False,
) -> tuple[list[dict], int, list[MatchDetail]]:
    """Resolve community PR authors to LiteLLM Discord guild members.

    Returns ``(matched_groups, skipped_count, match_details)``.
    Confirmed automatic matches are persisted in SQLite so they are computed once.
    """
    moment_iso = iso(now or datetime.now(UTC))
    guild_members = list(members) if members is not None else load_guild_members(guild, loop)
    member_by_id = {m.discord_id: m for m in guild_members}

    resolved: dict[str, tuple[dict, str, str, float]] = {}  # login_lower -> (group, discord_id, method, confidence)
    pending: list[dict] = []

    manual_map = getattr(config, "github_discord_map", None) or {}
    for group in groups:
        login = group["login"]
        login_key = login.strip().casefold()
        user_id = group.get("user_id")

        # Optional pre-existing Discord ID on group, /link-github store, or static map.
        discord_id = group.get("discord_id")
        if discord_id and is_discord_id(str(discord_id)):
            resolved[login_key] = (group, str(discord_id), "prelinked", 1.0)
            continue

        if store is not None and isinstance(user_id, int) and hasattr(store, "link_for_github_user"):
            link = store.link_for_github_user(user_id)
            if link and is_discord_id(link.get("discord_id")):
                resolved[login_key] = (group, str(link["discord_id"]), "link_store", 1.0)
                continue

        mapped_id = manual_map.get(login_key)
        if mapped_id and is_discord_id(str(mapped_id)):
            resolved[login_key] = (group, str(mapped_id), "config_map", 1.0)
            continue

        # Persisted SQLite automatic match (computed once).
        if store is not None and hasattr(store, "get_confirmed_match"):
            cached = store.get_confirmed_match(login)
            if cached and is_discord_id(cached.get("discord_id")):
                cid = str(cached["discord_id"])
                if not member_by_id or cid in member_by_id:
                    resolved[login_key] = (
                        group,
                        cid,
                        f"sqlite:{cached['method']}",
                        float(cached.get("confidence") or 1.0),
                    )
                    continue

        pending.append(group)

    if pending and guild_members:
        token = getattr(config, "github_token", None) or getattr(config, "github_actor_token", None)
        repository = getattr(config, "github_repository", "") or ""
        rate_limited = False
        ambiguous_candidates: list[CandidatePair] = []
        pending_by_login = {g["login"].strip().casefold(): g for g in pending}

        for group in pending:
            login = group["login"]
            login_key = login.strip().casefold()
            pull_numbers = [
                int(p["number"])
                for p in group.get("pulls", ())
                if isinstance(p, dict) and isinstance(p.get("number"), int)
            ]
            if skip_github_enrichment or rate_limited:
                profile = GitHubAuthorProfile(login=login, user_id=group.get("user_id"))
            else:
                profile, hit_limit = fetch_github_author_profile(
                    login,
                    repository,
                    pull_numbers,
                    token,
                    user_id=group.get("user_id"),
                    getter=github_getter,
                )
                if hit_limit:
                    rate_limited = True

            strong, candidates = score_author_against_members(profile, guild_members)
            if strong is not None:
                member, method, confidence = strong
                resolved[login_key] = (group, member.discord_id, method, confidence)
            elif candidates:
                ambiguous_candidates.extend(candidates)

        if ambiguous_candidates:
            llm_winners = disambiguate_candidates(
                config,
                ambiguous_candidates,
                jev_transport=jev_transport,
                haiku_complete=haiku_complete,
            )
            for login_key, (member, method, confidence) in llm_winners.items():
                if login_key not in resolved and login_key in pending_by_login:
                    resolved[login_key] = (pending_by_login[login_key], member.discord_id, method, confidence)

    # Enforce 1-to-1 Discord ID assignment (highest confidence wins; leave conflicts unmatched).
    claimed_discord: dict[str, tuple[str, float]] = {}
    for login_key, (_group, discord_id, _method, confidence) in sorted(
        resolved.items(), key=lambda item: (-item[1][3], item[0])
    ):
        existing = claimed_discord.get(discord_id)
        if existing is None or confidence > existing[1]:
            claimed_discord[discord_id] = (login_key, confidence)

    winning_logins = {login_key for login_key, _conf in claimed_discord.values()}

    matched_groups: list[dict] = []
    details: list[MatchDetail] = []
    for login_key in winning_logins:
        group, discord_id, method, confidence = resolved[login_key]
        # Persist newly confirmed automatic matches to SQLite so they are computed once.
        if (
            store is not None
            and hasattr(store, "save_confirmed_match")
            and method not in {"prelinked", "link_store", "config_map"}
            and not method.startswith("sqlite:")
        ):
            try:
                store.save_confirmed_match(
                    group["login"],
                    discord_id,
                    method=method,
                    confidence=confidence,
                    matched_at=moment_iso,
                    github_user_id=group.get("user_id"),
                )
            except Exception as error:
                logging.warning("Could not persist match for %s: %s", group["login"], error)

        matched_groups.append({
            **group,
            "discord_id": discord_id,
            "match_method": method,
            "match_confidence": confidence,
        })
        member = member_by_id.get(discord_id)
        details.append(MatchDetail(
            github_login=group["login"],
            discord_id=discord_id,
            method=method,
            confidence=confidence,
            discord_username=member.username if member else "",
            pr_count=len(group.get("pulls", ())),
        ))

    matched_groups.sort(key=lambda item: (-len(item["pulls"]), item["login"].lower()))
    details.sort(key=lambda item: (-item.pr_count, item.github_login.lower()))
    skipped_count = len(groups) - len(matched_groups)
    return matched_groups, skipped_count, details
