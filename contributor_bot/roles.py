"""Role rules for merged LiteLLM pull requests.

Edit the thresholds, path prefixes, and path tokens in this file. Discord role
names and snowflakes come from the environment (see ``ROLE_*`` in the README);
the keys below are stable and are what we store after a role is granted.

Calendar months are UTC. A role, once earned, stays earned: later months do
not take OSS Champion away, and we never remove Discord roles.
"""

from dataclasses import dataclass
from datetime import datetime

# How many merged human PRs it takes to earn each role.
CONTRIBUTOR_MIN_MERGED = 1
OSS_CHAMPION_MIN_MERGED = 10
OSS_CHAMPION_MIN_IN_MONTH = 3
PROVIDER_WIZARD_MIN_MERGED = 3

# A pull request counts for Provider Wizard when any changed path is under the prefix.
PROVIDER_WIZARD_PREFIX = "litellm/llms/"

# Agent Builder: any changed path containing one of these tokens (case-insensitive),
# or a pull request label exactly named "agents".
AGENT_PATH_TOKENS = ("mcp", "agent", "a2a")
AGENT_LABEL = "agents"

ROLE_KEYS = (
    "contributor",
    "oss_champion",
    "agent_builder",
    "provider_wizard",
)

DEFAULT_ROLE_NAMES = {
    "contributor": "Contributor",
    "oss_champion": "OSS Champion",
    "agent_builder": "Agent Builder",
    "provider_wizard": "Provider Wizard",
}

ROLE_ID_ENV = {
    "contributor": "ROLE_CONTRIBUTOR_ID",
    "oss_champion": "ROLE_OSS_CHAMPION_ID",
    "agent_builder": "ROLE_AGENT_BUILDER_ID",
    "provider_wizard": "ROLE_PROVIDER_WIZARD_ID",
}

ROLE_NAME_ENV = {
    "contributor": "ROLE_CONTRIBUTOR_NAME",
    "oss_champion": "ROLE_OSS_CHAMPION_NAME",
    "agent_builder": "ROLE_AGENT_BUILDER_NAME",
    "provider_wizard": "ROLE_PROVIDER_WIZARD_NAME",
}


@dataclass(frozen=True)
class MergedPull:
    number: int
    merged_at: datetime
    labels: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    is_bot: bool = False


def _human(pulls: list[MergedPull]) -> list[MergedPull]:
    return [pull for pull in pulls if not pull.is_bot]


def _under(path: str, prefix: str) -> bool:
    return path.startswith(prefix)


def _agent_path(path: str) -> bool:
    lowered = path.lower()
    return any(token in lowered for token in AGENT_PATH_TOKENS)


def _agent_label(labels: tuple[str, ...]) -> bool:
    return any(label.strip().lower() == AGENT_LABEL for label in labels)


def qualifies_contributor(pulls: list[MergedPull]) -> bool:
    return len(_human(pulls)) >= CONTRIBUTOR_MIN_MERGED


def qualifies_oss_champion(pulls: list[MergedPull]) -> bool:
    humans = _human(pulls)
    if len(humans) >= OSS_CHAMPION_MIN_MERGED:
        return True
    months: dict[tuple[int, int], int] = {}
    for pull in humans:
        # merged_at is normalized to UTC before it is stored or passed in.
        month = (pull.merged_at.year, pull.merged_at.month)
        months[month] = months.get(month, 0) + 1
    return any(count >= OSS_CHAMPION_MIN_IN_MONTH for count in months.values())


def qualifies_agent_builder(pulls: list[MergedPull]) -> bool:
    for pull in _human(pulls):
        if _agent_label(pull.labels) or any(_agent_path(path) for path in pull.files):
            return True
    return False


def qualifies_provider_wizard(pulls: list[MergedPull]) -> bool:
    matched = sum(
        1 for pull in _human(pulls) if any(_under(path, PROVIDER_WIZARD_PREFIX) for path in pull.files)
    )
    return matched >= PROVIDER_WIZARD_MIN_MERGED


RULES = (
    ("contributor", qualifies_contributor),
    ("oss_champion", qualifies_oss_champion),
    ("agent_builder", qualifies_agent_builder),
    ("provider_wizard", qualifies_provider_wizard),
)


def earned_role_keys(pulls: list[MergedPull]) -> list[str]:
    """Role keys this author has earned, in announcement order."""
    return [key for key, qualifies in RULES if qualifies(pulls)]
