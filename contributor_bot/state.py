import json
import os
from pathlib import Path
from typing import Any

from .config import Config


def load_state(config: Config) -> dict[str, Any]:
    path = Path(config.state_file)
    if not path.exists():
        return {
            "version": 1,
            "scope": config.scope,
            "last_message_id": None,
            "pull_requests": {},
            "contributor_ids": [],
            "outbox": [],
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read state file {path}: {error}") from error
    if (
        not isinstance(state, dict)
        or state.get("version") != 1
        or not isinstance(state.get("pull_requests"), dict)
        or not isinstance(state.get("contributor_ids"), list)
        or not isinstance(state.get("outbox"), list)
        or "last_message_id" not in state
    ):
        raise RuntimeError(f"Invalid contributor state in {path}; refusing to reset announcement history")
    if state.get("scope") != config.scope:
        raise RuntimeError("State belongs to different channels/repository; use a different STATE_FILE")
    return state


def save_state(path: str, state: dict[str, Any]) -> None:
    """Atomically retain all PRs and contributor IDs; never truncate dedupe history."""
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_name(f"{state_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        temporary_path.chmod(0o600)
        json.dump(state, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, state_path)
