import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .config import Config


def load_state(config: Config) -> dict[str, Any]:
    path = Path(config.state_file)
    if not path.exists():
        return {
            "version": 2,
            "scope": config.scope,
            "last_message_id": None,
            "pull_requests": {},
            "contributor_ids": [],
            "outbox": [],
            "weekly": None,
            "releases": {},
            "roundups": [],
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read state file {path}: {error}") from error
    if (
        not isinstance(state, dict)
        or state.get("version") not in {1, 2}
        or not isinstance(state.get("pull_requests"), dict)
        or not isinstance(state.get("contributor_ids"), list)
        or not isinstance(state.get("outbox"), list)
        or "last_message_id" not in state
    ):
        raise RuntimeError(f"Invalid contributor state in {path}; refusing to reset announcement history")
    if state.get("scope") != config.scope:
        raise RuntimeError("State belongs to different channels/repository; use a different STATE_FILE")
    if state["version"] == 1:
        if state["outbox"]:
            raise RuntimeError("Legacy all-time outbox needs manual review before weekly migration")
        state.update(version=2, weekly=None, releases={}, roundups=[])
    if (not isinstance(state.get("releases"), dict)
            or not isinstance(state.get("roundups"), list)
            or "weekly" not in state):
        raise RuntimeError("Invalid weekly state; refusing to reset history")
    window = state["weekly"]
    if window is not None:
        if (not isinstance(window, dict) or not all(key in window for key in
                ("start", "end", "time", "timezone", "merge_numbers", "prepared"))):
            raise RuntimeError("Invalid weekly window")
        if (window["timezone"], window["time"]) != (config.roundup_timezone, config.roundup_time):
            raise RuntimeError("Schedule changed; review the saved weekly boundaries before changing schedule")
    return state


def save_state(path: str, state: dict[str, Any]) -> None:
    """Atomically retain all PRs and contributor IDs; never truncate dedupe history."""
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Random, exclusively created 0600 temporary files cannot be pre-planted
    # as symlinks to overwrite application code or other files.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=state_path.parent,
            prefix=f".{state_path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(state, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, state_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
