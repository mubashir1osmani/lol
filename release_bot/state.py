import json
import os
from pathlib import Path


def _read_json(path: str) -> dict:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read state file {path}: {error}") from error


def _write_json(path: str, data: dict) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_name(f"{state_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.chmod(0o600)
    os.replace(temporary_path, state_path)


def load_ids(path: str, key: str) -> tuple[bool, list[str]]:
    """Return (whether this key was seen before, the IDs stored under it)."""
    data = _read_json(path)
    if key not in data:
        return False, []
    ids = data[key]
    return True, [str(item) for item in ids] if isinstance(ids, list) else []


def save_ids(path: str, key: str, ids: list[str]) -> None:
    data = _read_json(path)
    data[key] = list(dict.fromkeys(map(str, ids)))[-500:]
    _write_json(path, data)


def load_state(path: str) -> tuple[bool, list[str]]:
    state_path = Path(path)
    if not state_path.exists():
        return False, []
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        ids = data.get("postedReleaseIds", [])
        return True, [str(release_id) for release_id in ids] if isinstance(ids, list) else []
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read state file {path}: {error}") from error


def save_state(path: str, release_ids: list[str]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    unique_ids = list(dict.fromkeys(map(str, release_ids)))[-200:]
    temporary_path = state_path.with_name(f"{state_path.name}.tmp")
    temporary_path.write_text(
        json.dumps({"postedReleaseIds": unique_ids}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.chmod(0o600)
    os.replace(temporary_path, state_path)
