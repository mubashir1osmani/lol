import json
import os
from pathlib import Path


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
