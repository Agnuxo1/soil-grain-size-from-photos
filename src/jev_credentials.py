"""Safe TypeSafe/Jev credential discovery shared by local agents.

The resolver keeps secrets out of source code and logs. It accepts an explicit
environment override, the rescue orchestrator's secret files, and the user's
utility file as a last-resort local source. Callers receive key strings only in
memory and should never print them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import requests


KEY_RE = re.compile(r"\b(apikey_[A-Za-z0-9_]+)")
RESCUE_ROOT = Path(r"E:\Rescate-C-2026-09-21\Documents\JEV-Orchestrator")
UTILITY_FILE = Path(r"C:\Users\Windows-500GB\Desktop\UTILIDADES_HERRAMIENTAS_APIs.txt")
ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def _candidate_paths() -> Iterable[Path]:
    for variable in ("JEV_TYPESAFE_KEY_FILE", "TYPESAFE_API_KEY_FILE"):
        value = os.environ.get(variable)
        if value:
            yield Path(value)
    yield RESCUE_ROOT / "secrets" / "typesafe_api_keys.txt"
    yield RESCUE_ROOT / "secrets" / "typesafe_api_key.txt"
    yield Path("secrets") / "typesafe_api_keys.txt"
    yield Path("secrets") / "typesafe_api_key.txt"
    yield UTILITY_FILE


def _utility_block(text: str) -> str:
    marker = re.search(r"TypeSafe\s+AI\s+JEV", text, flags=re.IGNORECASE)
    if not marker:
        return ""
    block = text[marker.end() :]
    boundary = re.search(
        r"(?:^|\n)\s*(?:Nuevo\s+Github\s+Token|OpenAI\s+API|Anthropic|Google\s+AI)\b",
        block,
        flags=re.IGNORECASE,
    )
    return block[: boundary.start()] if boundary else block


def _keys_from_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return []
    source = _utility_block(text) if path.name.lower().startswith("utilidades_") else text
    return KEY_RE.findall(source)


def load_typesafe_keys() -> list[str]:
    """Return unique local TypeSafe keys in priority order without logging them."""

    found: list[str] = []
    env_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if env_key.startswith("apikey_"):
        found.append(env_key)
    seen_paths: set[Path] = set()
    for path in _candidate_paths():
        resolved = path.expanduser()
        if resolved in seen_paths or not resolved.exists():
            continue
        seen_paths.add(resolved)
        for key in _keys_from_file(resolved):
            if key not in found:
                found.append(key)
    return found


def load_first_typesafe_key() -> str:
    keys = load_typesafe_keys()
    if not keys:
        searched = ", ".join(str(path) for path in _candidate_paths())
        raise RuntimeError(f"No TypeSafe key found. Configure TYPESAFE_API_KEY or one of: {searched}")
    return keys[0]


def post_typesafe(payload: dict[str, object], *, timeout: float = 60.0) -> requests.Response:
    """POST one bounded request, rotating only across authentication failures."""

    keys = load_typesafe_keys()
    if not keys:
        raise RuntimeError("No TypeSafe key found in the configured local sources.")
    last_response: requests.Response | None = None
    for key in keys:
        response = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        last_response = response
        if response.status_code not in (401, 403):
            response.raise_for_status()
            return response
    assert last_response is not None
    last_response.raise_for_status()
    return last_response


def credential_status() -> dict[str, object]:
    """Return non-secret diagnostics suitable for logs or health checks."""

    existing = []
    seen: set[Path] = set()
    for path in _candidate_paths():
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            existing.append({"path": str(resolved), "key_count": len(_keys_from_file(resolved))})
    return {"key_count": len(load_typesafe_keys()), "sources": existing}
