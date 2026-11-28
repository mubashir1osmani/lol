"""Optional OpenTelemetry tracing for LiteLLM SDK calls (e.g., to LiteLLM Lens).

Tracing stays completely off unless ``OTEL_ENDPOINT`` or
``OTEL_EXPORTER_OTLP_ENDPOINT`` is set in the environment. When configured,
``"otel"`` is registered in ``litellm.callbacks`` before any completion call.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Mapping
from urllib.parse import urlsplit, urlunsplit


def tracing_endpoint(env: Mapping[str, str] | None = None) -> str:
    """Return the configured OTLP traces endpoint, or ``""`` when tracing is off.

    Matches LiteLLM's ``OpenTelemetryConfig.from_env()`` precedence:
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` overrides ``OTEL_ENDPOINT``.
    """
    source = os.environ if env is None else env
    for key in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_ENDPOINT"):
        value = (source.get(key) or "").strip()
        if value:
            return value
    return ""


def _safe_endpoint_for_log(endpoint: str) -> str:
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        return "<configured>"
    if not parsed.scheme or not parsed.netloc:
        return endpoint
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def configure_litellm_tracing(env: Mapping[str, str] | None = None) -> bool:
    """Enable LiteLLM's OpenTelemetry callback iff an OTLP endpoint is configured.

    No-op (does not import ``litellm`` or touch ``litellm.callbacks``) when
    neither ``OTEL_ENDPOINT`` nor ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set.
    Idempotent across repeated calls.
    """
    endpoint = tracing_endpoint(env)
    if not endpoint:
        return False
    try:
        import litellm
    except ImportError:
        logging.warning("OTEL endpoint is set, but litellm is not installed; tracing disabled")
        return False

    existing = getattr(litellm, "callbacks", None)
    callbacks = list(existing) if isinstance(existing, (list, tuple)) else []
    already_enabled = any(
        cb == "otel" or type(cb).__name__ == "OpenTelemetry"
        for cb in callbacks
    )
    if not already_enabled:
        callbacks.append("otel")
        litellm.callbacks = callbacks
        source = os.environ if env is None else env
        service_name = (source.get("OTEL_SERVICE_NAME") or "litellm").strip() or "litellm"
        logging.info(
            "LiteLLM OpenTelemetry tracing enabled (service=%s, endpoint=%s)",
            service_name,
            _safe_endpoint_for_log(endpoint),
        )
    return True


def build_trace_metadata(
    agent: str,
    *,
    session_id: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build LiteLLM ``metadata`` dict for grouping traces by feature and session.

    - ``generation_name`` sets the OpenTelemetry span name in ``litellm``.
    - ``session_id``, ``agent_name``, and ``tags`` tag the call for LiteLLM / Lens.
    - ``requester_metadata`` is preserved by ``litellm``'s
      ``StandardLoggingMetadata`` filter onto ``metadata.requester_metadata``.
    """
    feature = (agent or "bot").strip() or "bot"
    sid = (str(session_id).strip() if session_id is not None else "") or f"{feature}:{uuid.uuid4().hex[:12]}"
    requester: dict[str, str] = {
        "agent_name": feature,
        "session_id": sid,
    }
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            text = str(value).strip()
            if text:
                requester[str(key)] = text
    return {
        "generation_name": feature,
        "agent_name": feature,
        "session_id": sid,
        "tags": [feature],
        "requester_metadata": requester,
    }


def build_sdk_trace_metadata(
    agent: str,
    *,
    session_id: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Flat string-only metadata dict for OpenAI Agents SDK ``ModelSettings.metadata``."""
    feature = (agent or "bot").strip() or "bot"
    sid = (str(session_id).strip() if session_id is not None else "") or f"{feature}:{uuid.uuid4().hex[:12]}"
    metadata: dict[str, str] = {
        "generation_name": feature,
        "agent_name": feature,
        "session_id": sid,
    }
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            text = str(value).strip()
            if text:
                metadata[str(key)] = text
    return metadata
