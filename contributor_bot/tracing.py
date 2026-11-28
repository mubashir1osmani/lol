"""Optional OpenTelemetry + OpenInference tracing for LiteLLM Lens.

Tracing stays completely off (zero third-party imports, no-op spans) unless
``OTEL_ENDPOINT`` or ``OTEL_EXPORTER_OTLP_ENDPOINT`` (or
``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``) is set in the environment.

When configured, builds an OpenTelemetry ``TracerProvider`` backed by an OTLP
HTTP span exporter and instruments LiteLLM via
``openinference.instrumentation.litellm.LiteLLMInstrumentor`` (without adding
``"otel"`` to ``litellm.callbacks``, which Lens v0.6.0 hides from its run list).
Logical bot tasks are wrapped in OpenInference ``AGENT`` spans via
:func:`agent_span` so Lens displays one run per task with nested LLM calls.
"""

from __future__ import annotations

import atexit
import contextvars
import json
import logging
import os
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from urllib.parse import urlsplit, urlunsplit

OPENINFERENCE_SPAN_KIND_ATTR = "openinference.span.kind"
OPENINFERENCE_SPAN_KIND_AGENT = "AGENT"
OPENINFERENCE_AGENT_NAME_ATTR = "agent.name"
OPENINFERENCE_SESSION_ID_ATTR = "session.id"
OPENINFERENCE_INPUT_VALUE_ATTR = "input.value"
OPENINFERENCE_INPUT_MIME_ATTR = "input.mime_type"
OPENINFERENCE_OUTPUT_VALUE_ATTR = "output.value"
OPENINFERENCE_OUTPUT_MIME_ATTR = "output.mime_type"

_MAX_SPAN_VALUE_CHARS = 4000
_DEFAULT_SERVICE_NAME = "litellm-contributor-bot"

_guard = threading.Lock()
_configured = False
_tracer_provider = None
_tracer = None
_atexit_registered = False


def _endpoint_with_source(env: Mapping[str, str] | None = None) -> tuple[str, str]:
    source = os.environ if env is None else env
    for key in ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_ENDPOINT"):
        value = (source.get(key) or "").strip()
        if value:
            return value, key
    return "", ""


def tracing_endpoint(env: Mapping[str, str] | None = None) -> str:
    """Return the configured OTLP traces endpoint, or ``""`` when tracing is off.

    Precedence: ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` >
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` > ``OTEL_ENDPOINT``.
    """
    endpoint, _ = _endpoint_with_source(env)
    return endpoint


def normalize_otlp_traces_endpoint(endpoint: str, *, exact: bool = False) -> str:
    """Ensure OTLP HTTP base URLs end with ``/v1/traces`` without double-appending."""
    cleaned = (endpoint or "").strip()
    if not cleaned or exact:
        return cleaned
    try:
        parsed = urlsplit(cleaned)
    except ValueError:
        return cleaned
    if not parsed.scheme or not parsed.netloc:
        return cleaned
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/traces"):
        return cleaned
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/v1/traces", parsed.query, parsed.fragment))


def parse_otel_headers(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Parse ``OTEL_HEADERS`` (or ``OTEL_EXPORTER_OTLP_HEADERS``) as ``k=v,k2=v2`` with literal values.

    Splits on commas and then on the first ``=`` without URL-decoding so literal
    spaces and ``=`` padding inside bearer tokens or API keys are preserved intact.
    """
    source = os.environ if env is None else env
    raw = (source.get("OTEL_HEADERS") or "").strip()
    if not raw:
        raw = (source.get("OTEL_EXPORTER_OTLP_HEADERS") or "").strip()
    if not raw:
        return {}
    headers: dict[str, str] = {}
    for entry in raw.split(","):
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        clean_key = key.strip()
        clean_val = value.strip()
        if clean_key:
            headers[clean_key] = clean_val
    return headers


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


def shutdown_tracing() -> None:
    """Flush and shut down the active TracerProvider, if tracing was configured."""
    global _configured, _tracer_provider, _tracer
    with _guard:
        provider = _tracer_provider
        _configured = False
        _tracer_provider = None
        _tracer = None
    if provider is None:
        return
    force_flush = getattr(provider, "force_flush", None)
    if callable(force_flush):
        try:
            force_flush()
        except Exception:
            pass
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass


def _reset_tracing_state() -> None:
    """Reset module-level tracing state (intended for unit tests)."""
    global _configured, _tracer_provider, _tracer
    with _guard:
        _configured = False
        _tracer_provider = None
        _tracer = None


def get_tracer():
    """Return the active OTel tracer when tracing is configured, else ``None``."""
    return _tracer


def configure_litellm_tracing(env: Mapping[str, str] | None = None) -> bool:
    """Enable OpenTelemetry + OpenInference LiteLLM tracing iff an OTLP endpoint is set.

    No-op (does not import ``litellm``, ``opentelemetry``, or ``openinference``)
    when no OTLP endpoint env var is configured. Idempotent across repeated calls.
    Explicitly does NOT add ``"otel"`` to ``litellm.callbacks``.
    """
    global _configured, _tracer_provider, _tracer, _atexit_registered

    raw_endpoint, source_key = _endpoint_with_source(env)
    if not raw_endpoint:
        return False

    with _guard:
        if _configured:
            return True
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from openinference.instrumentation.litellm import LiteLLMInstrumentor
        except ImportError as error:
            logging.warning(
                "OTEL endpoint is set, but OpenTelemetry/OpenInference packages are not installed (%s); tracing disabled",
                error,
            )
            return False

        source = os.environ if env is None else env
        service_name = (source.get("OTEL_SERVICE_NAME") or _DEFAULT_SERVICE_NAME).strip() or _DEFAULT_SERVICE_NAME
        otlp_endpoint = normalize_otlp_traces_endpoint(
            raw_endpoint,
            exact=(source_key == "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"),
        )
        headers = parse_otel_headers(source)

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, headers=headers or None)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        LiteLLMInstrumentor().instrument(tracer_provider=provider)

        _tracer_provider = provider
        _tracer = provider.get_tracer("contributor_bot")
        _configured = True
        if not _atexit_registered:
            atexit.register(shutdown_tracing)
            _atexit_registered = True
        logging.info(
            "LiteLLM OpenInference tracing enabled (service=%s, endpoint=%s, headers=%d)",
            service_name,
            _safe_endpoint_for_log(otlp_endpoint),
            len(headers),
        )
        return True


def _serialize_span_value(value: object) -> tuple[str, str]:
    """Serialize an input/output value to ``(text, mime_type)`` bounded in size."""
    if isinstance(value, str):
        return value[:_MAX_SPAN_VALUE_CHARS], "text/plain"
    if isinstance(value, (dict, list, tuple)):
        try:
            encoded = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            encoded = str(value)
        return encoded[:_MAX_SPAN_VALUE_CHARS], "application/json"
    return str(value)[:_MAX_SPAN_VALUE_CHARS], "text/plain"


def _set_safe_attribute(span, key: str, value: object) -> None:
    if span is None or value is None:
        return
    if isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and not value:
            return
        span.set_attribute(key, value)
    else:
        text = str(value).strip()
        if text:
            span.set_attribute(key, text[:_MAX_SPAN_VALUE_CHARS])


class _SpanHandle:
    """Thin wrapper over an active OTel span (or no-op when tracing is off)."""

    __slots__ = ("_span", "session_id")

    def __init__(self, span=None, session_id: str = "") -> None:
        self._span = span
        self.session_id = session_id

    @property
    def active(self) -> bool:
        return self._span is not None

    def set_attribute(self, key: str, value: object) -> None:
        _set_safe_attribute(self._span, key, value)

    def set_input(self, value: object) -> None:
        if self._span is None or value is None:
            return
        text, mime = _serialize_span_value(value)
        if text:
            self._span.set_attribute(OPENINFERENCE_INPUT_VALUE_ATTR, text)
            self._span.set_attribute(OPENINFERENCE_INPUT_MIME_ATTR, mime)

    def set_output(self, value: object) -> None:
        if self._span is None or value is None:
            return
        text, mime = _serialize_span_value(value)
        if text:
            self._span.set_attribute(OPENINFERENCE_OUTPUT_VALUE_ATTR, text)
            self._span.set_attribute(OPENINFERENCE_OUTPUT_MIME_ATTR, mime)


_NOOP_SPAN = _SpanHandle(None, "")


def _using_session_cm(session_id: str):
    if not session_id:
        return nullcontext()
    try:
        from openinference.instrumentation import using_session
        return using_session(session_id)
    except Exception:
        return nullcontext()


def _set_span_status(span, *, ok: bool, description: str = "") -> None:
    set_status = getattr(span, "set_status", None)
    if not callable(set_status):
        return
    try:
        from opentelemetry.trace import Status, StatusCode
        code = StatusCode.OK if ok else StatusCode.ERROR
        set_status(Status(code, description[:500] if not ok and description else None))
    except Exception:
        pass


@contextmanager
def agent_span(
    name: str,
    *,
    session_id: str | None = None,
    input_value: object = None,
    attributes: Mapping[str, object] | None = None,
) -> Iterator[_SpanHandle]:
    """Wrap one logical bot task in an OpenInference ``AGENT`` span.

    Completely a zero-import no-op when tracing is not configured.
    When tracing is active, sets ``openinference.span.kind="AGENT"``,
    ``agent.name``, ``session.id``, and optional ``input.value`` / ``output.value``
    so LiteLLM Lens shows one top-level run per task with nested LLM spans.
    """
    tracer = _tracer
    if tracer is None:
        if not tracing_endpoint():
            yield _NOOP_SPAN
            return
        configure_litellm_tracing()
        tracer = _tracer
        if tracer is None:
            yield _NOOP_SPAN
            return

    agent_name = (name or "bot").strip() or "bot"
    sid = (str(session_id).strip() if session_id is not None else "") or f"{agent_name}:{uuid.uuid4().hex[:12]}"
    with tracer.start_as_current_span(agent_name) as span:
        handle = _SpanHandle(span, sid)
        span.set_attribute(OPENINFERENCE_SPAN_KIND_ATTR, OPENINFERENCE_SPAN_KIND_AGENT)
        span.set_attribute(OPENINFERENCE_AGENT_NAME_ATTR, agent_name)
        span.set_attribute(OPENINFERENCE_SESSION_ID_ATTR, sid)
        if input_value is not None:
            handle.set_input(input_value)
        if attributes:
            for key, value in attributes.items():
                handle.set_attribute(str(key), value)
        with _using_session_cm(sid):
            try:
                yield handle
            except Exception as exc:
                record_exc = getattr(span, "record_exception", None)
                if callable(record_exc):
                    try:
                        record_exc(exc)
                    except Exception:
                        pass
                _set_span_status(span, ok=False, description=str(exc))
                raise
            else:
                _set_span_status(span, ok=True)


def submit_with_context(pool, fn, *args, **kwargs):
    """Submit ``fn`` to a ThreadPoolExecutor while preserving OTel contextvars."""
    ctx = contextvars.copy_context()
    return pool.submit(ctx.run, fn, *args, **kwargs)


def build_trace_metadata(
    agent: str,
    *,
    session_id: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build LiteLLM ``metadata`` dict for grouping traces by feature and session."""
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
