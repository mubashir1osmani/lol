"""Optional OpenTelemetry + OpenInference tracing for LiteLLM Lens.

Tracing stays completely off (zero third-party imports, no-op spans) unless
``OTEL_ENDPOINT`` or ``OTEL_EXPORTER_OTLP_ENDPOINT`` (or
``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``) is set in the environment (or a
TracerProvider is installed explicitly in tests).

When configured, builds an OpenTelemetry ``TracerProvider`` backed by a
gzip-compressed OTLP HTTP span exporter and instruments LiteLLM via
``openinference.instrumentation.litellm.LiteLLMInstrumentor`` (without adding
``"otel"`` to ``litellm.callbacks``, which Lens v0.6.0 hides from its run list).

Every LLM call in the Aria bot runs inside exactly one of three top-level
OpenInference ``AGENT`` spans:
  - ``pr_review_agent``
  - ``support_agent``
  - ``engagement_agent``

Sub-tasks (message routing, classification, KB retrieval, tool calls,
role matching, weekly summary) are emitted as child spans (``CHAIN`` / ``TOOL``
/ ``RETRIEVER``) without ``agent.name`` so Lens displays exactly three agents.
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
OPENINFERENCE_SPAN_KIND_CHAIN = "CHAIN"
OPENINFERENCE_SPAN_KIND_TOOL = "TOOL"
OPENINFERENCE_SPAN_KIND_RETRIEVER = "RETRIEVER"
OPENINFERENCE_AGENT_NAME_ATTR = "agent.name"
OPENINFERENCE_TOOL_NAME_ATTR = "tool.name"
OPENINFERENCE_SESSION_ID_ATTR = "session.id"
OPENINFERENCE_INPUT_VALUE_ATTR = "input.value"
OPENINFERENCE_INPUT_MIME_ATTR = "input.mime_type"
OPENINFERENCE_OUTPUT_VALUE_ATTR = "output.value"
OPENINFERENCE_OUTPUT_MIME_ATTR = "output.mime_type"
AGENT_SOURCE_TYPE_ATTR = "agent.source.type"
AGENT_SOURCE_URL_ATTR = "agent.source.url"
AGENT_SOURCE_TITLE_ATTR = "agent.source.title"

AGENT_PR_REVIEW = "pr_review_agent"
AGENT_SUPPORT = "support_agent"
AGENT_ENGAGEMENT = "engagement_agent"
TOP_LEVEL_AGENTS = frozenset({AGENT_PR_REVIEW, AGENT_SUPPORT, AGENT_ENGAGEMENT})

_MAX_SPAN_VALUE_CHARS = 4000
_MAX_SOURCE_TITLE_CHARS = 80


def _is_snowflake(value: object) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return stripped.isdigit() and 1 <= len(stripped) <= 20 and not stripped.startswith("0")


def discord_source_url(
    guild_id: object,
    channel_id: object,
    message_id: object = None,
) -> str:
    """Build a Discord permalink (``https://discord.com/channels/<guild>/<channel>[/<message>]``)."""
    gid = str(guild_id or "").strip()
    cid = str(channel_id or "").strip()
    if not (_is_snowflake(gid) and _is_snowflake(cid)):
        return ""
    mid = str(message_id or "").strip()
    if _is_snowflake(mid):
        return f"https://discord.com/channels/{gid}/{cid}/{mid}"
    return f"https://discord.com/channels/{gid}/{cid}"


def truncate_source_title(
    text: object,
    *,
    fallback: str = "",
    limit: int = _MAX_SOURCE_TITLE_CHARS,
) -> str:
    """Normalize and truncate triggering text to ~80 chars for ``agent.source.title``."""
    raw = str(text or "")
    cleaned = " ".join(raw.split()).strip()
    if not cleaned:
        cleaned = " ".join(str(fallback or "").split()).strip()
    if not cleaned:
        return ""
    max_len = max(10, int(limit))
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."
_DEFAULT_SERVICE_NAME = "aria-litellm-discord"
_OTLP_EXPORT_TIMEOUT_SECONDS = 30
_OTLP_BATCH_MAX_SIZE = 16
_OTLP_EXPORT_TIMEOUT_MS = 30_000
_AGENT_FLUSH_TIMEOUT_MS = 5_000

_guard = threading.Lock()
_configured = False
_tracer_provider = None
_tracer = None
_atexit_registered = False

_ACTIVE_AGENT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "contributor_bot_active_agent", default=None
)
_ACTIVE_AGENT_SPAN_HANDLE: contextvars.ContextVar[_SpanHandle | None] = contextvars.ContextVar(
    "contributor_bot_active_agent_span_handle", default=None
)
_ACTIVE_SPAN_HANDLE: contextvars.ContextVar[_SpanHandle | None] = contextvars.ContextVar(
    "contributor_bot_active_span_handle", default=None
)


def current_agent_name() -> str | None:
    """Return the active top-level agent name in the current context, if any."""
    return _ACTIVE_AGENT.get()


def current_agent_span() -> _SpanHandle:
    """Return the active top-level AGENT span handle, or a no-op handle."""
    handle = _ACTIVE_AGENT_SPAN_HANDLE.get()
    return handle if handle is not None else _NOOP_SPAN


def current_span() -> _SpanHandle:
    """Return the innermost active span handle, or a no-op handle when tracing is off."""
    handle = _ACTIVE_SPAN_HANDLE.get()
    return handle if handle is not None else _NOOP_SPAN


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


def flush_tracing(timeout_millis: int = _AGENT_FLUSH_TIMEOUT_MS) -> bool:
    """Flush buffered spans to the exporter without shutting down the provider."""
    provider = _tracer_provider
    if provider is None:
        return False
    force_flush = getattr(provider, "force_flush", None)
    if not callable(force_flush):
        return False
    try:
        try:
            result = force_flush(timeout_millis=timeout_millis)
        except TypeError:
            result = force_flush()
        return True if result is None else bool(result)
    except Exception:
        logging.debug("Tracing force_flush failed", exc_info=True)
        return False


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


def _build_otlp_exporter(OTLPSpanExporter, endpoint: str, headers: dict[str, str]):
    """Build an OTLP HTTP exporter with gzip compression and 30s timeout when supported."""
    kwargs: dict[str, object] = {
        "endpoint": endpoint,
        "headers": headers or None,
    }
    compression = None
    try:
        from opentelemetry.exporter.otlp.proto.http import Compression
        compression = Compression.Gzip
    except Exception:
        compression = None
    if compression is not None:
        kwargs["compression"] = compression
    kwargs["timeout"] = _OTLP_EXPORT_TIMEOUT_SECONDS
    try:
        return OTLPSpanExporter(**kwargs)
    except TypeError:
        return OTLPSpanExporter(endpoint=endpoint, headers=headers or None)


def _build_batch_processor(BatchSpanProcessor, exporter):
    """Build a BatchSpanProcessor sized for multi-turn agent traces."""
    try:
        return BatchSpanProcessor(
            exporter,
            max_export_batch_size=_OTLP_BATCH_MAX_SIZE,
            export_timeout_millis=_OTLP_EXPORT_TIMEOUT_MS,
        )
    except TypeError:
        return BatchSpanProcessor(exporter)


def _instrument_litellm(LiteLLMInstrumentor, provider) -> None:
    """Instrument LiteLLM with TraceConfig that drops duplicate invocation_parameters bloat."""
    trace_config = None
    try:
        from openinference.instrumentation import TraceConfig
        trace_config = TraceConfig(hide_llm_invocation_parameters=True)
    except Exception:
        trace_config = None
    instrumentor = LiteLLMInstrumentor()
    if trace_config is not None:
        try:
            instrumentor.instrument(tracer_provider=provider, config=trace_config)
            return
        except TypeError:
            pass
    instrumentor.instrument(tracer_provider=provider)


def install_tracer_provider(provider, *, instrument_litellm: bool = False) -> None:
    """Install a custom TracerProvider (e.g. with InMemorySpanExporter in tests)."""
    global _configured, _tracer_provider, _tracer
    with _guard:
        if instrument_litellm:
            from openinference.instrumentation.litellm import LiteLLMInstrumentor
            _instrument_litellm(LiteLLMInstrumentor, provider)
        _tracer_provider = provider
        _tracer = provider.get_tracer("contributor_bot")
        _configured = True


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
        exporter = _build_otlp_exporter(OTLPSpanExporter, otlp_endpoint, headers)
        provider.add_span_processor(_build_batch_processor(BatchSpanProcessor, exporter))
        trace.set_tracer_provider(provider)
        _instrument_litellm(LiteLLMInstrumentor, provider)

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
    if isinstance(value, (bool, int, float)):
        span.set_attribute(key, value)
    elif isinstance(value, str):
        if value:
            span.set_attribute(key, value[:_MAX_SPAN_VALUE_CHARS])
    elif isinstance(value, (dict, list, tuple)):
        try:
            encoded = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            encoded = str(value)
        span.set_attribute(key, encoded[:_MAX_SPAN_VALUE_CHARS])
    else:
        text = str(value).strip()
        if text:
            span.set_attribute(key, text[:_MAX_SPAN_VALUE_CHARS])


def _sanitize_event_attributes(attributes: Mapping[str, object] | None) -> dict[str, object]:
    if not attributes:
        return {}
    clean: dict[str, object] = {}
    for raw_key, value in attributes.items():
        if value is None:
            continue
        key = str(raw_key)
        if isinstance(value, (bool, int, float)):
            clean[key] = value
        elif isinstance(value, str):
            if value:
                clean[key] = value[:_MAX_SPAN_VALUE_CHARS]
        elif isinstance(value, (dict, list, tuple)):
            try:
                encoded = json.dumps(value, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                encoded = str(value)
            clean[key] = encoded[:_MAX_SPAN_VALUE_CHARS]
        else:
            text = str(value).strip()
            if text:
                clean[key] = text[:_MAX_SPAN_VALUE_CHARS]
    return clean


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

    def set_attributes(self, attributes: Mapping[str, object] | None) -> None:
        if self._span is None or not attributes:
            return
        for key, value in attributes.items():
            if isinstance(value, (dict, list, tuple)):
                try:
                    encoded = json.dumps(value, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    encoded = str(value)
                _set_safe_attribute(self._span, str(key), encoded)
            else:
                _set_safe_attribute(self._span, str(key), value)

    def add_event(self, name: str, attributes: Mapping[str, object] | None = None) -> None:
        if self._span is None or not name:
            return
        add_event_fn = getattr(self._span, "add_event", None)
        if not callable(add_event_fn):
            return
        clean = _sanitize_event_attributes(attributes)
        try:
            add_event_fn(str(name), attributes=clean)
        except TypeError:
            try:
                add_event_fn(str(name), clean)
            except Exception:
                pass
        except Exception:
            pass

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


def record_span_event(name: str, attributes: Mapping[str, object] | None = None) -> None:
    """Attach a structured event to the innermost active span (no-op when tracing is off)."""
    handle = _ACTIVE_SPAN_HANDLE.get()
    if handle is not None and handle.active:
        handle.add_event(name, attributes)


def log_decision(
    event_name: str,
    message: str,
    *args,
    attributes: Mapping[str, object] | None = None,
    level: int = logging.INFO,
) -> None:
    """Log a decision AND mirror it onto the active OTel span (log -> span parity)."""
    logging.log(level, message, *args)
    handle = _ACTIVE_SPAN_HANDLE.get()
    if handle is None or not handle.active:
        return
    try:
        rendered = (message % args) if args else message
    except Exception:
        rendered = message
    payload: dict[str, object] = {
        "log.level": logging.getLevelName(level),
        "log.message": rendered[:_MAX_SPAN_VALUE_CHARS],
    }
    if attributes:
        payload.update(attributes)
    handle.add_event(event_name, payload)


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


def _resolve_tracer():
    tracer = _tracer
    if tracer is not None:
        return tracer
    if not tracing_endpoint():
        return None
    configure_litellm_tracing()
    return _tracer


@contextmanager
def child_span(
    name: str,
    *,
    kind: str = OPENINFERENCE_SPAN_KIND_CHAIN,
    input_value: object = None,
    attributes: Mapping[str, object] | None = None,
) -> Iterator[_SpanHandle]:
    """Open a child span (``CHAIN``, ``TOOL``, ``RETRIEVER``) without setting ``agent.name``.

    Completely a zero-import no-op when tracing is not configured.
    """
    tracer = _resolve_tracer()
    if tracer is None:
        yield _NOOP_SPAN
        return

    span_name = (name or "step").strip() or "step"
    span_kind = (kind or OPENINFERENCE_SPAN_KIND_CHAIN).strip() or OPENINFERENCE_SPAN_KIND_CHAIN
    with tracer.start_as_current_span(span_name) as span:
        handle = _SpanHandle(span, "")
        handle_token = _ACTIVE_SPAN_HANDLE.set(handle)
        span.set_attribute(OPENINFERENCE_SPAN_KIND_ATTR, span_kind)
        if input_value is not None:
            handle.set_input(input_value)
        if attributes:
            handle.set_attributes(attributes)
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
        finally:
            _ACTIVE_SPAN_HANDLE.reset(handle_token)


@contextmanager
def tool_span(
    name: str,
    *,
    input_value: object = None,
    attributes: Mapping[str, object] | None = None,
) -> Iterator[_SpanHandle]:
    """Open an OpenInference ``TOOL`` child span for a tool invocation."""
    merged: dict[str, object] = {OPENINFERENCE_TOOL_NAME_ATTR: (name or "tool").strip() or "tool"}
    if attributes:
        merged.update(attributes)
    with child_span(
        name,
        kind=OPENINFERENCE_SPAN_KIND_TOOL,
        input_value=input_value,
        attributes=merged,
    ) as handle:
        yield handle


@contextmanager
def agent_span(
    name: str,
    *,
    session_id: str | None = None,
    input_value: object = None,
    attributes: Mapping[str, object] | None = None,
    source_type: str | None = "discord",
    source_url: str | None = None,
    source_title: str | None = None,
) -> Iterator[_SpanHandle]:
    """Wrap one logical bot agent run in a top-level OpenInference ``AGENT`` span.

    Completely a zero-import no-op when tracing is not configured.
    When called while already inside an active ``AGENT`` span, opens a ``CHAIN``
    child span (without ``agent.name``) so Lens always shows only the top-level
    agent. Flushes the active TracerProvider when a root ``AGENT`` span exits so
    multi-turn runs are not dropped on batch delay or shutdown.
    """
    tracer = _resolve_tracer()
    if tracer is None:
        yield _NOOP_SPAN
        return

    agent_name = (name or "bot").strip() or "bot"
    if _ACTIVE_AGENT.get() is not None:
        with child_span(
            agent_name,
            kind=OPENINFERENCE_SPAN_KIND_CHAIN,
            input_value=input_value,
            attributes=attributes,
        ) as handle:
            yield handle
        return

    sid = (str(session_id).strip() if session_id is not None else "") or f"{agent_name}:{uuid.uuid4().hex[:12]}"
    token = _ACTIVE_AGENT.set(agent_name)
    try:
        with tracer.start_as_current_span(agent_name) as span:
            handle = _SpanHandle(span, sid)
            agent_handle_token = _ACTIVE_AGENT_SPAN_HANDLE.set(handle)
            handle_token = _ACTIVE_SPAN_HANDLE.set(handle)
            span.set_attribute(OPENINFERENCE_SPAN_KIND_ATTR, OPENINFERENCE_SPAN_KIND_AGENT)
            span.set_attribute(OPENINFERENCE_AGENT_NAME_ATTR, agent_name)
            span.set_attribute(OPENINFERENCE_SESSION_ID_ATTR, sid)
            if source_type:
                _set_safe_attribute(span, AGENT_SOURCE_TYPE_ATTR, str(source_type).strip())
            if source_url and str(source_url).strip().startswith("https://"):
                _set_safe_attribute(span, AGENT_SOURCE_URL_ATTR, str(source_url).strip())
            if source_title:
                clean_title = truncate_source_title(source_title)
                if clean_title:
                    _set_safe_attribute(span, AGENT_SOURCE_TITLE_ATTR, clean_title)
            if input_value is not None:
                handle.set_input(input_value)
            if attributes:
                handle.set_attributes(attributes)
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
                finally:
                    _ACTIVE_SPAN_HANDLE.reset(handle_token)
                    _ACTIVE_AGENT_SPAN_HANDLE.reset(agent_handle_token)
    finally:
        _ACTIVE_AGENT.reset(token)
        flush_tracing()


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
    """Build LiteLLM ``metadata`` dict for grouping traces by agent and session."""
    feature = (agent or current_agent_name() or "bot").strip() or "bot"
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
    feature = (agent or current_agent_name() or "bot").strip() or "bot"
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
