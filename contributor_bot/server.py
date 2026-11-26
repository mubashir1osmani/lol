"""HTTP endpoints for the GitHub webhook, GitHub OAuth, and Discord interactions."""

import html
import json
import logging
import threading
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .api import TransientAPIError
from .config import Config
from .interactions import handle_interaction
from .oauth import callback_page
from .review_job import schedule_webhook_recheck
from .rewards import handle_github_event
from .signatures import verify_discord_interaction, verify_github_signature
from .store import Store

MAX_BODY_BYTES = 2_000_000


class RewardsHandler(BaseHTTPRequestHandler):
    server_version = "litellm-contributor-bot"
    sys_version = "1"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        try:
            self._get()
        except Exception:
            logging.exception("HTTP GET failed")
            self._send(500, b"internal error", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        try:
            self._post()
        except Exception:
            logging.exception("HTTP POST failed")
            self._send(500, b'{"error":"internal error"}', "application/json")

    def _get(self) -> None:
        path, query = _target(self.path)
        if path == "/health":
            self._send(200, b'{"ok":true}', "application/json")
            return
        if path == "/github/oauth/callback":
            status, text = callback_page(self.server.config, self.server.store, query)
            self._send(status, _page(text), "text/html; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def _post(self) -> None:
        path, _query = _target(self.path)
        try:
            size = int(self.headers.get("Content-Length", ""))
        except (TypeError, ValueError):
            self._send(411, b"length required", "text/plain; charset=utf-8")
            return
        if size < 0 or size > MAX_BODY_BYTES:
            self._send(413, b"payload too large", "text/plain; charset=utf-8")
            return
        body = self.rfile.read(size) if size else b""
        if path == "/github/webhook":
            self._webhook(body)
            return
        if path == "/discord/interactions":
            self._interactions(body)
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def _webhook(self, body: bytes) -> None:
        if not verify_github_signature(
            self.server.config.github_webhook_secret or "",
            body,
            self.headers.get("X-Hub-Signature-256"),
        ):
            logging.warning("Rejected a GitHub webhook with a bad signature")
            self._send(401, b'{"error":"bad signature"}', "application/json")
            return
        event = self.headers.get("X-GitHub-Event") or ""
        delivery = self.headers.get("X-GitHub-Delivery") or ""
        logging.info("GitHub webhook %s delivery %s", event or "unknown", delivery or "unknown")
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send(400, b'{"error":"invalid json"}', "application/json")
            return
        if not isinstance(payload, dict):
            self._send(400, b'{"error":"invalid json"}', "application/json")
            return
        try:
            status = handle_github_event(self.server.config, self.server.store, event, payload)
        except TransientAPIError:
            logging.warning("GitHub webhook deferred")
            status = 500
        try:
            schedule_webhook_recheck(self.server.config, self.server.store, event, payload)
        except Exception:
            logging.exception("PR review webhook hook failed")
        if status == 200:
            response = b'{"ok":true}'
        elif status == 400:
            response = b'{"error":"invalid payload"}'
        else:
            response = b'{"error":"unavailable"}'
        self._send(status, response, "application/json")

    def _interactions(self, body: bytes) -> None:
        if not verify_discord_interaction(
            self.server.config.discord_public_key or "",
            self.headers.get("X-Signature-Ed25519"),
            self.headers.get("X-Signature-Timestamp"),
            body,
        ):
            logging.warning("Rejected a Discord interaction with a bad signature")
            self._send(401, b'{"error":"bad signature"}', "application/json")
            return
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send(400, b'{"error":"invalid json"}', "application/json")
            return
        if not isinstance(payload, dict):
            self._send(400, b'{"error":"invalid json"}', "application/json")
            return
        response = handle_interaction(self.server.config, self.server.store, payload)
        self._send(200, json.dumps(response).encode("utf-8"), "application/json")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        if getattr(self, "_sent", False):
            return
        self._sent = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        logging.info("http %s %s %s", self.command, urlsplit(self.path).path, code)


class RewardsHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, config: Config, store: Store):
        self.config = config
        self.store = store
        super().__init__(server_address, RewardsHandler)


def _target(raw_path: str) -> tuple[str, str]:
    parts = urlsplit(raw_path)
    return parts.path, parts.query


def _page(text: str) -> bytes:
    safe = html.escape(text)
    return (
        "<!doctype html><meta charset=\"utf-8\"><title>LiteLLM Discord</title>"
        f"<p>{safe}</p>"
    ).encode("utf-8")


def start_http_server(config: Config, store: Store | None = None) -> RewardsHTTPServer:
    owned = store or Store(config.roles_db_path)
    try:
        server = RewardsHTTPServer((config.http_host, config.http_port), config, owned)
    except Exception:
        if store is None:
            owned.close()
        raise
    thread = threading.Thread(target=server.serve_forever, name="contributor-roles-http", daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    logging.info("Contributor HTTP server listening on %s:%s", host, port)
    return server


def stop_http_server(server: RewardsHTTPServer) -> None:
    server.shutdown()
    server.server_close()
    server.store.close()
