import hashlib
import hmac
import http.client
import json
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from unittest import mock

from contributor_bot.api import TransientAPIError
from contributor_bot.config import get_config
from contributor_bot.interactions import SLASH_COMMANDS, handle_interaction
from contributor_bot.oauth import callback_page
from contributor_bot.rewards import handle_github_event
from contributor_bot.server import RewardsHTTPServer
from contributor_bot.signatures import (
    ed25519_verify,
    verify_discord_interaction,
    verify_github_signature,
)
from contributor_bot.store import Store
from tests.test_roles import roles_env

# RFC 8032 test vector 1 (empty message) and an OpenSSL Ed25519 signature over
# the timestamped Discord ping body "1700000000{"type":1}".
RFC_PUBLIC = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
RFC_SIGNATURE = bytes.fromhex(
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
    "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
)
DISCORD_PUBLIC = "279bf352371577251884cbc4c9e89a98f756cb7582483ac591eb6898d11fa279"
DISCORD_SIGNATURE = (
    "181de245e1ca438184d77004ae2f116440eb9c1fe9fb92c6b7b5b0fa0f0c642d"
    "4323eb421e1153933156b653e0f1dec6a4335a858c018bb60ef17f17e5a45c08"
)
DISCORD_TIMESTAMP = "1700000000"
DISCORD_BODY = b'{"type":1}'


def merged_event(number=7, login="octocat", user_id=5, bot=False, merged=True, labels=(), repo="BerriAI/litellm", action="closed"):
    return {
        "action": action,
        "pull_request": {
            "number": number,
            "merged": merged,
            "merged_at": "2026-11-02T03:04:05Z" if merged else None,
            "title": "add mcp tools",
            "user": {"login": login, "id": user_id, "type": "Bot" if bot else "User"},
            "labels": [{"name": label} for label in labels],
        },
        "repository": {"full_name": repo},
    }


class SignatureTests(unittest.TestCase):
    def test_github_hmac_accepts_only_the_exact_body(self):
        body = b'{"zen":"keep it secret"}'
        good = "sha256=" + hmac.new(b"webhook-secret-value", body, hashlib.sha256).hexdigest()
        self.assertTrue(verify_github_signature("webhook-secret-value", body, good))
        prefix, digest = good.split("=", 1)
        self.assertTrue(verify_github_signature("webhook-secret-value", body, prefix + "=" + digest.upper()))
        self.assertFalse(verify_github_signature("webhook-secret-value", body + b" ", good))
        self.assertFalse(verify_github_signature("other-secret-value", body, good))
        self.assertFalse(verify_github_signature("webhook-secret-value", body, "sha1=" + "ab" * 20))
        self.assertFalse(verify_github_signature("webhook-secret-value", body, None))
        self.assertFalse(verify_github_signature("webhook-secret-value", body, "sha256=abcd"))
        self.assertFalse(verify_github_signature("", body, good))

    def test_ed25519_matches_rfc_and_a_discord_ping_signature(self):
        self.assertTrue(ed25519_verify(RFC_PUBLIC, b"", RFC_SIGNATURE))
        self.assertFalse(ed25519_verify(RFC_PUBLIC, b"x", RFC_SIGNATURE))
        flipped = bytearray(RFC_SIGNATURE)
        flipped[0] ^= 1
        self.assertFalse(ed25519_verify(RFC_PUBLIC, b"", bytes(flipped)))
        self.assertTrue(verify_discord_interaction(
            DISCORD_PUBLIC, DISCORD_SIGNATURE, DISCORD_TIMESTAMP, DISCORD_BODY, now=int(DISCORD_TIMESTAMP),
        ))
        self.assertFalse(verify_discord_interaction(
            DISCORD_PUBLIC, DISCORD_SIGNATURE, DISCORD_TIMESTAMP, DISCORD_BODY, now=int(DISCORD_TIMESTAMP) + 301,
        ))
        self.assertFalse(verify_discord_interaction(
            DISCORD_PUBLIC, DISCORD_SIGNATURE, DISCORD_TIMESTAMP, b'{"type":2}', now=int(DISCORD_TIMESTAMP),
        ))


class WebhookBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = get_config(roles_env(
            self.directory.name,
            PR_COMMENT_INVITES_ENABLED="true",
            DISCORD_INVITE_URL="https://discord.gg/litellm",
        ))
        self.store = Store(self.config.roles_db_path)
        self.addCleanup(self.store.close)
        self.files = mock.patch("contributor_bot.rewards.fetch_pull_files", return_value=["litellm/proxy/mcp_server.py"])
        self.add = mock.patch("contributor_bot.rewards.add_member_role")
        self.post = mock.patch("contributor_bot.rewards.post_message", return_value={"id": "50", "channel_id": "789"})
        self.comment = mock.patch("contributor_bot.rewards.post_issue_comment", return_value={"id": 9})
        self.fetch_files = self.files.start()
        self.add_role = self.add.start()
        self.post_message = self.post.start()
        self.comment_on = self.comment.start()
        self.addCleanup(self.files.stop)
        self.addCleanup(self.add.stop)
        self.addCleanup(self.post.stop)
        self.addCleanup(self.comment.stop)

    def handle(self, payload, event="pull_request"):
        return handle_github_event(self.config, self.store, event, payload)

    def test_merged_human_pr_grants_roles_once_and_ignores_the_rest(self):
        self.store.link_account("111", "octocat", 5, "2026-11-01T00:00:00+00:00")
        self.assertEqual(self.handle(merged_event(labels=("agents",))), 200)
        self.assertEqual(self.handle(merged_event(labels=("agents",))), 200)
        self.assertEqual(
            {call.args[2] for call in self.add_role.call_args_list},
            {"1001", "1003"},
        )
        self.assertEqual(self.post_message.call_count, 1)
        content = self.post_message.call_args.args[2]["content"]
        self.assertEqual(
            content,
            "<@111> just shipped #7 and became a contributor and an agent builder 🎉",
        )
        self.assertEqual(self.post_message.call_args.args[2]["allowed_mentions"], {"parse": [], "users": ["111"]})
        saved = self.store.get_pull("BerriAI/litellm", 7)
        self.assertEqual(saved["files"], ["litellm/proxy/mcp_server.py"])
        self.assertEqual(saved["labels"], ["agents"])
        self.assertIsNotNone(saved["shoutout_at"])
        self.fetch_files.reset_mock()
        self.assertEqual(self.handle(merged_event()), 200)
        self.fetch_files.assert_not_called()
        self.post_message.assert_called_once()

        self.post_message.reset_mock()
        self.add_role.reset_mock()
        self.assertEqual(self.handle(merged_event(number=8, login="dependabot[bot]", user_id=1, bot=True)), 200)
        self.assertEqual(self.handle(merged_event(number=9, merged=False)), 200)
        self.assertEqual(self.handle(merged_event(number=10), event="issues"), 200)
        self.assertEqual(self.handle(merged_event(repo="other/repo")), 200)
        self.assertIsNone(self.store.get_pull("BerriAI/litellm", 8))
        self.add_role.assert_not_called()
        self.post_message.assert_not_called()
        self.comment_on.assert_not_called()

    def test_unlinked_author_gets_one_invite_and_retries_a_rate_limit(self):
        self.comment_on.side_effect = [TransientAPIError("rate limit"), {"id": 9}]
        self.assertEqual(self.handle(merged_event()), 500)
        self.assertEqual(self.handle(merged_event()), 200)
        self.assertEqual(self.handle(merged_event()), 200)
        self.assertEqual(self.comment_on.call_count, 2)
        body = self.comment_on.call_args.args[3]
        self.assertIn("https://discord.gg/litellm", body)
        self.assertIn("/link-github", body)
        self.assertTrue(body.startswith("hey @octocat"))
        self.add_role.assert_not_called()
        quiet = get_config(roles_env(self.directory.name))
        self.comment_on.reset_mock()
        self.assertEqual(handle_github_event(quiet, self.store, "pull_request", merged_event(number=11)), 200)
        self.comment_on.assert_not_called()


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = get_config(roles_env(self.directory.name))
        self.store = Store(self.config.roles_db_path)
        self.addCleanup(self.store.close)

    def command(self, name, discord_id="111", **extra):
        payload = {
            "type": 2,
            "guild_id": "333",
            "application_id": "444",
            "token": "interaction-token-value",
            "member": {"user": {"id": discord_id}},
            "data": {"name": name},
        }
        payload.update(extra)
        return handle_interaction(self.config, self.store, payload)

    def test_commands_and_link_flow(self):
        self.assertEqual([command["name"] for command in SLASH_COMMANDS], ["link-github", "unlink-github", "my-prs"])
        self.assertEqual(handle_interaction(self.config, self.store, {"type": 1}), {"type": 1})
        linked = self.command("link-github")
        self.assertEqual(linked["data"]["flags"], 64)
        self.assertIn("https://github.com/login/oauth/authorize?", linked["data"]["content"])
        self.assertIn("state=", linked["data"]["content"])
        self.assertNotIn("<@", linked["data"]["content"])
        self.assertIn("litellm discord", self.command("link-github", guild_id="999")["data"]["content"])
        self.assertIn("isn't linked", self.command("unlink-github")["data"]["content"])
        self.assertIn("isn't linked", self.command("my-prs")["data"]["content"])

    def test_oauth_callback_links_once_and_my_prs_reports_status(self):
        response = self.command("link-github")
        state = response["data"]["content"].split("state=", 1)[1].split("&", 1)[0].split()[0]
        with mock.patch("contributor_bot.oauth.exchange_github_oauth", return_value="gho_test") as exchange, mock.patch("contributor_bot.oauth.fetch_github_user", return_value={"login": "octocat", "id": 5}), mock.patch("contributor_bot.oauth.schedule_author_sync") as sync:
            status, text = callback_page(self.config, self.store, f"code=abc123456789&state={state}")
            self.assertEqual(callback_page(self.config, self.store, f"code=abc123456789&state={state}")[0], 400)
        self.assertEqual(status, 200)
        self.assertIn("octocat", text)
        exchange.assert_called_once()
        sync.assert_called_once()
        self.assertEqual(self.store.link_for_discord("111")["github_user_id"], 5)
        self.assertIn("already linked", self.store.link_account("222", "octocat", 5, "2026-11-03T00:00:00+00:00"))
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=4, github_user_id=5, github_login="octocat",
            merged_at="2026-11-02T00:00:00+00:00", title="docs", labels=[], files=["docs/a.md", "docs/b.md", "docs/c.md"],
        )
        # Docs-only files do not earn a path role. One merged PR earns Contributor.
        self.store.upsert_pull(
            repository="BerriAI/litellm", number=3, github_user_id=5, github_login="octocat",
            merged_at="2026-01-02T00:00:00+00:00", title="one", labels=[], files=["README.md"],
        )
        edits = []

        def inline_thread(target, args=(), name=None, daemon=None):
            class Started:
                def start(self_inner):
                    target(*args)
            return Started()

        with mock.patch("contributor_bot.interactions.threading.Thread", inline_thread), mock.patch("contributor_bot.interactions.fetch_open_pulls_for_author", return_value=([{"number": 8, "title": "open me"}], 1)), mock.patch("contributor_bot.interactions.fetch_pull_request", return_value={"number": 8, "title": "open me", "draft": False, "merged": False, "user": {"login": "octocat"}}), mock.patch("contributor_bot.interactions.fetch_pull_reviews", return_value=[{"user": {"id": 2}, "state": "CHANGES_REQUESTED"}]), mock.patch("contributor_bot.interactions.edit_original_interaction", side_effect=lambda *args: edits.append(args)):
            deferred = self.command("my-prs")
        self.assertEqual(deferred["type"], 5)
        self.assertIn("changes requested", edits[0][2]["content"])
        self.assertIn("merged: 2 recorded", edits[0][2]["content"])
        self.assertIn("contributor", edits[0][2]["content"])
        self.assertIn("unlinked", self.command("unlink-github")["data"]["content"])
        self.assertIsNone(self.store.link_for_discord("111"))


@contextmanager
def running(config, store):
    server = RewardsHTTPServer(("127.0.0.1", 0), config, store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def request(port, method, path, body=b"", headers=None):
    last_error = None
    for _ in range(30):
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            payload = response.read()
            status = response.status
            connection.close()
            return status, payload
        except (ConnectionRefusedError, http.client.RemoteDisconnected) as error:
            last_error = error
            time.sleep(0.02)
    raise AssertionError(last_error)


class HttpSignatureTests(unittest.TestCase):
    def test_webhook_rejects_a_bad_signature_and_accepts_a_good_one(self):
        with tempfile.TemporaryDirectory() as directory:
            config = get_config(roles_env(directory))
            store = Store(config.roles_db_path)
            body = json.dumps(merged_event()).encode()
            signature = "sha256=" + hmac.new(b"webhook-secret-value", body, hashlib.sha256).hexdigest()
            with running(config, store) as port, mock.patch("contributor_bot.rewards.fetch_pull_files", return_value=["README.md"]), mock.patch("contributor_bot.rewards.add_member_role"), mock.patch("contributor_bot.rewards.post_message"):
                health, health_body = request(port, "GET", "/health")
                rejected, _ = request(port, "POST", "/github/webhook", body, {
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": "sha256=" + "ab" * 32,
                })
                missing, _ = request(port, "POST", "/github/webhook", body, {
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "pull_request",
                })
                accepted, accepted_body = request(port, "POST", "/github/webhook", body, {
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": signature,
                    "X-GitHub-Delivery": "delivery-1",
                })
                ping_body = b'{"type":1}'
                discord_status, _ = request(port, "POST", "/discord/interactions", ping_body, {
                    "Content-Type": "application/json",
                    "X-Signature-Ed25519": "aa" * 64,
                    "X-Signature-Timestamp": "1700000000",
                })
            self.assertEqual(health, 200)
            self.assertEqual(health_body, b'{"ok":true}')
            self.assertEqual(rejected, 401)
            self.assertEqual(missing, 401)
            self.assertEqual(accepted, 200)
            self.assertEqual(accepted_body, b'{"ok":true}')
            self.assertEqual(discord_status, 401)
            self.assertIsNotNone(store.get_pull("BerriAI/litellm", 7))
            self.assertEqual(store.grant_keys("111"), set())
            store.close()


if __name__ == "__main__":
    unittest.main()
