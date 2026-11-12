import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request

from contributor_bot import api
from contributor_bot.config import Config
from contributor_bot.state import load_state, save_state


class NetworkBoundaryTests(unittest.TestCase):
    def test_unapproved_destinations_never_reach_network(self):
        urls = [
            "http://api.github.com/repos/BerriAI/litellm/pulls/1",
            "https://api.github.com.evil.test/repos/BerriAI/litellm/pulls/1",
            "https://api.github.com:444/repos/BerriAI/litellm/pulls/1",
            "https://user:password@api.github.com/repos/BerriAI/litellm/pulls/1",
            "https://169.254.169.254/latest/meta-data/",
            "https://localhost/admin",
            "https://api.github.com/repos/../litellm/pulls/1",
            "https://discord.com/api/v10/users/@me",
            "https://api.render.com/v1/services",
        ]
        with mock.patch.object(api, "_open_request") as request:
            for url in urls:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    api._http(Request(url))
        request.assert_not_called()

    def test_github_writes_and_discord_admin_operations_are_blocked(self):
        for url, method in [
            ("https://api.github.com/repos/BerriAI/litellm/pulls/1", "PATCH"),
            ("https://api.github.com/repos/BerriAI/litellm/pulls/1/merge", "PUT"),
            ("https://discord.com/api/v10/channels/123", "DELETE"),
            ("https://discord.com/api/v10/guilds/123/roles", "POST"),
        ]:
            with self.subTest(url=url, method=method), mock.patch.object(api, "_open_request") as request:
                with self.assertRaises(ValueError):
                    api._http(Request(url, method=method))
                request.assert_not_called()

    def test_redirects_never_return_a_credentialed_followup_request(self):
        request = Request(
            "https://discord.com/api/v10/channels/123",
            headers={"Authorization": "Bot secret"},
        )
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code), self.assertRaisesRegex(RuntimeError, "redirects"):
                api._NoRedirects().redirect_request(request, None, code, "redirect", {}, "https://evil.test")

    def test_response_size_is_bounded(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status, response.headers = 200, {}
        response.read.return_value = b"x" * (api.MAX_RESPONSE_BYTES + 1)
        with mock.patch.object(api, "_open_request", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "size limit"):
                api._http(Request("https://discord.com/api/v10/channels/123"))
        response.read.assert_called_once_with(api.MAX_RESPONSE_BYTES + 1)

    def test_upstream_error_bodies_do_not_leak_into_logs(self):
        url = "https://discord.com/api/v10/channels/123"
        error = HTTPError(url, 403, "Forbidden", {}, io.BytesIO(b"echoed-super-secret-token"))
        with mock.patch.object(api, "_open_request", side_effect=error):
            with self.assertRaises(RuntimeError) as raised:
                api._http(Request(url))
        self.assertEqual(str(raised.exception), "discord.com returned HTTP 403")
        self.assertNotIn("super-secret", str(raised.exception))


class CredentialAndStateTests(unittest.TestCase):
    def test_configuration_repr_redacts_tokens(self):
        config = Config("discord-secret", "123", "456", github_token="github-secret")
        self.assertNotIn("discord-secret", repr(config))
        self.assertNotIn("github-secret", repr(config))

    def test_preplanted_temporary_symlink_cannot_overwrite_other_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            victim = root / "bot.py"
            victim.write_text("original application")
            (root / "state.json.tmp").symlink_to(victim)
            config = Config("token", "123", "456", state_file=str(root / "state.json"))
            state = load_state(config)
            save_state(config.state_file, state)
            self.assertEqual(victim.read_text(), "original application")
            self.assertEqual(load_state(config), state)

    def test_mass_mentions_are_disabled_at_the_send_boundary(self):
        message = {"content": "@everyone <@&555> <@111>", "allowed_mentions": {"parse": ["everyone", "roles"], "roles": ["555"], "users": ["111"]}}
        with mock.patch.object(api, "_request_json", return_value={"id": "999", "channel_id": "123"}) as send:
            api.post_message("123", "token", message)
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(payload["allowed_mentions"], {"parse": [], "users": ["111"]})


if __name__ == "__main__":
    unittest.main()
