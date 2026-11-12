from dataclasses import replace
import unittest
from unittest import mock

from contributor_bot import api
from contributor_bot import main as bot_main
from contributor_bot.config import Config, get_config


class DeliveryTestTests(unittest.TestCase):
    def setUp(self):
        self.channel = {"id": "789", "name": "moderators-only", "type": 0, "guild_id": "111"}
        self.receipt = {"id": "999", "channel_id": "789"}

    def test_one_message_to_moderators_only_without_pings(self):
        with mock.patch.object(bot_main, "fetch_channel", return_value=self.channel) as fetch, mock.patch.object(bot_main, "post_message", return_value=self.receipt) as post:
            url = bot_main.send_test_message("789", "token")
        fetch.assert_called_once_with("789", "token")
        post.assert_called_once()
        channel_id, token, payload = post.call_args.args
        self.assertEqual((channel_id, token), ("789", "token"))
        self.assertEqual(payload["allowed_mentions"], {"parse": []})
        self.assertIn("delivery test", payload["content"])
        self.assertTrue(payload["enforce_nonce"])
        self.assertEqual(url, "https://discord.com/channels/111/789/999")

    def test_other_destinations_are_rejected_before_sending(self):
        for override in ({"name": "announcements"}, {"type": 15}, {"guild_id": None}):
            with self.subTest(override=override), mock.patch.object(bot_main, "fetch_channel", return_value=self.channel | override), mock.patch.object(bot_main, "post_message") as post:
                with self.assertRaisesRegex(ValueError, "#moderators-only"):
                    bot_main.send_test_message("789", "token")
                post.assert_not_called()

    def test_test_command_does_not_load_production_config_or_start_worker(self):
        with mock.patch.object(bot_main, "load_dotenv"), mock.patch.dict("os.environ", {"DISCORD_BOT_TOKEN": "token"}, clear=True), mock.patch.object(bot_main, "get_config") as config, mock.patch.object(bot_main, "poll_once") as poll, mock.patch.object(bot_main, "send_test_message", return_value="https://discord.com/channels/111/789/999") as send, mock.patch("builtins.print"):
            bot_main.main(["--test-channel-id", "789"])
        send.assert_called_once_with("789", "token")
        config.assert_not_called()
        poll.assert_not_called()

    def test_invalid_id_or_missing_token_never_calls_discord(self):
        with mock.patch.object(bot_main, "fetch_channel") as fetch:
            for channel_id, token in (("#moderators-only", "token"), ("789", "")):
                with self.subTest(channel_id=channel_id), self.assertRaises(ValueError):
                    bot_main.send_test_message(channel_id, token)
        fetch.assert_not_called()

    def test_paused_worker_does_not_poll_or_post(self):
        config = replace(Config("token", "123", "456"), announcements_enabled=False)
        with mock.patch.object(bot_main, "load_dotenv"), mock.patch.object(bot_main, "get_config", return_value=config), mock.patch.object(bot_main.signal, "signal"), mock.patch.object(bot_main.threading, "Event") as event, mock.patch.object(bot_main, "poll_once") as poll, mock.patch.object(bot_main, "post_message") as post:
            bot_main.main([])
        event.return_value.wait.assert_called_once_with()
        poll.assert_not_called()
        post.assert_not_called()

    def test_pause_configuration_is_explicit(self):
        env = {"DISCORD_BOT_TOKEN": "token", "OSS_CONTRIBUTIONS_CHANNEL_ID": "123", "ANNOUNCEMENTS_CHANNEL_ID": "456"}
        self.assertTrue(get_config(env).announcements_enabled)
        self.assertFalse(get_config(env | {"ANNOUNCEMENTS_ENABLED": "false"}).announcements_enabled)
        with self.assertRaisesRegex(ValueError, "ANNOUNCEMENTS_ENABLED"):
            get_config(env | {"ANNOUNCEMENTS_ENABLED": "oops"})

    def test_discord_receipt_required_before_reporting_success(self):
        for receipt in (None, {}, {"id": "999", "channel_id": "123"}):
            with self.subTest(receipt=receipt), mock.patch.object(api, "_request_json", return_value=receipt):
                with self.assertRaisesRegex(RuntimeError, "receipt"):
                    api.post_message("789", "token", {"content": "test"})


if __name__ == "__main__":
    unittest.main()
