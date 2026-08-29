import unittest

from release_bot.api import release_message
from release_bot.config import Config, get_config
from release_bot.poller import poll_once


def release(release_id: int, tag: str) -> dict:
    return {"id": release_id, "tag_name": tag, "draft": False, "prerelease": False}


class ConfigTests(unittest.TestCase):
    def test_litellm_defaults(self) -> None:
        config = get_config(
            {"DISCORD_BOT_TOKEN": "token", "DISCORD_CHANNEL_ID": "123"}
        )
        self.assertEqual(config.github_repository, "BerriAI/litellm")
        self.assertEqual(config.poll_interval_seconds, 300)
        self.assertTrue(config.include_prereleases)

    def test_missing_credentials(self) -> None:
        with self.assertRaisesRegex(ValueError, "DISCORD_BOT_TOKEN"):
            get_config({})


class MessageTests(unittest.TestCase):
    def test_embed_is_bounded(self) -> None:
        message = release_message(
            {
                "tag_name": "v1.2.3",
                "name": "A release",
                "body": "x" * 10_000,
                "html_url": "https://github.com/BerriAI/litellm/releases/tag/v1.2.3",
                "prerelease": False,
                "published_at": "2026-01-01T00:00:00Z",
                "author": {"login": "octocat"},
            },
            "BerriAI/litellm",
        )
        self.assertIn("v1.2.3", message["content"])
        self.assertLessEqual(len(message["embeds"][0]["description"]), 3_800)
        self.assertEqual(message["allowed_mentions"], {"parse": []})


class PollerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(discord_token="token", discord_channel_id="channel")

    def test_first_run_posts_latest_and_seeds_state(self) -> None:
        posted: list[str] = []
        saved: list[str] = []
        poll_once(
            self.config,
            lambda _repo, _token: [release(3, "v3"), release(2, "v2")],
            lambda _channel, _token, item, _repo: posted.append(item["tag_name"]),
            lambda _path: (False, []),
            lambda _path, ids: saved.extend(ids),
        )
        self.assertEqual(posted, ["v3"])
        self.assertEqual(saved, ["3", "2"])

    def test_later_runs_post_unseen_oldest_first(self) -> None:
        posted: list[str] = []
        poll_once(
            self.config,
            lambda _repo, _token: [
                release(4, "v4"),
                release(3, "v3"),
                release(2, "v2"),
            ],
            lambda _channel, _token, item, _repo: posted.append(item["tag_name"]),
            lambda _path: (True, ["2"]),
            lambda _path, _ids: None,
        )
        self.assertEqual(posted, ["v3", "v4"])


if __name__ == "__main__":
    unittest.main()
