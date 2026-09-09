import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from release_bot import api
from release_bot.agent import feed_prompt, release_prompt, try_complete
from release_bot.api import apply_mention, release_message
from release_bot.config import Config, get_config
from release_bot.feeds import feed_message, parse_feed, poll_feeds_once
from release_bot.poller import poll_once
from release_bot.state import load_ids, save_ids
from release_bot.threads import poll_threads_once


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


class MentionTests(unittest.TestCase):
    def test_role_mention_is_whitelisted(self) -> None:
        message = apply_mention(
            {"content": "hello", "allowed_mentions": {"parse": []}}, "555", False
        )
        self.assertTrue(message["content"].startswith("<@&555>"))
        self.assertEqual(message["allowed_mentions"], {"parse": [], "roles": ["555"]})

    def test_everyone_mention(self) -> None:
        message = apply_mention({"content": "hello"}, None, True)
        self.assertTrue(message["content"].startswith("@everyone"))
        self.assertEqual(message["allowed_mentions"], {"parse": ["everyone"]})

    def test_no_mention_configured(self) -> None:
        message = apply_mention(
            {"content": "hello", "allowed_mentions": {"parse": []}}, None, False
        )
        self.assertEqual(message["content"], "hello")
        self.assertEqual(message["allowed_mentions"], {"parse": []})


class AgentTests(unittest.TestCase):
    def test_prompts_include_source_material(self) -> None:
        prompt = release_prompt(
            {"tag_name": "v1.0.0", "name": "Big one", "body": "Added GPT-6"},
            "BerriAI/litellm",
        )
        self.assertIn("v1.0.0", prompt)
        self.assertIn("Added GPT-6", prompt)
        blurb = feed_prompt(
            {"title": "Post", "link": "https://x", "summary": "Details"}, "blog"
        )
        self.assertIn("Details", blurb)

    def test_try_complete_without_agent_returns_none(self) -> None:
        self.assertIsNone(try_complete(None, "anything"))

    def test_try_complete_swallows_failures(self) -> None:
        class Exploding:
            def complete(self, _prompt: str) -> str:
                raise RuntimeError("api down")

        self.assertIsNone(try_complete(Exploding(), "anything"))

    def test_release_message_prefers_digest(self) -> None:
        message = release_message(
            {
                "tag_name": "v1.2.3",
                "body": "raw notes",
                "html_url": "https://github.com/BerriAI/litellm/releases/tag/v1.2.3",
            },
            "BerriAI/litellm",
            digest="Claude's summary",
        )
        self.assertEqual(message["embeds"][0]["description"], "Claude's summary")


RSS_SAMPLE = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
  <channel>
    <title>liteLLM Blog</title>
    <item>
      <title><![CDATA[Newest post]]></title>
      <link>https://docs.litellm.ai/blog/newest</link>
      <guid>https://docs.litellm.ai/blog/newest</guid>
      <description><![CDATA[<p>Some <b>HTML</b> summary.</p>]]></description>
    </item>
    <item>
      <title>Older post</title>
      <link>https://docs.litellm.ai/blog/older</link>
      <guid>https://docs.litellm.ai/blog/older</guid>
      <description>Plain summary</description>
    </item>
  </channel>
</rss>
"""

ATOM_SAMPLE = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Social</title>
  <entry>
    <id>tag:social,1</id>
    <title>We shipped a thing</title>
    <link rel="alternate" href="https://example.com/post/1"/>
    <summary>Announcement text</summary>
  </entry>
</feed>
"""


class FeedTests(unittest.TestCase):
    def test_parses_rss_and_strips_html(self) -> None:
        entries = parse_feed(RSS_SAMPLE)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["title"], "Newest post")
        self.assertEqual(entries[0]["summary"], "Some  HTML  summary.")

    def test_parses_atom(self) -> None:
        entries = parse_feed(ATOM_SAMPLE)
        self.assertEqual(entries[0]["link"], "https://example.com/post/1")
        self.assertEqual(entries[0]["title"], "We shipped a thing")

    def test_blog_message_shape(self) -> None:
        entry = parse_feed(RSS_SAMPLE)[0]
        message = feed_message(entry, "blog")
        self.assertIn("blog", message["content"].lower())
        self.assertEqual(
            message["embeds"][0]["url"], "https://docs.litellm.ai/blog/newest"
        )
        self.assertEqual(message["allowed_mentions"], {"parse": []})

    def test_first_poll_seeds_then_posts_only_new(self) -> None:
        config = Config(
            discord_token="token",
            discord_channel_id="channel",
            blog_feed_url="https://docs.litellm.ai/blog/rss.xml",
        )
        posted: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            state_file = str(Path(directory) / "feeds.json")
            older_only = parse_feed(RSS_SAMPLE)[1:]
            both = parse_feed(RSS_SAMPLE)

            poll_feeds_once(
                config,
                lambda _url: older_only,
                lambda _c, _t, message: posted.append(message["embeds"][0]["title"]),
                state_file,
            )
            self.assertEqual(posted, [])

            poll_feeds_once(
                config,
                lambda _url: both,
                lambda _c, _t, message: posted.append(message["embeds"][0]["title"]),
                state_file,
            )
        self.assertEqual(posted, ["Newest post"])


class ThreadTests(unittest.TestCase):
    def make_config(self, **overrides) -> Config:
        values = {
            "discord_token": "token",
            "discord_channel_id": "channel",
            "discord_guild_id": "42",
            "support_user_id": "777",
        }
        values.update(overrides)
        return Config(**values)

    def test_seeds_then_adds_user_to_new_threads(self) -> None:
        added: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            state_file = str(Path(directory) / "threads.json")
            existing = [{"id": "1", "parent_id": "9", "name": "old"}]
            with_new = existing + [{"id": "2", "parent_id": "9", "name": "help me"}]

            poll_threads_once(
                self.make_config(),
                lambda _guild, _token: existing,
                lambda thread_id, _user, _token: added.append(thread_id),
                state_file,
            )
            self.assertEqual(added, [])

            poll_threads_once(
                self.make_config(),
                lambda _guild, _token: with_new,
                lambda thread_id, _user, _token: added.append(thread_id),
                state_file,
            )
        self.assertEqual(added, ["2"])

    def test_filters_by_support_channels(self) -> None:
        added: list[str] = []
        config = self.make_config(support_channel_ids=("9",))
        with tempfile.TemporaryDirectory() as directory:
            state_file = str(Path(directory) / "threads.json")
            poll_threads_once(
                config, lambda _g, _t: [], lambda *_args: None, state_file
            )
            poll_threads_once(
                config,
                lambda _g, _t: [
                    {"id": "5", "parent_id": "9", "name": "support"},
                    {"id": "6", "parent_id": "8", "name": "off-topic"},
                ],
                lambda thread_id, _user, _token: added.append(thread_id),
                state_file,
            )
        self.assertEqual(added, ["5"])

    def test_disabled_without_support_user(self) -> None:
        config = Config(discord_token="token", discord_channel_id="channel")
        poll_threads_once(
            config,
            lambda _g, _t: self.fail("should not fetch"),
            lambda *_args: self.fail("should not add"),
            "unused.json",
        )


class StateTests(unittest.TestCase):
    def test_keys_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_file = str(Path(directory) / "state.json")
            save_ids(state_file, "feed-a", ["1", "2"])
            save_ids(state_file, "feed-b", ["3"])
            self.assertEqual(load_ids(state_file, "feed-a"), (True, ["1", "2"]))
            self.assertEqual(load_ids(state_file, "feed-b"), (True, ["3"]))
            self.assertEqual(load_ids(state_file, "feed-c"), (False, []))
            data = json.loads(Path(state_file).read_text())
            self.assertEqual(set(data), {"feed-a", "feed-b"})


class ConditionalGetTests(unittest.TestCase):
    def setUp(self) -> None:
        api._conditional_cache.clear()

    def test_304_returns_cached_body(self) -> None:
        responses = iter(
            [
                (200, {"ETag": '"abc"'}, b"fresh body"),
                (304, {}, b""),
            ]
        )
        sent_headers: list[dict] = []

        def fake_http(request, attempts=2):
            sent_headers.append(dict(request.headers))
            return next(responses)

        with mock.patch.object(api, "_http", fake_http):
            first = api.conditional_get("https://example.com/feed", {})
            second = api.conditional_get("https://example.com/feed", {})
        self.assertEqual(first, b"fresh body")
        self.assertEqual(second, b"fresh body")
        self.assertEqual(sent_headers[1].get("If-none-match"), '"abc"')


class LlmConfigTests(unittest.TestCase):
    def base_env(self) -> dict[str, str]:
        return {"DISCORD_BOT_TOKEN": "token", "DISCORD_CHANNEL_ID": "123"}

    def test_litellm_proxy_settings(self) -> None:
        config = get_config(
            self.base_env()
            | {
                "LITELLM_PROXY_BASE_URL": "https://proxy.example.com",
                "LITELLM_PROXY_API_KEY": "sk-proxy",
            }
        )
        self.assertEqual(config.llm_base_url, "https://proxy.example.com")
        self.assertEqual(config.anthropic_api_key, "sk-proxy")

    def test_anthropic_key_still_works_without_proxy(self) -> None:
        config = get_config(self.base_env() | {"ANTHROPIC_API_KEY": "sk-ant"})
        self.assertIsNone(config.llm_base_url)
        self.assertEqual(config.anthropic_api_key, "sk-ant")


class NewConfigTests(unittest.TestCase):
    def base_env(self) -> dict[str, str]:
        return {"DISCORD_BOT_TOKEN": "token", "DISCORD_CHANNEL_ID": "123"}

    def test_blog_feed_on_by_default(self) -> None:
        config = get_config(self.base_env())
        self.assertEqual(
            config.feeds, (("blog", "https://docs.litellm.ai/blog/rss.xml"),)
        )
        self.assertEqual(config.updates_channel_id, "123")

    def test_support_user_requires_guild(self) -> None:
        env = self.base_env() | {"SUPPORT_USER_ID": "777"}
        with self.assertRaisesRegex(ValueError, "DISCORD_GUILD_ID"):
            get_config(env)

    def test_full_configuration(self) -> None:
        env = self.base_env() | {
            "DISCORD_GUILD_ID": "42",
            "SUPPORT_USER_ID": "777",
            "SUPPORT_CHANNEL_IDS": "9, 10",
            "UPDATES_CHANNEL_ID": "456",
            "SOCIAL_FEED_URLS": "https://nitter.net/litellm/rss",
        }
        config = get_config(env)
        self.assertEqual(config.support_channel_ids, ("9", "10"))
        self.assertEqual(config.updates_channel_id, "456")
        self.assertEqual(
            config.feeds,
            (
                ("blog", "https://docs.litellm.ai/blog/rss.xml"),
                ("social", "https://nitter.net/litellm/rss"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
