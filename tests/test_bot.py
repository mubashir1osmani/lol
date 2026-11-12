from dataclasses import replace
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request

from contributor_bot import api
from contributor_bot.config import Config, get_config
from contributor_bot.poller import extract_pr_numbers
from contributor_bot.state import load_state, save_state


def message(message_id, user_id="111", numbers=(1,), **extra):
    return {
        "id": str(message_id),
        "author": {"id": user_id, "bot": False},
        "content": " ".join(f"https://github.com/BerriAI/litellm/pull/{n}" for n in numbers),
        **extra,
    }


def pull(number=1, merged=True, login="octocat", **extra):
    return {"number": number, "merged": merged, "user": {"login": login}, **extra}


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.env = {
            "DISCORD_BOT_TOKEN": "token",
            "OSS_CONTRIBUTIONS_CHANNEL_ID": "123",
            "ANNOUNCEMENTS_CHANNEL_ID": "456",
        }

    def test_only_contributor_settings_are_loaded(self):
        config = get_config(self.env | {"MENTION_EVERYONE": "true", "ANTHROPIC_API_KEY": "unused"})
        self.assertEqual(config.github_repository, "BerriAI/litellm")
        self.assertEqual(config.poll_interval_seconds, 300)
        self.assertEqual(config.state_file, ".data/contributors.json")
        self.assertFalse(hasattr(config, "mention_everyone"))
        self.assertFalse(hasattr(config, "anthropic_api_key"))

    def test_required_channels_are_explicit_and_distinct(self):
        for key in self.env:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                get_config({name: value for name, value in self.env.items() if name != key})
        with self.assertRaisesRegex(ValueError, "must differ"):
            get_config(self.env | {"ANNOUNCEMENTS_CHANNEL_ID": "123"})
        with self.assertRaisesRegex(ValueError, "channel ID"):
            get_config(self.env | {"OSS_CONTRIBUTIONS_CHANNEL_ID": "#oss-contributions"})

    def test_invalid_poll_intervals_are_rejected(self):
        for interval in ("0", "-1", "nan", "inf", "oops"):
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, "positive"):
                get_config(self.env | {"POLL_INTERVAL_MINUTES": interval})

    def test_mapping_validation(self):
        config = get_config(self.env | {"GITHUB_DISCORD_MAP": '{"OctoCat":"789"}'})
        self.assertEqual(config.github_discord_map, {"octocat": "789"})
        for mapping in ('[]', '{"octocat":123}', '{"octocat":"@everyone"}', 'oops'):
            with self.subTest(mapping=mapping), self.assertRaisesRegex(ValueError, "GITHUB_DISCORD_MAP"):
                get_config(self.env | {"GITHUB_DISCORD_MAP": mapping})
        self.assertEqual(get_config(self.env).github_discord_map, {})


class ExtractionTests(unittest.TestCase):
    def test_markdown_queries_fragments_paths_and_embeds(self):
        data = message(1, content=(
            "[My PR](https://github.com/berriai/LiteLLM/pull/42). "
            "<https://github.com/BerriAI/litellm/pull/42/files#diff> "
            "https://github.com/BerriAI/litellm/pull/43?diff=split"
        ), embeds=[{"url": "https://github.com/BerriAI/litellm/pull/44"}])
        self.assertEqual(extract_pr_numbers(data, "BerriAI/litellm"), [42, 43, 44])

    def test_rejects_other_repositories_issues_and_lookalike_hosts(self):
        data = message(1, content=" ".join([
            "https://github.com/other/litellm/pull/1",
            "https://github.com/BerriAI/litellm/issues/2",
            "https://github.com.evil.test/BerriAI/litellm/pull/3",
            "https://evil.test/github.com/BerriAI/litellm/pull/4",
            "https://github.com@evil.test/BerriAI/litellm/pull/5",
            "https://evil@github.com/BerriAI/litellm/pull/6",
            "https://github.com/BerriAI/litellm/pull/7oops",
        ]))
        self.assertEqual(extract_pr_numbers(data, "BerriAI/litellm"), [])


class APITests(unittest.TestCase):
    def setUp(self):
        api._conditional_cache.clear()

    def test_complete_history_paginates_beyond_one_hundred_messages(self):
        pages = [[message(n) for n in range(250, 150, -1)],
                 [message(n) for n in range(150, 50, -1)],
                 [message(n) for n in range(50, 0, -1)]]
        with mock.patch.object(api, "_request_json", side_effect=pages) as fetch:
            messages = list(api.fetch_channel_messages("123", "token"))
        self.assertEqual(len(messages), 250)
        self.assertEqual(fetch.call_count, 3)
        self.assertIn("before=151", fetch.call_args_list[1].args[0].full_url)
        self.assertIn("before=51", fetch.call_args_list[2].args[0].full_url)

    def test_incremental_history_does_not_skip_more_than_one_page(self):
        pages = [[message(n) for n in range(250, 150, -1)],
                 [message(n) for n in range(150, 50, -1)]]
        with mock.patch.object(api, "_request_json", side_effect=pages) as fetch:
            messages = list(api.fetch_channel_messages("123", "token", "100"))
        self.assertEqual({int(m["id"]) for m in messages}, set(range(101, 251)))
        self.assertEqual(fetch.call_count, 2)

    def test_failed_history_request_propagates(self):
        with mock.patch.object(api, "_request_json", side_effect=RuntimeError("403")):
            with self.assertRaisesRegex(RuntimeError, "403"):
                list(api.fetch_channel_messages("123", "token"))

    def test_github_reads_pull_api_and_requires_a_merge_flag(self):
        body = json.dumps(pull()).encode()
        with mock.patch.object(api, "conditional_get", return_value=body) as fetch:
            self.assertTrue(api.fetch_pull_request("BerriAI/litellm", 1, "github-token")["merged"])
        url, headers = fetch.call_args.args
        self.assertEqual(url, "https://api.github.com/repos/BerriAI/litellm/pulls/1")
        self.assertEqual(headers["Authorization"], "Bearer github-token")
        with mock.patch.object(api, "conditional_get", return_value=b'{"state":"closed"}'):
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                api.fetch_pull_request("BerriAI/litellm", 1, None)

    def test_post_preserves_exact_user_allowlist_and_nonce(self):
        payload = {"content": "Thanks <@111>", "allowed_mentions": {"parse": [], "users": ["111"]}, "nonce": "abc", "enforce_nonce": True}
        receipt = {"id": "789", "channel_id": "456"}
        with mock.patch.object(api, "_request_json", return_value=receipt) as post:
            self.assertEqual(api.post_message("456", "token", payload), receipt)
        request = post.call_args.args[0]
        self.assertEqual(request.full_url, "https://discord.com/api/v10/channels/456/messages")
        self.assertEqual(request.method, "POST")
        self.assertEqual(json.loads(request.data), payload)
        self.assertEqual(request.headers["Authorization"], "Bot token")

    def test_conditional_requests_reuse_body_and_drop_stale_etag(self):
        responses = [(200, {"ETag": '"abc"'}, b"first"), (304, {}, b""), (200, {}, b"new"), (200, {}, b"last")]
        with mock.patch.object(api, "_http", side_effect=responses) as fetch:
            self.assertEqual(api.conditional_get("https://example.com/pr", {}), b"first")
            self.assertEqual(api.conditional_get("https://example.com/pr", {}), b"first")
            self.assertEqual(api.conditional_get("https://example.com/pr", {}), b"new")
            self.assertEqual(api.conditional_get("https://example.com/pr", {}), b"last")
        self.assertEqual(fetch.call_args_list[1].args[0].headers["If-none-match"], '"abc"')
        self.assertNotIn("If-none-match", fetch.call_args_list[3].args[0].headers)

    def test_rate_limit_defers_to_next_poll_without_immediate_retry(self):
        url = "https://api.github.com/repos/BerriAI/litellm/pulls/1"
        error = HTTPError(url, 403, "rate limit", {"X-RateLimit-Remaining": "0"}, io.BytesIO(b"rate limit"))
        with mock.patch.object(api, "_open_request", side_effect=error) as request:
            with self.assertRaises(api.TransientAPIError):
                api._http(Request(url))
        request.assert_called_once()

    def test_discord_short_rate_limit_waits_then_resumes(self):
        error = HTTPError("https://discord.com", 429, "rate limit", {}, io.BytesIO(b'{"retry_after":0.75}'))
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status, response.headers = 200, {}
        response.read.return_value = b"[]"
        with mock.patch.object(api, "_open_request", side_effect=[error, response]), mock.patch.object(api.time, "sleep") as sleep:
            self.assertEqual(api._http(Request("https://discord.com/api/v10/channels/123/messages")), (200, {}, b"[]"))
        sleep.assert_called_once_with(0.75)


class StateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "state.json"
        self.config = Config("token", "123", "456", state_file=str(self.path))

    def test_dedupe_history_is_not_truncated(self):
        state = load_state(self.config)
        state["pull_requests"] = {str(n): {"announced": True} for n in range(1, 1001)}
        state["contributor_ids"] = [str(n) for n in range(1, 1001)]
        save_state(str(self.path), state)
        self.assertEqual(load_state(self.config), state)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_corrupt_or_old_release_state_is_never_silently_reset(self):
        for content in ("{broken", "[]", '{"postedReleaseIds":["1"]}'):
            with self.subTest(content=content):
                self.path.write_text(content)
                with self.assertRaises(RuntimeError):
                    load_state(self.config)

    def test_state_cannot_be_reused_for_other_channels_or_repositories(self):
        save_state(str(self.path), load_state(self.config))
        for override in ({"announcements_channel_id": "999"}, {"contributions_channel_id": "999"}, {"github_repository": "other/repo"}):
            with self.subTest(override=override), self.assertRaisesRegex(RuntimeError, "different channels/repository"):
                load_state(replace(self.config, **override))


if __name__ == "__main__":
    unittest.main()
