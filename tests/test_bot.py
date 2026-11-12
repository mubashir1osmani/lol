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
from contributor_bot.poller import extract_pr_numbers, poll_once
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
        with self.assertRaisesRegex(ValueError, "requires"):
            get_config(self.env | {"CONTRIBUTOR_ATTRIBUTION": "github_mapping"})


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


class PollerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config = Config("token", "123", "456", state_file=str(Path(directory.name) / "state.json"))
        self.messages = mock.Mock(return_value=[])
        self.github = mock.Mock(side_effect=lambda _repo, number, _token: pull(number))
        self.post = mock.Mock()

    def poll(self):
        poll_once(self.config, self.messages, self.github, self.post)

    def test_first_run_thanks_all_merged_contributors_once_not_closed_prs(self):
        self.messages.return_value = [message(13, "222", (3,)), message(12, "111", (2,)), message(11)]
        self.github.side_effect = lambda _repo, n, _token: pull(n, merged=n != 3, state="closed")
        self.poll()
        self.post.assert_called_once()
        channel, token, payload = self.post.call_args.args
        self.assertEqual((channel, token), ("456", "token"))
        self.assertEqual(payload["allowed_mentions"], {"parse": [], "users": ["111"]})
        self.assertEqual(payload["content"].count("<@111>"), 1)
        self.assertIn("2 PRs", payload["content"])
        self.assertNotIn("<@222>", payload["content"])
        state = load_state(self.config)
        self.assertEqual(state["contributor_ids"], ["111"])
        self.assertEqual(state["last_message_id"], "13")
        self.assertFalse(state["pull_requests"]["3"]["announced"])

    def test_restarts_do_not_recheck_or_reannounce_merged_prs(self):
        self.messages.return_value = [message(10)]
        self.poll()
        self.messages.return_value = []
        self.poll()
        self.post.assert_called_once()
        self.github.assert_called_once_with("BerriAI/litellm", 1, None)
        self.messages.assert_called_with("123", "token", "10")

    def test_old_unmerged_pr_is_rechecked_without_new_messages(self):
        self.messages.return_value = [message(10)]
        self.github.side_effect = [pull(merged=False), pull(merged=True)]
        self.poll()
        self.post.assert_not_called()
        self.messages.return_value = []
        self.poll()
        self.post.assert_called_once()
        self.assertEqual(load_state(self.config)["contributor_ids"], ["111"])

    def test_reposts_credit_earliest_human_not_bot_or_webhook(self):
        self.messages.return_value = [
            message(15, "333"), message(14, "222"), message(13, "111"),
            message(12, author={"id": "999", "bot": True}),
            message(11, "888", webhook_id="12"),
        ]
        self.poll()
        self.assertEqual(self.post.call_args.args[2]["allowed_mentions"]["users"], ["111"])
        self.github.assert_called_once()
        self.messages.return_value = [message(16, "555")]
        self.poll()
        self.post.assert_called_once()

    def test_same_contributor_can_be_thanked_for_a_new_pr(self):
        self.messages.return_value = [message(10)]
        self.poll()
        self.messages.return_value = [message(11, numbers=(2,))]
        self.poll()
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(load_state(self.config)["contributor_ids"], ["111"])

    def test_mapping_credits_actual_github_author_not_sharer(self):
        self.config = replace(self.config, github_discord_map={"octocat": "789"})
        self.messages.return_value = [message(10)]
        self.github.side_effect = None
        self.github.return_value = pull(login="OctoCat")
        self.poll()
        self.assertEqual(self.post.call_args.args[2]["allowed_mentions"]["users"], ["789"])

    def test_mapping_only_mode_holds_unmapped_prs_until_mapping_added(self):
        self.config = replace(self.config, contributor_attribution="github_mapping", github_discord_map={"someone": "999"})
        self.messages.return_value = [message(10)]
        with self.assertLogs(level="WARNING"):
            self.poll()
        self.post.assert_not_called()
        self.config = replace(self.config, github_discord_map={"octocat": "789"})
        self.messages.return_value = []
        self.poll()
        self.post.assert_called_once()
        self.github.assert_called_once()

    def test_failed_github_lookup_does_not_lose_discovered_pr(self):
        self.messages.return_value = [message(10)]
        self.github.side_effect = [api.TransientAPIError("rate limit"), pull()]
        with self.assertLogs(level="WARNING"):
            self.poll()
        self.post.assert_not_called()
        self.messages.return_value = []
        self.poll()
        self.post.assert_called_once()

    def test_one_missing_pr_does_not_block_other_contributors(self):
        self.messages.return_value = [message(10, numbers=(1, 2))]
        self.github.side_effect = [RuntimeError("404"), pull(2)]
        with self.assertLogs(level="ERROR"):
            self.poll()
        self.post.assert_called_once()
        self.assertFalse(load_state(self.config)["pull_requests"]["1"]["announced"])

    def test_rate_limited_backfill_does_not_starve_later_prs(self):
        self.messages.return_value = [message(10, numbers=(1, 2, 3))]
        self.github.side_effect = [pull(1, merged=False), api.TransientAPIError("rate limit")]
        with self.assertLogs(level="WARNING"):
            self.poll()
        self.messages.return_value = []
        self.github.reset_mock()
        self.github.side_effect = [pull(2), pull(3), pull(1, merged=False)]
        self.poll()
        self.assertEqual([call.args[1] for call in self.github.call_args_list], [2, 3, 1])
        self.assertIn("2 PRs", self.post.call_args.args[2]["content"])

    def test_history_failure_does_not_advance_checkpoint(self):
        self.messages.return_value = [message(10)]
        self.poll()
        def failing_history(*_args):
            yield message(12, numbers=(2,))
            raise api.TransientAPIError("history interrupted")
        self.messages.side_effect = failing_history
        with self.assertRaises(api.TransientAPIError):
            self.poll()
        self.assertEqual(load_state(self.config)["last_message_id"], "10")
        self.post.assert_called_once()

    def test_failed_send_preserves_outbox_and_retries_same_nonce(self):
        self.messages.return_value = [message(10)]
        self.post.side_effect = [api.TransientAPIError("timeout"), None]
        with self.assertRaises(api.TransientAPIError):
            self.poll()
        state = load_state(self.config)
        self.assertEqual(state["contributor_ids"], [])
        self.assertFalse(state["pull_requests"]["1"]["announced"])
        self.assertEqual(len(state["outbox"]), 1)
        payload = self.post.call_args.args[2]
        self.messages.return_value = []
        self.poll()
        self.assertEqual(self.post.call_args.args[2], payload)
        self.assertTrue(payload["enforce_nonce"])
        self.assertLessEqual(len(payload["nonce"]), 25)
        self.assertEqual(load_state(self.config)["outbox"], [])
        self.assertEqual(load_state(self.config)["contributor_ids"], ["111"])

    def test_large_batches_are_bounded_and_resume_after_partial_failure(self):
        user_ids = [str(100000000000000000 + n) for n in range(1, 151)]
        self.messages.return_value = [message(n, user_id, (n,)) for n, user_id in enumerate(user_ids, 1)]
        self.post.side_effect = [None, api.TransientAPIError("failed second batch")]
        with self.assertRaises(api.TransientAPIError):
            self.poll()
        first_payload = self.post.call_args_list[0].args[2]
        retry_payload = self.post.call_args_list[1].args[2]
        self.assertEqual(load_state(self.config)["contributor_ids"], first_payload["allowed_mentions"]["users"])
        self.messages.return_value = []
        self.post.reset_mock(side_effect=True)
        self.poll()
        self.assertEqual(self.post.call_args_list[0].args[2], retry_payload)
        all_payloads = [first_payload] + [call.args[2] for call in self.post.call_args_list]
        mentioned = []
        for payload in all_payloads:
            self.assertLessEqual(len(payload["content"].encode("utf-16-le")) // 2, 2000)
            self.assertLessEqual(len(payload["allowed_mentions"]["users"]), 100)
            self.assertEqual(payload["allowed_mentions"]["parse"], [])
            mentioned.extend(payload["allowed_mentions"]["users"])
        self.assertEqual(mentioned, user_ids)
        self.assertEqual(load_state(self.config)["contributor_ids"], user_ids)

    def test_empty_channel_does_not_post(self):
        self.poll()
        self.post.assert_not_called()
        self.github.assert_not_called()
        self.assertEqual(load_state(self.config)["contributor_ids"], [])


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
        error = HTTPError("https://api.github.com", 403, "rate limit", {"X-RateLimit-Remaining": "0"}, io.BytesIO(b"rate limit"))
        with mock.patch.object(api, "urlopen", side_effect=error) as request:
            with self.assertRaises(api.TransientAPIError):
                api._http(Request("https://api.github.com"))
        request.assert_called_once()

    def test_discord_short_rate_limit_waits_then_resumes(self):
        error = HTTPError("https://discord.com", 429, "rate limit", {}, io.BytesIO(b'{"retry_after":0.75}'))
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status, response.headers = 200, {}
        response.read.return_value = b"[]"
        with mock.patch.object(api, "urlopen", side_effect=[error, response]), mock.patch.object(api.time, "sleep") as sleep:
            self.assertEqual(api._http(Request("https://discord.com")), (200, {}, b"[]"))
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
