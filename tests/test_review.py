"""PR pre-review: detection, bot filtering, verdicts, and Discord splitting.

GitHub and litellm are mocked. These tests do not post or call the network.
"""

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock
from urllib.request import Request

from contributor_bot import api, review_github
from contributor_bot.config import Config, get_config
from contributor_bot.interactions import REVIEW_SLASH_COMMANDS, SLASH_COMMANDS, handle_interaction
from contributor_bot.review_agent import tool_submit_verdict
from contributor_bot.review_github import Snapshot, gather_snapshot
from contributor_bot.review_job import handle_channel_message, perform_review, poll_review_channels, schedule_webhook_recheck
from contributor_bot.review_llm import chunk_diff, complete_litellm, parse_review_json, run_llm_review
from contributor_bot.reviews import (
    DEFAULT_REVIEW_BOTS,
    asks_for_review,
    assess_gate,
    cooldown_remaining,
    discord_length,
    findings_from_comments,
    normalize_thread,
    recheck_number,
    render_status,
    review_chunks,
    should_review_message,
    split_discord_messages,
    summarize_comment,
    tag_line,
)
from contributor_bot.schedule import UTC, iso
from contributor_bot.store import Store

HEAD = "ab" * 20
OLD = "cd" * 20
BOTS = DEFAULT_REVIEW_BOTS


def _base_env():
    return {
        "DISCORD_BOT_TOKEN": "token",
        "OSS_CONTRIBUTIONS_CHANNEL_ID": "123",
        "ANNOUNCEMENTS_CHANNEL_ID": "456",
    }


def _review_env(**extra):
    return _base_env() | {
        "PR_REVIEW_ENABLED": "true",
        "DISCORD_GUILD_ID": "333",
        "DISCORD_APPLICATION_ID": "444",
        "DISCORD_PUBLIC_KEY": "ab" * 32,
        "REVIEW_CHANNEL_IDS": "789",
        "REVIEW_MODEL": "openai/gpt-4o-mini",
        **extra,
    }


def _review(login, sha=HEAD, body="no new issues", state="COMMENTED"):
    return {"user": {"login": login}, "commit_id": sha, "body": body, "state": state}


def _thread(login, body, *, resolved=False, outdated=False, path="litellm/cost_calculator.py", sha=HEAD):
    return {
        "resolved": resolved,
        "outdated": outdated,
        "path": path,
        "comments": [{"login": login, "body": body, "commit_id": sha}],
    }


def _gate(**overrides):
    payload = {
        "bots": BOTS,
        "head_sha": HEAD,
        "draft": False,
        "merged": False,
        "state": "open",
        "mergeable": True,
        "mergeable_state": "clean",
        "reviews": [_review(login) for login in BOTS],
        "comments": [],
        "threads": [],
        "check_runs": [{"name": "pytest", "status": "completed", "conclusion": "success"}],
        "combined_state": None,
    }
    payload.update(overrides)
    return assess_gate(**payload)


class DetectionTests(unittest.TestCase):
    def test_pull_urls_and_review_asks(self):
        url = "https://github.com/BerriAI/litellm/pull/12"
        message = {"content": f"please review {url}", "author": {"id": "111", "bot": False}}
        self.assertEqual(should_review_message(message, "BerriAI/litellm", require_ask=False), [12])
        self.assertEqual(should_review_message(message, "BerriAI/litellm", require_ask=True), [12])
        bare = {"content": url, "author": {"bot": False}}
        self.assertEqual(should_review_message(bare, "BerriAI/litellm", require_ask=False), [12])
        self.assertEqual(should_review_message(bare, "BerriAI/litellm", require_ask=True), [])
        self.assertTrue(asks_for_review("ptal on this"))
        self.assertTrue(asks_for_review("can someone take a look"))
        self.assertFalse(asks_for_review("shipped the docs today"))
        self.assertEqual(should_review_message({"content": "please review", "author": {"bot": False}}, "BerriAI/litellm", require_ask=False), [])

    def test_ignores_bots_other_repos_and_non_messages(self):
        url = "https://github.com/BerriAI/litellm/pull/12"
        self.assertEqual(should_review_message({"content": url, "author": {"bot": True}}, "BerriAI/litellm", require_ask=False), [])
        self.assertEqual(should_review_message({"content": url, "author": {"bot": False}, "webhook_id": "9"}, "BerriAI/litellm", require_ask=False), [])
        self.assertEqual(should_review_message({"content": url, "author": {"bot": False}, "type": 7}, "BerriAI/litellm", require_ask=False), [])
        other = {"content": "https://github.com/other/litellm/pull/4", "author": {"bot": False}}
        self.assertEqual(should_review_message(other, "BerriAI/litellm", require_ask=False), [])
        many = " ".join(f"https://github.com/BerriAI/litellm/pull/{n}" for n in range(1, 6))
        self.assertEqual(should_review_message({"content": many, "author": {"bot": False}}, "BerriAI/litellm", require_ask=False), [1, 2, 3])

    def test_webhook_recheck_events(self):
        self.assertEqual(recheck_number("pull_request", {"action": "synchronize", "pull_request": {"number": 5}}), 5)
        self.assertEqual(recheck_number("pull_request", {"action": "ready_for_review", "pull_request": {"number": 6}}), 6)
        self.assertIsNone(recheck_number("pull_request", {"action": "closed", "pull_request": {"number": 5}}))
        self.assertEqual(recheck_number("pull_request_review", {"action": "submitted", "pull_request": {"number": 9}}), 9)
        self.assertIsNone(recheck_number("pull_request_review", {"action": "dismissed", "pull_request": {"number": 9}}))
        self.assertIsNone(recheck_number("ping", {}))
        self.assertEqual(recheck_number("check_run", {
            "action": "completed",
            "check_run": {"pull_requests": [{"number": 15}]},
        }), 15)
        self.assertEqual(recheck_number("check_suite", {
            "action": "completed",
            "check_suite": {"pull_requests": [{"number": 16}]},
        }), 16)
        self.assertIsNone(recheck_number("check_run", {"action": "created", "check_run": {"pull_requests": [{"number": 15}]}}))


class BotFilterAndVerdictTests(unittest.TestCase):
    def test_comment_summary_strips_bot_markup(self):
        greptile = (
            '<a href=""><img alt="P1" src="https://greptile-static-assets.s3.amazonaws.com/badges/p1.svg"></a>'
            " **Batch results fail to load**  If deployment prices are missing"
        )
        self.assertEqual(summarize_comment(greptile), "Batch results fail to load If deployment prices are missing")
        bugbot = "<!-- BUGBOT_REVIEW --> ### Batch cost returns a float\n\n**Medium Severity**"
        self.assertTrue(summarize_comment(bugbot).startswith("Batch cost returns a float"))
        self.assertNotIn("BUGBOT", summarize_comment(bugbot))

    def test_graphql_thread_shape_and_human_threads_are_ignored(self):
        raw = {
            "isResolved": False,
            "isOutdated": False,
            "path": "litellm/proxy/server.py",
            "comments": {"nodes": [{
                "author": {"login": "veria-ai[bot]"},
                "body": "**Medium: Check fallback categories** before exempting",
                "commit": {"oid": HEAD},
            }]},
        }
        thread = normalize_thread(raw)
        self.assertEqual(thread["comments"][0]["login"], "veria-ai[bot]")
        self.assertFalse(thread["resolved"])
        gate = _gate(threads=[raw, _thread("someone", "nit", path="README.md")])
        self.assertFalse(gate.ready)
        self.assertTrue(any("veria-ai[bot]: unresolved on litellm/proxy/server.py" in reason for reason in gate.reasons))
        self.assertFalse(any(reason.startswith("someone") for reason in gate.reasons))

    def test_ready_when_bots_rereviewed_ci_green_and_mergeable(self):
        gate = _gate(threads=[
            _thread("cursor[bot]", "old", resolved=True),
            _thread("greptile-apps[bot]", "old", outdated=True),
        ])
        self.assertTrue(gate.ready)
        self.assertEqual(gate.reasons, ())
        text = render_status(gate, number=12, title="add a timeout", url="https://github.com/BerriAI/litellm/pull/12", head_sha=HEAD)
        self.assertEqual(text, "looks good")
        self.assertEqual(len(text.splitlines()), 1)
        self.assertNotIn("✅", text)

    def test_needs_work_lists_unresolved_bots_ci_draft_and_conflicts(self):
        stale = _review("cursor[bot]", body="<!-- BUGBOT_REVIEW --> <!-- BUGBOT_REVIEW_STALE --> stale")
        gate = _gate(
            draft=True,
            mergeable=False,
            mergeable_state="dirty",
            reviews=[stale, _review("greptile-apps[bot]", sha=OLD), _review("codecov[bot]")],
            threads=[_thread("greptile-apps[bot]", "**Batch results fail to load**", sha=OLD)],
            check_runs=[
                {"name": "pytest", "status": "completed", "conclusion": "failure"},
                {"name": "lint", "status": "in_progress", "conclusion": None},
                {"name": "docs", "status": "completed", "conclusion": "skipped"},
            ],
        )
        self.assertFalse(gate.ready)
        text = "\n".join(gate.reasons)
        self.assertIn("still a draft", text)
        self.assertIn("merge conflicts", text)
        self.assertIn("ci failing: pytest", text)
        self.assertNotIn("ci still running", text)
        self.assertIn("cursor[bot] has not re-reviewed the latest commit", text)
        self.assertIn("greptile-apps[bot]: unresolved on litellm/cost_calculator.py - Batch results fail to load", text)
        self.assertIn("veria-ai[bot] has not reviewed this pr", text)
        self.assertNotIn("codecov", text)
        rendered = render_status(gate, number=12, title="fix @everyone\nnow", url="https://github.com/BerriAI/litellm/pull/12", head_sha=HEAD)
        self.assertTrue(rendered.startswith("needs work\n- "))
        self.assertLessEqual(len(rendered.splitlines()) - 1, 5)
        self.assertNotIn("fix these before a maintainer looks:", rendered)
        self.assertNotIn("@everyone", rendered)

    def test_resolution_fallback_counts_only_latest_head_comments(self):
        comments = [
            {"user": {"login": "cursor[bot]"}, "commit_id": HEAD, "path": "a.py", "body": "real bug", "in_reply_to_id": None},
            {"user": {"login": "cursor[bot]"}, "commit_id": HEAD, "path": "a.py", "body": "thanks", "in_reply_to_id": 9},
            {"user": {"login": "cursor[bot]"}, "commit_id": OLD, "path": "a.py", "body": "old bug"},
            {"user": {"login": "codecov[bot]"}, "commit_id": HEAD, "path": "a.py", "body": "coverage"},
            {"user": {"login": "cursor[bot]"}, "commit_id": HEAD, "path": "a.py", "body": "<!-- BUGBOT_REVIEW_STALE --> ignore"},
        ]
        found = findings_from_comments(comments, BOTS, HEAD)
        self.assertEqual([item.summary for item in found], ["real bug"])
        gate = _gate(threads=None, comments=comments, reviews=[_review(login) for login in BOTS])
        self.assertFalse(gate.ready)
        self.assertIn("couldn't read review-thread resolution from github", gate.reasons)
        self.assertTrue(any("real bug" in reason for reason in gate.reasons))

    def test_closed_merged_and_ci_states(self):
        self.assertEqual(_gate(merged=True).reasons, ("already merged",))
        self.assertEqual(_gate(state="closed").reasons, ("pr is closed",))
        pending = _gate(check_runs=[{"name": "pytest", "status": "queued", "conclusion": None}])
        self.assertIn("ci still running: pytest", pending.reasons)
        missing = _gate(check_runs=[], combined_state=None)
        self.assertIn("ci has not reported yet", missing.reasons)
        unknown = _gate(check_runs=None, combined_state=None)
        self.assertIn("couldn't read ci checks", unknown.reasons)
        self.assertIn("mergeability isn't known yet", _gate(mergeable=None, mergeable_state=None).reasons)
        green_status = _gate(check_runs=None, combined_state="success")
        self.assertNotIn("couldn't read ci checks", green_status.reasons)
        self.assertTrue(green_status.ready)


class SplitAndTagTests(unittest.TestCase):
    def test_messages_split_on_the_discord_limit(self):
        short = split_discord_messages("needs work\n- ci failing: pytest")
        self.assertEqual(short, ["needs work\n- ci failing: pytest"])
        long = "a" * 4500
        parts = split_discord_messages(long, limit=2000)
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(parts), long)
        for part in parts:
            self.assertLessEqual(discord_length(part), 2000)
        lines = "\n".join(f"line {index} " + ("x" * 40) for index in range(80))
        chunks = split_discord_messages(lines)
        self.assertTrue(all(discord_length(chunk) <= 2000 for chunk in chunks))
        self.assertIn("line 0", chunks[0])
        emoji = "🙂" * 5
        emoji_parts = split_discord_messages(emoji, limit=4)
        self.assertEqual("".join(emoji_parts), emoji)
        self.assertTrue(all(discord_length(part) <= 4 for part in emoji_parts))
        self.assertEqual(split_discord_messages("   \n"), [])

    def test_review_chunks_and_tags(self):
        chunks = review_chunks("needs work\n- still a draft", "written review:\n" + ("risk " * 800))
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks), 8)
        self.assertTrue(all(discord_length(chunk) <= 2000 for chunk in chunks))
        self.assertIsNone(tag_line(None, "555", ready=False))
        self.assertEqual(tag_line("111", "555", ready=False), "<@111>")
        self.assertEqual(tag_line("111", "555", ready=True), "<@111> <@&555>")
        self.assertEqual(tag_line(None, "555", ready=True), "<@&555>")

    def test_cooldown(self):
        now = datetime(2026, 11, 26, tzinfo=timezone.utc)
        record = {"last_reviewed_at": iso(now - timedelta(seconds=100))}
        self.assertEqual(cooldown_remaining(record, 600, now), 500)
        self.assertEqual(cooldown_remaining(record, 600, now + timedelta(seconds=600)), 0)
        self.assertEqual(cooldown_remaining(None, 600, now), 0)


class LlmAndGithubMockTests(unittest.TestCase):
    def test_diff_chunks_and_json_parsing(self):
        files = [f"diff --git a/f{i}.py b/f{i}.py\n" + ("+x\n" * 30) for i in range(5)]
        chunks, omitted = chunk_diff("".join(files), chunk_chars=80, max_chunks=2)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(omitted)
        self.assertEqual(parse_review_json('```json\n{"summary": "adds a timeout", "risks": ["race"], "missing_tests": [], "checklist": ["run pytest"]}\n```')["summary"], "adds a timeout")
        self.assertEqual(parse_review_json("just some prose")["summary"], "just some prose")
        calls = []

        def complete(model, messages):
            calls.append((model, messages[-1]["content"]))
            return json.dumps({"summary": "adds a timeout", "risks": [], "missing_tests": ["none"], "checklist": ["run pytest"]})

        reviewed = run_llm_review("diff --git a/a.py b/a.py\n+timeout\n", "openai/gpt-4o-mini", complete)
        self.assertEqual(reviewed["summary"], "adds a timeout")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "openai/gpt-4o-mini")

    def test_completion_uses_the_litellm_sdk(self):
        fake = mock.Mock()
        fake.completion.return_value = {"choices": [{"message": {"content": "hi from the model"}}]}
        with mock.patch.dict("sys.modules", {"litellm": fake}):
            self.assertEqual(complete_litellm("openai/gpt-4o-mini", [{"role": "user", "content": "diff"}]), "hi from the model")
        fake.completion.assert_called_once()
        self.assertEqual(fake.completion.call_args.kwargs["model"], "openai/gpt-4o-mini")
        self.assertEqual(fake.completion.call_args.kwargs["temperature"], 0.2)

    def test_diff_fetch_is_a_get(self):
        with mock.patch.object(api, "_http", return_value=(200, {}, b"diff --git a/a.py b/a.py\n+x\n")) as http:
            text = api.fetch_pull_diff("BerriAI/litellm", 12, "token")
        request = http.call_args.args[0]
        self.assertIsInstance(request, Request)
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.get_header("Accept"), "application/vnd.github.diff")
        self.assertTrue(request.full_url.endswith("/pulls/12"))
        self.assertNotIn("merge", request.full_url)
        self.assertEqual(text.splitlines()[0], "diff --git a/a.py b/a.py")

    def test_gather_only_reads_reviews_checks_and_threads(self):
        calls = []

        def fake_get(url, token, pace=False):
            calls.append(url)
            if url.endswith("/pulls/12"):
                return {
                    "number": 12, "title": "add a timeout", "state": "open", "draft": False,
                    "merged": False, "mergeable": True, "mergeable_state": "clean",
                    "user": {"login": "octocat", "id": 5}, "head": {"sha": HEAD},
                }
            if "/reviews" in url or url.endswith("/comments") or "/comments?" in url:
                return []
            if "/check-runs" in url:
                return {"total_count": 1, "check_runs": [{"name": "pytest", "status": "completed", "conclusion": "success"}]}
            if url.endswith("/status"):
                return {"state": "pending", "total_count": 0, "statuses": []}
            raise AssertionError(url)

        def fake_graphql(token, query, variables):
            self.assertNotIn("mutation", query.lower())
            self.assertIn("reviewThreads", query)
            self.assertEqual(variables["number"], 12)
            return {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": [], "pageInfo": {"hasNextPage": False}}}}}}

        with mock.patch("contributor_bot.review_github.github_get_json", side_effect=fake_get), \
                mock.patch("contributor_bot.review_github.github_graphql", side_effect=fake_graphql), \
                mock.patch("contributor_bot.review_github.fetch_pull_diff", return_value="diff --git a/a.py b/a.py\n"), \
                mock.patch("contributor_bot.api.post_issue_comment", side_effect=AssertionError("github write")):
            snapshot = gather_snapshot("BerriAI/litellm", 12, "token")
        self.assertEqual(snapshot.head_sha, HEAD)
        self.assertEqual(snapshot.threads, [])
        self.assertEqual(snapshot.check_runs[0]["name"], "pytest")
        self.assertIsNone(snapshot.combined_state)
        self.assertTrue(calls)
        self.assertTrue(all("api.github.com" in url and "/merge" not in url for url in calls))
        with mock.patch("contributor_bot.review_github.github_graphql", return_value={"errors": [{"message": "nope"}]}):
            self.assertIsNone(review_github._threads("BerriAI/litellm", 12, "token"))


def _fake_sdk(on_run):
    """A stand-in for the Agents SDK. Tests never import openai-agents."""
    calls = {"n": 0, "model": None, "settings": {}}

    class LitellmModel:
        def __init__(self, model, **kwargs):
            calls["model"] = model

    class ModelSettings:
        def __init__(self, **kwargs):
            calls["settings"] = kwargs

    class Agent:
        def __init__(self, **kwargs):
            calls["instructions"] = kwargs.get("instructions")

    class Runner:
        @staticmethod
        async def run(agent, incoming, context=None, max_turns=None, hooks=None):
            calls["n"] += 1
            calls["max_turns"] = max_turns
            on_run(context)
            return type("Result", (), {"final_output": "done"})()

    def function_tool(func=None, **kwargs):
        def deco(fn):
            return fn
        if callable(func):
            return deco(func)
        return deco

    class RunHooks:
        pass

    class RunContextWrapper:
        pass

    sdk = type("SDK", (), {})()
    sdk.Agent = Agent
    sdk.Runner = Runner
    sdk.ModelSettings = ModelSettings
    sdk.LitellmModel = LitellmModel
    sdk.function_tool = function_tool
    sdk.RunContextWrapper = RunContextWrapper
    sdk.RunHooks = RunHooks
    sdk.MaxTurnsExceeded = type("MaxTurnsExceeded", (Exception,), {})
    sdk.set_tracing_disabled = lambda disabled: calls.__setitem__("tracing_disabled", disabled)
    return sdk, calls


class DiscordPostTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(str(Path(self.directory.name) / "roles.db"))
        self.addCleanup(self.store.close)
        self.config = replace(
            Config("token", "123", "456", github_token="gh"),
            pr_review_enabled=True,
            review_channel_ids=("321",),
            review_model="openai/gpt-4o-mini",
            review_bot_logins=BOTS,
            review_maintainer_role_id="555",
            review_cooldown_seconds=600,
            discord_guild_id="333",
            discord_application_id="444",
        )

    def _snapshot(self, **overrides):
        payload = dict(
            number=12, title="add a timeout", head_sha=HEAD, draft=False, merged=False,
            state="open", mergeable=True, mergeable_state="clean", author_login="octocat",
            author_id=5, reviews=[_review(login) for login in BOTS], comments=[], threads=[],
            check_runs=[{"name": "pytest", "status": "completed", "conclusion": "success"}],
            combined_state=None, diff="diff --git a/a.py b/a.py\n+timeout\n",
        )
        payload.update(overrides)
        return Snapshot(**payload)

    def test_review_posts_in_discord_and_does_not_write_to_github(self):
        self.store.link_account("111", "octocat", 5, "2026-11-01T00:00:00+00:00")
        verdict = {
            "verdict": "looks good",
            "findings": [],
        }
        sdk, calls = _fake_sdk(lambda context: tool_submit_verdict(context, verdict))

        with mock.patch("contributor_bot.review_agent.gather_snapshot", return_value=self._snapshot()) as gather, \
                mock.patch("contributor_bot.review_agent._load_sdk", return_value=sdk), \
                mock.patch("contributor_bot.review_job.fetch_recent_messages", return_value=[]), \
                mock.patch("contributor_bot.review_job.start_public_thread", return_value={"id": "900"}) as thread, \
                mock.patch("contributor_bot.review_agent.post_message", return_value={"id": "1", "channel_id": "900"}) as post, \
                mock.patch("contributor_bot.api.post_issue_comment", side_effect=AssertionError("github write")) as comment, \
                mock.patch("contributor_bot.api._http", side_effect=AssertionError("network")):
            result = perform_review(self.config, self.store, 12, channel_id="321", message_id="654", source="new")
        self.assertEqual(result["status"], "posted")
        self.assertEqual(result["thread_id"], "900")
        gather.assert_called_once()
        self.assertGreaterEqual(calls["n"], 1)
        self.assertEqual(calls["model"], "openai/gpt-4o-mini")
        self.assertGreaterEqual(calls["settings"]["max_tokens"], 4096)
        thread.assert_called_once()
        comment.assert_not_called()
        contents = [call.args[2]["content"] for call in post.call_args_list]
        self.assertEqual(contents[0], "looks good")
        self.assertNotIn("✅", "\n".join(contents))
        self.assertIn("<@111>", contents[-1])
        self.assertIn("<@&555>", contents[-1])
        self.assertEqual(post.call_args_list[-1].kwargs["mention_roles"], ["555"])
        for content in contents:
            self.assertLessEqual(discord_length(content), 2000)
        self.assertEqual(self.store.review_thread("BerriAI/litellm", 12)["thread_id"], "900")

    def test_needs_work_does_not_ping_the_maintainer_role(self):
        snapshot = self._snapshot(
            draft=True,
            reviews=[],
            check_runs=[{"name": "pytest", "status": "completed", "conclusion": "failure"}],
        )
        sdk, _calls = _fake_sdk(lambda context: tool_submit_verdict(context, {"summary": "draft", "blocking": []}))

        with mock.patch("contributor_bot.review_agent.gather_snapshot", return_value=snapshot), \
                mock.patch("contributor_bot.review_agent._load_sdk", return_value=sdk), \
                mock.patch("contributor_bot.review_job.fetch_recent_messages", return_value=[]), \
                mock.patch("contributor_bot.review_job.start_public_thread", return_value={"id": "900"}), \
                mock.patch("contributor_bot.review_agent.post_message", return_value={"id": "1", "channel_id": "900"}) as post:
            result = perform_review(self.config, self.store, 12, channel_id="321", message_id="654", source="new")
        self.assertEqual(result["status"], "posted")
        contents = [call.args[2]["content"] for call in post.call_args_list]
        self.assertTrue(contents[0].startswith("needs work"))
        self.assertIn("still a draft", contents[0])
        self.assertTrue(all(call.kwargs.get("mention_roles") in (None, []) for call in post.call_args_list))
        self.assertNotIn("<@&555>", "\n".join(contents))

    def test_recheck_respects_the_per_pr_cooldown(self):
        now = datetime.now(UTC)
        self.store.save_review_thread(
            repository="BerriAI/litellm", number=12, channel_id="321", message_id="654",
            thread_id="900", created_at=iso(now),
        )
        self.store.mark_review_done("BerriAI/litellm", 12, iso(now), HEAD)
        with mock.patch("contributor_bot.review_agent.run_review_session") as session, \
                mock.patch("contributor_bot.review_job._post") as posted:
            result = perform_review(self.config, self.store, 12, source="recheck")
        session.assert_not_called()
        posted.assert_called_once()
        self.assertEqual(result["status"], "cooldown")
        self.assertIn("<#900>", result["detail"])

    def test_first_poll_does_not_backfill_and_a_new_link_is_scheduled(self):
        scheduled = []
        message = {
            "id": "200",
            "type": 0,
            "author": {"id": "111", "bot": False},
            "content": "please review https://github.com/BerriAI/litellm/pull/12",
        }
        with mock.patch("contributor_bot.review_job.fetch_latest_message_id", return_value="100") as latest, \
                mock.patch("contributor_bot.review_job.fetch_channel_messages", return_value=[message]) as history, \
                mock.patch("contributor_bot.review_job.acknowledge_message") as ack, \
                mock.patch("contributor_bot.review_job._spawn", side_effect=lambda target: scheduled.append(target)):
            poll_review_channels(self.config, self.store)
            self.assertEqual(self.store.review_cursor("321"), "100")
            latest.assert_called_once()
            history.assert_not_called()
            self.assertEqual(scheduled, [])
            poll_review_channels(self.config, self.store)
        self.assertEqual(len(scheduled), 1)
        ack.assert_called_once_with(self.config, "321", "200")
        self.assertEqual(self.store.review_cursor("321"), "200")

    def test_webhook_recheck_only_updates_an_existing_thread(self):
        scheduled = []
        payload = {
            "action": "synchronize",
            "pull_request": {"number": 12},
            "repository": {"full_name": "BerriAI/litellm"},
        }
        with mock.patch("contributor_bot.review_job._spawn", side_effect=lambda target: scheduled.append(target)):
            schedule_webhook_recheck(self.config, self.store, "pull_request", payload)
            self.assertEqual(scheduled, [])
            turned_on = replace(self.config, review_webhook_recheck=True)
            schedule_webhook_recheck(turned_on, self.store, "pull_request", payload)
            self.assertEqual(scheduled, [])
            self.store.save_review_thread(
                repository="BerriAI/litellm", number=12, channel_id="321", message_id="654",
                thread_id="900", created_at="2026-11-26T00:00:00+00:00",
            )
            schedule_webhook_recheck(turned_on, self.store, "pull_request", payload)
            schedule_webhook_recheck(turned_on, self.store, "pull_request", {"action": "closed", "pull_request": {"number": 12}, "repository": {"full_name": "BerriAI/litellm"}})
            schedule_webhook_recheck(turned_on, self.store, "check_run", {
                "action": "completed",
                "check_run": {"pull_requests": [{"number": 12}]},
                "repository": {"full_name": "BerriAI/litellm"},
            })
        self.assertEqual(len(scheduled), 2)

    def test_channel_message_reuses_one_thread(self):
        self.store.save_review_thread(
            repository="BerriAI/litellm", number=12, channel_id="321", message_id="654",
            thread_id="900", created_at="2026-11-26T00:00:00+00:00",
        )
        self.store.mark_review_done("BerriAI/litellm", 12, iso(datetime.now(UTC)), HEAD)
        message = {"id": "201", "author": {"bot": False}, "content": "https://github.com/BerriAI/litellm/pull/12"}
        with mock.patch("contributor_bot.review_job._spawn") as spawn:
            handle_channel_message(self.config, self.store, "321", message)
        spawn.assert_not_called()


class ReviewConfigTests(unittest.TestCase):
    def test_review_is_off_unless_enabled(self):
        config = get_config(_base_env() | {"REVIEW_MODEL": "not a model", "PR_REVIEW_ENABLED": "false"})
        self.assertFalse(config.pr_review_enabled)
        self.assertEqual(config.review_bot_logins, ())
        self.assertIsNone(config.github_oauth_client_id)
        enabled = get_config(_review_env())
        self.assertEqual(enabled.review_bot_logins, DEFAULT_REVIEW_BOTS)
        self.assertEqual(enabled.review_channel_ids, ("789",))
        self.assertEqual(enabled.review_model, "openai/gpt-4o-mini")
        self.assertEqual(enabled.review_cooldown_seconds, 600)
        self.assertFalse(enabled.review_webhook_recheck)
        self.assertFalse(enabled.contributor_roles_enabled)
        self.assertNotIn("token", repr(enabled))
        with self.assertRaisesRegex(ValueError, "REVIEW_MODEL"):
            get_config(_review_env(REVIEW_MODEL=""))
        with self.assertRaisesRegex(ValueError, "REVIEW_CHANNEL_IDS"):
            get_config(_review_env(REVIEW_CHANNEL_IDS=""))
        with self.assertRaisesRegex(ValueError, "GITHUB_WEBHOOK_SECRET"):
            get_config(_review_env(REVIEW_WEBHOOK_RECHECK="true"))
        secret = get_config(_review_env(REVIEW_WEBHOOK_RECHECK="true", GITHUB_WEBHOOK_SECRET="webhook-secret-value"))
        self.assertEqual(secret.github_webhook_secret, "webhook-secret-value")
        self.assertNotIn("webhook-secret-value", repr(secret))
        custom = get_config(_review_env(REVIEW_BOT_LOGINS="Cursor[bot], greptile-apps[bot]"))
        self.assertEqual(custom.review_bot_logins, ("cursor[bot]", "greptile-apps[bot]"))

    def test_review_commands_stay_off_the_role_list_until_enabled(self):
        self.assertEqual([command["name"] for command in SLASH_COMMANDS], ["link-github", "unlink-github", "my-prs"])
        self.assertEqual([command["name"] for command in REVIEW_SLASH_COMMANDS], ["review-pr", "recheck"])
        config = get_config(_review_env(PR_REVIEW_ENABLED="false"))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(str(Path(directory.name) / "roles.db"))
        self.addCleanup(store.close)
        # Roles are off, so the guild check still needs the review flag's guild.
        # With the flag off, guild id is unset and the command is rejected first
        # only when roles/review did not load a guild. Use an enabled-roles-less
        # config that still has a guild by turning review on and then replacing the flag.
        configured = replace(get_config(_review_env()), pr_review_enabled=False)
        response = handle_interaction(configured, store, {
            "type": 2,
            "guild_id": "333",
            "application_id": "444",
            "member": {"user": {"id": "111"}},
            "data": {"name": "review-pr", "options": [{"name": "url", "value": "https://github.com/BerriAI/litellm/pull/12"}]},
        })
        self.assertIn("off", response["data"]["content"])
        self.assertEqual(config.pr_review_enabled, False)


if __name__ == "__main__":
    unittest.main()
