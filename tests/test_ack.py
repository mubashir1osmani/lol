"""Ack reactions and the Discord path that posts a verdict after them."""

import inspect
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from contributor_bot.ack import (
    EYES,
    FALLBACK_T,
    FALLBACK_Y,
    T_EMOJI_NAMES,
    Y_EMOJI_NAMES,
    acknowledge_message,
    apply_reactions,
    bind_ack_reactions,
    parse_ack_reactions,
    reset_ack_cache,
    resolve_ack_reactions,
)
from contributor_bot.config import Config, get_config
from contributor_bot.review_agent import tool_submit_verdict
from contributor_bot.interactions import handle_interaction
from contributor_bot.review_job import handle_channel_message, handle_thread_message
from contributor_bot.store import Store
import contributor_bot.review_job as review_job


def _config(**overrides):
    config = replace(
        Config("token", "123", "456", github_token="gh"),
        pr_review_enabled=True,
        review_channel_ids=("321",),
        review_model="openai/gpt-4o-mini",
        review_bot_logins=("cursor[bot]", "greptile-apps[bot]", "veria-ai[bot]"),
        review_maintainer_role_id="555",
        discord_guild_id="333",
        discord_application_id="444",
        review_cooldown_seconds=600,
    )
    return replace(config, **overrides) if overrides else config


class AckResolutionTests(unittest.TestCase):
    def tearDown(self):
        reset_ack_cache()

    def test_default_names_and_fallbacks(self):
        self.assertEqual(T_EMOJI_NAMES, ("t", "letter_t", "litellm_t", "regional_indicator_t"))
        self.assertEqual(Y_EMOJI_NAMES, ("y", "letter_y", "litellm_y", "regional_indicator_y"))
        api, line = resolve_ack_reactions(None, [])
        self.assertEqual(api, (EYES, FALLBACK_T, FALLBACK_Y))
        self.assertIn("no guild emoji named t, letter_t, litellm_t, regional_indicator_t", line)
        guild = [("Letter_T", "111111111111111111"), ("y", "222222222222222222")]
        api, line = resolve_ack_reactions(None, guild)
        self.assertEqual(api, (EYES, "Letter_T:111111111111111111", "y:222222222222222222"))
        self.assertIn("Letter_T:111111111111111111 (letter_t)", line)
        custom, _line = resolve_ack_reactions(("👀", "<:wave:333333333333333333>", "y"), guild)
        self.assertEqual(custom, (EYES, "wave:333333333333333333", "y:222222222222222222"))

    def test_explicit_list_rejects_a_broken_entry_and_logs_once(self):
        with self.assertRaises(ValueError):
            parse_ack_reactions("👀,,y")
        with self.assertRaises(ValueError):
            parse_ack_reactions("<:not-an-emoji>")
        config = _config(ack_reactions=None)
        with self.assertLogs(level="INFO") as logs:
            bind_ack_reactions(config, [("t", "111111111111111111"), ("y", "222222222222222222")])
            bind_ack_reactions(config, [("t", "111111111111111111"), ("y", "222222222222222222")])
        joined = "\n".join(logs.output)
        self.assertEqual(joined.count("Ack reactions:"), 1)
        self.assertIn("t:111111111111111111", joined)

    def test_a_forbidden_reaction_does_not_stop_the_next_one(self):
        reset_ack_cache()
        bind_ack_reactions(_config(), [("t", "111111111111111111"), ("y", "222222222222222222")])
        seen = []

        def react(channel_id, message_id, emoji, token):
            seen.append(emoji)
            if emoji == EYES:
                raise RuntimeError("discord.com returned HTTP 403")

        with mock.patch("contributor_bot.ack.add_reaction", side_effect=react):
            apply_reactions(_config(), "321", "654")
        self.assertEqual(seen, [EYES, "t:111111111111111111", "y:222222222222222222"])


class DiscordAckFlowTests(unittest.TestCase):
    def setUp(self):
        reset_ack_cache()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(str(Path(self.directory.name) / "roles.db"))
        self.addCleanup(self.store.close)
        self.config = _config(review_classify=False)
        bind_ack_reactions(self.config, [("t", "111111111111111111"), ("y", "222222222222222222")])

    def tearDown(self):
        reset_ack_cache()

    def _join_acks(self):
        for thread in threading.enumerate():
            if thread.name == "ack-react":
                thread.join(timeout=2)

    def test_review_request_reacts_then_posts_a_verdict_in_a_thread(self):
        discord = SimpleNamespace(reactions=[], posts=[], threads=[])

        def add_reaction(channel_id, message_id, emoji, token):
            discord.reactions.append((channel_id, message_id, emoji))

        def post_message(channel_id, token, message, mention_roles=None):
            discord.posts.append((channel_id, message["content"], mention_roles))
            return {"id": str(len(discord.posts)), "channel_id": channel_id}

        def start_thread(channel_id, message_id, token, name):
            discord.threads.append((channel_id, message_id, name))
            return {"id": "900"}

        order = []

        def session(config, store, number, thread_id, mode, history=""):
            for thread in threading.enumerate():
                if thread.name == "ack-react":
                    thread.join(timeout=2)
            order.append(("agent", tuple(discord.reactions)))
            from contributor_bot.review_agent import ReviewContext
            ctx = ReviewContext(config=config, store=store, number=number, thread_id=thread_id)
            tool_submit_verdict(ctx, {"summary": "the gate is clean", "blocking": []})
            from contributor_bot.review_agent import AgentResult
            return AgentResult("posted", verdict_posted=True, ready=True, head_sha="ab" * 20)

        message = {
            "id": "654",
            "author": {"id": "111", "bot": False},
            "content": "please review https://github.com/BerriAI/litellm/pull/12",
        }
        with mock.patch("contributor_bot.ack.add_reaction", side_effect=add_reaction), \
                mock.patch("contributor_bot.review_agent.post_message", side_effect=post_message), \
                mock.patch("contributor_bot.review_job.start_public_thread", side_effect=start_thread), \
                mock.patch("contributor_bot.review_job.fetch_recent_messages", return_value=[]), \
                mock.patch("contributor_bot.review_job.run_review_session", side_effect=session), \
                mock.patch("contributor_bot.review_agent.gather_snapshot", return_value=SimpleNamespace(
                    number=12, title="add a timeout", head_sha="ab" * 20, draft=False, merged=False,
                    state="open", mergeable=True, mergeable_state="clean", author_login="octocat",
                    author_id=5, base_sha="cd" * 20, reviews=[], comments=[], threads=[],
                    check_runs=[{"name": "pytest", "status": "completed", "conclusion": "success"}],
                    combined_state=None,
                )):
            handle_channel_message(self.config, self.store, "321", message)
            self._join_acks()
            for thread in threading.enumerate():
                if thread.name == "pr-review":
                    thread.join(timeout=3)
        self.assertEqual([item[2] for item in discord.reactions], [
            EYES, "t:111111111111111111", "y:222222222222222222",
        ])
        self.assertEqual(discord.reactions[0][0:2], ("321", "654"))
        self.assertEqual(discord.threads, [("321", "654", "pr 12 review")])
        self.assertTrue(order)
        self.assertEqual(order[0][1], tuple(discord.reactions))
        self.assertTrue(any("needs work" in content or "ready for maintainer" in content for _channel, content, _roles in discord.posts))
        self.assertNotIn("run_agent_loop", inspect.getsource(review_job))

    def test_recheck_without_a_message_does_not_react_and_check_again_does(self):
        self.store.save_review_thread(
            repository="BerriAI/litellm", number=12, channel_id="321", message_id="654",
            thread_id="900", created_at="2026-11-26T00:00:00+00:00",
        )
        order = []

        def note(*args):
            if args[-1]:
                order.append("react")

        with mock.patch("contributor_bot.review_job.acknowledge_message", side_effect=note), \
                mock.patch("contributor_bot.review_job._spawn"):
            response = handle_interaction(self.config, self.store, {
                "type": 2,
                "guild_id": "333",
                "application_id": "444",
                "channel_id": "900",
                "token": "tokentoken",
                "member": {"user": {"id": "111"}},
                "data": {"name": "recheck", "options": []},
                "channel": {"id": "900", "type": 11, "parent_id": "777"},
            })
        self.assertEqual(response["type"], 5)
        self.assertEqual(order, [])
        with mock.patch("contributor_bot.review_job.acknowledge_message", side_effect=note), \
                mock.patch("contributor_bot.review_job._spawn"):
            handle_interaction(self.config, self.store, {
                "type": 2,
                "guild_id": "333",
                "application_id": "444",
                "channel_id": "900",
                "token": "tokentoken",
                "member": {"user": {"id": "111"}},
                "message": {"id": "7777"},
                "data": {"name": "recheck", "options": []},
                "channel": {"id": "900", "type": 11, "parent_id": "777"},
            })
        self.assertEqual(order, ["react"])
        with mock.patch("contributor_bot.review_job.acknowledge_message", side_effect=note), \
                mock.patch("contributor_bot.review_job.perform_review", return_value={"status": "posted"}) as review:
            handle_thread_message(self.config, self.store, {
                "id": "8888",
                "channel_id": "900",
                "content": "check again",
                "author": {"id": "111", "bot": False},
            })
        self.assertEqual(order, ["react", "react"])
        review.assert_called_once()
        self.assertEqual(review.call_args.kwargs["source"], "recheck")

    def test_classifier_defaults_to_haiku(self):
        env = {
            "DISCORD_BOT_TOKEN": "token",
            "OSS_CONTRIBUTIONS_CHANNEL_ID": "123",
            "ANNOUNCEMENTS_CHANNEL_ID": "456",
            "PR_REVIEW_ENABLED": "true",
            "DISCORD_GUILD_ID": "333",
            "DISCORD_APPLICATION_ID": "444",
            "DISCORD_PUBLIC_KEY": "ab" * 32,
            "REVIEW_CHANNEL_IDS": "789",
            "REVIEW_MODEL": "anthropic/claude-sonnet-4-5",
        }
        config = get_config(env)
        self.assertEqual(config.review_classifier_model, "anthropic/claude-haiku-4-5")
        self.assertIsNone(config.ack_reactions)
        overridden = get_config(env | {"REVIEW_CLASSIFIER_MODEL": "openai/gpt-4o-mini", "ACK_REACTIONS": "👀,t,y"})
        self.assertEqual(overridden.review_classifier_model, "openai/gpt-4o-mini")
        self.assertEqual(overridden.ack_reactions, ("👀", "t", "y"))


class AcknowledgeStartsBeforeTheAgentTests(unittest.TestCase):
    def test_acknowledge_returns_before_the_reaction_finishes(self):
        started = threading.Event()
        release = threading.Event()

        def react(*args):
            started.set()
            release.wait(timeout=2)

        reset_ack_cache()
        config = _config()
        bind_ack_reactions(config, [])
        with mock.patch("contributor_bot.ack.add_reaction", side_effect=react):
            acknowledge_message(config, "321", "654")
            self.assertTrue(started.wait(timeout=2))
            release.set()
            for thread in threading.enumerate():
                if thread.name == "ack-react":
                    thread.join(timeout=2)
        reset_ack_cache()
